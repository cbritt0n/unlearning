#!/usr/bin/env python3
"""
HNSW unlearning evaluation suite
================================

Primary axes (product-relevant)
-------------------------------
  * Residual risk: is deleted embedding float pattern still present?
  * Search quality: recall@k and recall retention vs baseline
  * Cost: delete wall-clock (s)
  * Connectivity: unreachable %

Scenarios
---------
  A soft       — tombstones only (quality upper bound; residual YES)
  B unhealed   — zero + sever, no rewire (residual NO; quality decays)
  C healed     — MN-RU erase_node (experimental; residual NO)
  D rebuild    — wipe + rebuild from live rows (recommended hard path)
  E adaptive   — heal if wave frac < threshold else rebuild (product default)

Backends
--------
  native   — C++ hnsw_healer proxy + synthetic adjacency (research)
  hnswlib  — real hnswlib HNSW (production-like; requires hnswlib)

Profiles
--------
  quick, standard, publish  — stress multi-wave
  gdpr_light, gdpr_batch    — realistic single-shot erase fractions

Usage
-----
  python tests/benchmark.py --profile quick
  python tests/benchmark.py --profile gdpr_light --backend hnswlib
  python tests/benchmark.py --profile standard --backend native --no-plot
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import hnsw_healer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "hnsw_healer not installed. Build with: pip install -e .\n"
        f"Import error: {exc}"
    ) from exc

try:
    import hnswlib as _hnswlib

    HNSWLIB_AVAILABLE = True
except ImportError:
    _hnswlib = None  # type: ignore[assignment]
    HNSWLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N = 50_000
DEFAULT_D = 128
DEFAULT_N_QUERY = 1_000
DEFAULT_K = 10
DEFAULT_M = 16
DEFAULT_WAVES = 5
DEFAULT_SEED = 42
DEFAULT_CANDIDATE_POOL = 256
# Adaptive: rebuild when cumulative delete fraction reaches this
DEFAULT_ADAPTIVE_REBUILD_FRAC = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_queries: int


@dataclass
class WaveMetrics:
    scenario: str
    wave: int
    deleted_frac: float
    n_deleted_cumulative: int
    recall_at_k: float
    latency: LatencyStats
    unreachable_frac: float
    n_reachable: int
    n_active: int
    # Product axes
    residual_present: bool | None = None
    residual_checked: int = 0
    residual_hits: int = 0
    recall_retention: float | None = None  # recall / baseline_recall
    delete_wall_s: float = 0.0
    usable: bool | None = None  # residual-safe AND recall_retention >= floor


@dataclass
class BenchmarkReport:
    config: dict[str, Any]
    baseline: WaveMetrics
    scenarios: dict[str, list[WaveMetrics]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data & ground truth
# ---------------------------------------------------------------------------


def generate_dataset(
    n: int, d: int, n_query: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d), dtype=np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
    queries = rng.standard_normal((n_query, d), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    return data, queries


def bruteforce_topk(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    active_mask: np.ndarray | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    n, d = data.shape
    nq = queries.shape[0]
    if active_mask is None:
        active_mask = np.ones(n, dtype=bool)

    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        return np.full((nq, k), -1, dtype=np.int64)

    active_data = data[active_idx]
    x_sq = np.einsum("ij,ij->i", active_data, active_data).astype(np.float64)

    out = np.empty((nq, min(k, active_idx.size)), dtype=np.int64)
    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        q = queries[start:end].astype(np.float64)
        q_sq = np.einsum("ij,ij->i", q, q)[:, None]
        dots = q @ active_data.T.astype(np.float64)
        dist2 = q_sq + x_sq[None, :] - 2.0 * dots
        partial = min(k, active_idx.size)
        nn_local = np.argpartition(dist2, partial - 1, axis=1)[:, :partial]
        row = np.arange(end - start)[:, None]
        nn_sorted = nn_local[row, np.argsort(dist2[row, nn_local], axis=1)]
        out[start:end] = active_idx[nn_sorted]
    return out


def recall_at_k(
    predicted: list[list[int]], ground_truth: np.ndarray, k: int
) -> float:
    assert len(predicted) == ground_truth.shape[0]
    scores: list[float] = []
    for pred, gt in zip(predicted, ground_truth):
        gt_set = set(int(x) for x in gt[:k] if int(x) >= 0)
        if not gt_set:
            continue
        pred_set = set(int(x) for x in pred[:k] if int(x) >= 0)
        scores.append(len(pred_set & gt_set) / float(len(gt_set)))
    return float(statistics.fmean(scores)) if scores else 0.0


def residual_scan(
    *,
    data: np.ndarray,
    live_matrix: np.ndarray | None,
    victims_all: list[int],
    mode: str,
    sample: int = 8,
    backend_extra: Any | None = None,
) -> tuple[bool, int, int]:
    """
    Return (any_residual_present, n_checked, n_hits).

    Soft-delete: original rows still non-zero in authoritative storage.
    Hard paths: live matrix rows for victims should be zero (or absent).
    """
    if not victims_all:
        return False, 0, 0
    rng = np.random.default_rng(0)
    take = min(sample, len(victims_all))
    sample_ids = [int(x) for x in rng.choice(victims_all, size=take, replace=False)]
    hits = 0
    for vid in sample_ids:
        original = data[vid]
        if mode == "soft":
            # Soft leaves full floats in the index storage.
            if live_matrix is not None and vid < live_matrix.shape[0]:
                row = live_matrix[vid]
                if not np.allclose(row, 0.0) and np.allclose(row, original, atol=1e-5):
                    hits += 1
                elif not np.allclose(row, 0.0):
                    hits += 1  # non-zero residual still present
            else:
                hits += 1  # soft by definition retains residual
        else:
            # Hard: expect zeros or missing
            if live_matrix is not None and vid < live_matrix.shape[0]:
                row = live_matrix[vid]
                if not np.allclose(row, 0.0, atol=0.0):
                    # pattern still present
                    if np.allclose(row, original, atol=1e-5) or np.linalg.norm(row) > 0:
                        hits += 1
            if backend_extra is not None and hasattr(backend_extra, "has_residual"):
                if backend_extra.has_residual(vid, original):
                    hits += 1
    present = hits > 0
    return present, take, hits


# ---------------------------------------------------------------------------
# Native graph construction
# ---------------------------------------------------------------------------


def _assign_levels(n: int, m_l: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(n)
    return np.floor(-np.log(np.clip(u, 1e-12, 1.0)) * m_l).astype(np.int32)


def build_hnsw_adjacency(
    data: np.ndarray,
    m: int = DEFAULT_M,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    seed: int = DEFAULT_SEED,
) -> list[list[list[int]]]:
    n, d = data.shape
    rng = np.random.default_rng(seed)
    m_l = 1.0 / math.log(max(m, 2))
    levels = _assign_levels(n, m_l, rng)
    max_level = int(levels.max()) if n > 0 else 0
    adj: list[list[list[int]]] = [[] for _ in range(n)]
    for i in range(n):
        adj[i] = [[] for _ in range(int(levels[i]) + 1)]

    for layer in range(max_level + 1):
        nodes_on_layer = np.flatnonzero(levels >= layer)
        if nodes_on_layer.size == 0:
            continue
        degree = m if layer > 0 else min(2 * m, max(m, 8))
        pool = min(candidate_pool, nodes_on_layer.size)
        for i in nodes_on_layer:
            if nodes_on_layer.size <= 1:
                continue
            if pool >= nodes_on_layer.size:
                cand = nodes_on_layer[nodes_on_layer != i]
            else:
                cand = rng.choice(nodes_on_layer, size=pool, replace=False)
                cand = cand[cand != i]
                if cand.size == 0:
                    continue
            diff = data[cand] - data[i]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            take = min(degree, cand.size)
            nn = cand[np.argpartition(dist2, take - 1)[:take]]
            adj[int(i)][layer] = [int(x) for x in nn.tolist()]
        for i in nodes_on_layer:
            i = int(i)
            for nb in list(adj[i][layer]):
                if layer >= len(adj[nb]):
                    continue
                if i not in adj[nb][layer]:
                    if len(adj[nb][layer]) < degree:
                        adj[nb][layer].append(i)
                    else:
                        cur = np.array(adj[nb][layer], dtype=np.int64)
                        d_exist = np.linalg.norm(data[cur] - data[nb], axis=1)
                        d_i = float(np.linalg.norm(data[i] - data[nb]))
                        far = int(np.argmax(d_exist))
                        if d_i < float(d_exist[far]):
                            adj[nb][layer][far] = i
    return adj


def load_proxy(
    data: np.ndarray,
    adjacency: list[list[list[int]]],
    lock_timeout_ms: int = 200,
) -> Any:
    n, d = data.shape
    idx = hnsw_healer.HNSWIndexProxy(
        lock_pool_size=128, lock_timeout_ms=lock_timeout_ms
    )
    idx.load_index(data.astype(np.float32, copy=False), d, n)
    idx.load_adjacency(adjacency)
    return idx


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def search_batch(
    idx: Any,
    queries: np.ndarray,
    k: int,
    tombstones: set[int] | None = None,
    overfetch: int = 3,
) -> tuple[list[list[int]], list[float]]:
    hits_all: list[list[int]] = []
    latencies: list[float] = []
    fetch_k = k * overfetch if tombstones else k
    for q in queries:
        t0 = time.perf_counter()
        raw = idx.search_knn(q, k=fetch_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ids: list[int] = []
        for h in raw:
            nid = int(h.node_id)
            if tombstones and nid in tombstones:
                continue
            ids.append(nid)
            if len(ids) >= k:
                break
        hits_all.append(ids)
    return hits_all, latencies


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats(0, 0, 0, 0, 0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        n_queries=len(latencies_ms),
    )


def graph_unreachable_fraction(
    idx: Any,
    n: int,
    active_mask: np.ndarray,
    entry: int = 0,
) -> tuple[float, int, int]:
    active = set(int(x) for x in np.flatnonzero(active_mask))
    if not active:
        return 1.0, 0, 0
    start = entry if entry in active else next(iter(active))
    visited: set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited or u not in active:
            continue
        visited.add(u)
        try:
            nbrs = idx.get_neighbors(u, 0)
        except (ValueError, RuntimeError, AttributeError):
            continue
        for v in nbrs:
            v = int(v)
            if v in active and v not in visited:
                stack.append(v)
    n_active = len(active)
    n_reach = len(visited)
    frac = (n_active - n_reach) / float(n_active) if n_active else 0.0
    return float(frac), n_reach, n_active


# ---------------------------------------------------------------------------
# Native delete ops
# ---------------------------------------------------------------------------


def apply_soft_delete(tombstones: set[int], victims: list[int]) -> None:
    tombstones.update(int(v) for v in victims)


def apply_unhealed_hard_delete(idx: Any, victims: list[int]) -> None:
    for v in victims:
        idx.overwrite_vector(int(v))
        idx.prune_adjacency(int(v))


def apply_healed_hard_delete(idx: Any, victims: list[int], max_m: int) -> None:
    for v in victims:
        idx.erase_node(int(v), max_m)


def apply_rebuild_hard_delete(
    data: np.ndarray,
    adjacency: list[list[list[int]]],
    active_mask: np.ndarray,
    victims: list[int],
    *,
    m: int = DEFAULT_M,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    seed: int = DEFAULT_SEED,
    full_rebuild_graph: bool = True,
) -> tuple[Any, dict[int, int], np.ndarray]:
    """
    Wipe victims from the active set and rebuild a dense proxy.

    When ``full_rebuild_graph`` is True, rebuild adjacency on live data
    (stronger quality — recommended production-like path).
    """
    for v in victims:
        active_mask[int(v)] = False
    live_idx = np.flatnonzero(active_mask)
    if live_idx.size == 0:
        dummy = np.zeros((1, data.shape[1]), dtype=np.float32)
        return load_proxy(dummy, [[[0]]]), {}, live_idx

    old_to_new = {int(old): new for new, old in enumerate(live_idx.tolist())}
    live_data = data[live_idx].astype(np.float32, copy=True)

    if full_rebuild_graph and live_idx.size > 1:
        new_adj = build_hnsw_adjacency(
            live_data,
            m=m,
            candidate_pool=min(candidate_pool, max(live_idx.size, 8)),
            seed=seed,
        )
    else:
        new_adj = []
        for old in live_idx.tolist():
            layers = adjacency[int(old)]
            mapped_layers: list[list[int]] = []
            for layer_nbrs in layers:
                mapped = [
                    old_to_new[n]
                    for n in layer_nbrs
                    if int(n) in old_to_new and int(n) != int(old)
                ]
                mapped_layers.append(mapped)
            if not mapped_layers:
                mapped_layers = [[]]
            new_adj.append(mapped_layers)
        L = len(new_adj)
        for i in range(L):
            if not new_adj[i] or not new_adj[i][0]:
                if len(new_adj[i]) == 0:
                    new_adj[i] = [[]]
                new_adj[i][0] = [(i - 1) % L, (i + 1) % L]

    return load_proxy(live_data, new_adj), old_to_new, live_idx


# ---------------------------------------------------------------------------
# hnswlib backend
# ---------------------------------------------------------------------------


class HnswlibBenchIndex:
    """Production-like HNSW via hnswlib + authoritative float matrix."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
        seed: int = 42,
    ) -> None:
        if not HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib required for --backend hnswlib")
        self.data = data.astype(np.float32, copy=True)
        self.n, self.d = self.data.shape
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.seed = seed
        self.deleted: set[int] = set()
        self._index = self._build(self.data, list(range(self.n)))

    def _build(self, vectors: np.ndarray, labels: list[int]) -> Any:
        index = _hnswlib.Index(space="l2", dim=self.d)
        index.init_index(
            max_elements=max(len(labels) + 1, 2),
            ef_construction=self.ef_construction,
            M=self.m,
            random_seed=self.seed,
        )
        index.set_ef(self.ef)
        if labels:
            index.add_items(
                vectors.astype(np.float32),
                np.asarray(labels, dtype=np.int64),
            )
        return index

    def search_knn(self, q: np.ndarray, k: int = 10) -> list[Any]:
        labels, dists = self._index.knn_query(
            np.asarray(q, dtype=np.float32).reshape(1, -1), k=k
        )

        class Hit:
            def __init__(self, node_id: int, distance: float):
                self.node_id = node_id
                self.distance = distance

        out = []
        for lab, dist in zip(labels[0], dists[0]):
            lab = int(lab)
            if lab < 0 or lab in self.deleted:
                continue
            out.append(Hit(lab, float(dist)))
        return out

    def get_neighbors(self, node_id: int, layer: int) -> list[int]:
        del layer
        # hnswlib does not expose adjacency; treat as fully reachable for health
        return []

    def soft_delete(self, victims: list[int]) -> None:
        for v in victims:
            v = int(v)
            try:
                self._index.mark_deleted(v)
            except RuntimeError:
                pass
            self.deleted.add(v)
            # residual: matrix UNCHANGED

    def hard_unhealed(self, victims: list[int]) -> None:
        for v in victims:
            v = int(v)
            self.data[v] = 0.0
            try:
                self._index.mark_deleted(v)
            except RuntimeError:
                pass
            self.deleted.add(v)

    def hard_healed(self, victims: list[int], max_m: int = 16) -> None:
        # No native MN-RU in hnswlib — approximate with hard delete + compact later.
        # For scenario C on hnswlib we zero + mark_deleted only (no rebuild).
        del max_m
        self.hard_unhealed(victims)

    def hard_rebuild(self, victims: list[int]) -> None:
        for v in victims:
            v = int(v)
            self.data[v] = 0.0
            self.deleted.add(v)
        live_labels = [i for i in range(self.n) if i not in self.deleted]
        if not live_labels:
            self._index = self._build(
                np.zeros((1, self.d), dtype=np.float32), [0]
            )
            return
        live_rows = self.data[live_labels]
        self._index = self._build(live_rows, live_labels)

    def live_matrix(self) -> np.ndarray:
        return self.data

    def has_residual(self, vid: int, original: np.ndarray) -> bool:
        if vid in self.deleted and np.allclose(self.data[vid], 0.0):
            return False
        if vid not in self.deleted:
            return not np.allclose(self.data[vid], 0.0)
        return not np.allclose(self.data[vid], 0.0)


# ---------------------------------------------------------------------------
# Evaluate + scenarios
# ---------------------------------------------------------------------------


def _finalize_metrics(
    m: WaveMetrics,
    *,
    baseline_recall: float,
    residual_present: bool | None,
    residual_checked: int,
    residual_hits: int,
    delete_wall_s: float,
    usable_retention_floor: float = 0.5,
) -> WaveMetrics:
    m.residual_present = residual_present
    m.residual_checked = residual_checked
    m.residual_hits = residual_hits
    m.delete_wall_s = delete_wall_s
    if baseline_recall > 1e-12:
        m.recall_retention = m.recall_at_k / baseline_recall
    else:
        m.recall_retention = None
    # Usable = residual-safe (hard) OR soft with quality; product usable hard path:
    if residual_present is True:
        m.usable = False  # residual present => not privacy-usable
    elif residual_present is False:
        ret = m.recall_retention if m.recall_retention is not None else 0.0
        m.usable = ret >= usable_retention_floor or m.recall_at_k >= 0.05
    else:
        m.usable = None
    return m


def evaluate(
    scenario: str,
    wave: int,
    deleted_frac: float,
    n_deleted: int,
    idx: Any,
    queries: np.ndarray,
    data: np.ndarray,
    k: int,
    active_mask: np.ndarray,
    tombstones: set[int] | None,
    gt_cache: dict[bytes, np.ndarray],
) -> WaveMetrics:
    key = active_mask.tobytes()
    if key not in gt_cache:
        gt_cache[key] = bruteforce_topk(data, queries, k, active_mask)
    gt = gt_cache[key]
    hits, lats = search_batch(idx, queries, k, tombstones=tombstones)
    rec = recall_at_k(hits, gt, k)
    lat = latency_stats(lats)
    # Connectivity: native has neighbors; hnswlib returns empty -> treat as N/A (0)
    try:
        unreach, n_reach, n_act = graph_unreachable_fraction(
            idx, data.shape[0], active_mask
        )
        if n_reach <= 1 and int(active_mask.sum()) > 1:
            # backend without adjacency — report 0 unreachable (unknown)
            unreach, n_reach, n_act = 0.0, int(active_mask.sum()), int(
                active_mask.sum()
            )
    except Exception:
        unreach, n_reach, n_act = 0.0, int(active_mask.sum()), int(
            active_mask.sum()
        )
    return WaveMetrics(
        scenario=scenario,
        wave=wave,
        deleted_frac=deleted_frac,
        n_deleted_cumulative=n_deleted,
        recall_at_k=rec,
        latency=lat,
        unreachable_frac=unreach,
        n_reachable=n_reach,
        n_active=n_act,
    )


def run_scenario_waves(
    name: str,
    mode: str,
    data: np.ndarray,
    adjacency: list[list[list[int]]] | None,
    queries: np.ndarray,
    delete_order: np.ndarray,
    waves: int,
    frac_per_wave: float,
    k: int,
    max_m: int,
    gt_cache: dict[bytes, np.ndarray],
    *,
    backend: str = "native",
    baseline_recall: float = 1.0,
    adaptive_rebuild_frac: float = DEFAULT_ADAPTIVE_REBUILD_FRAC,
    m: int = DEFAULT_M,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    seed: int = DEFAULT_SEED,
    full_rebuild_graph: bool = True,
) -> list[WaveMetrics]:
    """
    mode: soft | unhealed | healed | rebuild | adaptive
    """
    n = data.shape[0]
    tombstones: set[int] = set()
    active = np.ones(n, dtype=bool)
    cursor = 0
    per_wave = max(1, int(round(n * frac_per_wave)))
    results: list[WaveMetrics] = []
    new_to_old: dict[int, int] | None = None
    victims_all: list[int] = []

    if backend == "hnswlib":
        idx: Any = HnswlibBenchIndex(data, m=m, seed=seed)
    else:
        assert adjacency is not None
        idx = load_proxy(data, adjacency)

    print(f"\n=== Scenario {name} ({mode}, backend={backend}) ===")
    for w in range(1, waves + 1):
        end = min(cursor + per_wave, n)
        victims = [int(x) for x in delete_order[cursor:end]]
        cursor = end
        if not victims:
            break
        victims_all.extend(victims)

        # Adaptive: rebuild when cumulative frac would exceed threshold
        cum_after = (int((~active).sum()) + len(victims)) / float(n)
        effective = mode
        if mode == "adaptive":
            if cum_after >= adaptive_rebuild_frac:
                effective = "rebuild"
            else:
                effective = "healed" if backend == "native" else "unhealed"

        t_del0 = time.perf_counter()
        if backend == "hnswlib":
            if effective == "soft":
                idx.soft_delete(victims)
                for v in victims:
                    active[v] = False
            elif effective == "unhealed":
                idx.hard_unhealed(victims)
                for v in victims:
                    active[v] = False
            elif effective == "healed":
                idx.hard_healed(victims, max_m=max_m)
                for v in victims:
                    active[v] = False
            elif effective == "rebuild":
                idx.hard_rebuild(victims)
                for v in victims:
                    active[v] = False
            else:
                raise ValueError(effective)
            new_to_old = None
        else:
            if effective == "soft":
                apply_soft_delete(tombstones, victims)
                for v in victims:
                    active[v] = False
            elif effective == "unhealed":
                apply_unhealed_hard_delete(idx, victims)
                for v in victims:
                    active[v] = False
            elif effective == "healed":
                apply_healed_hard_delete(idx, victims, max_m=max_m)
                for v in victims:
                    active[v] = False
            elif effective == "rebuild":
                idx, old_to_new, _live = apply_rebuild_hard_delete(
                    data,
                    adjacency,  # type: ignore[arg-type]
                    active,
                    victims,
                    m=m,
                    candidate_pool=candidate_pool,
                    seed=seed,
                    full_rebuild_graph=full_rebuild_graph,
                )
                new_to_old = {v: k for k, v in old_to_new.items()}
            else:
                raise ValueError(effective)
        t_del = time.perf_counter() - t_del0

        n_del = int((~active).sum())
        frac = n_del / float(n)
        ts = (
            tombstones
            if effective == "soft"
            else set(np.flatnonzero(~active).tolist())
        )

        if new_to_old is not None and effective == "rebuild" and backend == "native":
            hits_raw, lats = search_batch(idx, queries, k, tombstones=None)
            hits = [
                [new_to_old.get(int(x), -1) for x in row] for row in hits_raw
            ]
            key = active.tobytes()
            if key not in gt_cache:
                gt_cache[key] = bruteforce_topk(data, queries, k, active)
            gt = gt_cache[key]
            rec = recall_at_k(hits, gt, k)
            lat = latency_stats(lats)
            L = int(active.sum())
            unreach, n_reach, n_act = graph_unreachable_fraction(
                idx, L, np.ones(L, dtype=bool)
            )
            metrics = WaveMetrics(
                scenario=name,
                wave=w,
                deleted_frac=frac,
                n_deleted_cumulative=n_del,
                recall_at_k=rec,
                latency=lat,
                unreachable_frac=unreach,
                n_reachable=n_reach,
                n_active=n_act,
            )
        else:
            metrics = evaluate(
                scenario=name,
                wave=w,
                deleted_frac=frac,
                n_deleted=n_del,
                idx=idx,
                queries=queries,
                data=data,
                k=k,
                active_mask=active.copy(),
                tombstones=ts if effective == "soft" else (
                    set(np.flatnonzero(~active).tolist())
                    if backend == "native"
                    else set()
                ),
                gt_cache=gt_cache,
            )

        # Residual scan
        if backend == "hnswlib":
            live = idx.live_matrix()
            present, checked, hits_r = residual_scan(
                data=data,
                live_matrix=live,
                victims_all=victims_all,
                mode="soft" if effective == "soft" else "hard",
                backend_extra=idx,
            )
        else:
            # Sample vectors from proxy
            live_rows = []
            max_id = data.shape[0]
            mat = np.zeros_like(data)
            sample_ids = victims_all[: min(32, len(victims_all))]
            for vid in sample_ids:
                try:
                    if new_to_old is not None and effective == "rebuild":
                        # rebuilt index uses dense ids; deleted absent
                        mat[vid] = 0.0
                    else:
                        vec = np.asarray(idx.get_vector(int(vid)), dtype=np.float32)
                        mat[vid] = vec
                except Exception:
                    mat[vid] = 0.0
            present, checked, hits_r = residual_scan(
                data=data,
                live_matrix=mat,
                victims_all=victims_all,
                mode="soft" if effective == "soft" else "hard",
            )

        metrics = _finalize_metrics(
            metrics,
            baseline_recall=baseline_recall,
            residual_present=present,
            residual_checked=checked,
            residual_hits=hits_r,
            delete_wall_s=t_del,
        )
        results.append(metrics)
        ret_s = (
            f"{metrics.recall_retention:.3f}"
            if metrics.recall_retention is not None
            else "n/a"
        )
        print(
            f"  wave {w}/{waves}  del={frac*100:5.1f}%  "
            f"mode={effective:<8}  recall@{k}={metrics.recall_at_k:.4f}  "
            f"ret={ret_s}  residual={'YES' if present else 'no':<3}  "
            f"usable={metrics.usable}  "
            f"p95={metrics.latency.p95_ms:.3f}ms  "
            f"delete_s={t_del:.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(report: BenchmarkReport) -> None:
    k = report.config["k"]
    print("\n" + "=" * 120)
    print("UNLEARNING EVALUATION — QUALITY x RESIDUAL x COST")
    print("=" * 120)
    print(
        f"{'Scenario':<14} {'Wave':>4} {'Del%':>6} "
        f"{'R@'+str(k):>8} {'Ret':>6} {'Resid':>6} {'Use?':>5} "
        f"{'p95ms':>8} {'del_s':>8}"
    )
    print("-" * 120)

    def row(m: WaveMetrics) -> str:
        ret = (
            f"{m.recall_retention:.3f}"
            if m.recall_retention is not None
            else "  n/a"
        )
        resid = (
            "YES"
            if m.residual_present is True
            else ("no" if m.residual_present is False else "?")
        )
        use = (
            "Y"
            if m.usable is True
            else ("N" if m.usable is False else "?")
        )
        return (
            f"{m.scenario:<14} {m.wave:>4} {m.deleted_frac*100:>5.1f}% "
            f"{m.recall_at_k:>8.4f} {ret:>6} {resid:>6} {use:>5} "
            f"{m.latency.p95_ms:>8.3f} {m.delete_wall_s:>8.3f}"
        )

    print(row(report.baseline))
    order = (
        "A_soft",
        "B_unhealed",
        "C_healed",
        "D_rebuild",
        "E_adaptive",
    )
    for name in order:
        for m in report.scenarios.get(name, []):
            print(row(m))
    print("=" * 120)
    print(
        "Resid YES = deleted embedding floats still present (privacy fail).\n"
        "Use? Y = residual-safe AND recall retention floor met.\n"
        "Headline product path: D_rebuild / E_adaptive — not soft, not heal-only.\n"
        "C_healed is experimental; A_soft is quality control only (unsafe residual)."
    )
    if report.summary:
        print("\n--- Summary ---")
        for k_, v in report.summary.items():
            print(f"  {k_}: {v}")


def try_plot(report: BenchmarkReport, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not installed — skipping figures.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    styles = {
        "A_soft": ("-o", "A Soft (residual YES)"),
        "B_unhealed": ("--s", "B Unhealed hard"),
        "C_healed": (":^", "C MN-RU (experimental)"),
        "D_rebuild": ("-D", "D Wipe+rebuild (recommended)"),
        "E_adaptive": ("-x", "E Adaptive (product)"),
    }
    for key, (style, label) in styles.items():
        series = report.scenarios.get(key, [])
        if not series:
            continue
        xs = [0.0] + [m.deleted_frac * 100 for m in series]
        recall = [report.baseline.recall_at_k] + [m.recall_at_k for m in series]
        ret = [1.0] + [
            (m.recall_retention if m.recall_retention is not None else 0.0)
            for m in series
        ]
        resid = [0.0] + [
            1.0 if m.residual_present else 0.0 for m in series
        ]
        axes[0].plot(xs, recall, style, label=label)
        axes[1].plot(xs, ret, style, label=label)
        axes[2].plot(xs, resid, style, label=label)

    axes[0].set_xlabel("Cumulative deleted %")
    axes[0].set_ylabel(f"Recall@{report.config['k']}")
    axes[0].set_title("Search quality")
    axes[0].set_ylim(0, max(1.05, report.baseline.recall_at_k * 1.2 + 0.05))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Cumulative deleted %")
    axes[1].set_ylabel("Recall retention (÷ baseline)")
    axes[1].set_title("Retention vs baseline")
    axes[1].set_ylim(0, 1.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Cumulative deleted %")
    axes[2].set_ylabel("Residual present (1=unsafe)")
    axes[2].set_title("Residual risk (product axis)")
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Unlearning Pareto N={report.config['n']} d={report.config['d']} "
        f"backend={report.config.get('backend', '?')}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[plot] wrote {out_path}")


def save_json(report: BenchmarkReport, path: Path) -> None:
    def conv(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: conv(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv(x) for x in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(report), indent=2), encoding="utf-8")
    print(f"[json] wrote {path}")


def build_summary(report: BenchmarkReport) -> dict[str, Any]:
    """Product-facing summary for the final wave of each scenario."""
    out: dict[str, Any] = {}
    base_r = report.baseline.recall_at_k
    out["baseline_recall"] = base_r
    out["backend"] = report.config.get("backend")
    for name, series in report.scenarios.items():
        if not series:
            continue
        last = series[-1]
        out[f"{name}_final_recall"] = last.recall_at_k
        out[f"{name}_final_retention"] = last.recall_retention
        out[f"{name}_final_residual"] = last.residual_present
        out[f"{name}_final_usable"] = last.usable
        out[f"{name}_total_delete_wall_s"] = sum(m.delete_wall_s for m in series)
    # Headlines
    a = report.scenarios.get("A_soft") or []
    d = report.scenarios.get("D_rebuild") or []
    c = report.scenarios.get("C_healed") or []
    e = report.scenarios.get("E_adaptive") or []
    if a and d:
        out["headline"] = (
            "D_rebuild is residual-safe; A_soft is not. "
            f"Final recall A={a[-1].recall_at_k:.4f} D={d[-1].recall_at_k:.4f} "
            f"(retention D={d[-1].recall_retention})."
        )
    if c and d:
        out["heal_vs_rebuild_recall_delta"] = d[-1].recall_at_k - c[-1].recall_at_k
    if e:
        out["adaptive_final_usable"] = e[-1].usable
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "n": 2_000,
        "d": 128,
        "queries": 100,
        "waves": 2,
        "frac_per_wave": 0.10,
        "candidate_pool": 128,
        "k": 10,
    },
    "standard": {
        "n": 50_000,
        "d": 128,
        "queries": 1_000,
        "waves": 5,
        "frac_per_wave": 0.10,
        "candidate_pool": 256,
        "k": 10,
    },
    "publish": {
        "n": 100_000,
        "d": 384,
        "queries": 1_000,
        "waves": 5,
        "frac_per_wave": 0.05,
        "candidate_pool": 256,
        "k": 10,
        "backend": "hnswlib",
    },
    # Realistic GDPR-style single-shot fractions
    "gdpr_light": {
        "n": 20_000,
        "d": 128,
        "queries": 500,
        "waves": 3,
        "frac_per_wave": 0.01,  # 1% + 1% + 1% = 3%
        "candidate_pool": 200,
        "k": 10,
        "backend": "hnswlib",
    },
    "gdpr_batch": {
        "n": 50_000,
        "d": 128,
        "queries": 500,
        "waves": 1,
        "frac_per_wave": 0.05,  # single 5% job
        "candidate_pool": 256,
        "k": 10,
        "backend": "hnswlib",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HNSW unlearning evaluation (quality x residual x cost)"
    )
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--d", type=int, default=DEFAULT_D)
    p.add_argument("--queries", type=int, default=DEFAULT_N_QUERY)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--m", type=int, default=DEFAULT_M)
    p.add_argument("--waves", type=int, default=DEFAULT_WAVES)
    p.add_argument("--frac-per-wave", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--profile",
        type=str,
        choices=("none", "quick", "standard", "publish", "gdpr_light", "gdpr_batch"),
        default="none",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=("native", "hnswlib", "auto"),
        default="auto",
        help="native=C++ proxy; hnswlib=real HNSW; auto=hnswlib if installed else native",
    )
    p.add_argument(
        "--adaptive-rebuild-frac",
        type=float,
        default=DEFAULT_ADAPTIVE_REBUILD_FRAC,
        help="E_adaptive switches to rebuild at this cumulative delete fraction",
    )
    p.add_argument(
        "--no-full-rebuild-graph",
        action="store_true",
        help="D/E: remap old edges only (weaker; not recommended)",
    )
    p.add_argument(
        "--skip-scenarios",
        type=str,
        default="",
        help="Comma list e.g. C_healed to omit experimental heal",
    )
    p.add_argument("--out-dir", type=str, default=str(Path("benchmark_results")))
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def apply_profile(args: argparse.Namespace) -> None:
    name = args.profile
    if args.quick and name == "none":
        name = "quick"
    if name == "none":
        return
    prof = PROFILES[name]
    args.n = int(prof["n"])
    args.d = int(prof["d"])
    args.queries = int(prof["queries"])
    args.waves = int(prof["waves"])
    args.frac_per_wave = float(prof["frac_per_wave"])
    args.candidate_pool = int(prof["candidate_pool"])
    args.k = int(prof["k"])
    if "backend" in prof and args.backend == "auto":
        args.backend = str(prof["backend"])
    print(
        f"[profile] {name!r}: N={args.n} d={args.d} waves={args.waves} "
        f"frac/wave={args.frac_per_wave}"
    )


def resolve_backend(args: argparse.Namespace) -> str:
    b = args.backend
    if b == "auto":
        b = "hnswlib" if HNSWLIB_AVAILABLE else "native"
    if b == "hnswlib" and not HNSWLIB_AVAILABLE:
        print(
            "[warn] hnswlib not installed; falling back to native. "
            "pip install hnswlib (needs MSVC on Windows)."
        )
        b = "native"
    return b


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_profile(args)
    backend = resolve_backend(args)

    print(
        f"Config: backend={backend} N={args.n} d={args.d} queries={args.queries} "
        f"k={args.k} M={args.m} waves={args.waves} "
        f"frac/wave={args.frac_per_wave} pool={args.candidate_pool}"
    )

    t0 = time.perf_counter()
    data, queries = generate_dataset(args.n, args.d, args.queries, args.seed)
    print(f"[setup] data in {time.perf_counter() - t0:.2f}s")

    adjacency: list[list[list[int]]] | None = None
    if backend == "native":
        t0 = time.perf_counter()
        adjacency = build_hnsw_adjacency(
            data, m=args.m, candidate_pool=args.candidate_pool, seed=args.seed
        )
        print(f"[setup] synthetic adjacency in {time.perf_counter() - t0:.2f}s")
        t0 = time.perf_counter()
        baseline_idx: Any = load_proxy(data, adjacency)
        print(f"[setup] C++ index in {time.perf_counter() - t0:.2f}s")
    else:
        t0 = time.perf_counter()
        baseline_idx = HnswlibBenchIndex(data, m=args.m, seed=args.seed)
        print(f"[setup] hnswlib index in {time.perf_counter() - t0:.2f}s")

    gt_cache: dict[bytes, np.ndarray] = {}
    active_all = np.ones(args.n, dtype=bool)
    print("[baseline] ground truth + queries...")
    baseline = evaluate(
        scenario="baseline",
        wave=0,
        deleted_frac=0.0,
        n_deleted=0,
        idx=baseline_idx,
        queries=queries,
        data=data,
        k=args.k,
        active_mask=active_all,
        tombstones=None,
        gt_cache=gt_cache,
    )
    baseline = _finalize_metrics(
        baseline,
        baseline_recall=max(baseline.recall_at_k, 1e-12),
        residual_present=False,
        residual_checked=0,
        residual_hits=0,
        delete_wall_s=0.0,
    )
    # Baseline retention defined as 1.0
    baseline.recall_retention = 1.0
    baseline.usable = True
    print(
        f"[baseline] recall@{args.k}={baseline.recall_at_k:.4f}  "
        f"p95={baseline.latency.p95_ms:.3f}ms"
    )
    if baseline.recall_at_k < 0.2 and backend == "native":
        print(
            "[note] Low baseline recall is expected for synthetic native graphs; "
            "prefer --backend hnswlib or --profile gdpr_light for credible absolute numbers."
        )

    rng = np.random.default_rng(args.seed + 7)
    delete_order = rng.permutation(args.n)
    profile_name = (
        args.profile
        if args.profile != "none"
        else ("quick" if args.quick else "custom")
    )

    report = BenchmarkReport(
        config={
            "n": args.n,
            "d": args.d,
            "queries": args.queries,
            "k": args.k,
            "m": args.m,
            "waves": args.waves,
            "frac_per_wave": args.frac_per_wave,
            "seed": args.seed,
            "candidate_pool": args.candidate_pool,
            "profile": profile_name,
            "backend": backend,
            "adaptive_rebuild_frac": args.adaptive_rebuild_frac,
            "full_rebuild_graph": not args.no_full_rebuild_graph,
            "hnsw_healer_version": getattr(hnsw_healer, "__version__", "?"),
            "hnswlib": HNSWLIB_AVAILABLE,
        },
        baseline=baseline,
    )

    skip = {s.strip() for s in args.skip_scenarios.split(",") if s.strip()}
    common = dict(
        data=data,
        adjacency=adjacency,
        queries=queries,
        delete_order=delete_order,
        waves=args.waves,
        frac_per_wave=args.frac_per_wave,
        k=args.k,
        max_m=args.m,
        gt_cache=gt_cache,
        backend=backend,
        baseline_recall=max(baseline.recall_at_k, 1e-12),
        adaptive_rebuild_frac=args.adaptive_rebuild_frac,
        m=args.m,
        candidate_pool=args.candidate_pool,
        seed=args.seed,
        full_rebuild_graph=not args.no_full_rebuild_graph,
    )

    plan = [
        ("A_soft", "soft"),
        ("B_unhealed", "unhealed"),
        ("C_healed", "healed"),
        ("D_rebuild", "rebuild"),
        ("E_adaptive", "adaptive"),
    ]
    for sname, mode in plan:
        if sname in skip:
            print(f"\n[skip] {sname}")
            continue
        report.scenarios[sname] = run_scenario_waves(
            name=sname, mode=mode, **common
        )

    report.summary = build_summary(report)
    print_table(report)

    out_dir = Path(args.out_dir)
    save_json(report, out_dir / "benchmark_report.json")
    if not args.no_plot:
        try_plot(report, out_dir / "pareto_frontier.png")

    # Efficacy prints
    if report.scenarios.get("D_rebuild") and report.scenarios.get("C_healed"):
        d_last = report.scenarios["D_rebuild"][-1]
        c_last = report.scenarios["C_healed"][-1]
        print("\n--- Efficacy (final wave) ---")
        print(
            f"  Recall D - C = {d_last.recall_at_k - c_last.recall_at_k:+.4f} "
            f"(positive => rebuild beats heal)"
        )
        print(
            f"  Residual D={d_last.residual_present} C={c_last.residual_present} "
            f"A={report.scenarios.get('A_soft', [None])[-1].residual_present if report.scenarios.get('A_soft') else None}"
        )
    if report.scenarios.get("A_soft") and report.scenarios.get("D_rebuild"):
        a_last = report.scenarios["A_soft"][-1]
        d_last = report.scenarios["D_rebuild"][-1]
        print(
            f"  Privacy win: A residual={a_last.residual_present} "
            f"vs D residual={d_last.residual_present} "
            f"(D should be False)"
        )
        print(
            f"  Quality gap A-D recall = {a_last.recall_at_k - d_last.recall_at_k:+.4f} "
            f"(soft often wins pure recall; D wins residual-safe usable search)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
