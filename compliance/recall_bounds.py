r"""
Formal recall bounds under adversarial deletions
================================================

This module states **provable relationships** between graph connectivity
after hard deletes and worst-case k-NN recall for *greedy* / beam search
on the remaining graph, and provides empirical checkers used by tests and
``tests/benchmark.py``.

Model
-----
Consider the base layer of an HNSW-like graph as an undirected graph
\(G=(V,E)\) on active (non-deleted) nodes. Let \(G'\) be the graph after
deleting a set \(D \subset V\) and applying a rewiring rule R.

Definitions
~~~~~~~~~~~
* **Exact recall@k** for query \(q\):
  \( R_k(q) = |A_k(q) \\cap S_k(q)| / k \)
  where \(A_k\) is ANN result set and \(S_k\) is exact kNN among active nodes.

* **Navigability failure**: a node \(u\) is unreachable from entry \(s\) in \(G'\).

Theorem 1 (Soft-delete baseline)
--------------------------------
If deletions are *logical only* (tombstones) and the search algorithm never
returns tombstoned ids but **traverses** them, then the geometric graph on
active nodes is unchanged, so:

    E[R_k] = E[R_k^{pre}]   (up to finite-sample noise from query set)

when the same ANN algorithm and ef/k are used. Soft-delete is the
**upper envelope** for connectivity-preserving policies.

Theorem 2 (Hard delete without rewiring)
----------------------------------------
Let \(D\) be deleted. For each edge \(\{u,v\}\) with \(u \\in D\) or
\(v \\in D\), remove it. Let \(U\) be the set of active nodes unreachable
from entry \(s\) in the remaining graph. Then for any query whose exact
nearest neighbor \(t^* \\in U\), greedy search from \(s\) **cannot** return
\(t^*\) (no path). Hence:

    R_1(q) = 0  whenever  t^*(q) \\in U.

Averaging over uniform queries supported on active nodes:

    E[R_1]  \\le  1 - |U| / |V \\setminus D|

This is the **fragmentation upper bound** for unhealed hard delete.

Theorem 3 (Degree-bounded rewiring — MN-RU style)
-------------------------------------------------
Suppose after isolating \(q \\in D\), every former neighbor \(u \\in N(q)\)
is offered edges to other members of \(N(q)\) under max degree \(M\), and
at least the closest available pair is connected when both have residual
capacity (or successful swap). Then the subgraph induced by \(N(q)\) gains
edges that are a subgraph of the Delaunay graph on those points under L2
(heuristic, not full Delaunay).

**Guarantee (local):** If \(|N(q)| \\ge 2\) and at least one pair both have
degree \(< M\) before rewiring, then that pair becomes connected and the
number of connected components among \(N(q)\) does **not increase** relative
to the star through \(q\) after \(q\)'s removal *in the two-node case*, and
is non-increasing whenever the MN-RU pass inserts a spanning tree on
\(N(q)\) (possible when \(|N(q)|-1\) edges fit under capacity).

**Global formal bound (conservative):** Let \(\\alpha\\) be the fraction of
deleted mass \(|D|/|V|\) and assume the pre-delete base layer is
\(M\)-regular and expanders with Cheeger constant \(h > 0\). After random
deletes without heal, expected cut edges removed are \(\\Theta(\\alpha M)\).
With MN-RU local replacement, expected extra edges reinserted among
orphans are at least \(c \\cdot \\min(M, \\bar d_{orphan})\\) per deleted
node for some \(c \\in (0,1]\) depending on capacity. Empirically we track:

    gap(α) = E[R_k^{healed}] - E[R_k^{unhealed}]  \\ge  0

and

    unreach_healed(α)  \\le  unreach_unhealed(α)

which are the **testable** formal claims validated in
``tests/test_recall_bounds.py`` and the benchmark harness.

Theorem 4 (Adversarial deletes)
-------------------------------
An adversary that deletes a vertex cut of size \(b\) (balanced separator)
can force \(\\Omega(|V|)\) unreachability without rewiring. MN-RU only
repairs **local** stars of deleted nodes; it does **not** claim to restore
expansion against adaptive cut deletions. Therefore:

    For adaptive cut adversaries, only soft-delete or global rebuild
    (compact) restores formal navigability.

Operational corollary: after large adversarial or bulk deletes, call
``compact()`` / full rebuild in addition to MN-RU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FragmentationBound:
    """Theorem 2 style bound for unhealed deletes."""

    n_active: int
    n_unreachable: int
    recall_at_1_upper_bound: float

    @staticmethod
    def from_counts(n_active: int, n_unreachable: int) -> "FragmentationBound":
        if n_active <= 0:
            return FragmentationBound(0, 0, 0.0)
        u = min(n_unreachable, n_active)
        # E[R_1] <= 1 - |U|/|V_active|
        return FragmentationBound(
            n_active=n_active,
            n_unreachable=u,
            recall_at_1_upper_bound=1.0 - (u / float(n_active)),
        )


@dataclass
class RecallComparison:
    """Empirical gap between policies at a fixed delete fraction."""

    delete_fraction: float
    recall_soft: float
    recall_unhealed: float
    recall_healed: float
    unreach_unhealed: float
    unreach_healed: float

    def healed_dominates_unhealed_recall(self, tol: float = 1e-9) -> bool:
        return self.recall_healed + tol >= self.recall_unhealed

    def healed_dominates_unhealed_reach(self, tol: float = 1e-9) -> bool:
        return self.unreach_healed <= self.unreach_unhealed + tol

    def soft_is_upper_envelope(self, tol: float = 0.05) -> bool:
        """Soft recall should be >= healed within tolerance (noise)."""
        return self.recall_soft + tol >= self.recall_healed


def exact_recall_at_k(
    predicted: Sequence[Sequence[int]],
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Mean |pred ∩ gt| / |gt| over queries."""
    scores: list[float] = []
    for pred, gt in zip(predicted, ground_truth):
        gt_set = {int(x) for x in gt[:k] if int(x) >= 0}
        if not gt_set:
            continue
        pred_set = {int(x) for x in pred[:k] if int(x) >= 0}
        scores.append(len(pred_set & gt_set) / float(len(gt_set)))
    return float(np.mean(scores)) if scores else 0.0


def bruteforce_topk(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Exact L2 top-k among active rows."""
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        return np.full((queries.shape[0], k), -1, dtype=np.int64)
    active = data[active_idx]
    x_sq = np.einsum("ij,ij->i", active, active)
    out = np.empty((queries.shape[0], min(k, active_idx.size)), dtype=np.int64)
    for i, q in enumerate(queries):
        q = q.astype(np.float64)
        dist = x_sq + np.dot(q, q) - 2.0 * (active @ q)
        take = min(k, active_idx.size)
        part = np.argpartition(dist, take - 1)[:take]
        order = part[np.argsort(dist[part])]
        out[i] = active_idx[order]
    return out


def bfs_unreachable_fraction(
    adjacency: list[list[int]],
    active_mask: np.ndarray,
    entry: int = 0,
) -> tuple[float, int, int]:
    """Layer-0 adjacency as list of neighbor lists; active_mask bool."""
    active = set(int(x) for x in np.flatnonzero(active_mask))
    if not active:
        return 1.0, 0, 0
    start = entry if entry in active else next(iter(active))
    seen: set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen or u not in active:
            continue
        seen.add(u)
        if u < len(adjacency):
            for v in adjacency[u]:
                v = int(v)
                if v in active and v not in seen:
                    stack.append(v)
    n_act = len(active)
    n_reach = len(seen)
    return (n_act - n_reach) / float(n_act), n_reach, n_act


def assert_theorem2_bound(
    recall_at_1: float,
    n_active: int,
    n_unreachable: int,
    *,
    tol: float = 1e-6,
) -> FragmentationBound:
    """
    Check empirical recall@1 does not exceed Theorem 2 upper bound.
    """
    bound = FragmentationBound.from_counts(n_active, n_unreachable)
    if recall_at_1 > bound.recall_at_1_upper_bound + tol:
        raise AssertionError(
            f"recall@1={recall_at_1:.4f} exceeds fragmentation bound "
            f"{bound.recall_at_1_upper_bound:.4f} "
            f"(unreachable={n_unreachable}/{n_active})"
        )
    return bound
