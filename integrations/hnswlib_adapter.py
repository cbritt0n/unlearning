"""
hnswlib hard-delete adapter
===========================

Enterprise Python stacks (and many Chroma paths) sit on **hnswlib**. Soft
delete there is ``mark_deleted(label)`` — the float row remains in the
binary. This adapter adds:

1. **Physical zero** of the authoritative float matrix (and optional native
   MN-RU heal on a mirrored ``HNSWIndexProxy``).
2. **hnswlib ``mark_deleted``** so ANN search stops returning the id.
3. **``compact()``** — rebuild hnswlib from non-deleted rows so residual
   floats are not left in the on-disk/in-memory index blob.

Install optional extra::

    pip install hnsw-healer[hnswlib]
    # or: pip install hnswlib
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from integrations.backends import BackendEraseResult
from integrations.id_registry import CollectionIdRegistry

logger = logging.getLogger(__name__)

try:
    import hnswlib

    HNSWLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    hnswlib = None  # type: ignore[assignment]
    HNSWLIB_AVAILABLE = False

try:
    import hnsw_healer

    HEALER_AVAILABLE = True
except ImportError:  # pragma: no cover
    hnsw_healer = None  # type: ignore[assignment]
    HEALER_AVAILABLE = False


class HnswlibHardDeleteAdapter:
    """
    Dual-structure store: dense float matrix + hnswlib index + optional heal mirror.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    space:
        hnswlib space: ``l2``, ``cosine``, or ``ip``.
    max_elements:
        hnswlib capacity.
    M, ef_construction, ef:
        Standard hnswlib HNSW hyperparameters.
    enable_heal_mirror:
        If True and ``hnsw_healer`` is importable, maintain a parallel
        ``HNSWIndexProxy`` for MN-RU graph healing on hard delete.
    collection:
        Default collection name for backend protocol calls.
    """

    def __init__(
        self,
        dim: int,
        *,
        space: str = "l2",
        max_elements: int = 10_000,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        enable_heal_mirror: bool = True,
        collection: str = "default",
        registry: CollectionIdRegistry | None = None,
        seed: int = 42,
    ) -> None:
        if not HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is required for HnswlibHardDeleteAdapter. "
                "Install with: pip install hnswlib"
            )
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = int(dim)
        self.space = space
        self.max_elements = int(max_elements)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef = int(ef)
        self.collection = collection
        self.registry = registry or CollectionIdRegistry()
        self.registry.ensure_collection(collection)
        self._seed = seed

        self._deleted: set[int] = set()
        self._labels: list[int] = []
        # Authoritative residual-sensitive storage (row = label index path).
        # We keep a dense map label -> row via _label_to_row.
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._label_to_row: dict[int, int] = {}
        self._row_to_label: list[int] = []

        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            random_seed=seed,
        )
        self._index.set_ef(ef)

        self._heal_mirror = None
        if enable_heal_mirror and HEALER_AVAILABLE:
            self._heal_mirror = hnsw_healer.HNSWIndexProxy(
                lock_pool_size=64, lock_timeout_ms=100
            )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add(
        self,
        external_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        collection: str | None = None,
    ) -> list[int]:
        """
        Add vectors under business ids. Returns allocated internal labels.
        """
        coll = collection or self.collection
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"expected dim={self.dim}, got {vecs.shape[1]}"
            )
        if len(external_ids) != vecs.shape[0]:
            raise ValueError("external_ids and vectors length mismatch")

        labels: list[int] = []
        for ext, row in zip(external_ids, vecs):
            label = self.registry.register(coll, ext)
            if label in self._label_to_row and label not in self._deleted:
                raise ValueError(
                    f"label {label} already present; drop or use new id"
                )
            self._append_row(label, row)
            labels.append(label)

        active_labels = np.array(
            [lab for lab in labels if lab not in self._deleted],
            dtype=np.int64,
        )
        if active_labels.size:
            rows = np.stack(
                [self._vectors[self._label_to_row[int(lab)]] for lab in active_labels]
            )
            self._index.add_items(rows, active_labels)

        self._sync_heal_mirror()
        return labels

    def _append_row(self, label: int, vector: np.ndarray) -> None:
        if label in self._label_to_row:
            # Reuse row slot (e.g. re-add after delete in tests)
            row_i = self._label_to_row[label]
            self._vectors[row_i] = vector
            self._deleted.discard(label)
            return
        row_i = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._label_to_row[label] = row_i
        self._row_to_label.append(label)
        self._labels.append(label)

    # ------------------------------------------------------------------
    # Hard delete backend protocol
    # ------------------------------------------------------------------

    def hard_delete_label(
        self,
        collection: str,
        label: int,
        *,
        max_m: int = 16,
    ) -> BackendEraseResult:
        del collection
        label = int(label)
        if label not in self._label_to_row:
            return BackendEraseResult(
                success=False,
                label=label,
                message="unknown label",
            )

        # 1) Physical zero in authoritative matrix (residual mitigation).
        row_i = self._label_to_row[label]
        original = self._vectors[row_i].copy()
        self._vectors[row_i] = 0.0
        bytes_wiped = int(original.nbytes)

        # 2) Soft-delete in hnswlib so search stops returning it.
        try:
            if label not in self._deleted:
                self._index.mark_deleted(label)
        except RuntimeError as exc:
            # Already deleted or not in index — continue wiping storage.
            logger.debug("hnswlib.mark_deleted(%s): %s", label, exc)

        self._deleted.add(label)

        # 3) MN-RU heal on mirror when available (graph quality for dual search).
        msg = "zeroed+mark_deleted"
        if self._heal_mirror is not None and self._heal_mirror.is_loaded:
            try:
                # Map label → dense row id used in the mirror (0..n-1).
                mirror_id = row_i
                if mirror_id < self._heal_mirror.num_elements:
                    self._heal_mirror.erase_node(mirror_id, max_m)
                    msg += "+mn_ru_heal"
            except Exception as exc:  # noqa: BLE001
                logger.warning("heal mirror erase failed: %s", exc)
                msg += f"+heal_error({exc})"

        return BackendEraseResult(
            success=True,
            label=label,
            bytes_wiped=bytes_wiped,
            message=msg,
            compacted=False,
        )

    def verify_zeroed(self, collection: str, label: int) -> bool:
        del collection
        label = int(label)
        if label not in self._label_to_row:
            return False
        row = self._vectors[self._label_to_row[label]]
        return bool(np.all(row == 0.0))

    # ------------------------------------------------------------------
    # Search / compact
    # ------------------------------------------------------------------

    def query(
        self, vector: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, distances) from hnswlib (deleted ids excluded)."""
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        labels, dists = self._index.knn_query(q, k=k)
        return labels[0], dists[0]

    def compact(self) -> int:
        """
        Rebuild hnswlib from non-deleted, non-zero rows.

        This is the step that removes residual floats from the **hnswlib**
        binary structure after ``mark_deleted``. Returns number of live items.
        """
        live_labels: list[int] = []
        live_rows: list[np.ndarray] = []
        for lab, row_i in self._label_to_row.items():
            if lab in self._deleted:
                continue
            row = self._vectors[row_i]
            if np.all(row == 0.0):
                continue
            live_labels.append(lab)
            live_rows.append(row.copy())

        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        cap = max(self.max_elements, len(live_labels) + 1)
        self._index.init_index(
            max_elements=cap,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self._seed,
        )
        self._index.set_ef(self.ef)
        if live_labels:
            self._index.add_items(
                np.stack(live_rows).astype(np.float32),
                np.array(live_labels, dtype=np.int64),
            )
        # Deleted set still tracks logical deletes for the matrix.
        self._sync_heal_mirror()
        logger.info("hnswlib compact: %s live items", len(live_labels))
        return len(live_labels)

    def get_vector(self, label: int) -> np.ndarray:
        return self._vectors[self._label_to_row[int(label)]].copy()

    @property
    def deleted_labels(self) -> set[int]:
        return set(self._deleted)

    def _sync_heal_mirror(self) -> None:
        """Reload native proxy with current matrix + simple ring adjacency."""
        if self._heal_mirror is None or self._vectors.shape[0] == 0:
            return
        n, d = self._vectors.shape
        data = self._vectors.astype(np.float32, copy=True)
        # Zero already-deleted rows explicitly.
        for lab in self._deleted:
            if lab in self._label_to_row:
                data[self._label_to_row[lab]] = 0.0
        self._heal_mirror.load_index(data, d, n)
        # Lightweight navigable base layer for heal experiments.
        for i in range(n):
            nbrs = [(i - 1) % n, (i + 1) % n, (i + 2) % n]
            self._heal_mirror.set_neighbors(i, 0, nbrs)

    def save_index(self, path: str) -> None:
        """Persist hnswlib index file (call ``compact()`` first for residual-free)."""
        self._index.save_index(path)

    def load_index(self, path: str, max_elements: int | None = None) -> None:
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.load_index(
            path, max_elements=max_elements or self.max_elements
        )
        self._index.set_ef(self.ef)
