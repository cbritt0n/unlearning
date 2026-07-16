"""
FAISS IndexHNSW hard-delete backend
===================================

Implements ``HardDeleteBackend`` for FAISS HNSW indexes.

Strategy
--------
FAISS ``IndexHNSW*`` does not expose a stable public "zero this vector in
place and rewire links" API across all builds. This adapter keeps an
**authoritative float32 matrix** (residual-sensitive source of truth) and a
FAISS HNSW index used for search:

1. On hard delete: zero the matrix row, mark label deleted, optional MN-RU
   heal on a mirrored ``HNSWIndexProxy``.
2. Rebuild (or surgically drop) the FAISS index from live rows via
   ``compact()`` so residual floats are not left in the FAISS structure.

Install::

    pip install hnsw-healer[faiss]
    # or: pip install faiss-cpu
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from integrations.backends import BackendEraseResult
from integrations.id_registry import CollectionIdRegistry

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False

try:
    import hnsw_healer

    HEALER_AVAILABLE = True
except ImportError:  # pragma: no cover
    hnsw_healer = None  # type: ignore[assignment]
    HEALER_AVAILABLE = False


class FaissHNSWHardDeleteAdapter:
    """
    Dual store: dense float matrix + FAISS IndexHNSWFlat + optional heal mirror.

    Labels are integer ids registered via ``CollectionIdRegistry`` (or
    sequential ints when adding without a registry external id).
    """

    def __init__(
        self,
        dim: int,
        *,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        metric: str = "l2",
        enable_heal_mirror: bool = True,
        collection: str = "default",
        registry: CollectionIdRegistry | None = None,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for FaissHNSWHardDeleteAdapter. "
                "Install with: pip install faiss-cpu"
            )
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = int(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.metric = metric.lower()
        self.collection = collection
        self.registry = registry or CollectionIdRegistry()
        self.registry.ensure_collection(collection)

        self._deleted: set[int] = set()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._label_to_row: dict[int, int] = {}
        self._row_to_label: list[int] = []

        self._index = self._new_index()
        self._heal_mirror = None
        if enable_heal_mirror and HEALER_AVAILABLE:
            self._heal_mirror = hnsw_healer.HNSWIndexProxy(
                lock_pool_size=64, lock_timeout_ms=100
            )

    def _new_index(self):
        metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.metric in ("ip", "inner_product")
            else faiss.METRIC_L2
        )
        # IndexHNSWFlat owns float storage + HNSW links.
        index = faiss.IndexHNSWFlat(self.dim, self.M, metric)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        # ID map so we address vectors by business labels.
        return faiss.IndexIDMap2(index)

    def _base_hnsw(self):
        # Unwrap IDMap → IndexHNSWFlat
        return getattr(self._index, "index", self._index)

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
        coll = collection or self.collection
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vecs.shape[1]}")
        if len(external_ids) != vecs.shape[0]:
            raise ValueError("external_ids and vectors length mismatch")

        labels: list[int] = []
        rows_to_add: list[np.ndarray] = []
        ids_to_add: list[int] = []

        for ext, row in zip(external_ids, vecs):
            label = self.registry.register(coll, ext)
            self._append_row(label, row)
            labels.append(label)
            if label not in self._deleted:
                rows_to_add.append(row.astype(np.float32, copy=False))
                ids_to_add.append(label)

        if ids_to_add:
            xb = np.stack(rows_to_add).astype(np.float32)
            ida = np.asarray(ids_to_add, dtype=np.int64)
            self._index.add_with_ids(xb, ida)

        self._sync_heal_mirror()
        return labels

    def _append_row(self, label: int, vector: np.ndarray) -> None:
        if label in self._label_to_row:
            row_i = self._label_to_row[label]
            self._vectors[row_i] = vector
            self._deleted.discard(label)
            return
        row_i = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._label_to_row[label] = row_i
        self._row_to_label.append(label)

    # ------------------------------------------------------------------
    # HardDeleteBackend
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
                success=False, label=label, message="unknown label"
            )

        row_i = self._label_to_row[label]
        original = self._vectors[row_i].copy()
        self._vectors[row_i] = 0.0
        bytes_wiped = int(original.nbytes)
        self._deleted.add(label)

        # Prefer FAISS remove_ids when available (IndexIDMap2 supports it).
        removed = False
        try:
            sel = faiss.IDSelectorBatch(
                np.array([label], dtype=np.int64)
            )
            nrem = self._index.remove_ids(sel)
            removed = nrem > 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("faiss.remove_ids failed (%s); compact later", exc)

        msg = "zeroed"
        msg += "+remove_ids" if removed else "+pending_compact"

        if self._heal_mirror is not None and self._heal_mirror.is_loaded:
            try:
                if row_i < self._heal_mirror.num_elements:
                    self._heal_mirror.erase_node(row_i, max_m)
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
        return bool(np.all(self._vectors[self._label_to_row[label]] == 0.0))

    # ------------------------------------------------------------------
    # Search / compact / persistence
    # ------------------------------------------------------------------

    def query(
        self, vector: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        base = self._base_hnsw()
        if hasattr(base, "hnsw"):
            base.hnsw.efSearch = self.ef_search
        dists, labels = self._index.search(q, k)
        return labels[0], dists[0]

    def compact(self) -> int:
        """
        Rebuild FAISS HNSW from non-deleted, non-zero rows.

        Guarantees residual floats for deleted labels are not present in the
        FAISS index structure (beyond the authoritative zeroed matrix).
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

        self._index = self._new_index()
        if live_labels:
            xb = np.stack(live_rows).astype(np.float32)
            ida = np.asarray(live_labels, dtype=np.int64)
            self._index.add_with_ids(xb, ida)

        self._sync_heal_mirror()
        logger.info("faiss compact: %s live items", len(live_labels))
        return len(live_labels)

    def get_vector(self, label: int) -> np.ndarray:
        return self._vectors[self._label_to_row[int(label)]].copy()

    @property
    def deleted_labels(self) -> set[int]:
        return set(self._deleted)

    def _sync_heal_mirror(self) -> None:
        if self._heal_mirror is None or self._vectors.shape[0] == 0:
            return
        n, d = self._vectors.shape
        data = self._vectors.astype(np.float32, copy=True)
        for lab in self._deleted:
            if lab in self._label_to_row:
                data[self._label_to_row[lab]] = 0.0
        self._heal_mirror.load_index(data, d, n)
        for i in range(n):
            self._heal_mirror.set_neighbors(
                i, 0, [(i - 1) % n, (i + 1) % n, (i + 2) % n]
            )

    def save_index(self, path: str) -> None:
        faiss.write_index(self._index, path)

    def load_index(self, path: str) -> None:
        self._index = faiss.read_index(path)
        base = self._base_hnsw()
        if hasattr(base, "hnsw"):
            base.hnsw.efSearch = self.ef_search
