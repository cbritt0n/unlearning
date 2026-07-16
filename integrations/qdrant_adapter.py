"""
Qdrant hard-delete adapter (rebuild-based v1)
=============================================

Qdrant point delete is often enough for search filters, but residual risk
depends on segment rewrites / snapshots. This adapter keeps an
**authoritative float32 matrix** (same residual model as hnswlib/FAISS) and:

1. Zeros the matrix row for the label.
2. Deletes the point from Qdrant by business id (point id = external id or
   integer label string).
3. ``compact()`` rebuilds the Qdrant collection from live non-zero rows so
   residual vectors are not left in the serving index.

Install::

    pip install hnsw-healer[qdrant]
    # or: pip install qdrant-client

For unit tests without a server, use ``InMemoryQdrantClient``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

import numpy as np

from integrations.backends import BackendEraseResult
from integrations.id_registry import CollectionIdRegistry

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[misc, assignment]
    qmodels = None  # type: ignore[assignment]
    QDRANT_AVAILABLE = False


class QdrantClientProtocol(Protocol):
    """Minimal surface used by the adapter (real client or in-memory)."""

    def upsert_points(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None: ...

    def delete_points(self, collection: str, ids: Sequence[str]) -> None: ...

    def recreate_collection(
        self, collection: str, dim: int, distance: str = "Cosine"
    ) -> None: ...

    def count_points(self, collection: str) -> int: ...

    def get_point_vector(
        self, collection: str, point_id: str
    ) -> list[float] | None: ...


class InMemoryQdrantClient:
    """Process-local stand-in for tests (no network)."""

    def __init__(self) -> None:
        # collection -> {id -> vector}
        self._data: dict[str, dict[str, np.ndarray]] = {}
        self._dims: dict[str, int] = {}

    def recreate_collection(
        self, collection: str, dim: int, distance: str = "Cosine"
    ) -> None:
        del distance
        self._data[collection] = {}
        self._dims[collection] = int(dim)

    def upsert_points(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        del payloads
        vecs = np.asarray(vectors, dtype=np.float32)
        if collection not in self._data:
            self.recreate_collection(collection, vecs.shape[1])
        store = self._data[collection]
        for i, pid in enumerate(ids):
            store[str(pid)] = vecs[i].copy()

    def delete_points(self, collection: str, ids: Sequence[str]) -> None:
        store = self._data.get(collection, {})
        for pid in ids:
            store.pop(str(pid), None)

    def count_points(self, collection: str) -> int:
        return len(self._data.get(collection, {}))

    def get_point_vector(
        self, collection: str, point_id: str
    ) -> list[float] | None:
        store = self._data.get(collection, {})
        row = store.get(str(point_id))
        if row is None:
            return None
        return row.astype(np.float32).tolist()


class QdrantSdkClient:
    """Adapter over official ``qdrant-client``."""

    def __init__(self, client: Any) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client required. pip install qdrant-client"
            )
        self._client = client

    def recreate_collection(
        self, collection: str, dim: int, distance: str = "Cosine"
    ) -> None:
        dist = {
            "cosine": qmodels.Distance.COSINE,
            "euclid": qmodels.Distance.EUCLID,
            "dot": qmodels.Distance.DOT,
        }.get(distance.lower(), qmodels.Distance.COSINE)
        self._client.recreate_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=dist),
        )

    def upsert_points(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        points = []
        for i, pid in enumerate(ids):
            payload = dict(payloads[i]) if payloads is not None else {}
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(pid),
                    vector=vecs[i].tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=collection, points=points)

    def delete_points(self, collection: str, ids: Sequence[str]) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(
                points=[self._point_id(i) for i in ids]
            ),
        )

    def count_points(self, collection: str) -> int:
        res = self._client.count(collection_name=collection, exact=True)
        return int(getattr(res, "count", 0))

    def get_point_vector(
        self, collection: str, point_id: str
    ) -> list[float] | None:
        pts = self._client.retrieve(
            collection_name=collection,
            ids=[self._point_id(point_id)],
            with_vectors=True,
        )
        if not pts:
            return None
        vec = pts[0].vector
        if isinstance(vec, dict):
            # named vectors — take first
            vec = next(iter(vec.values()))
        return list(vec) if vec is not None else None

    @staticmethod
    def _point_id(pid: str):
        # Qdrant accepts UUID or unsigned int; use hash for arbitrary strings.
        s = str(pid)
        if s.isdigit():
            return int(s)
        # stable positive int64-ish from hash
        return abs(hash(s)) % (2**63 - 1)


class QdrantHardDeleteAdapter:
    """
    Dual store: dense float matrix + Qdrant collection + id registry.
    """

    def __init__(
        self,
        dim: int,
        *,
        collection: str = "default",
        registry: CollectionIdRegistry | None = None,
        client: QdrantClientProtocol | None = None,
        url: str | None = None,
        distance: str = "Cosine",
        ensure_collection: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.collection = collection
        self.registry = registry or CollectionIdRegistry()
        self.registry.ensure_collection(collection)
        self.distance = distance
        self._deleted: set[int] = set()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._label_to_row: dict[int, int] = {}
        self._label_to_ext: dict[int, str] = {}

        if client is not None:
            self._client = client
        elif url is not None:
            if not QDRANT_AVAILABLE:
                raise ImportError(
                    "qdrant-client required for URL mode. "
                    "pip install qdrant-client"
                )
            self._client = QdrantSdkClient(QdrantClient(url=url))
        else:
            self._client = InMemoryQdrantClient()

        if ensure_collection:
            self._client.recreate_collection(
                collection, dim, distance=distance
            )

    def add(
        self,
        external_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        collection: str | None = None,
        payloads: Sequence[dict[str, Any]] | None = None,
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
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for ext, row in zip(external_ids, vecs):
            label = self.registry.register(coll, ext)
            self._append_row(label, row)
            self._label_to_ext[label] = str(ext)
            self._deleted.discard(label)
            labels.append(label)
            ids.append(str(ext))
            rows.append(row)

        self._client.upsert_points(
            coll, ids, np.stack(rows), payloads=payloads
        )
        return labels

    def _append_row(self, label: int, vector: np.ndarray) -> None:
        if label in self._label_to_row:
            self._vectors[self._label_to_row[label]] = vector
            return
        row_i = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._label_to_row[label] = row_i

    def hard_delete_label(
        self,
        collection: str,
        label: int,
        *,
        max_m: int = 16,
    ) -> BackendEraseResult:
        del max_m
        label = int(label)
        if label not in self._label_to_row:
            return BackendEraseResult(
                success=False, label=label, message="unknown label"
            )
        row_i = self._label_to_row[label]
        original = self._vectors[row_i].copy()
        self._vectors[row_i] = 0.0
        self._deleted.add(label)
        ext = self._label_to_ext.get(label)
        if ext is None:
            try:
                ext = self.registry.external_of(collection, label)
                self._label_to_ext[label] = ext
            except Exception:  # noqa: BLE001
                ext = str(label)
        try:
            self._client.delete_points(collection or self.collection, [ext])
            msg = "zeroed+qdrant_delete"
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant delete failed: %s", exc)
            msg = f"zeroed+qdrant_error({exc})"
            return BackendEraseResult(
                success=False,
                label=label,
                bytes_wiped=int(original.nbytes),
                message=msg,
            )
        return BackendEraseResult(
            success=True,
            label=label,
            bytes_wiped=int(original.nbytes),
            message=msg,
            compacted=False,
        )

    def verify_zeroed(self, collection: str, label: int) -> bool:
        del collection
        label = int(label)
        if label not in self._label_to_row:
            return False
        return bool(np.all(self._vectors[self._label_to_row[label]] == 0.0))

    def get_vector(self, label: int) -> np.ndarray:
        return self._vectors[self._label_to_row[int(label)]].copy()

    def compact(self) -> int:
        """Recreate Qdrant collection from non-deleted, non-zero matrix rows."""
        live_ids: list[str] = []
        live_rows: list[np.ndarray] = []
        for lab, row_i in self._label_to_row.items():
            if lab in self._deleted:
                continue
            row = self._vectors[row_i]
            if np.all(row == 0.0):
                continue
            ext = self._label_to_ext.get(lab, str(lab))
            live_ids.append(ext)
            live_rows.append(row.copy())

        self._client.recreate_collection(
            self.collection, self.dim, distance=self.distance
        )
        if live_ids:
            self._client.upsert_points(
                self.collection, live_ids, np.stack(live_rows)
            )
        logger.info("qdrant compact: %s live points", len(live_ids))
        return len(live_ids)

    @property
    def deleted_labels(self) -> set[int]:
        return set(self._deleted)
