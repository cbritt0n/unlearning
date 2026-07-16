"""
Weaviate hard-delete adapter (rebuild-based v1)
===============================================

Keeps an authoritative float32 matrix and dual-writes to Weaviate objects.
Hard delete zeros the matrix and removes the object; ``compact()`` batch
re-imports live rows into a fresh class (or clears + reinserts).

Uses a small client protocol so tests run without a Weaviate server.

Install::

    pip install hnsw-healer[weaviate]
    # or: pip install weaviate-client
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

import numpy as np

from integrations.backends import BackendEraseResult
from integrations.id_registry import CollectionIdRegistry

logger = logging.getLogger(__name__)


class WeaviateClientProtocol(Protocol):
    def ensure_class(self, class_name: str, dim: int) -> None: ...

    def upsert_objects(
        self,
        class_name: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        properties: Sequence[dict[str, Any]] | None = None,
    ) -> None: ...

    def delete_objects(self, class_name: str, ids: Sequence[str]) -> None: ...

    def clear_class(self, class_name: str) -> None: ...

    def count(self, class_name: str) -> int: ...


class InMemoryWeaviateClient:
    def __init__(self) -> None:
        self._objs: dict[str, dict[str, np.ndarray]] = {}
        self._dims: dict[str, int] = {}

    def ensure_class(self, class_name: str, dim: int) -> None:
        self._objs.setdefault(class_name, {})
        self._dims[class_name] = int(dim)

    def upsert_objects(
        self,
        class_name: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        properties: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        del properties
        vecs = np.asarray(vectors, dtype=np.float32)
        self.ensure_class(class_name, vecs.shape[1])
        store = self._objs[class_name]
        for i, oid in enumerate(ids):
            store[str(oid)] = vecs[i].copy()

    def delete_objects(self, class_name: str, ids: Sequence[str]) -> None:
        store = self._objs.get(class_name, {})
        for oid in ids:
            store.pop(str(oid), None)

    def clear_class(self, class_name: str) -> None:
        dim = self._dims.get(class_name, 0)
        self._objs[class_name] = {}
        if dim:
            self._dims[class_name] = dim

    def count(self, class_name: str) -> int:
        return len(self._objs.get(class_name, {}))


class WeaviateHardDeleteAdapter:
    def __init__(
        self,
        dim: int,
        *,
        collection: str = "Document",
        registry: CollectionIdRegistry | None = None,
        client: WeaviateClientProtocol | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        # Weaviate class names are PascalCase; collection key stays as given.
        self.collection = collection
        self.class_name = collection[0].upper() + collection[1:] if collection else "Document"
        self.registry = registry or CollectionIdRegistry()
        self.registry.ensure_collection(collection)
        self._client = client or InMemoryWeaviateClient()
        self._client.ensure_class(self.class_name, dim)
        self._deleted: set[int] = set()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._label_to_row: dict[int, int] = {}
        self._label_to_ext: dict[int, str] = {}

    def add(
        self,
        external_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        collection: str | None = None,
        properties: Sequence[dict[str, Any]] | None = None,
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
            if label in self._label_to_row:
                self._vectors[self._label_to_row[label]] = row
            else:
                row_i = self._vectors.shape[0]
                self._vectors = np.vstack(
                    [self._vectors, row.reshape(1, -1)]
                )
                self._label_to_row[label] = row_i
            self._label_to_ext[label] = str(ext)
            self._deleted.discard(label)
            labels.append(label)
            ids.append(str(ext))
            rows.append(row)

        self._client.upsert_objects(
            self.class_name, ids, np.stack(rows), properties=properties
        )
        return labels

    def hard_delete_label(
        self,
        collection: str,
        label: int,
        *,
        max_m: int = 16,
    ) -> BackendEraseResult:
        del collection, max_m
        label = int(label)
        if label not in self._label_to_row:
            return BackendEraseResult(
                success=False, label=label, message="unknown label"
            )
        row_i = self._label_to_row[label]
        original = self._vectors[row_i].copy()
        self._vectors[row_i] = 0.0
        self._deleted.add(label)
        ext = self._label_to_ext.get(label, str(label))
        try:
            self._client.delete_objects(self.class_name, [ext])
            msg = "zeroed+weaviate_delete"
        except Exception as exc:  # noqa: BLE001
            return BackendEraseResult(
                success=False,
                label=label,
                bytes_wiped=int(original.nbytes),
                message=f"zeroed+weaviate_error({exc})",
            )
        return BackendEraseResult(
            success=True,
            label=label,
            bytes_wiped=int(original.nbytes),
            message=msg,
        )

    def verify_zeroed(self, collection: str, label: int) -> bool:
        del collection
        if int(label) not in self._label_to_row:
            return False
        return bool(
            np.all(self._vectors[self._label_to_row[int(label)]] == 0.0)
        )

    def get_vector(self, label: int) -> np.ndarray:
        return self._vectors[self._label_to_row[int(label)]].copy()

    def compact(self) -> int:
        live_ids: list[str] = []
        live_rows: list[np.ndarray] = []
        for lab, row_i in self._label_to_row.items():
            if lab in self._deleted:
                continue
            row = self._vectors[row_i]
            if np.all(row == 0.0):
                continue
            live_ids.append(self._label_to_ext.get(lab, str(lab)))
            live_rows.append(row.copy())
        self._client.clear_class(self.class_name)
        self._client.ensure_class(self.class_name, self.dim)
        if live_ids:
            self._client.upsert_objects(
                self.class_name, live_ids, np.stack(live_rows)
            )
        logger.info("weaviate compact: %s live objects", len(live_ids))
        return len(live_ids)
