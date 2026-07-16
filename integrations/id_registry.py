"""
Collection-scoped identity map: business IDs ↔ internal HNSW labels.

Enterprise deletes arrive as ``user_id`` / document UUIDs, not dense
integer node indices. This registry is the control-plane glue.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class IdMappingError(KeyError):
    """Raised when a collection or external ID is unknown or conflicts."""


@dataclass(frozen=True)
class ResolvedId:
    collection: str
    external_id: str
    label: int


class CollectionIdRegistry:
    """
    Bidirectional map for multi-tenant / multi-collection deployments.

    Storage shape (JSON on disk when persisted)::

        {
          "collections": {
            "users": {"alice": 0, "bob": 1},
            "docs":  {"doc-1": 0}
          },
          "next_label": {"users": 2, "docs": 1}
        }

    Labels are **per-collection** dense integers suitable for HNSW node ids
    inside that collection's index backend.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # collection -> {external_id -> label}
        self._forward: dict[str, dict[str, int]] = {}
        # collection -> {label -> external_id}
        self._reverse: dict[str, dict[int, str]] = {}
        self._next_label: dict[str, int] = {}

    def ensure_collection(self, collection: str) -> None:
        coll = self._norm_collection(collection)
        with self._lock:
            self._forward.setdefault(coll, {})
            self._reverse.setdefault(coll, {})
            self._next_label.setdefault(coll, 0)

    def register(
        self,
        collection: str,
        external_id: str,
        label: int | None = None,
    ) -> int:
        """
        Register ``external_id`` in ``collection``.

        If ``label`` is omitted, allocates the next dense label.
        Returns the label.
        """
        coll = self._norm_collection(collection)
        ext = self._norm_external(external_id)
        with self._lock:
            self.ensure_collection(coll)
            existing = self._forward[coll].get(ext)
            if existing is not None:
                if label is not None and label != existing:
                    raise IdMappingError(
                        f"id {ext!r} already mapped to {existing}, "
                        f"not {label}"
                    )
                return existing

            if label is None:
                label = self._next_label[coll]
                self._next_label[coll] = label + 1
            else:
                if label in self._reverse[coll]:
                    raise IdMappingError(
                        f"label {label} already used in {coll!r} by "
                        f"{self._reverse[coll][label]!r}"
                    )
                self._next_label[coll] = max(self._next_label[coll], label + 1)

            self._forward[coll][ext] = label
            self._reverse[coll][label] = ext
            return label

    def register_many(
        self,
        collection: str,
        external_ids: Iterable[str],
    ) -> list[int]:
        return [self.register(collection, eid) for eid in external_ids]

    def resolve(self, collection: str, external_id: str) -> ResolvedId:
        coll = self._norm_collection(collection)
        ext = self._norm_external(external_id)
        with self._lock:
            try:
                label = self._forward[coll][ext]
            except KeyError as exc:
                raise IdMappingError(
                    f"unknown id {ext!r} in collection {coll!r}"
                ) from exc
            return ResolvedId(collection=coll, external_id=ext, label=label)

    def resolve_many(
        self, collection: str, external_ids: Iterable[str]
    ) -> list[ResolvedId]:
        return [self.resolve(collection, eid) for eid in external_ids]

    def external_of(self, collection: str, label: int) -> str:
        coll = self._norm_collection(collection)
        with self._lock:
            try:
                return self._reverse[coll][label]
            except KeyError as exc:
                raise IdMappingError(
                    f"unknown label {label} in collection {coll!r}"
                ) from exc

    def drop(self, collection: str, external_id: str) -> int:
        """Remove mapping after a successful hard delete. Returns label."""
        resolved = self.resolve(collection, external_id)
        with self._lock:
            del self._forward[resolved.collection][resolved.external_id]
            del self._reverse[resolved.collection][resolved.label]
        return resolved.label

    def contains(self, collection: str, external_id: str) -> bool:
        coll = self._norm_collection(collection)
        ext = self._norm_external(external_id)
        with self._lock:
            return ext in self._forward.get(coll, {})

    def labels(self, collection: str) -> list[int]:
        coll = self._norm_collection(collection)
        with self._lock:
            return sorted(self._reverse.get(coll, {}).keys())

    def external_ids(self, collection: str) -> list[str]:
        coll = self._norm_collection(collection)
        with self._lock:
            return sorted(self._forward.get(coll, {}).keys())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            payload = {
                "collections": {
                    c: dict(m) for c, m in self._forward.items()
                },
                "next_label": dict(self._next_label),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._forward = {
                c: {str(k): int(v) for k, v in m.items()}
                for c, m in payload.get("collections", {}).items()
            }
            self._reverse = {
                c: {lab: ext for ext, lab in m.items()}
                for c, m in self._forward.items()
            }
            self._next_label = {
                c: int(v) for c, v in payload.get("next_label", {}).items()
            }
            for c, m in self._forward.items():
                self._next_label.setdefault(
                    c, (max(m.values()) + 1) if m else 0
                )

    @staticmethod
    def _norm_collection(collection: str) -> str:
        coll = (collection or "").strip()
        if not coll:
            raise ValueError("collection must be a non-empty string")
        return coll

    @staticmethod
    def _norm_external(external_id: str) -> str:
        ext = str(external_id).strip()
        if not ext:
            raise ValueError("external_id must be a non-empty string")
        return ext
