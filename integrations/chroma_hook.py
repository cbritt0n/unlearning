"""
ChromaDB hard-delete hook
=========================

Chroma's default ``collection.delete(ids=...)`` is a **metadata soft-delete**
relative to residual vector risk when the underlying ANN segment still holds
floats. This wrapper:

1. Resolves business ids through ``ErasureService`` / backend (physical wipe,
   auto-compact, residual proof).
2. Then calls Chroma's delete so metadata and query filters stay consistent.

Install::

    pip install hnsw-healer[chroma]
    # or: pip install chromadb
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from integrations.erase_service import ErasureReceipt, ErasureService

logger = logging.getLogger(__name__)

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    CHROMA_AVAILABLE = False


class ChromaHardDeleteCollection:
    """
    Drop-in style wrapper around a Chroma collection.

    Typical wiring::

        client = chromadb.Client()
        raw = client.get_or_create_collection("docs")
        wrapped = ChromaHardDeleteCollection(raw, erase_service, collection="docs")
        wrapped.add(ids=[...], embeddings=[...], documents=[...])
        wrapped.delete(ids=["user-42"], reason="gdpr_art_17")
        # register-on-add is automatic; compact + residual proof run inside
        # ErasureService (no separate register/compact calls required).
    """

    def __init__(
        self,
        collection: Any,
        erase_service: ErasureService,
        *,
        collection_name: str | None = None,
        fail_closed: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        collection:
            A Chroma collection object (must expose ``add`` / ``delete`` / ``query``).
        erase_service:
            Configured ``ErasureService`` pointing at your hard-delete backend.
        collection_name:
            Registry collection key; defaults to ``collection.name``.
        fail_closed:
            If True (default), refuse Chroma metadata delete when hard-erase
            fails (including compact / residual-proof failures).
        """
        self._collection = collection
        self._erase = erase_service
        self._name = collection_name or getattr(collection, "name", "default")
        self.fail_closed = fail_closed

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw(self) -> Any:
        """Underlying Chroma collection (escape hatch)."""
        return self._collection

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]] | None = None,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Dual-write: register + backend matrix (when available), then Chroma.

        Id registration is always performed — either via ``backend.add``
        (which registers) or via the registry alone when there is no
        vector backend ``add``.
        """
        backend = getattr(self._erase, "backend", None)
        dual_wrote = False
        if embeddings is not None and hasattr(backend, "add"):
            import numpy as np

            backend.add(
                list(ids),
                np.asarray(embeddings, dtype=np.float32),
                collection=self._name,
            )
            dual_wrote = True

        if not dual_wrote:
            # Registry-only path (native backend without matrix add).
            for eid in ids:
                self._erase.registry.register(self._name, eid)

        self._collection.add(
            ids=list(ids),
            embeddings=list(embeddings) if embeddings is not None else None,
            documents=list(documents) if documents is not None else None,
            metadatas=list(metadatas) if metadatas is not None else None,
            **kwargs,
        )

    def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        reason: str | None = None,
        request_id: str | None = None,
        max_m: int = 16,
        where: dict | None = None,
        compact: Any = None,
        residual_proof: Any = None,
        **kwargs: Any,
    ) -> ErasureReceipt | None:
        """
        Hard-erase then Chroma-delete (fail-closed by default).

        Only id-based deletes are hard-erased. Filter-only deletes (``where``)
        without ids fall back to Chroma behavior and log a residual warning.
        """
        if ids:
            delete_kwargs: dict[str, Any] = {
                "reason": reason,
                "request_id": request_id,
                "max_m": max_m,
            }
            if compact is not None:
                delete_kwargs["compact"] = compact
            if residual_proof is not None:
                delete_kwargs["residual_proof"] = residual_proof

            receipt = self._erase.delete(
                self._name,
                list(ids),
                **delete_kwargs,
            )
            if not receipt.success and self.fail_closed:
                raise RuntimeError(
                    "hard erase failed; refusing Chroma soft-delete to avoid "
                    f"residual vectors: status={receipt.status} "
                    f"errors={receipt.errors}"
                )
            # Chroma metadata / segment bookkeeping only after hard success
            # (or fail_closed=False).
            self._collection.delete(ids=list(ids), **kwargs)
            return receipt

        logger.warning(
            "Chroma delete without explicit ids (where=%s) cannot guarantee "
            "physical vector wipe; residual embeddings may remain in ANN storage",
            where,
        )
        self._collection.delete(where=where, **kwargs)
        return None

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._collection.query(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._collection.get(*args, **kwargs)

    @staticmethod
    def require_chromadb() -> None:
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
