"""
Enterprise integration layer for HNSW Healer.

Use this package to plug physical erasure + graph healing into business
identity (collection + external IDs) and popular vector stores.

Public entry points
-------------------
ErasureService
    Stable delete API: ``delete(collection, ids, ...)``.
CollectionIdRegistry
    Maps ``(collection, external_id) → internal label``.
HnswlibHardDeleteAdapter
    hnswlib-backed store with mark_deleted + physical zero + compact.
ChromaHardDeleteCollection
    Wraps a Chroma collection so ``delete()`` hard-erases first.
"""

from integrations.erase_service import ErasureReceipt, ErasureService
from integrations.id_registry import CollectionIdRegistry, IdMappingError

__all__ = [
    "CollectionIdRegistry",
    "ErasureReceipt",
    "ErasureService",
    "IdMappingError",
    "HnswlibHardDeleteAdapter",
    "FaissHNSWHardDeleteAdapter",
    "ChromaHardDeleteCollection",
    "ReplicaFanoutCoordinator",
    "InPlaceVendorSession",
    "ErasureWorkflow",
    "ErasureWorkflowStore",
    "ErasureWorkflowRunner",
    "QdrantHardDeleteAdapter",
    "WeaviateHardDeleteAdapter",
    "recommend_delete_strategy",
]


def __getattr__(name: str):
    # Lazy imports so core installs work without optional deps.
    if name == "HnswlibHardDeleteAdapter":
        from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

        return HnswlibHardDeleteAdapter
    if name == "FaissHNSWHardDeleteAdapter":
        from integrations.faiss_adapter import FaissHNSWHardDeleteAdapter

        return FaissHNSWHardDeleteAdapter
    if name == "ChromaHardDeleteCollection":
        from integrations.chroma_hook import ChromaHardDeleteCollection

        return ChromaHardDeleteCollection
    if name == "ReplicaFanoutCoordinator":
        from integrations.replica_fanout import ReplicaFanoutCoordinator

        return ReplicaFanoutCoordinator
    if name == "InPlaceVendorSession":
        from integrations.vendor_attach import InPlaceVendorSession

        return InPlaceVendorSession
    if name in (
        "ErasureWorkflow",
        "ErasureWorkflowStore",
        "ErasureWorkflowRunner",
    ):
        from integrations import workflow as _wf

        return getattr(_wf, name)
    if name == "QdrantHardDeleteAdapter":
        from integrations.qdrant_adapter import QdrantHardDeleteAdapter

        return QdrantHardDeleteAdapter
    if name == "WeaviateHardDeleteAdapter":
        from integrations.weaviate_adapter import WeaviateHardDeleteAdapter

        return WeaviateHardDeleteAdapter
    if name == "recommend_delete_strategy":
        from integrations.delete_strategy import recommend_delete_strategy

        return recommend_delete_strategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
