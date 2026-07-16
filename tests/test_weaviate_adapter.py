"""Weaviate hard-delete adapter tests (in-memory)."""

from __future__ import annotations

import numpy as np

from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.weaviate_adapter import (
    InMemoryWeaviateClient,
    WeaviateHardDeleteAdapter,
)


def test_weaviate_hard_delete_and_compact() -> None:
    reg = CollectionIdRegistry()
    client = InMemoryWeaviateClient()
    store = WeaviateHardDeleteAdapter(
        dim=6,
        collection="Document",
        registry=reg,
        client=client,
    )
    vecs = np.random.randn(5, 6).astype(np.float32)
    store.add([f"o{i}" for i in range(5)], vecs)
    assert client.count("Document") == 5

    svc = ErasureService(reg, store, drop_mappings=False)
    receipt = svc.delete("Document", ["o1", "o2"], residual_proof="off")
    assert receipt.success
    assert receipt.compacted
    assert client.count("Document") == 3
