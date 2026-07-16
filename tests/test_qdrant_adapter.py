"""Qdrant hard-delete adapter tests (in-memory client, no server)."""

from __future__ import annotations

import numpy as np

from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.qdrant_adapter import (
    InMemoryQdrantClient,
    QdrantHardDeleteAdapter,
)


def test_qdrant_hard_delete_and_compact() -> None:
    reg = CollectionIdRegistry()
    client = InMemoryQdrantClient()
    store = QdrantHardDeleteAdapter(
        dim=8,
        collection="docs",
        registry=reg,
        client=client,
    )
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 8), dtype=np.float32)
    ids = [f"d{i}" for i in range(10)]
    store.add(ids, vecs)

    assert client.count_points("docs") == 10
    label = reg.resolve("docs", "d3").label
    assert not np.all(store.get_vector(label) == 0.0)

    svc = ErasureService(reg, store, drop_mappings=False)
    receipt = svc.delete(
        "docs", ["d3"], residual_proof="sample", compact="always"
    )
    assert receipt.success
    assert receipt.compacted
    assert store.verify_zeroed("docs", label)
    assert client.count_points("docs") == 9
    assert client.get_point_vector("docs", "d3") is None


def test_qdrant_batch_register_on_add() -> None:
    reg = CollectionIdRegistry()
    store = QdrantHardDeleteAdapter(
        dim=4, collection="c", registry=reg, client=InMemoryQdrantClient()
    )
    store.add(["a", "b"], np.eye(2, 4).astype(np.float32))
    assert reg.contains("c", "a")
    assert reg.contains("c", "b")
