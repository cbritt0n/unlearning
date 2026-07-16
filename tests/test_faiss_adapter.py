"""FAISS IndexHNSW hard-delete adapter tests (skip if faiss missing)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from integrations.erase_service import ErasureService
from integrations.faiss_adapter import FaissHNSWHardDeleteAdapter
from integrations.id_registry import CollectionIdRegistry


def test_faiss_hard_delete_zeros_and_search() -> None:
    reg = CollectionIdRegistry()
    adapter = FaissHNSWHardDeleteAdapter(
        dim=8,
        M=16,
        registry=reg,
        collection="users",
        enable_heal_mirror=True,
    )
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((12, 8), dtype=np.float32)
    ids = [f"u{i}" for i in range(12)]
    adapter.add(ids, vecs)

    label = reg.resolve("users", "u4").label
    assert not np.all(adapter.get_vector(label) == 0.0)

    svc = ErasureService(reg, adapter, drop_mappings=False)
    receipt = svc.delete("users", ["u4"], reason="test")
    assert receipt.success
    assert receipt.compacted is True  # auto-compact once per batch
    assert adapter.verify_zeroed("users", label)

    labels, _ = adapter.query(vecs[4], k=5)
    assert label not in set(int(x) for x in labels if int(x) >= 0)
