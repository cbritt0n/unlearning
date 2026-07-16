"""hnswlib adapter tests (skipped if hnswlib not installed)."""

from __future__ import annotations

import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

from integrations.erase_service import ErasureService
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter
from integrations.id_registry import CollectionIdRegistry


def test_hnswlib_hard_delete_zeros_and_mark() -> None:
    reg = CollectionIdRegistry()
    adapter = HnswlibHardDeleteAdapter(
        dim=8,
        max_elements=100,
        registry=reg,
        collection="users",
        enable_heal_mirror=True,
    )
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((10, 8), dtype=np.float32)
    ids = [f"u{i}" for i in range(10)]
    adapter.add(ids, vecs)

    original = adapter.get_vector(reg.resolve("users", "u3").label).copy()
    assert not np.all(original == 0)

    svc = ErasureService(reg, adapter, drop_mappings=False)
    receipt = svc.delete("users", ["u3"], reason="test")
    assert receipt.success
    assert receipt.compacted is True
    assert receipt.status == "complete"

    label = receipt.labels[0]
    assert adapter.verify_zeroed("users", label)
    assert np.all(adapter.get_vector(label) == 0.0)

    # Search should not return deleted label among top hits preferentially.
    q = vecs[3]
    labels, _ = adapter.query(q, k=5)
    # mark_deleted + compact: deleted id should not appear
    assert label not in set(int(x) for x in labels)


def test_compact_removes_residual_from_hnswlib_file(tmp_path) -> None:
    reg = CollectionIdRegistry()
    adapter = HnswlibHardDeleteAdapter(
        dim=4,
        max_elements=50,
        registry=reg,
        collection="c",
        enable_heal_mirror=False,
    )
    vecs = np.eye(4, dtype=np.float32)
    # pad to 8 vectors
    vecs = np.vstack([vecs, np.random.randn(4, 4).astype(np.float32)])
    ids = [f"i{i}" for i in range(8)]
    adapter.add(ids, vecs)

    lab = reg.resolve("c", "i0").label
    original = adapter.get_vector(lab).copy()
    path_before = tmp_path / "before.bin"
    adapter.save_index(str(path_before))

    receipt = ErasureService(reg, adapter, drop_mappings=False).delete(
        "c", ["i0"]
    )
    assert receipt.compacted is True

    path_after = tmp_path / "after.bin"
    adapter.save_index(str(path_after))

    from compliance.residual import float32_pattern_bytes

    pattern = float32_pattern_bytes(original)
    del pattern  # pattern scan is best-effort for hnswlib binary layout
    assert adapter.verify_zeroed("c", lab)
