"""Tests for ErasureService + native backend (receipt v2, compact, proof)."""

from __future__ import annotations

import numpy as np
import pytest

import hnsw_healer
from integrations.erase_service import RECEIPT_VERSION, ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.native_backend import NativeHealerBackend


@pytest.fixture
def loaded_native() -> None:
    n, d = 30, 8
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(
            i, 0, [(i + 1) % n, (i + 2) % n]
        )
    yield


def test_erase_by_external_id(loaded_native: None) -> None:
    reg = CollectionIdRegistry()
    for i in range(30):
        reg.register("users", f"u{i}", label=i)

    svc = ErasureService(reg, NativeHealerBackend(), drop_mappings=True)
    before = hnsw_healer.default_index().get_vector(5)
    assert any(x != 0 for x in before)

    receipt = svc.delete(
        "users", ["u5"], reason="test_erasure", request_id="req-1"
    )
    assert receipt.success
    assert receipt.status == "complete"
    assert receipt.receipt_version == RECEIPT_VERSION
    assert receipt.labels == [5]
    assert receipt.signature
    assert receipt.residual_proof is not None
    assert receipt.residual_proof["passed"] is True
    assert all(x == 0.0 for x in hnsw_healer.default_index().get_vector(5))
    assert not reg.contains("users", "u5")


def test_idempotent_unknown_id(loaded_native: None) -> None:
    reg = CollectionIdRegistry()
    reg.register("users", "only", label=0)
    svc = ErasureService(reg, NativeHealerBackend())
    receipt = svc.delete("users", ["missing"], idempotent=True)
    assert "missing" in receipt.errors[0]
    assert receipt.status == "partial"
    assert receipt.success is False


def test_receipt_v2_fields_and_proof_off(loaded_native: None) -> None:
    reg = CollectionIdRegistry()
    for i in range(30):
        reg.register("c", f"id{i}", label=i)
    svc = ErasureService(reg, NativeHealerBackend())
    receipt = svc.delete("c", ["id3"], residual_proof="off")
    assert receipt.success
    assert receipt.status == "complete"
    assert receipt.receipt_version == 2
    assert receipt.residual_proof is not None
    assert receipt.residual_proof["mode"] == "off"
    assert receipt.residual_proof["passed"] is None


def test_batch_delete_single_compact_hnswlib() -> None:
    hnswlib = pytest.importorskip("hnswlib")
    del hnswlib
    from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

    reg = CollectionIdRegistry()
    adapter = HnswlibHardDeleteAdapter(
        dim=8,
        max_elements=50,
        registry=reg,
        collection="users",
        enable_heal_mirror=False,
    )
    vecs = np.random.randn(10, 8).astype(np.float32)
    ids = [f"u{i}" for i in range(10)]
    adapter.add(ids, vecs)

    compact_calls = {"n": 0}
    original_compact = adapter.compact

    def counting_compact() -> int:
        compact_calls["n"] += 1
        return original_compact()

    adapter.compact = counting_compact  # type: ignore[method-assign]

    svc = ErasureService(reg, adapter, drop_mappings=False)
    receipt = svc.delete(
        "users",
        ["u1", "u2", "u3"],
        compact="auto",
        residual_proof="sample",
    )
    assert receipt.success
    assert receipt.status == "complete"
    assert receipt.compacted is True
    assert compact_calls["n"] == 1
    assert len(receipt.labels) == 3


def test_compact_never_still_zeros() -> None:
    hnswlib = pytest.importorskip("hnswlib")
    del hnswlib
    from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

    reg = CollectionIdRegistry()
    adapter = HnswlibHardDeleteAdapter(
        dim=4,
        max_elements=20,
        registry=reg,
        collection="c",
        enable_heal_mirror=False,
    )
    adapter.add(["a", "b"], np.eye(2, 4).astype(np.float32))
    svc = ErasureService(reg, adapter, drop_mappings=False)
    receipt = svc.delete("c", ["a"], compact="never", residual_proof="off")
    # Without compact, status is still complete when compact=never
    assert receipt.success
    assert receipt.compacted is False
    assert adapter.verify_zeroed("c", receipt.labels[0])


def test_residual_proof_fail_closed() -> None:
    """Backend that zeros verify but returns non-zero get_vector after erase."""

    class EvilBackend:
        def __init__(self) -> None:
            self._v = {0: [1.0, 2.0]}

        def hard_delete_label(self, collection, label, *, max_m=16):
            from integrations.backends import BackendEraseResult

            self._v[label] = [0.0, 0.0]
            return BackendEraseResult(success=True, label=label, bytes_wiped=8)

        def verify_zeroed(self, collection, label) -> bool:
            return True

        def get_vector(self, label):
            # Lie after first read used for snapshot — return non-zero for proof
            return [9.0, 9.0]

    reg = CollectionIdRegistry()
    reg.register("c", "x", label=0)
    svc = ErasureService(reg, EvilBackend(), drop_mappings=False)
    receipt = svc.delete("c", ["x"], residual_proof="full", compact="never")
    assert receipt.success is False
    assert receipt.status == "partial"
    assert any("residual_proof" in e for e in receipt.errors)
