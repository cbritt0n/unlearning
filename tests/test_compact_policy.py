"""Coalesced compact policy tests."""

from __future__ import annotations

from integrations.backends import BackendEraseResult
from integrations.compact_policy import CompactCoalescePolicy
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry


class CountingBackend:
    def __init__(self) -> None:
        self.compacts = 0
        self._v = {i: [float(i + 1), 1.0] for i in range(20)}

    def hard_delete_label(self, collection, label, *, max_m=16):
        self._v[label] = [0.0, 0.0]
        return BackendEraseResult(True, label, bytes_wiped=8, message="z")

    def verify_zeroed(self, collection, label):
        return all(x == 0.0 for x in self._v[label])

    def get_vector(self, label):
        return list(self._v[label])

    def compact(self):
        self.compacts += 1
        return sum(1 for v in self._v.values() if any(x != 0 for x in v))


def test_coalesce_every_n(monkeypatch) -> None:
    # Disable adaptive "always compact" so coalesce policy is observable.
    monkeypatch.setenv("HEALER_ADAPTIVE_COMPACT", "0")
    policy = CompactCoalescePolicy(mode="coalesce", every_n=3, max_age_s=9999)
    reg = CollectionIdRegistry()
    backend = CountingBackend()
    for i in range(10):
        reg.register("c", f"id{i}", label=i)
    svc = ErasureService(
        reg,
        backend,
        drop_mappings=False,
        default_compact="auto",
        default_residual_proof="off",
        compact_policy=policy,
    )
    r1 = svc.delete("c", ["id0"], compact="auto")
    assert r1.success
    assert r1.compacted is False
    assert backend.compacts == 0
    r2 = svc.delete("c", ["id1"], compact="auto")
    assert r2.compacted is False
    r3 = svc.delete("c", ["id2"], compact="auto")
    assert r3.compacted is True
    assert backend.compacts == 1


def test_force_compact(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ADAPTIVE_COMPACT", "0")
    policy = CompactCoalescePolicy(mode="coalesce", every_n=100, max_age_s=9999)
    reg = CollectionIdRegistry()
    backend = CountingBackend()
    reg.register("c", "x", label=0)
    svc = ErasureService(
        reg,
        backend,
        drop_mappings=False,
        default_residual_proof="off",
        compact_policy=policy,
    )
    svc.delete("c", ["x"], compact="never")
    assert backend.compacts == 0
    out = svc.force_compact()
    assert out["compacted"] is True
    assert backend.compacts == 1
