"""Batch delete: exactly one compact() call per ErasureService.delete()."""

from __future__ import annotations

from integrations.backends import BackendEraseResult
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry


class CountingCompactBackend:
    def __init__(self, n: int = 10) -> None:
        self.compact_calls = 0
        self._vecs = {i: [float(i + 1), 1.0, 1.0, 1.0] for i in range(n)}

    def hard_delete_label(self, collection, label, *, max_m=16):
        self._vecs[label] = [0.0, 0.0, 0.0, 0.0]
        return BackendEraseResult(
            success=True, label=label, bytes_wiped=16, message="zeroed"
        )

    def verify_zeroed(self, collection, label) -> bool:
        return all(x == 0.0 for x in self._vecs[label])

    def get_vector(self, label):
        return list(self._vecs[label])

    def compact(self) -> int:
        self.compact_calls += 1
        return sum(1 for v in self._vecs.values() if any(x != 0.0 for x in v))


def test_one_compact_per_batch_delete() -> None:
    reg = CollectionIdRegistry()
    backend = CountingCompactBackend(10)
    for i in range(10):
        reg.register("users", f"u{i}", label=i)

    svc = ErasureService(reg, backend, drop_mappings=False)
    receipt = svc.delete(
        "users",
        ["u1", "u2", "u3", "u4"],
        compact="auto",
        residual_proof="off",
    )
    assert receipt.success
    assert receipt.compacted is True
    assert backend.compact_calls == 1
    assert len(receipt.labels) == 4


def test_compact_never_zero_calls() -> None:
    reg = CollectionIdRegistry()
    backend = CountingCompactBackend(5)
    for i in range(5):
        reg.register("users", f"u{i}", label=i)
    svc = ErasureService(reg, backend, drop_mappings=False)
    receipt = svc.delete(
        "users", ["u0", "u1"], compact="never", residual_proof="off"
    )
    assert receipt.success
    assert receipt.compacted is False
    assert backend.compact_calls == 0
