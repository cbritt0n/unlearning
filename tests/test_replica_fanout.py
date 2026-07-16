"""Multi-replica delete fan-out reference workers."""

from __future__ import annotations

from integrations.replica_fanout import (
    DeleteIntent,
    InProcessTransport,
    ReplicaFanoutCoordinator,
    ReplicaWorker,
)


def test_inprocess_fanout_quorum() -> None:
    applied: dict[str, list[str]] = {"a": [], "b": [], "c": []}

    def make_apply(rid: str):
        def _apply(collection, ids, reason, max_m):
            applied[rid].extend(ids)
            return True, f"{rid}-ok"

        return _apply

    workers = {
        rid: ReplicaWorker(rid, make_apply(rid)) for rid in ("a", "b", "c")
    }
    transport = InProcessTransport(workers)
    coord = ReplicaFanoutCoordinator(
        ["a", "b", "c"], transport, quorum=2
    )
    result = coord.publish(
        "users", ["u1", "u2"], reason="gdpr", request_id="req-1"
    )
    assert result.quorum_met
    assert result.successes == 3
    assert applied["a"] == ["u1", "u2"]
    assert applied["c"] == ["u1", "u2"]


def test_idempotent_replay() -> None:
    calls = {"n": 0}

    def apply(collection, ids, reason, max_m):
        calls["n"] += 1
        return True, "ok"

    w = ReplicaWorker("r1", apply)
    intent = DeleteIntent(
        request_id="same",
        collection="c",
        external_ids=["x"],
    )
    a1 = w.handle(intent)
    a2 = w.handle(intent)
    assert a1.success and a2.success
    assert a2.message == "idempotent_replay"
    assert calls["n"] == 1


def test_quorum_failure() -> None:
    def ok(collection, ids, reason, max_m):
        return True, "ok"

    def bad(collection, ids, reason, max_m):
        return False, "fail"

    workers = {
        "a": ReplicaWorker("a", ok),
        "b": ReplicaWorker("b", bad),
        "c": ReplicaWorker("c", bad),
    }
    coord = ReplicaFanoutCoordinator(
        ["a", "b", "c"], InProcessTransport(workers), quorum=2
    )
    result = coord.publish("c", ["id"])
    assert not result.quorum_met
    assert result.successes == 1
