"""File queue transport tests."""

from __future__ import annotations

from pathlib import Path

from integrations.queue_transports import (
    FileQueueTransport,
    enqueue_delete_intent,
)


def test_file_queue_roundtrip(tmp_path: Path) -> None:
    q = FileQueueTransport(tmp_path)
    mid = enqueue_delete_intent(
        q,
        collection="docs",
        external_ids=["u1", "u2"],
        request_id="req-9",
        reason="gdpr",
    )
    assert mid
    msgs = q.dequeue(max_messages=5)
    assert len(msgs) == 1
    assert msgs[0].body["request_id"] == "req-9"
    assert msgs[0].body["external_ids"] == ["u1", "u2"]
    q.ack(msgs[0])
    assert q.dequeue() == []
    assert list((tmp_path / "queue" / "done").glob("*.json"))
