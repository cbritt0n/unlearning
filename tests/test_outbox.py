"""File outbox durability tests."""

from __future__ import annotations

from pathlib import Path

from integrations.outbox import FileOutbox, OutboxDispatcher, OutboxEnvelope


def test_enqueue_dispatch_done(tmp_path: Path) -> None:
    box = FileOutbox(tmp_path)
    env = box.enqueue(
        collection="c",
        external_ids=["a"],
        request_id="req-1",
    )
    assert len(box.list_pending()) == 1

    def ok(e: OutboxEnvelope):
        return {"success": True, "message": "ok"}

    disp = OutboxDispatcher(box, ok)
    result = disp.dispatch_pending()
    assert result["done"] == 1
    assert result["pending_left"] == 0
    assert (tmp_path / "outbox" / "done" / f"{env.envelope_id}.json").is_file()


def test_dispatch_failure_moves_after_max(tmp_path: Path) -> None:
    box = FileOutbox(tmp_path)
    env = box.enqueue(
        collection="c", external_ids=["a"], request_id="req-2"
    )

    def bad(e: OutboxEnvelope):
        return {"success": False, "message": "nope"}

    disp = OutboxDispatcher(box, bad, max_attempts=1)
    result = disp.dispatch_pending()
    assert result["failed"] == 1
    assert (tmp_path / "outbox" / "failed" / f"{env.envelope_id}.json").is_file()
