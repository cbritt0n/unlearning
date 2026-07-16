"""
Durable delete-intent outbox
============================

Filesystem outbox so replica fan-out survives process crashes:

1. After local hard-delete, ``enqueue`` writes intent JSON under ``outbox/pending/``.
2. A worker (or ``OutboxDispatcher.dispatch_pending``) sends to replicas.
3. On success, file moves to ``outbox/done/``; on permanent failure → ``outbox/failed/``.

This is a reference implementation (no Redis/SQS required). Swap the store
for SQS/NATS in production while keeping the same envelope shape.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class OutboxEnvelope:
    envelope_id: str
    request_id: str
    collection: str
    external_ids: list[str]
    reason: str | None = None
    max_m: int = 16
    attempts: int = 0
    created_unix_ns: int = field(default_factory=time.time_ns)
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutboxEnvelope":
        return cls(
            envelope_id=str(data["envelope_id"]),
            request_id=str(data["request_id"]),
            collection=str(data["collection"]),
            external_ids=list(data["external_ids"]),
            reason=data.get("reason"),
            max_m=int(data.get("max_m", 16)),
            attempts=int(data.get("attempts", 0)),
            created_unix_ns=int(data.get("created_unix_ns", 0)),
            last_error=str(data.get("last_error", "")),
        )


class FileOutbox:
    """JSON file outbox under ``root/outbox/{pending,done,failed}``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "outbox" / "pending"
        self.done = self.root / "outbox" / "done"
        self.failed = self.root / "outbox" / "failed"
        for d in (self.pending, self.done, self.failed):
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        collection: str,
        external_ids: Sequence[str],
        request_id: str,
        reason: str | None = None,
        max_m: int = 16,
    ) -> OutboxEnvelope:
        env = OutboxEnvelope(
            envelope_id=str(uuid.uuid4()),
            request_id=request_id,
            collection=collection,
            external_ids=[str(x) for x in external_ids],
            reason=reason,
            max_m=max_m,
        )
        path = self.pending / f"{env.envelope_id}.json"
        path.write_text(
            json.dumps(env.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return env

    def list_pending(self) -> list[OutboxEnvelope]:
        out: list[OutboxEnvelope] = []
        for p in sorted(self.pending.glob("*.json")):
            out.append(
                OutboxEnvelope.from_dict(
                    json.loads(p.read_text(encoding="utf-8"))
                )
            )
        return out

    def _path(self, folder: Path, envelope_id: str) -> Path:
        return folder / f"{envelope_id}.json"

    def mark_done(self, env: OutboxEnvelope) -> None:
        src = self._path(self.pending, env.envelope_id)
        dst = self._path(self.done, env.envelope_id)
        if src.is_file():
            shutil.move(str(src), str(dst))

    def mark_failed(self, env: OutboxEnvelope, error: str) -> None:
        env.attempts += 1
        env.last_error = error
        src = self._path(self.pending, env.envelope_id)
        dst = self._path(self.failed, env.envelope_id)
        if src.is_file():
            src.write_text(
                json.dumps(env.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            shutil.move(str(src), str(dst))

    def requeue_failed(self, envelope_id: str) -> None:
        src = self._path(self.failed, envelope_id)
        dst = self._path(self.pending, envelope_id)
        if src.is_file():
            shutil.move(str(src), str(dst))


# apply_fn(envelope) -> dict with success bool
DispatchFn = Callable[[OutboxEnvelope], dict[str, Any]]


class OutboxDispatcher:
    """Process pending outbox messages with a dispatch callback."""

    def __init__(
        self,
        outbox: FileOutbox,
        dispatch_fn: DispatchFn,
        *,
        max_attempts: int = 5,
    ) -> None:
        self.outbox = outbox
        self.dispatch_fn = dispatch_fn
        self.max_attempts = max_attempts

    def dispatch_pending(self, *, limit: int = 100) -> dict[str, Any]:
        done = 0
        failed = 0
        errors: list[str] = []
        for env in self.outbox.list_pending()[:limit]:
            try:
                result = self.dispatch_fn(env)
                ok = bool(result.get("success", result.get("quorum_met", True)))
                if ok:
                    self.outbox.mark_done(env)
                    done += 1
                else:
                    msg = str(result.get("message", result))
                    if env.attempts + 1 >= self.max_attempts:
                        self.outbox.mark_failed(env, msg)
                        failed += 1
                    else:
                        env.attempts += 1
                        env.last_error = msg
                        path = self.outbox._path(  # noqa: SLF001
                            self.outbox.pending, env.envelope_id
                        )
                        path.write_text(
                            json.dumps(env.to_dict(), indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        errors.append(msg)
            except Exception as exc:  # noqa: BLE001
                logger.exception("outbox dispatch failed")
                if env.attempts + 1 >= self.max_attempts:
                    self.outbox.mark_failed(env, str(exc))
                    failed += 1
                else:
                    env.attempts += 1
                    env.last_error = str(exc)
                    path = self.outbox._path(  # noqa: SLF001
                        self.outbox.pending, env.envelope_id
                    )
                    path.write_text(
                        json.dumps(env.to_dict(), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    errors.append(str(exc))
        return {
            "done": done,
            "failed": failed,
            "errors": errors,
            "pending_left": len(self.outbox.list_pending()),
        }


def make_outbox_replica_hook(
    outbox: FileOutbox,
    *,
    dispatch_now: OutboxDispatcher | None = None,
):
    """
    Workflow replica hook: enqueue intent; optionally dispatch immediately.
    """

    def _hook(
        collection: str, ids: Sequence[str], request_id: str
    ) -> dict[str, Any]:
        env = outbox.enqueue(
            collection=collection,
            external_ids=ids,
            request_id=request_id,
        )
        result: dict[str, Any] = {
            "enqueued": True,
            "envelope_id": env.envelope_id,
            "request_id": request_id,
        }
        if dispatch_now is not None:
            disp = dispatch_now.dispatch_pending(limit=1)
            result["dispatch"] = disp
            if disp.get("failed"):
                raise RuntimeError(f"outbox dispatch failed: {disp}")
        return result

    return _hook
