"""
Durable queue transports for replica outbox fan-out
===================================================

Complements ``FileOutbox`` with interchangeable backends:

- ``FileQueueTransport`` — same directory layout as FileOutbox envelopes
- ``RedisListTransport`` — optional redis RPUSH/BLPOP (requires redis)
- ``SqsTransport`` — optional AWS SQS (requires boto3)

Envelope JSON matches ``OutboxEnvelope.to_dict()``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    message_id: str
    body: dict[str, Any]
    receipt_handle: str = ""
    enqueued_unix_ns: int = field(default_factory=time.time_ns)


class QueueTransport(Protocol):
    def enqueue(self, body: dict[str, Any]) -> str:
        """Return message id."""
        ...

    def dequeue(self, *, max_messages: int = 1, wait_s: float = 0) -> list[QueueMessage]:
        ...

    def ack(self, message: QueueMessage) -> None:
        ...


class FileQueueTransport:
    """Directory-based queue (pending/ → processing/ → done/)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "queue" / "pending"
        self.processing = self.root / "queue" / "processing"
        self.done = self.root / "queue" / "done"
        for d in (self.pending, self.processing, self.done):
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(self, body: dict[str, Any]) -> str:
        mid = str(uuid.uuid4())
        msg = QueueMessage(message_id=mid, body=body)
        path = self.pending / f"{mid}.json"
        path.write_text(
            json.dumps(asdict(msg), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return mid

    def dequeue(
        self, *, max_messages: int = 1, wait_s: float = 0
    ) -> list[QueueMessage]:
        del wait_s
        out: list[QueueMessage] = []
        for path in sorted(self.pending.glob("*.json"))[:max_messages]:
            data = json.loads(path.read_text(encoding="utf-8"))
            msg = QueueMessage(
                message_id=str(data["message_id"]),
                body=dict(data["body"]),
                receipt_handle=str(path),
                enqueued_unix_ns=int(data.get("enqueued_unix_ns", 0)),
            )
            dest = self.processing / path.name
            path.replace(dest)
            msg.receipt_handle = str(dest)
            out.append(msg)
        return out

    def ack(self, message: QueueMessage) -> None:
        src = Path(message.receipt_handle) if message.receipt_handle else None
        if src and src.is_file():
            dest = self.done / src.name
            src.replace(dest)


class RedisListTransport:
    """
    Redis list queue: RPUSH enqueue, LPOP dequeue.

    Install: ``pip install redis``
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key: str = "hnsw_healer:delete_outbox",
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._r = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "redis package required: pip install redis"
                ) from exc
            self._r = redis.Redis.from_url(url, decode_responses=True)
        self.key = key

    def enqueue(self, body: dict[str, Any]) -> str:
        mid = str(uuid.uuid4())
        payload = json.dumps(
            {"message_id": mid, "body": body, "enqueued_unix_ns": time.time_ns()}
        )
        self._r.rpush(self.key, payload)
        return mid

    def dequeue(
        self, *, max_messages: int = 1, wait_s: float = 0
    ) -> list[QueueMessage]:
        out: list[QueueMessage] = []
        for _ in range(max_messages):
            if wait_s > 0 and hasattr(self._r, "blpop"):
                item = self._r.blpop(self.key, timeout=max(1, int(wait_s)))
                raw = item[1] if item else None
            else:
                raw = self._r.lpop(self.key)
            if not raw:
                break
            data = json.loads(raw)
            out.append(
                QueueMessage(
                    message_id=str(data["message_id"]),
                    body=dict(data["body"]),
                    receipt_handle=str(data["message_id"]),
                    enqueued_unix_ns=int(data.get("enqueued_unix_ns", 0)),
                )
            )
        return out

    def ack(self, message: QueueMessage) -> None:
        # LPOP already removed; ack is a no-op (at-most-once). For at-least-once
        # use a processing list + visibility timeout in production.
        del message


class SqsTransport:
    """
    AWS SQS transport.

    Install: ``pip install boto3``
    """

    def __init__(
        self,
        queue_url: str,
        *,
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self.queue_url = queue_url
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "boto3 required: pip install boto3"
                ) from exc
            self._client = boto3.client("sqs", region_name=region_name)

    def enqueue(self, body: dict[str, Any]) -> str:
        mid = str(uuid.uuid4())
        resp = self._client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(
                {
                    "message_id": mid,
                    "body": body,
                    "enqueued_unix_ns": time.time_ns(),
                }
            ),
        )
        return str(resp.get("MessageId", mid))

    def dequeue(
        self, *, max_messages: int = 1, wait_s: float = 0
    ) -> list[QueueMessage]:
        resp = self._client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=int(min(max(wait_s, 0), 20)),
        )
        out: list[QueueMessage] = []
        for m in resp.get("Messages", []) or []:
            data = json.loads(m["Body"])
            out.append(
                QueueMessage(
                    message_id=str(data.get("message_id", m["MessageId"])),
                    body=dict(data.get("body", data)),
                    receipt_handle=str(m["ReceiptHandle"]),
                    enqueued_unix_ns=int(data.get("enqueued_unix_ns", 0)),
                )
            )
        return out

    def ack(self, message: QueueMessage) -> None:
        self._client.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=message.receipt_handle,
        )


def enqueue_delete_intent(
    transport: QueueTransport,
    *,
    collection: str,
    external_ids: Sequence[str],
    request_id: str,
    reason: str | None = None,
    max_m: int = 16,
) -> str:
    """Helper: push a standard delete-intent body onto any transport."""
    return transport.enqueue(
        {
            "kind": "hnsw_healer.delete_intent",
            "request_id": request_id,
            "collection": collection,
            "external_ids": list(external_ids),
            "reason": reason,
            "max_m": max_m,
        }
    )
