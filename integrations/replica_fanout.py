r"""
Multi-replica delete fan-out
============================

Physical wipe on one node is insufficient if another replica still holds
``v`` (latent embedding). This module is a **reference implementation** of
intent fan-out + quorum acknowledgement.

Typical flow
------------
1. Primary runs local ``ErasureService.delete`` (or WAL hard-delete).
2. ``ReplicaFanoutCoordinator.publish`` sends the same intent to workers.
3. Each worker applies hard-delete on its local backend and ACKs.
4. Coordinator waits for quorum before marking the GDPR request complete.

Transports
----------
- ``InProcessTransport`` — unit tests / single binary multi-backend demos
- ``HttpReplicaTransport`` — POST JSON to peer unlearning sidecars
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

logger = logging.getLogger(__name__)


@dataclass
class DeleteIntent:
    """Replicated hard-delete intent (idempotent by request_id)."""

    request_id: str
    collection: str
    external_ids: list[str]
    reason: str | None = None
    max_m: int = 16
    timestamp_unix_ns: int = field(default_factory=time.time_ns)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str | bytes) -> "DeleteIntent":
        data = json.loads(raw)
        return DeleteIntent(
            request_id=str(data["request_id"]),
            collection=str(data["collection"]),
            external_ids=list(data["external_ids"]),
            reason=data.get("reason"),
            max_m=int(data.get("max_m", 16)),
            timestamp_unix_ns=int(
                data.get("timestamp_unix_ns", time.time_ns())
            ),
        )


@dataclass
class ReplicaAck:
    replica_id: str
    request_id: str
    success: bool
    message: str = ""
    latency_ms: float = 0.0


class ReplicaTransport(Protocol):
    def send(self, replica_id: str, intent: DeleteIntent) -> ReplicaAck:
        ...


class ReplicaWorker:
    """
    Applies delete intents on a local erase callback.

    ``apply_fn(collection, ids, reason, max_m) -> (success, message)``
    typically wraps ``ErasureService.delete``.
    """

    def __init__(
        self,
        replica_id: str,
        apply_fn: Callable[..., tuple[bool, str]],
    ) -> None:
        self.replica_id = replica_id
        self._apply = apply_fn
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def handle(self, intent: DeleteIntent) -> ReplicaAck:
        t0 = time.perf_counter()
        with self._lock:
            if intent.request_id in self._seen:
                return ReplicaAck(
                    replica_id=self.replica_id,
                    request_id=intent.request_id,
                    success=True,
                    message="idempotent_replay",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
        try:
            ok, msg = self._apply(
                intent.collection,
                intent.external_ids,
                intent.reason,
                intent.max_m,
            )
            if ok:
                with self._lock:
                    self._seen.add(intent.request_id)
            return ReplicaAck(
                replica_id=self.replica_id,
                request_id=intent.request_id,
                success=bool(ok),
                message=msg,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("replica %s failed", self.replica_id)
            return ReplicaAck(
                replica_id=self.replica_id,
                request_id=intent.request_id,
                success=False,
                message=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )


class InProcessTransport:
    """Routes intents to in-process ``ReplicaWorker`` instances."""

    def __init__(self, workers: dict[str, ReplicaWorker]) -> None:
        self._workers = dict(workers)

    def send(self, replica_id: str, intent: DeleteIntent) -> ReplicaAck:
        if replica_id not in self._workers:
            return ReplicaAck(
                replica_id=replica_id,
                request_id=intent.request_id,
                success=False,
                message=f"unknown replica {replica_id}",
            )
        return self._workers[replica_id].handle(intent)


class HttpReplicaTransport:
    """
    POST ``DeleteIntent`` JSON to peer sidecars.

    Each replica URL should accept ``POST {path}`` with the intent body and
    return ``{"success": bool, "message": str}``.
    """

    def __init__(
        self,
        endpoints: dict[str, str],
        *,
        timeout_s: float = 30.0,
        path: str = "/v1/internal/replica/delete",
    ) -> None:
        # replica_id -> base URL (e.g. http://replica-b:8000)
        self.endpoints = dict(endpoints)
        self.timeout_s = timeout_s
        self.path = path

    def send(self, replica_id: str, intent: DeleteIntent) -> ReplicaAck:
        base = self.endpoints.get(replica_id)
        if not base:
            return ReplicaAck(
                replica_id=replica_id,
                request_id=intent.request_id,
                success=False,
                message="unknown replica endpoint",
            )
        url = base.rstrip("/") + self.path
        t0 = time.perf_counter()
        data = intent.to_json().encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return ReplicaAck(
                replica_id=replica_id,
                request_id=intent.request_id,
                success=bool(body.get("success")),
                message=str(body.get("message", "")),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError) as exc:
            return ReplicaAck(
                replica_id=replica_id,
                request_id=intent.request_id,
                success=False,
                message=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )


@dataclass
class FanoutResult:
    request_id: str
    quorum: int
    acks: list[ReplicaAck]
    quorum_met: bool

    @property
    def successes(self) -> int:
        return sum(1 for a in self.acks if a.success)


class ReplicaFanoutCoordinator:
    """
    Publish delete intents to all replicas and wait for quorum.

    Parameters
    ----------
    replica_ids:
        Peer replica identifiers (excluding or including primary).
    transport:
        ``InProcessTransport`` or ``HttpReplicaTransport``.
    quorum:
        Minimum successful ACKs required (default: majority of replicas).
    """

    def __init__(
        self,
        replica_ids: Sequence[str],
        transport: ReplicaTransport,
        *,
        quorum: int | None = None,
        max_workers: int = 8,
    ) -> None:
        self.replica_ids = list(replica_ids)
        if not self.replica_ids:
            raise ValueError("replica_ids must be non-empty")
        self.transport = transport
        self.quorum = (
            quorum
            if quorum is not None
            else max(1, (len(self.replica_ids) // 2) + 1)
        )
        self.max_workers = max_workers

    def publish(
        self,
        collection: str,
        external_ids: Sequence[str],
        *,
        reason: str | None = None,
        max_m: int = 16,
        request_id: str | None = None,
    ) -> FanoutResult:
        intent = DeleteIntent(
            request_id=request_id or str(uuid.uuid4()),
            collection=collection,
            external_ids=[str(x) for x in external_ids],
            reason=reason,
            max_m=max_m,
        )
        acks: list[ReplicaAck] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {
                pool.submit(self.transport.send, rid, intent): rid
                for rid in self.replica_ids
            }
            for fut in as_completed(futs):
                acks.append(fut.result())

        successes = sum(1 for a in acks if a.success)
        result = FanoutResult(
            request_id=intent.request_id,
            quorum=self.quorum,
            acks=acks,
            quorum_met=successes >= self.quorum,
        )
        if not result.quorum_met:
            logger.warning(
                "fanout quorum not met for %s (%s/%s)",
                intent.request_id,
                successes,
                self.quorum,
            )
        return result

    def publish_after_local(
        self,
        local_success: bool,
        collection: str,
        external_ids: Sequence[str],
        **kwargs: Any,
    ) -> FanoutResult:
        """
        Convenience: only fan out if the primary local delete succeeded.
        """
        if not local_success:
            rid = kwargs.get("request_id") or str(uuid.uuid4())
            return FanoutResult(
                request_id=str(rid),
                quorum=self.quorum,
                acks=[],
                quorum_met=False,
            )
        return self.publish(collection, external_ids, **kwargs)
