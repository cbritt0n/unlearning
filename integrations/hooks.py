"""
First-class workflow hooks
==========================

Reference implementations for non-live erasure steps:

- ``HttpWebhookHook`` — POST JSON to document-store / backup / generic URLs
- ``CryptoShredHookFactory`` — wraps ``CryptoShredVault`` / KMS vault
- ``HttpReplicaFanoutHook`` — durable-friendly wrapper around fan-out
- ``BackupPolicyHook`` — records backup-ack policy decision

Wire these into ``ErasureWorkflowRunner`` (see docs/HOOKS.md).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

logger = logging.getLogger(__name__)


class HttpWebhookHook:
    """
    Generic HTTP POST webhook.

    Body shape::

        {
          "event": "document_store_delete" | "backup_ack" | ...,
          "collection": "...",
          "ids": ["..."],
          "request_id": "...",
          "reason": "...",
          "timestamp_unix_ns": 0
        }
    """

    def __init__(
        self,
        url: str,
        *,
        event: str,
        timeout_s: float = 10.0,
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.event = event
        self.timeout_s = timeout_s
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Type", "application/json")
        if api_key:
            self.headers["X-API-Key"] = api_key

    def __call__(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        request_id: str = "",
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "event": self.event,
            "collection": collection,
            "ids": list(ids),
            "request_id": request_id,
            "reason": reason,
            "timestamp_unix_ns": time.time_ns(),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.url, data=data, headers=self.headers, method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", 200)
                if status >= 400:
                    raise RuntimeError(f"webhook HTTP {status}: {body[:200]}")
                return {
                    "ok": True,
                    "http_status": status,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "response": body[:500],
                    "url": self.url,
                    "event": self.event,
                }
        except urlerror.HTTPError as exc:
            raise RuntimeError(
                f"webhook HTTP {exc.code}: {exc.read()[:200]!r}"
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"webhook unreachable: {exc}") from exc

    def as_document_store_hook(self):
        def _hook(collection: str, ids: Sequence[str]) -> dict[str, Any]:
            return self(collection, ids)

        return _hook

    def as_backup_ack_hook(self):
        def _hook(collection: str, ids: Sequence[str]) -> dict[str, Any]:
            return self(collection, ids)

        return _hook


def make_crypto_shred_hook(vault: Any):
    """
    Build a crypto-shred hook from ``CryptoShredVault`` or ``KmsCryptoShredVault``.

    Expects ``vault.shred(entity_id)`` returning a receipt-like object or bool.
    """

    def _hook(ids: Sequence[str]) -> dict[str, Any]:
        results = []
        for eid in ids:
            try:
                if hasattr(vault, "shred"):
                    r = vault.shred(str(eid))
                    if hasattr(r, "shredded"):
                        results.append(
                            {
                                "entity_id": str(eid),
                                "shredded": bool(r.shredded),
                                "message": getattr(r, "message", "ok"),
                            }
                        )
                    else:
                        results.append(
                            {"entity_id": str(eid), "shredded": bool(r)}
                        )
                else:
                    raise TypeError("vault has no shred()")
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "entity_id": str(eid),
                        "shredded": False,
                        "error": str(exc),
                    }
                )
        failed = [r for r in results if not r.get("shredded")]
        if failed:
            raise RuntimeError(f"crypto_shred failed for: {failed}")
        return {"ok": True, "results": results}

    return _hook


def make_replica_fanout_hook(coordinator: Any):
    """
    Wrap ``ReplicaFanoutCoordinator.publish`` for workflow use.
    """

    def _hook(
        collection: str, ids: Sequence[str], request_id: str
    ) -> dict[str, Any]:
        result = coordinator.publish(
            collection, list(ids), request_id=request_id
        )
        # Coordinator may return object or dict
        if hasattr(result, "quorum_met"):
            payload = {
                "quorum_met": bool(result.quorum_met),
                "acks": [
                    {
                        "replica_id": getattr(a, "replica_id", "?"),
                        "success": getattr(a, "success", False),
                        "message": getattr(a, "message", ""),
                    }
                    for a in getattr(result, "acks", [])
                ],
            }
            if not result.quorum_met:
                raise RuntimeError(f"replica quorum not met: {payload}")
            return payload
        if isinstance(result, dict):
            if not result.get("quorum_met", True):
                raise RuntimeError(f"replica quorum not met: {result}")
            return result
        return {"result": str(result)}

    return _hook


class LocalBackupAckHook:
    """
    Dev/test backup ack: records that operator policy accepted backup expiry.

    Production should replace with a webhook that checks snapshot retention.
    """

    def __init__(self, *, auto_ack: bool = True) -> None:
        self.auto_ack = auto_ack
        self.acks: list[dict[str, Any]] = []

    def __call__(
        self, collection: str, ids: Sequence[str]
    ) -> dict[str, Any]:
        if not self.auto_ack:
            raise RuntimeError(
                "backup ack requires operator confirmation (auto_ack=False)"
            )
        rec = {
            "collection": collection,
            "ids": list(ids),
            "acked": True,
            "policy": "local_auto_ack_dev_only",
            "timestamp_unix_ns": time.time_ns(),
        }
        self.acks.append(rec)
        return rec
