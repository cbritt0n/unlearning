"""
Multi-tenant isolation
======================

Isolates durable state per tenant:

  {HEALER_DATA_DIR}/tenants/{tenant_id}/
      index.bin, index.wal, id_registry.json,
      receipts.jsonl, workflows/, outbox/

Signing keys can be derived per tenant from a master key (HMAC) so tenant A
cannot forge tenant B receipts without the master secret.

HTTP: pass ``X-Tenant-ID`` (or ``tenant`` query) when ``HEALER_MULTI_TENANT=1``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
from pathlib import Path
from typing import Any

_TENANT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$")


def multi_tenant_enabled() -> bool:
    return os.environ.get("HEALER_MULTI_TENANT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def validate_tenant_id(tenant_id: str) -> str:
    tid = (tenant_id or "").strip()
    if not tid:
        raise ValueError("tenant_id required")
    if not _TENANT_RE.match(tid):
        raise ValueError(
            "tenant_id must be 1-64 chars: alphanumeric, _ . : -"
        )
    return tid


def derive_tenant_signing_key(
    master_key: bytes, tenant_id: str
) -> bytes:
    """HMAC-SHA256 derive a per-tenant signing key from master."""
    tid = validate_tenant_id(tenant_id)
    return hmac.new(master_key, tid.encode("utf-8"), hashlib.sha256).digest()


class TenantContext:
    """Resolved tenant paths and secrets."""

    def __init__(
        self,
        tenant_id: str,
        data_dir: Path,
        signing_key: bytes,
    ) -> None:
        self.tenant_id = tenant_id
        self.data_dir = data_dir
        self.signing_key = signing_key
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def receipts_path(self) -> Path:
        return self.data_dir / "receipts.jsonl"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "id_registry.json"


class TenantManager:
    """
    Resolves tenant roots under ``base_dir/tenants/{id}``.

    When multi-tenant is disabled, ``get(None)`` returns the base_dir itself
    (single-tenant mode).
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        master_signing_key: bytes | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        key = master_signing_key or os.environ.get(
            "HEALER_SIGNING_KEY", "dev-only-placeholder-signing-key"
        )
        self._master = (
            key if isinstance(key, bytes) else str(key).encode("utf-8")
        )
        self._lock = threading.RLock()
        self._cache: dict[str, TenantContext] = {}

    def get(self, tenant_id: str | None) -> TenantContext:
        if not multi_tenant_enabled() or not tenant_id:
            # Single-tenant: use base data dir
            return TenantContext(
                tenant_id=tenant_id or "default",
                data_dir=self.base_dir,
                signing_key=self._master,
            )
        tid = validate_tenant_id(tenant_id)
        with self._lock:
            if tid not in self._cache:
                tdir = self.base_dir / "tenants" / tid
                sk = derive_tenant_signing_key(self._master, tid)
                self._cache[tid] = TenantContext(tid, tdir, sk)
            return self._cache[tid]

    def list_tenants(self) -> list[str]:
        root = self.base_dir / "tenants"
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())
