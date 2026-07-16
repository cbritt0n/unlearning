"""Multi-tenant path and key derivation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from integrations.tenancy import (
    TenantManager,
    derive_tenant_signing_key,
    validate_tenant_id,
)


def test_validate_tenant_id() -> None:
    assert validate_tenant_id("acme_prod") == "acme_prod"
    with pytest.raises(ValueError):
        validate_tenant_id("../etc")


def test_derive_keys_differ() -> None:
    master = b"master-secret-key-material"
    a = derive_tenant_signing_key(master, "tenant-a")
    b = derive_tenant_signing_key(master, "tenant-b")
    assert a != b
    assert len(a) == 32


def test_tenant_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HEALER_MULTI_TENANT", "1")
    mgr = TenantManager(tmp_path, master_signing_key=b"k" * 16)
    ta = mgr.get("alpha")
    tb = mgr.get("beta")
    assert ta.data_dir != tb.data_dir
    assert ta.data_dir.is_dir()
    assert "alpha" in mgr.list_tenants()
