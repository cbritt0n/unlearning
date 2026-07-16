"""API key middleware and production signing-key guards."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import hnsw_healer
import api.main as main_mod
from api.auth import DEFAULT_SIGNING_KEY, validate_production_secrets
from api.main import app, with_lock_retry
from api.persistence import PersistenceEngine
from integrations.erase_service import ErasureService
from integrations.native_backend import NativeHealerBackend


@pytest.fixture
def client_with_key(persistence_engine, isolated_data_dir, monkeypatch):
    monkeypatch.setenv("HEALER_API_KEY", "test-secret-key")
    monkeypatch.delenv("HEALER_ENV", raising=False)
    monkeypatch.setenv("HEALER_ALLOW_INSECURE", "0")
    # Re-import middleware behavior reads env at request time.
    with TestClient(app, raise_server_exceptions=True) as c:
        main_mod.engine = PersistenceEngine(
            isolated_data_dir, lock_retry=with_lock_retry
        )
        main_mod._erasure_service = ErasureService(
            main_mod.id_registry,
            NativeHealerBackend(),
            persistence=main_mod.engine,
        )
        yield c


def test_health_public_with_auth(client_with_key: TestClient) -> None:
    r = client_with_key.get("/health")
    assert r.status_code == 200


def test_delete_requires_api_key(client_with_key: TestClient) -> None:
    r = client_with_key.post("/delete", json={"node_id": 0})
    assert r.status_code == 401


def test_delete_accepts_api_key_header(
    client_with_key: TestClient, persistence_engine
) -> None:
    n, d = 10, 4
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])
    persistence_engine.save_initial_checkpoint()

    r = client_with_key.post(
        "/delete",
        json={"node_id": 1, "max_m": 4},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert r.status_code == 200, r.text


def test_delete_accepts_bearer(
    client_with_key: TestClient, persistence_engine
) -> None:
    n, d = 10, 4
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])
    persistence_engine.save_initial_checkpoint()

    r = client_with_key.post(
        "/delete",
        json={"node_id": 2, "max_m": 4},
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert r.status_code == 200, r.text


def test_production_rejects_default_signing_key(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ENV", "production")
    monkeypatch.setenv("HEALER_SIGNING_KEY", DEFAULT_SIGNING_KEY)
    monkeypatch.setenv("HEALER_API_KEY", "something")
    monkeypatch.delenv("HEALER_ALLOW_INSECURE", raising=False)
    with pytest.raises(RuntimeError, match="non-default HEALER_SIGNING_KEY"):
        validate_production_secrets()


def test_production_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ENV", "production")
    monkeypatch.setenv("HEALER_SIGNING_KEY", "a-strong-production-signing-key")
    monkeypatch.delenv("HEALER_API_KEY", raising=False)
    monkeypatch.delenv("HEALER_ALLOW_INSECURE", raising=False)
    with pytest.raises(RuntimeError, match="HEALER_API_KEY"):
        validate_production_secrets()


def test_production_ok_with_secrets(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ENV", "production")
    monkeypatch.setenv("HEALER_SIGNING_KEY", "a-strong-production-signing-key")
    monkeypatch.setenv("HEALER_API_KEY", "prod-api-key")
    monkeypatch.delenv("HEALER_ALLOW_INSECURE", raising=False)
    validate_production_secrets()  # does not raise
