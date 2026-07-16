"""Unit tests for FastAPI endpoints, enterprise routes, and lock-retry."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import hnsw_healer
import api.main as main_mod
from api.main import app, with_lock_retry
from api.persistence import PersistenceEngine
from integrations.erase_service import ErasureService
from integrations.native_backend import NativeHealerBackend


@pytest.fixture
def client(persistence_engine, isolated_data_dir) -> TestClient:
    """HTTP client with lifespan; re-bind engine after startup."""
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


@pytest.fixture
def loaded_index(persistence_engine) -> dict:
    """Load a tiny index, checkpoint, return shape metadata."""
    n, d = 50, 8
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        nbrs = [(i - 1) % n, (i + 1) % n, (i + 2) % n]
        hnsw_healer.default_index().set_neighbors(i, 0, nbrs)
    persistence_engine.save_initial_checkpoint()
    return {"n": n, "d": d}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "lock_pool_size" in body
    assert "lock_timeout_ms" in body
    assert "wal_path" in body
    assert "index_path" in body


def test_delete_success(client: TestClient, loaded_index: dict) -> None:
    response = client.post("/delete", json={"node_id": 7, "max_m": 8})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "erased"
    assert data["node_id"] == 7
    assert isinstance(data["signature"], str)
    assert len(data["signature"]) == 64
    assert data.get("message") is not None
    assert "retries" in data
    assert data.get("transaction_id") is not None

    hnsw_healer.load_index_file(
        str(main_mod.engine.index_path)  # type: ignore[union-attr]
    )
    assert all(v == 0.0 for v in hnsw_healer.default_index().get_vector(7))


def test_delete_rejects_negative_node_id(client: TestClient) -> None:
    response = client.post("/delete", json={"node_id": -1})
    assert response.status_code == 422


def test_delete_requires_node_id(client: TestClient) -> None:
    response = client.post("/delete", json={})
    assert response.status_code == 422


def test_search_with_retry(client: TestClient, loaded_index: dict) -> None:
    query = np.zeros(loaded_index["d"], dtype=np.float32).tolist()
    response = client.post("/search", json={"query": query, "k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert len(data["hits"]) <= 3


def test_register_ids(client: TestClient) -> None:
    response = client.post(
        "/v1/ids/register",
        json={"collection": "users", "ids": ["alice", "bob"], "labels": [0, 1]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["collection"] == "users"
    assert body["mapping"]["alice"] == 0
    assert body["mapping"]["bob"] == 1


def test_register_ids_length_mismatch(client: TestClient) -> None:
    response = client.post(
        "/v1/ids/register",
        json={"collection": "users", "ids": ["a"], "labels": [0, 1]},
    )
    assert response.status_code == 400


def test_enterprise_delete_by_external_id(
    client: TestClient, loaded_index: dict
) -> None:
    reg = client.post(
        "/v1/ids/register",
        json={
            "collection": "users",
            "ids": [f"u{i}" for i in range(loaded_index["n"])],
            "labels": list(range(loaded_index["n"])),
        },
    )
    assert reg.status_code == 200

    # Re-bind erasure service so it uses current registry after register.
    main_mod._erasure_service = ErasureService(
        main_mod.id_registry,
        NativeHealerBackend(),
        persistence=main_mod.engine,
    )

    response = client.post(
        "/v1/collections/users/delete",
        json={
            "collection": "users",
            "ids": ["u7"],
            "reason": "gdpr_art_17",
            "request_id": "req-test-7",
            "max_m": 8,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["request_id"] == "req-test-7"
    assert data["labels"] == [7]
    assert data["bytes_wiped_total"] > 0
    assert len(data["signature"]) == 64
    assert data["errors"] == []
    assert data.get("receipt_version") == 2
    assert data.get("status") == "complete"
    assert "residual_proof" in data

    assert all(v == 0.0 for v in hnsw_healer.default_index().get_vector(7))
    assert not main_mod.id_registry.contains("users", "u7")


def test_enterprise_delete_collection_mismatch(
    client: TestClient, loaded_index: dict
) -> None:
    response = client.post(
        "/v1/collections/users/delete",
        json={"collection": "other", "ids": ["x"]},
    )
    assert response.status_code == 400


def test_with_lock_retry_succeeds_after_contention() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise hnsw_healer.LockContentionError("simulated contention")
        return "ok"

    assert with_lock_retry(flaky, max_attempts=5, base_delay_s=0.001) == "ok"
    assert calls["n"] == 3


def test_with_lock_retry_exhausts() -> None:
    def always_fail() -> None:
        raise hnsw_healer.LockContentionError("still busy")

    with pytest.raises(hnsw_healer.LockContentionError):
        with_lock_retry(always_fail, max_attempts=3, base_delay_s=0.001)


def test_ingest_vectors_and_delete(
    client: TestClient, persistence_engine
) -> None:
    response = client.post(
        "/v1/vectors/ingest",
        json={
            "collection": "docs",
            "ids": ["a", "b", "c"],
            "vectors": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.1, 0.2, 0.3],
            ],
            "replace_index": True,
            "checkpoint": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert "a" in body["mapping"]

    main_mod._erasure_service = ErasureService(
        main_mod.id_registry,
        NativeHealerBackend(),
        persistence=main_mod.engine,
        receipt_log=None,
        default_residual_proof="off",
    )
    deleted = client.post(
        "/v1/collections/docs/delete",
        json={
            "collection": "docs",
            "ids": ["b"],
            "reason": "test",
            "residual_proof": "off",
        },
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["success"] is True


def test_metrics_endpoints(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "hnsw_healer_info" in r.text
    j = client.get("/v1/metrics")
    assert j.status_code == 200
    assert "counters" in j.json()


def test_workflow_api_create_and_export(
    client: TestClient, loaded_index: dict
) -> None:
    reg = client.post(
        "/v1/ids/register",
        json={
            "collection": "users",
            "ids": [f"u{i}" for i in range(loaded_index["n"])],
            "labels": list(range(loaded_index["n"])),
        },
    )
    assert reg.status_code == 200
    main_mod._erasure_service = ErasureService(
        main_mod.id_registry,
        NativeHealerBackend(),
        persistence=main_mod.engine,
    )
    main_mod._workflow_runner = None
    main_mod._workflow_store = None

    created = client.post(
        "/v1/erasure-requests",
        json={
            "collection": "users",
            "ids": ["u4"],
            "reason": "gdpr_art_17",
            "request_id": "wf-api-1",
            "advance": True,
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["request_id"] == "wf-api-1"
    assert body["status"] in ("complete", "blocked", "in_progress")

    exported = client.get("/v1/erasure-requests/wf-api-1/export")
    assert exported.status_code == 200
    assert exported.json()["kind"] == "hnsw_healer.erasure_export"
