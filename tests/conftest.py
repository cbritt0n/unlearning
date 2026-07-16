"""
Shared pytest fixtures for the hnsw-healer test suite.

Ensures each test gets an isolated data directory and a clean enterprise
id registry / erasure service when exercising the FastAPI app.
"""

from __future__ import annotations

import pytest

import api.main as main_mod
from api.main import with_lock_retry
from api.persistence import PersistenceEngine
from integrations.id_registry import CollectionIdRegistry


@pytest.fixture(autouse=True)
def _safe_healer_env(monkeypatch):
    """
    Isolate auth/signing env so developer machine settings do not break tests.
    Individual tests may override these via monkeypatch.
    """
    monkeypatch.delenv("HEALER_API_KEY", raising=False)
    monkeypatch.delenv("HEALER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("HEALER_ENV", "development")
    monkeypatch.delenv("HEALER_ALLOW_INSECURE", raising=False)
    # Residual proofs on by default in product; tests may override.
    monkeypatch.setenv("HEALER_RESIDUAL_PROOF", "sample")
    yield


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point HEALER_DATA_DIR at a temp directory for the test process."""
    monkeypatch.setenv("HEALER_DATA_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def fresh_registry():
    """Replace the process-global id registry with an empty one."""
    previous = main_mod.id_registry
    main_mod.id_registry = CollectionIdRegistry()
    main_mod._erasure_service = None
    yield main_mod.id_registry
    main_mod.id_registry = previous
    main_mod._erasure_service = None


@pytest.fixture
def persistence_engine(isolated_data_dir, fresh_registry):
    """PersistenceEngine bound to the isolated data dir."""
    eng = PersistenceEngine(isolated_data_dir, lock_retry=with_lock_retry)
    main_mod.engine = eng
    main_mod._bootstrap_info = {}
    main_mod._erasure_service = None
    main_mod._workflow_store = None
    main_mod._workflow_runner = None
    yield eng
    main_mod.engine = None
    main_mod._erasure_service = None
    main_mod._workflow_store = None
    main_mod._workflow_runner = None
