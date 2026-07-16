"""Erasure workflow persistence + runner tests."""

from __future__ import annotations

import numpy as np
import pytest

import hnsw_healer
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.native_backend import NativeHealerBackend
from integrations.workflow import (
    ErasureWorkflow,
    ErasureWorkflowRunner,
    ErasureWorkflowStore,
)


@pytest.fixture
def loaded_native() -> None:
    n, d = 20, 8
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])
    yield


def test_workflow_create_advance_export(tmp_path, loaded_native: None) -> None:
    reg = CollectionIdRegistry()
    for i in range(20):
        reg.register("users", f"u{i}", label=i)
    svc = ErasureService(reg, NativeHealerBackend())
    store = ErasureWorkflowStore(tmp_path)
    runner = ErasureWorkflowRunner(store, svc)

    wf = runner.create(
        "users",
        ["u2", "u3"],
        reason="gdpr_art_17",
        request_id="ticket-1",
    )
    assert wf.status == "open"
    assert (tmp_path / "workflows" / "ticket-1.json").is_file()

    wf = runner.advance("ticket-1")
    assert wf.status == "complete"
    assert wf.receipt is not None
    assert wf.receipt["success"] is True
    assert wf.step("live_hard_delete").status == "done"
    assert wf.step("residual_proof").status in ("done", "skipped")

    package = wf.export_package()
    assert package["closeable"] is True
    assert package["summary"]["request_id"] == "ticket-1"

    reloaded = store.load("ticket-1")
    assert reloaded.status == "complete"


def test_workflow_blocked_without_required_hook(tmp_path, loaded_native: None) -> None:
    reg = CollectionIdRegistry()
    for i in range(20):
        reg.register("users", f"u{i}", label=i)
    svc = ErasureService(reg, NativeHealerBackend())
    store = ErasureWorkflowStore(tmp_path)
    runner = ErasureWorkflowRunner(store, svc)

    runner.create(
        "users",
        ["u1"],
        request_id="need-backup",
        require_backup_ack=True,
    )
    wf = runner.advance("need-backup")
    assert wf.status == "blocked"
    assert wf.step("backup_policy").status == "failed"
