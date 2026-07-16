"""Append-only receipt log tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import hnsw_healer
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.native_backend import NativeHealerBackend
from integrations.receipt_log import AppendOnlyReceiptLog


def test_receipt_log_appends(tmp_path: Path) -> None:
    n, d = 12, 4
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])

    log = AppendOnlyReceiptLog(tmp_path / "receipts.jsonl")
    reg = CollectionIdRegistry()
    for i in range(n):
        reg.register("c", f"id{i}", label=i)
    svc = ErasureService(
        reg,
        NativeHealerBackend(),
        receipt_log=log,
        default_residual_proof="off",
    )
    r1 = svc.delete("c", ["id1"], request_id="r-1")
    r2 = svc.delete("c", ["id2"], request_id="r-2")
    assert r1.success and r2.success
    assert log.count() == 2
    found = log.find_by_request_id("r-1")
    assert len(found) == 1
    assert found[0]["labels"] == [1]
