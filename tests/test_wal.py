"""Unit tests for WAL records and crash-recovery helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hnsw_healer
from api.persistence import PersistenceEngine
from api.wal import WriteAheadLog


def test_wal_begin_commit_roundtrip(tmp_path: Path) -> None:
    wal = WriteAheadLog(tmp_path / "index.wal")
    begin = wal.append_begin_delete(42, max_m=8)
    assert begin.target_node_id == 42
    assert begin.max_m == 8
    assert len(begin.checksum) == 32

    assert len(wal.uncommitted_deletes()) == 1
    wal.append_commit(begin.transaction_id)
    assert wal.uncommitted_deletes() == []


def test_wal_checksum_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "index.wal"
    wal = WriteAheadLog(path)
    wal.append_begin_delete(1, max_m=4)

    raw = bytearray(path.read_bytes())
    # Flip a byte inside the checksum region of the first record.
    raw[-1] ^= 0xFF
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="checksum"):
        list(WriteAheadLog(path).iter_records())


def test_hard_delete_writes_index_and_commit(tmp_path: Path) -> None:
    n, d = 20, 4
    data = np.random.randn(n, d).astype(np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(
            i, 0, [(i + 1) % n, (i + 2) % n]
        )

    eng = PersistenceEngine(tmp_path)
    eng.save_initial_checkpoint()
    assert (tmp_path / "index.bin").is_file()

    result = eng.hard_delete_and_heal(3, max_m=8)
    assert result.success
    assert result.transaction_id > 0
    assert eng.wal.uncommitted_deletes() == []

    # Vector should be zeros in live memory.
    vec = hnsw_healer.default_index().get_vector(3)
    assert all(v == 0.0 for v in vec)

    # Reload from disk — zeros must persist (no resurrection).
    hnsw_healer.load_index_file(str(tmp_path / "index.bin"))
    vec2 = hnsw_healer.default_index().get_vector(3)
    assert all(v == 0.0 for v in vec2)


def test_recovery_replays_uncommitted(tmp_path: Path) -> None:
    n, d = 15, 4
    data = np.ones((n, d), dtype=np.float32)
    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])

    eng = PersistenceEngine(tmp_path)
    eng.save_initial_checkpoint()

    # Simulate crash: WAL BEGIN without mutation / commit.
    begin = eng.wal.append_begin_delete(5, max_m=4)
    assert eng.wal.uncommitted_deletes()

    # Fresh engine bootstrap must replay.
    eng2 = PersistenceEngine(tmp_path)
    info = eng2.bootstrap()
    assert info["index_loaded"] is True
    assert any(
        r["node_id"] == 5 for r in info["recovered_transactions"]
    )
    assert eng2.wal.uncommitted_deletes() == []

    vec = hnsw_healer.default_index().get_vector(5)
    assert all(v == 0.0 for v in vec)
    # Original crash tx should be retired (committed).
    committed_ids = {
        rec.transaction_id
        for rec in eng2.wal.iter_records()
        if rec.__class__.__name__ == "CommitRecord"
    }
    assert begin.transaction_id in committed_ids
