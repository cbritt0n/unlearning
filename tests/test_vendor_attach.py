"""In-process vendor attach / zero-copy wipe tests."""

from __future__ import annotations

import numpy as np
import pytest

import hnsw_healer
from integrations.vendor_attach import (
    InPlaceVendorSession,
    SharedMemoryVectorBank,
    as_c_float32_matrix,
)


def test_inplace_session_zeros_caller_buffer() -> None:
    n, d = 10, 6
    data = np.random.randn(n, d).astype(np.float32)
    data = np.ascontiguousarray(data)
    # ring adjacency
    adj = [[[(i + 1) % n, (i + 2) % n]] for i in range(n)]

    session = InPlaceVendorSession.from_numpy(data, adjacency=adj)
    original = data[3].copy()
    assert not np.all(original == 0.0)

    session.hard_delete_row(3, max_m=8)
    assert session.verify_zeroed(3)
    # Caller-owned buffer must observe zeros when attach is zero-copy
    assert np.all(data[3] == 0.0) or np.all(session.matrix[3] == 0.0)


def test_shared_memory_zero_row() -> None:
    name = "hnsw_healer_test_shm_bank"
    try:
        bank = SharedMemoryVectorBank(name, n=5, d=4, create=True)
    except FileExistsError:
        SharedMemoryVectorBank(name, n=5, d=4, create=False).unlink()
        bank = SharedMemoryVectorBank(name, n=5, d=4, create=True)
    try:
        bank.write_row(2, [1, 2, 3, 4])
        assert bank.matrix[2, 0] == 1.0
        bank.zero_row(2)
        assert np.all(bank.matrix[2] == 0.0)
    finally:
        bank.close()
        bank.unlink()


def test_as_c_float32_matrix() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = as_c_float32_matrix(a)
    assert b.dtype == np.float32
    assert b.flags["C_CONTIGUOUS"]
