"""Residual-data proof suite: live zeros + checkpoint pattern absence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hnsw_healer
from compliance.residual import (
    float32_pattern_bytes,
    prove_vector_erased,
    vector_all_zeros,
)
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.native_backend import NativeHealerBackend


def test_vector_all_zeros() -> None:
    assert vector_all_zeros([0.0, 0.0])
    assert not vector_all_zeros([0.0, 1e-3])


def test_live_and_file_residual_after_erase(tmp_path: Path) -> None:
    n, d = 20, 16
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d), dtype=np.float32)
    # Make row 7 distinctive for pattern search.
    data[7] = np.linspace(1.0, 2.0, d, dtype=np.float32)

    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])

    original = np.array(hnsw_healer.default_index().get_vector(7), dtype=np.float32)
    pattern = float32_pattern_bytes(original)

    # Checkpoint before delete should contain the pattern once saved.
    pre = tmp_path / "pre.bin"
    hnsw_healer.save_index(str(pre))
    assert pattern in pre.read_bytes()

    reg = CollectionIdRegistry()
    for i in range(n):
        reg.register("c", f"id{i}", label=i)
    svc = ErasureService(reg, NativeHealerBackend())
    receipt = svc.delete("c", ["id7"])
    assert receipt.success

    live = hnsw_healer.default_index().get_vector(7)
    post = tmp_path / "post.bin"
    hnsw_healer.save_index(str(post))

    proof = prove_vector_erased(
        label_or_id="id7",
        live_vector=live,
        original_vector=original,
        checkpoint_path=post,
    )
    assert proof.live_all_zeros
    assert proof.file_pattern_absent is True
    assert proof.passed

    # Pre-delete checkpoint still has the pattern (expected residual surface).
    from compliance.residual import file_contains_bytes

    assert file_contains_bytes(pre, pattern)
