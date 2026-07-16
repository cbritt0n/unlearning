#!/usr/bin/env python3
"""
Lightweight residual attack contrast demo
=========================================

Shows (without full Vec2Text) that soft-delete leaves the original float32
pattern recoverable from an index snapshot, while hard-delete + residual
proof does not.

Usage::

    python examples/attack_demo/run.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import hnsw_healer
except ImportError:
    print("Build native module: pip install -e .", file=sys.stderr)
    sys.exit(1)

from compliance.residual import (
    file_contains_bytes,
    float32_pattern_bytes,
    prove_vector_erased,
)
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry
from integrations.native_backend import NativeHealerBackend


def main() -> int:
    n, d = 32, 16
    rng = np.random.default_rng(7)
    data = rng.standard_normal((n, d), dtype=np.float32)
    # Distinctive victim row
    data[5] = np.linspace(3.0, 9.0, d, dtype=np.float32)
    original = data[5].copy()
    pattern = float32_pattern_bytes(original)

    hnsw_healer.load_index(data, d, n)
    for i in range(n):
        hnsw_healer.default_index().set_neighbors(i, 0, [(i + 1) % n])

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        soft_path = td_path / "soft.bin"
        hard_path = td_path / "hard.bin"

        # --- Soft-delete simulation: checkpoint still has the pattern ---
        hnsw_healer.save_index(str(soft_path))
        soft_has = file_contains_bytes(soft_path, pattern)
        print("=== Soft-delete residual risk ===")
        print(f"  checkpoint contains original float32 pattern: {soft_has}")
        print(
            "  (metadata tombstone would hide the id from search, but the "
            "bytes remain invertible via Vec2Text-class models)"
        )

        # --- Hard-delete with residual proof ---
        reg = CollectionIdRegistry()
        for i in range(n):
            reg.register("docs", f"id{i}", label=i)
        svc = ErasureService(
            reg, NativeHealerBackend(), default_residual_proof="full"
        )
        receipt = svc.delete("docs", ["id5"], residual_proof="full")
        live = hnsw_healer.default_index().get_vector(5)
        hnsw_healer.save_index(str(hard_path))
        proof = prove_vector_erased(
            label_or_id="id5",
            live_vector=live,
            original_vector=original,
            checkpoint_path=hard_path,
        )

        print("\n=== Hard-delete + residual proof ===")
        print(f"  receipt.success: {receipt.success}")
        print(f"  live all zeros:  {proof.live_all_zeros}")
        print(f"  pattern absent:  {proof.file_pattern_absent}")
        print(f"  proof.passed:    {proof.passed}")
        print(f"  details: {proof.details}")

        if not soft_has:
            print("\n[warn] soft checkpoint unexpectedly missing pattern")
            return 1
        if not proof.passed or not receipt.success:
            print("\n[fail] hard-delete residual proof did not pass")
            return 1

        print("\nOK: soft-delete retains pattern; hard-delete removes it.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
