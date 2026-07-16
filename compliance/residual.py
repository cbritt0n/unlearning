"""
Residual-data proof helpers
===========================

After a hard delete, verify that:

1. Live API ``get_vector`` (or backend matrix row) is all zeros.
2. A durable checkpoint file no longer contains the original float pattern
   as a contiguous byte sequence (best-effort scan).

This supports security reviews of the Vec2Text residual threat model.
It is **not** a formal cryptographic proof against all forensic techniques.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class ResidualProof:
    """Result of a residual scan for one erased embedding."""

    label_or_id: str
    live_all_zeros: bool
    file_pattern_absent: bool | None
    original_norm: float
    details: str

    @property
    def passed(self) -> bool:
        if self.file_pattern_absent is None:
            return self.live_all_zeros
        return self.live_all_zeros and self.file_pattern_absent


def vector_all_zeros(vec: Sequence[float], *, atol: float = 0.0) -> bool:
    arr = np.asarray(vec, dtype=np.float64)
    if atol <= 0:
        return bool(np.all(arr == 0.0))
    return bool(np.all(np.abs(arr) <= atol))


def float32_pattern_bytes(vector: Sequence[float]) -> bytes:
    """Little-endian float32 byte image of the embedding (as stored on disk)."""
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes(order="C")


def file_contains_bytes(path: Path, pattern: bytes) -> bool:
    """
    Return True if ``pattern`` occurs anywhere in the file.

    Used to detect whether an old embedding blob still sits in index.bin
    or an hnswlib save file. False negatives possible if the store
    quantizes, encrypts, or reorders components.
    """
    if not pattern or not path.is_file():
        return False
    data = path.read_bytes()
    return pattern in data


def prove_vector_erased(
    *,
    label_or_id: str,
    live_vector: Sequence[float],
    original_vector: Sequence[float] | None = None,
    checkpoint_path: str | Path | None = None,
    atol: float = 0.0,
) -> ResidualProof:
    """
    Build a residual proof for one erased vector.

    Parameters
    ----------
    live_vector:
        Embedding read back after erase (must be zeros).
    original_vector:
        Pre-delete snapshot used for file pattern search. Required for
        ``file_pattern_absent`` to be meaningful.
    checkpoint_path:
        Optional path to ``index.bin`` / hnswlib file to scan.
    """
    live_ok = vector_all_zeros(live_vector, atol=atol)
    orig_norm = float(
        np.linalg.norm(np.asarray(original_vector, dtype=np.float64))
        if original_vector is not None
        else 0.0
    )

    file_ok: bool | None = None
    details_parts = []
    if not live_ok:
        details_parts.append("live vector still has non-zero components")
    else:
        details_parts.append("live vector is all zeros")

    if checkpoint_path is not None and original_vector is not None:
        path = Path(checkpoint_path)
        pattern = float32_pattern_bytes(original_vector)
        # Avoid trivial matches for all-zero originals.
        if orig_norm == 0.0:
            file_ok = True
            details_parts.append("original was zero; file scan skipped")
        elif not path.is_file():
            file_ok = None
            details_parts.append(f"checkpoint missing: {path}")
        else:
            present = file_contains_bytes(path, pattern)
            file_ok = not present
            details_parts.append(
                "original float32 pattern ABSENT from checkpoint"
                if file_ok
                else "ORIGINAL PATTERN STILL PRESENT in checkpoint"
            )

    return ResidualProof(
        label_or_id=str(label_or_id),
        live_all_zeros=live_ok,
        file_pattern_absent=file_ok,
        original_norm=orig_norm,
        details="; ".join(details_parts),
    )


def prove_batch_zeroed(
    items: list[tuple[str, Sequence[float]]],
    *,
    atol: float = 0.0,
) -> list[ResidualProof]:
    """Convenience: live-only zero checks for many labels."""
    return [
        prove_vector_erased(label_or_id=i, live_vector=v, atol=atol)
        for i, v in items
    ]


def pack_f32_le(values: Sequence[float]) -> bytes:
    """Explicit struct pack (tests / debugging)."""
    return b"".join(struct.pack("<f", float(x)) for x in values)
