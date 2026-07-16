"""
Backend protocol for physical hard-delete implementations.

ErasureService talks only to this interface so hnswlib, the native proxy,
or future adapters (FAISS, Qdrant) plug in uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class BackendEraseResult:
    """Outcome of erasing one internal label inside a backend."""

    success: bool
    label: int
    bytes_wiped: int = 0
    message: str | None = None
    compacted: bool = False


@runtime_checkable
class HardDeleteBackend(Protocol):
    """Minimal surface required by ErasureService."""

    def hard_delete_label(
        self,
        collection: str,
        label: int,
        *,
        max_m: int = 16,
    ) -> BackendEraseResult:
        """Physically erase ``label`` in ``collection`` (or global index)."""
        ...

    def verify_zeroed(self, collection: str, label: int) -> bool:
        """Return True if the embedding storage for ``label`` is all zeros."""
        ...


@runtime_checkable
class CompactableBackend(Protocol):
    """Optional extension: rebuild ANN structure without deleted rows."""

    def compact(self) -> int:
        """Rebuild without residual deleted rows. Returns live count."""
        ...


@runtime_checkable
class VectorReadableBackend(Protocol):
    """Optional extension: read back a row for residual proofs."""

    def get_vector(self, label: int) -> Any:
        """Return the live embedding for ``label`` (zeros after erase)."""
        ...
