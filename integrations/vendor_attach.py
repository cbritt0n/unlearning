"""
Deeper in-process attach to vendor memory layouts
=================================================

Beyond copy/compact, production unlearning sometimes needs to **operate on
the live float buffer** owned by another library (zero-copy).

This module provides:

1. ``attach_numpy_to_proxy`` — pin a C-contiguous float32 array into
   ``HNSWIndexProxy`` via ``attach_index`` (no copy; caller retains ownership).
2. ``SharedMemoryVectorBank`` — OS shared-memory segment for multi-process
   readers/writers with hard-delete zeroing in place.
3. ``HnswlibLayoutImporter`` / ``FaissLayoutImporter`` — pull live vectors
   from vendor indexes into an attachable matrix (export path when vendors
   do not expose raw pointers safely).
4. ``InPlaceVendorSession`` — combine attach + adjacency install + erase
   against the **same** physical buffer the application already holds.

Safety
------
After ``attach_index``, the numpy/shared buffer **must outlive** the proxy.
Hard-delete zeros that buffer in place — any other view of the same memory
observes zeros (which is the unlearning goal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import hnsw_healer

    HEALER_AVAILABLE = True
except ImportError:  # pragma: no cover
    hnsw_healer = None  # type: ignore[assignment]
    HEALER_AVAILABLE = False


def require_healer() -> None:
    if not HEALER_AVAILABLE:
        raise ImportError("hnsw_healer native module required")


def as_c_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return C-contiguous float32 (N, D), copying only if needed."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def attach_numpy_to_proxy(
    proxy: Any,
    vectors: np.ndarray,
    *,
    adjacency: list | None = None,
) -> np.ndarray:
    """
    Zero-copy attach of a float32 matrix into ``HNSWIndexProxy``.

    Returns the array actually attached (may be a contiguous copy if the
    input was non-contiguous — in that case the copy is the live buffer).

    Notes
    -----
    C++ ``attach_index`` does not extend Python object lifetime. Hold a
    strong reference to the returned array for as long as ``proxy`` is used
    (``InPlaceVendorSession`` does this for you).
    """
    require_healer()
    mat = as_c_float32_matrix(vectors)
    n, d = mat.shape
    if hasattr(proxy, "attach_index"):
        proxy.attach_index(mat, d, n)
    elif hasattr(hnsw_healer, "attach_index_buffer"):
        hnsw_healer.attach_index_buffer(mat, d, n)
    else:  # pragma: no cover — old builds
        logger.warning(
            "attach_index not available; falling back to load_index (copy)"
        )
        proxy.load_index(mat, d, n)
    if adjacency is not None:
        proxy.load_adjacency(adjacency)
    return mat


# ---------------------------------------------------------------------------
# Shared memory bank
# ---------------------------------------------------------------------------


class SharedMemoryVectorBank:
    """
    Cross-process float32 matrix in ``shared_memory``.

    Hard-delete zeros rows in place so every process mapping the segment
    observes erasure without compact.
    """

    def __init__(
        self,
        name: str,
        n: int,
        d: int,
        *,
        create: bool = True,
    ) -> None:
        self.name = name
        self.n = int(n)
        self.d = int(d)
        nbytes = self.n * self.d * 4
        if create:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=nbytes
            )
            self._created = True
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._created = False
        self.matrix = np.ndarray(
            (self.n, self.d), dtype=np.float32, buffer=self._shm.buf
        )
        if create:
            self.matrix[:] = 0.0

    def write_row(self, label: int, vector: Sequence[float]) -> None:
        self.matrix[int(label)] = np.asarray(vector, dtype=np.float32)

    def zero_row(self, label: int) -> int:
        """Physical wipe of one row. Returns bytes wiped."""
        lab = int(label)
        self.matrix[lab] = 0.0
        return self.d * 4

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._created:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Vendor importers (export live floats when pointers are unavailable)
# ---------------------------------------------------------------------------


class HnswlibLayoutImporter:
    """Export vectors from an hnswlib.Index via get_items (copy out)."""

    def __init__(self, index: Any) -> None:
        self.index = index

    def export_matrix(self, labels: Sequence[int]) -> np.ndarray:
        labs = list(int(x) for x in labels)
        # hnswlib get_items returns list/array of vectors
        items = self.index.get_items(labs)
        return as_c_float32_matrix(np.asarray(items, dtype=np.float32))


class FaissLayoutImporter:
    """
    Export vectors from a FAISS index that supports ``reconstruct``.

    Works with IndexFlat / IndexHNSWFlat / IDMap wrappers that forward
    reconstruct.
    """

    def __init__(self, index: Any) -> None:
        self.index = index

    def export_matrix(self, n: int) -> np.ndarray:
        dim = self.index.d
        out = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            try:
                out[i] = self.index.reconstruct(i)
            except Exception:
                # IDMap may need reconstruct from inner index
                inner = getattr(self.index, "index", None)
                if inner is None:
                    raise
                out[i] = inner.reconstruct(i)
        return out


# ---------------------------------------------------------------------------
# Session: attach + erase against application-owned buffer
# ---------------------------------------------------------------------------


@dataclass
class InPlaceVendorSession:
    """
    Hold a live matrix attached (or loaded) into a proxy for in-place wipe.
    """

    proxy: Any
    matrix: np.ndarray  # strong ref — must outlive proxy use
    owns_copy: bool

    @classmethod
    def from_numpy(
        cls,
        vectors: np.ndarray,
        *,
        adjacency: list | None = None,
        proxy: Any | None = None,
    ) -> "InPlaceVendorSession":
        require_healer()
        proxy = proxy or hnsw_healer.HNSWIndexProxy()
        mat = as_c_float32_matrix(vectors)
        owns = mat is not vectors and not np.shares_memory(mat, vectors)
        # Prefer zero-copy attach when available on the extension.
        attached = attach_numpy_to_proxy(proxy, mat, adjacency=adjacency)
        owns = owns or (attached is not vectors)
        return cls(proxy=proxy, matrix=attached, owns_copy=owns)

    def hard_delete_row(self, label: int, *, max_m: int = 16) -> int:
        """
        Zero row ``label`` in the attached buffer and run MN-RU heal.
        Returns bytes wiped.
        """
        lab = int(label)
        self.matrix[lab] = 0.0
        bytes_wiped = self.matrix.shape[1] * 4
        # Also run native erase for adjacency heal (overwrite is redundant
        # but keeps heal path consistent).
        if self.proxy.is_loaded:
            try:
                self.proxy.erase_node(lab, max_m)
            except Exception:
                # If attach used fallback load, erase_node zeros proxy memory
                # which is the same buffer only when attach is real.
                self.proxy.overwrite_vector(lab)
                self.proxy.heal_graph_structure(lab, max_m)
        return bytes_wiped

    def verify_zeroed(self, label: int) -> bool:
        return bool(np.all(self.matrix[int(label)] == 0.0))
