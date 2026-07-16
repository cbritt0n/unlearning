"""
HardDeleteBackend over the process-local ``hnsw_healer`` proxy.

Used when the application loads vectors directly into the native module
(without hnswlib). Collection is accepted for API symmetry; the default
proxy is single-index (one collection per process) unless you manage
multiple ``HNSWIndexProxy`` instances yourself.
"""

from __future__ import annotations

import hnsw_healer

from integrations.backends import BackendEraseResult


class NativeHealerBackend:
    """Erase labels on ``hnsw_healer.default_index()`` (or a provided proxy)."""

    def __init__(self, index=None) -> None:
        self._index = index  # None → default_index() at call time

    def _idx(self):
        return self._index if self._index is not None else hnsw_healer.default_index()

    def hard_delete_label(
        self,
        collection: str,
        label: int,
        *,
        max_m: int = 16,
    ) -> BackendEraseResult:
        del collection  # single shared native index in the default deployment
        result = self._idx().erase_node(int(label), int(max_m))
        return BackendEraseResult(
            success=bool(getattr(result, "success", False)),
            label=int(label),
            bytes_wiped=int(getattr(result, "bytes_wiped", 0)),
            message=getattr(result, "message", None),
        )

    def verify_zeroed(self, collection: str, label: int) -> bool:
        del collection
        vec = self._idx().get_vector(int(label))
        return all(float(x) == 0.0 for x in vec)

    def get_vector(self, label: int):
        """Live embedding for residual proofs (list/array of floats)."""
        return self._idx().get_vector(int(label))
