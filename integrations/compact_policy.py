"""
Coalesced compact policy
========================

Full ANN rebuild (``compact()``) after every delete batch is correct but
expensive under high churn. This policy decides when to compact:

- ``always`` — every successful batch (legacy auto behavior)
- ``never`` — never compact via policy (caller may compact later)
- ``coalesce`` — compact when pending deletes ≥ N **or** age ≥ T seconds

Pending deletes that have not yet been compacted are tracked so residual
proofs still see live zeros (matrix wipe happens immediately; compact
only rebuilds the ANN structure).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

CompactPolicyName = Literal["always", "never", "coalesce"]


def policy_from_env() -> "CompactCoalescePolicy":
    name = os.environ.get("HEALER_COMPACT_POLICY", "always").strip().lower()
    every_n = int(os.environ.get("HEALER_COMPACT_EVERY_N", "32"))
    max_age_s = float(os.environ.get("HEALER_COMPACT_MAX_AGE_S", "60"))
    if name in ("never", "off", "0"):
        return CompactCoalescePolicy(mode="never")
    if name in ("coalesce", "batch", "deferred"):
        return CompactCoalescePolicy(
            mode="coalesce", every_n=every_n, max_age_s=max_age_s
        )
    return CompactCoalescePolicy(mode="always")


@dataclass
class CompactDecision:
    should_compact: bool
    reason: str
    pending_deletes: int


class CompactCoalescePolicy:
    """
    Stateful compact scheduler shared by ``ErasureService`` instances.
    """

    def __init__(
        self,
        *,
        mode: CompactPolicyName = "always",
        every_n: int = 32,
        max_age_s: float = 60.0,
    ) -> None:
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        if max_age_s < 0:
            raise ValueError("max_age_s must be >= 0")
        self.mode = mode
        self.every_n = int(every_n)
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._pending = 0
        self._first_pending_mono: float | None = None

    def note_deletes(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            if self._pending == 0:
                self._first_pending_mono = time.monotonic()
            self._pending += int(n)

    def decide(self, *, force: bool = False) -> CompactDecision:
        with self._lock:
            pending = self._pending
            if force or self.mode == "always":
                return CompactDecision(True, "always" if not force else "force", pending)
            if self.mode == "never":
                return CompactDecision(False, "never", pending)
            # coalesce
            if pending <= 0:
                return CompactDecision(False, "no_pending", 0)
            if pending >= self.every_n:
                return CompactDecision(True, f"every_n>={self.every_n}", pending)
            age = 0.0
            if self._first_pending_mono is not None:
                age = time.monotonic() - self._first_pending_mono
            if self.max_age_s > 0 and age >= self.max_age_s:
                return CompactDecision(
                    True, f"max_age>={self.max_age_s}s", pending
                )
            return CompactDecision(
                False, f"waiting pending={pending} age={age:.2f}s", pending
            )

    def mark_compacted(self) -> None:
        with self._lock:
            self._pending = 0
            self._first_pending_mono = None

    def pending_count(self) -> int:
        with self._lock:
            return self._pending
