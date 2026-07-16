"""
Append-only erasure receipt log
================================

Persists every signed ``ErasureReceipt`` as one JSON line under
``HEALER_DATA_DIR/receipts.jsonl`` (or a tenant-scoped path).

Audits should not depend solely on the application database — this log
is intentionally immutable (append-only; no update/delete API).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator


class AppendOnlyReceiptLog:
    """Thread-safe JSONL receipt store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Create empty file so existence checks are stable.
        if not self.path.is_file():
            self.path.touch()

    def append(self, receipt: dict[str, Any]) -> None:
        line = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def iter_receipts(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                yield json.loads(raw)

    def count(self) -> int:
        return sum(1 for _ in self.iter_receipts())

    def find_by_request_id(self, request_id: str) -> list[dict[str, Any]]:
        rid = str(request_id)
        return [
            r for r in self.iter_receipts() if r.get("request_id") == rid
        ]
