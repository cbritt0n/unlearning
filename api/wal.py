"""
Append-only Write-Ahead Log for hard-delete transactions.

Binary record layout (little-endian), fixed-size for crash-safe scanning:

BEGIN_DELETE (type=1) — 72 bytes
  0-3   magic          b"WDEL"
  4     rec_type       u8 = 1
  5-7   reserved       3 x u8
  8-15  timestamp      u64  (unix ns)
  16-23 transaction_id u64
  24-31 target_node_id u64
  32-35 max_m          u32
  36-39 reserved       u32
  40-71 sha256_checksum of payload
        payload = timestamp || transaction_id || target_node_id || max_m
        (each field little-endian as stored)

COMMIT (type=2) — 56 bytes
  0-3   magic          b"WCMT"
  4     rec_type       u8 = 2
  5-7   reserved       3 x u8
  8-15  timestamp      u64
  16-23 transaction_id u64
  24-55 sha256_checksum of payload
        payload = timestamp || transaction_id
"""

from __future__ import annotations

import hashlib
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# Record type codes
REC_BEGIN_DELETE = 1
REC_COMMIT = 2

MAGIC_BEGIN = b"WDEL"
MAGIC_COMMIT = b"WCMT"

BEGIN_FMT = "<4sB3sQQQII32s"  # 4+1+3+8+8+8+4+4+32 = 72
BEGIN_SIZE = struct.calcsize(BEGIN_FMT)
COMMIT_FMT = "<4sB3sQQ32s"  # 4+1+3+8+8+32 = 56
COMMIT_SIZE = struct.calcsize(COMMIT_FMT)

assert BEGIN_SIZE == 72
assert COMMIT_SIZE == 56


@dataclass(frozen=True)
class BeginDeleteRecord:
    timestamp: int
    transaction_id: int
    target_node_id: int
    max_m: int
    checksum: bytes


@dataclass(frozen=True)
class CommitRecord:
    timestamp: int
    transaction_id: int
    checksum: bytes


def _unix_ns() -> int:
    return time.time_ns()


def _checksum_begin(
    timestamp: int, transaction_id: int, target_node_id: int, max_m: int
) -> bytes:
    payload = struct.pack("<QQQI", timestamp, transaction_id, target_node_id, max_m)
    return hashlib.sha256(payload).digest()


def _checksum_commit(timestamp: int, transaction_id: int) -> bytes:
    payload = struct.pack("<QQ", timestamp, transaction_id)
    return hashlib.sha256(payload).digest()


def _fsync_file(f: BinaryIO) -> None:
    f.flush()
    os.fsync(f.fileno())


class WriteAheadLog:
    """
    Thread-safe, append-only WAL for hard-delete durability.

    Every BEGIN must be durable on disk *before* the C++ mutation starts.
    COMMIT is appended only after an atomic index.bin replace succeeds.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tx_counter = self._recover_max_tx_id()

    def _recover_max_tx_id(self) -> int:
        max_id = 0
        if not self.path.is_file():
            return 0
        for rec in self.iter_records():
            if isinstance(rec, BeginDeleteRecord):
                max_id = max(max_id, rec.transaction_id)
            elif isinstance(rec, CommitRecord):
                max_id = max(max_id, rec.transaction_id)
        return max_id

    def next_transaction_id(self) -> int:
        with self._lock:
            self._tx_counter += 1
            return self._tx_counter

    def append_begin_delete(
        self,
        target_node_id: int,
        *,
        max_m: int = 16,
        transaction_id: int | None = None,
        timestamp: int | None = None,
    ) -> BeginDeleteRecord:
        """
        Append a BEGIN_DELETE record and fsync before returning.

        Must complete successfully before any in-memory hard delete starts.
        """
        if target_node_id < 0:
            raise ValueError("target_node_id must be non-negative")
        if max_m < 1:
            raise ValueError("max_m must be positive")

        with self._lock:
            tx = (
                transaction_id
                if transaction_id is not None
                else (self._tx_counter + 1)
            )
            if transaction_id is None:
                self._tx_counter = tx
            else:
                self._tx_counter = max(self._tx_counter, tx)

            ts = timestamp if timestamp is not None else _unix_ns()
            digest = _checksum_begin(ts, tx, target_node_id, max_m)
            blob = struct.pack(
                BEGIN_FMT,
                MAGIC_BEGIN,
                REC_BEGIN_DELETE,
                b"\x00\x00\x00",
                ts,
                tx,
                target_node_id,
                max_m,
                0,
                digest,
            )
            assert len(blob) == BEGIN_SIZE

            with open(self.path, "ab", buffering=0) as f:
                f.write(blob)
                _fsync_file(f)

            return BeginDeleteRecord(
                timestamp=ts,
                transaction_id=tx,
                target_node_id=target_node_id,
                max_m=max_m,
                checksum=digest,
            )

    def append_commit(
        self,
        transaction_id: int,
        *,
        timestamp: int | None = None,
    ) -> CommitRecord:
        """Append a COMMIT marker after a successful atomic index flush."""
        with self._lock:
            ts = timestamp if timestamp is not None else _unix_ns()
            digest = _checksum_commit(ts, transaction_id)
            blob = struct.pack(
                COMMIT_FMT,
                MAGIC_COMMIT,
                REC_COMMIT,
                b"\x00\x00\x00",
                ts,
                transaction_id,
                digest,
            )
            assert len(blob) == COMMIT_SIZE

            with open(self.path, "ab", buffering=0) as f:
                f.write(blob)
                _fsync_file(f)

            return CommitRecord(
                timestamp=ts,
                transaction_id=transaction_id,
                checksum=digest,
            )

    def iter_records(self):
        """
        Yield validated BeginDeleteRecord / CommitRecord from the WAL file.

        Scanning strategy
        -----------------
        Records are fixed-size and self-describing via a 4-byte magic prefix
        (``WDEL`` vs ``WCMT``). We read an 8-byte header first so we know
        which record length to pull next. A short read at EOF is treated as a
        torn write from a crash mid-append and is ignored (safe: recovery
        only cares about fully durable BEGIN/COMMIT pairs).
        """
        if not self.path.is_file():
            return

        with open(self.path, "rb") as f:
            while True:
                head = f.read(8)
                if not head:
                    break
                if len(head) < 8:
                    # Torn header at EOF — ignore incomplete tail.
                    break

                magic = head[:4]
                rec_type = head[4]

                # ---- BEGIN_DELETE: intent to erase target_node_id --------
                if magic == MAGIC_BEGIN and rec_type == REC_BEGIN_DELETE:
                    rest = f.read(BEGIN_SIZE - 8)
                    if len(rest) < BEGIN_SIZE - 8:
                        # Torn BEGIN body — not durable; stop scanning.
                        break
                    blob = head + rest
                    (
                        _mag,
                        _typ,
                        _res,
                        ts,
                        tx,
                        node,
                        max_m,
                        _pad,
                        checksum,
                    ) = struct.unpack(BEGIN_FMT, blob)
                    # Reject bit-flips / partial overwrites of the payload.
                    expect = _checksum_begin(ts, tx, node, max_m)
                    if checksum != expect:
                        raise ValueError(
                            f"WAL checksum mismatch on BEGIN tx={tx}"
                        )
                    yield BeginDeleteRecord(
                        timestamp=ts,
                        transaction_id=tx,
                        target_node_id=node,
                        max_m=max_m,
                        checksum=checksum,
                    )
                # ---- COMMIT: index.bin flush for this transaction is done
                elif magic == MAGIC_COMMIT and rec_type == REC_COMMIT:
                    rest = f.read(COMMIT_SIZE - 8)
                    if len(rest) < COMMIT_SIZE - 8:
                        break
                    blob = head + rest
                    (_mag, _typ, _res, ts, tx, checksum) = struct.unpack(
                        COMMIT_FMT, blob
                    )
                    expect = _checksum_commit(ts, tx)
                    if checksum != expect:
                        raise ValueError(
                            f"WAL checksum mismatch on COMMIT tx={tx}"
                        )
                    yield CommitRecord(
                        timestamp=ts,
                        transaction_id=tx,
                        checksum=checksum,
                    )
                else:
                    # Unknown magic mid-stream: refuse to guess (safer than
                    # skipping and silently losing recovery intent).
                    raise ValueError(
                        f"WAL corrupt: unknown magic/type {magic!r}/{rec_type}"
                    )

    def uncommitted_deletes(self) -> list[BeginDeleteRecord]:
        """
        Return BEGIN_DELETE records with no matching COMMIT.

        Used by crash recovery to re-apply hard deletes that may not have
        reached a durable index.bin commit.
        """
        committed: set[int] = set()
        begins: dict[int, BeginDeleteRecord] = {}

        for rec in self.iter_records():
            if isinstance(rec, BeginDeleteRecord):
                begins[rec.transaction_id] = rec
            elif isinstance(rec, CommitRecord):
                committed.add(rec.transaction_id)

        pending = [
            begins[tx]
            for tx in sorted(begins.keys())
            if tx not in committed
        ]
        return pending
