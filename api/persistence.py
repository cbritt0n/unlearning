"""
Crash-resilient persistence: WAL + atomic index.bin commit + recovery.

Pipeline for hard_delete_and_heal
---------------------------------
1. Append BEGIN_DELETE to index.wal (fsync)          — intent durable
2. C++ erase_node / heal_graph_structure            — in-memory mutation
3. C++ save_to_file(index.bin.tmp)                  — serialize snapshot
4. os.replace(index.bin.tmp, index.bin)             — atomic publish
5. Append COMMIT to index.wal (fsync)               — intent retired

On API startup, uncommitted BEGIN records are replayed so a crash between
(1) and (5) cannot resurrect a deleted vector from an old checkpoint.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hnsw_healer

from api.wal import BeginDeleteRecord, WriteAheadLog

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.environ.get(
    "HEALER_DATA_DIR",
    str(Path.cwd() / "data"),
)

INDEX_BIN = "index.bin"
INDEX_BIN_TMP = "index.bin.tmp"
INDEX_WAL = "index.wal"


@dataclass
class HardDeleteResult:
    success: bool
    node_id: int
    transaction_id: int
    message: str | None
    bytes_wiped: int
    recovered: bool = False


class PersistenceEngine:
    """
    Owns the on-disk layout and coordinates WAL + atomic index commits.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str] | None = None,
        *,
        lock_retry: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / INDEX_BIN
        self.index_tmp_path = self.data_dir / INDEX_BIN_TMP
        self.wal_path = self.data_dir / INDEX_WAL
        self.wal = WriteAheadLog(self.wal_path)
        self._io_lock = threading.Lock()
        self._lock_retry = lock_retry or (lambda fn: fn())

    # ------------------------------------------------------------------
    # Atomic index commit
    # ------------------------------------------------------------------

    def atomic_flush_index(self) -> None:
        """
        Serialize mutated C++ memory to index.bin.tmp then atomically
        replace production index.bin via os.replace.
        """
        with self._io_lock:
            tmp = self.index_tmp_path
            final = self.index_path

            # Remove stale temp from a previous crash mid-write.
            if tmp.exists():
                tmp.unlink()

            # C++ writes the full snapshot into the temp path.
            self._lock_retry(
                lambda: hnsw_healer.save_index(str(tmp))
            )

            # Ensure temp contents hit durable media before rename.
            with open(tmp, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            # Best-effort directory fsync (POSIX); ignore on platforms
            # that do not support it.
            try:
                dir_fd = os.open(str(self.data_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (AttributeError, OSError, NotImplementedError):
                pass

            # Atomic publish: crash mid-replace leaves either old or new.
            os.replace(str(tmp), str(final))

            try:
                dir_fd = os.open(str(self.data_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (AttributeError, OSError, NotImplementedError):
                pass

    # ------------------------------------------------------------------
    # Hard delete with WAL
    # ------------------------------------------------------------------

    def hard_delete_and_heal(
        self,
        node_id: int,
        max_m: int = 16,
        *,
        recovered: bool = False,
    ) -> HardDeleteResult:
        """
        Durable hard delete: WAL BEGIN → C++ mutate → atomic flush → COMMIT.
        """
        begin = self.wal.append_begin_delete(node_id, max_m=max_m)
        logger.info(
            "WAL BEGIN tx=%s node=%s max_m=%s recovered=%s",
            begin.transaction_id,
            node_id,
            max_m,
            recovered,
        )

        try:
            result = self._lock_retry(
                lambda: hnsw_healer.erase_node(node_id, max_m)
            )
        except Exception:
            # BEGIN stays uncommitted → recovery will retry on next boot.
            logger.exception(
                "C++ erase failed for tx=%s node=%s; WAL left uncommitted",
                begin.transaction_id,
                node_id,
            )
            raise

        if not getattr(result, "success", False):
            raise RuntimeError(
                getattr(result, "message", "erase_node reported failure")
            )

        try:
            self.atomic_flush_index()
        except Exception:
            logger.exception(
                "atomic flush failed for tx=%s; WAL left uncommitted",
                begin.transaction_id,
            )
            raise

        self.wal.append_commit(begin.transaction_id)
        logger.info("WAL COMMIT tx=%s node=%s", begin.transaction_id, node_id)

        return HardDeleteResult(
            success=True,
            node_id=node_id,
            transaction_id=begin.transaction_id,
            message=getattr(result, "message", None),
            bytes_wiped=int(getattr(result, "bytes_wiped", 0)),
            recovered=recovered,
        )

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def load_index_if_present(self) -> bool:
        """Load index.bin into the default C++ proxy when the file exists."""
        if not self.index_path.is_file():
            logger.info("No index.bin at %s — starting empty", self.index_path)
            return False
        hnsw_healer.load_index_file(str(self.index_path))
        logger.info(
            "Loaded index.bin (%s elements)",
            hnsw_healer.default_index().num_elements,
        )
        return True

    def recover_uncommitted(self) -> list[HardDeleteResult]:
        """
        Re-run hard_delete_and_heal for every BEGIN without a COMMIT.

        Guarantees state convergence after a crash between WAL intent and
        durable index publish. Idempotent on already-zeroed nodes.
        """
        pending = self.wal.uncommitted_deletes()
        if not pending:
            logger.info("WAL recovery: no uncommitted delete transactions")
            return []

        if not hnsw_healer.default_index().is_loaded:
            # Intent without a base checkpoint: cannot heal an empty index.
            # Leave WAL pending until an index is loaded/created.
            logger.warning(
                "WAL has %s uncommitted delete(s) but no index loaded; "
                "deferring recovery",
                len(pending),
            )
            return []

        results: list[HardDeleteResult] = []
        for rec in pending:
            logger.warning(
                "WAL recovery: replaying tx=%s node=%s max_m=%s",
                rec.transaction_id,
                rec.target_node_id,
                rec.max_m,
            )
            # Replay uses a *new* BEGIN+COMMIT pair so the original
            # uncommitted BEGIN is closed by a fresh successful cycle.
            # First close the old tx only after successful re-apply:
            # we re-apply with a new transaction, then also commit the old
            # orphaned tx id to retire it without double-work ambiguity.
            try:
                result = self._replay_one(rec)
                results.append(result)
            except Exception:
                logger.exception(
                    "WAL recovery failed for tx=%s node=%s",
                    rec.transaction_id,
                    rec.target_node_id,
                )
                raise
        return results

    def _replay_one(self, rec: BeginDeleteRecord) -> HardDeleteResult:
        """
        Re-apply a single uncommitted delete and retire both the new and
        original transaction ids.
        """
        # Mutate + flush under a new BEGIN (standard durable path).
        # Use internal steps so we can COMMIT the original tx id as well.
        begin = self.wal.append_begin_delete(
            rec.target_node_id, max_m=rec.max_m
        )
        result = self._lock_retry(
            lambda: hnsw_healer.erase_node(rec.target_node_id, rec.max_m)
        )
        if not getattr(result, "success", False):
            raise RuntimeError(
                getattr(result, "message", "recovery erase failed")
            )
        self.atomic_flush_index()
        self.wal.append_commit(begin.transaction_id)
        # Retire the crash-orphaned transaction so it is not replayed again.
        if rec.transaction_id != begin.transaction_id:
            self.wal.append_commit(rec.transaction_id)

        return HardDeleteResult(
            success=True,
            node_id=int(rec.target_node_id),
            transaction_id=begin.transaction_id,
            message=getattr(result, "message", None),
            bytes_wiped=int(getattr(result, "bytes_wiped", 0)),
            recovered=True,
        )

    def bootstrap(self) -> dict[str, Any]:
        """
        Startup sequence: load checkpoint, then recover uncommitted WAL.
        """
        loaded = self.load_index_if_present()
        recovered = self.recover_uncommitted()
        return {
            "index_loaded": loaded,
            "index_path": str(self.index_path),
            "wal_path": str(self.wal_path),
            "recovered_transactions": [
                {
                    "transaction_id": r.transaction_id,
                    "node_id": r.node_id,
                    "message": r.message,
                }
                for r in recovered
            ],
        }

    def save_initial_checkpoint(self) -> None:
        """
        Persist the current in-memory index as index.bin without a WAL
        delete (used after bulk load / test fixtures).
        """
        if not hnsw_healer.default_index().is_loaded:
            raise RuntimeError("cannot checkpoint: no index loaded")
        self.atomic_flush_index()
