"""
Stable enterprise erase API keyed by collection + external IDs.

Business code should call this — not raw ``hnsw_healer.erase_node(int)``.

Example::

    svc = ErasureService(registry, backend, persistence=engine)
    receipt = svc.delete("users", ["user-42"], reason="gdpr_erasure_request")
    assert receipt.status == "complete"
    assert receipt.compacted or not receipt.backend_supports_compact
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from integrations.backends import BackendEraseResult, HardDeleteBackend
from integrations.compact_policy import (
    CompactCoalescePolicy,
    policy_from_env,
)
from integrations.id_registry import CollectionIdRegistry, IdMappingError

logger = logging.getLogger(__name__)

try:
    from api.metrics import METRICS
except ImportError:  # pragma: no cover
    METRICS = None  # type: ignore[assignment]

# Optional durable path (process-local healer + WAL)
try:
    from api.persistence import PersistenceEngine
except ImportError:  # pragma: no cover
    PersistenceEngine = None  # type: ignore[misc, assignment]

RECEIPT_VERSION = 2

ReceiptStatus = Literal["complete", "partial", "failed"]
CompactMode = Literal["auto", "always", "never"] | bool
ResidualProofMode = Literal["off", "sample", "full"]


def _env_residual_proof_mode() -> ResidualProofMode:
    raw = os.environ.get("HEALER_RESIDUAL_PROOF", "sample").strip().lower()
    if raw in ("off", "0", "false", "no"):
        return "off"
    if raw in ("full", "all"):
        return "full"
    return "sample"


def _normalize_compact(compact: CompactMode) -> CompactMode:
    if compact is True:
        return "always"
    if compact is False:
        return "never"
    if compact in ("auto", "always", "never"):
        return compact
    raise ValueError(
        f"compact must be auto|always|never|bool, got {compact!r}"
    )


@dataclass
class ResidualProofSummary:
    """Aggregate residual-proof result attached to an erasure receipt."""

    mode: str
    passed: bool | None
    checked: int = 0
    failed: int = 0
    proofs: list[dict[str, Any]] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErasureReceipt:
    """
    Audit-friendly proof that a hard-delete was requested and applied.

    Persist this in your compliance log store; it is not a legal certificate
    by itself but is designed for export into DPIA / audit packages.

    Receipt schema v2 adds ``status``, ``compacted``, ``residual_proof``,
    and ``receipt_version`` so auditors can tell complete vs partial erasures.
    """

    request_id: str
    collection: str
    external_ids: list[str]
    labels: list[int]
    success: bool
    reason: str | None
    timestamp_unix_ns: int
    bytes_wiped_total: int
    backend_messages: list[str] = field(default_factory=list)
    transaction_ids: list[int] = field(default_factory=list)
    signature: str = ""
    errors: list[str] = field(default_factory=list)
    # --- schema v2 ---
    receipt_version: int = RECEIPT_VERSION
    status: ReceiptStatus = "failed"
    compacted: bool = False
    residual_proof: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ErasureService:
    """
    Control-plane hard delete: resolve IDs → backend wipe → optional compact
    → residual proof → optional WAL.

    Parameters
    ----------
    registry:
        Collection-scoped ID map.
    backend:
        Object implementing ``HardDeleteBackend`` (native proxy, hnswlib, …).
    persistence:
        Optional ``PersistenceEngine`` for WAL + atomic index.bin (native path).
    signing_key:
        HMAC key for receipt signatures (defaults to HEALER_SIGNING_KEY).
    drop_mappings:
        If True (default), remove ID mappings after successful erase.
    default_compact:
        Default compact policy for ``delete()`` (``auto`` / ``always`` / ``never``).
        When ``auto`` and a ``compact_policy`` is set to coalesce mode, rebuilds
        are deferred until every-N or max-age thresholds fire.
    default_residual_proof:
        Default residual-proof mode; ``None`` reads ``HEALER_RESIDUAL_PROOF``.
    receipt_log:
        Optional append-only receipt log (JSONL).
    compact_policy:
        Optional ``CompactCoalescePolicy`` (default from env).
    metrics:
        Optional metrics registry (default process ``METRICS``).
    """

    def __init__(
        self,
        registry: CollectionIdRegistry,
        backend: HardDeleteBackend,
        *,
        persistence: Any | None = None,
        signing_key: bytes | None = None,
        drop_mappings: bool = True,
        default_compact: CompactMode = "auto",
        default_residual_proof: ResidualProofMode | None = None,
        receipt_log: Any | None = None,
        compact_policy: CompactCoalescePolicy | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.persistence = persistence
        self.drop_mappings = drop_mappings
        self.default_compact = _normalize_compact(default_compact)
        self.default_residual_proof = default_residual_proof
        self.receipt_log = receipt_log
        self.compact_policy = compact_policy or policy_from_env()
        self.metrics = metrics if metrics is not None else METRICS
        key = signing_key or os.environ.get(
            "HEALER_SIGNING_KEY", "dev-only-placeholder-signing-key"
        )
        self._signing_key = (
            key if isinstance(key, bytes) else str(key).encode("utf-8")
        )

    @property
    def backend_supports_compact(self) -> bool:
        return callable(getattr(self.backend, "compact", None))

    def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        reason: str | None = None,
        request_id: str | None = None,
        max_m: int = 16,
        idempotent: bool = True,
        compact: CompactMode | None = None,
        residual_proof: ResidualProofMode | None = None,
        checkpoint_path: str | None = None,
    ) -> ErasureReceipt:
        """
        Hard-delete one or more business IDs in ``collection``.

        Parameters
        ----------
        collection:
            Logical collection / tenant index name.
        ids:
            External identifiers (user ids, document ids, …).
        reason:
            Free-text compliance reason (e.g. ``gdpr_art_17``).
        request_id:
            Caller-supplied correlation id; generated if omitted.
        max_m:
            HNSW max degree for MN-RU heal backends.
        idempotent:
            If True, unknown / already-dropped IDs become soft errors
            instead of failing the whole batch.
        compact:
            ``auto`` (default): compact once after the batch if the backend
            exposes ``compact()``. ``always`` / ``never`` / bool override.
            One compact per ``delete()`` call — not per id.
        residual_proof:
            ``off`` | ``sample`` | ``full``. Default from service / env
            (``HEALER_RESIDUAL_PROOF``, default ``sample``).
            Fail-closed: proof failure marks the receipt unsuccessful.
        checkpoint_path:
            Optional on-disk index path for residual file-pattern scan.
        """
        req = request_id or str(uuid.uuid4())
        ts = time.time_ns()
        ext_ids = [str(i).strip() for i in ids if str(i).strip()]
        labels: list[int] = []
        messages: list[str] = []
        errors: list[str] = []
        tx_ids: list[int] = []
        bytes_total = 0
        erased_ext: list[str] = []
        originals: dict[int, list[float]] = {}

        # Adaptive compact: wipe+rebuild preferred over heal-only (see benchmarks).
        compact_arg = self.default_compact if compact is None else compact
        try:
            from integrations.delete_strategy import apply_strategy_to_delete_kwargs

            idx_size = max(len(ext_ids), 1)
            try:
                idx_size = max(
                    idx_size, len(self.registry.labels(collection)) + len(ext_ids)
                )
            except Exception:  # noqa: BLE001
                pass
            if hasattr(self.backend, "_vectors"):
                try:
                    idx_size = max(
                        idx_size, int(self.backend._vectors.shape[0])  # noqa: SLF001
                    )
                except Exception:  # noqa: BLE001
                    pass
            adapted = apply_strategy_to_delete_kwargs(
                delete_count=len(ext_ids),
                index_size=idx_size,
                backend_supports_compact=self.backend_supports_compact,
                backend_supports_heal=hasattr(self.backend, "_heal_mirror")
                or self.persistence is not None,
                compact=compact_arg if compact is not None else None,
            )
            if adapted.get("compact") is not None and (
                compact is None or compact in ("auto", True)
            ):
                compact_arg = adapted["compact"]
                strat = adapted.get("strategy") or {}
                messages.append(
                    f"strategy:{strat.get('action', '?')}({strat.get('reason', '')})"
                )
        except Exception:  # noqa: BLE001
            pass

        compact_mode = _normalize_compact(compact_arg)
        proof_mode: ResidualProofMode
        if residual_proof is not None:
            proof_mode = residual_proof
        elif self.default_residual_proof is not None:
            proof_mode = self.default_residual_proof
        else:
            proof_mode = _env_residual_proof_mode()

        # Snapshot originals for residual proofs before mutation.
        need_originals = proof_mode != "off" and hasattr(
            self.backend, "get_vector"
        )

        for ext in ext_ids:
            try:
                resolved = self.registry.resolve(collection, ext)
            except IdMappingError as exc:
                if idempotent:
                    errors.append(f"skip {ext}: {exc}")
                    continue
                raise

            try:
                if need_originals:
                    try:
                        vec = self.backend.get_vector(resolved.label)  # type: ignore[attr-defined]
                        originals[resolved.label] = [
                            float(x) for x in vec
                        ]
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "could not snapshot vector for residual proof: %s",
                            exc,
                        )

                if self.persistence is not None and hasattr(
                    self.persistence, "hard_delete_and_heal"
                ):
                    # Durable native path (WAL + index.bin).
                    result = self.persistence.hard_delete_and_heal(
                        resolved.label, max_m=max_m
                    )
                    br = BackendEraseResult(
                        success=bool(result.success),
                        label=resolved.label,
                        bytes_wiped=int(result.bytes_wiped),
                        message=result.message,
                    )
                    if getattr(result, "transaction_id", None) is not None:
                        tx_ids.append(int(result.transaction_id))
                else:
                    br = self.backend.hard_delete_label(
                        collection, resolved.label, max_m=max_m
                    )

                if not br.success:
                    errors.append(
                        f"erase failed for {ext} (label={resolved.label}): "
                        f"{br.message}"
                    )
                    continue

                if not self.backend.verify_zeroed(collection, resolved.label):
                    errors.append(
                        f"post-condition failed: label {resolved.label} "
                        f"not fully zeroed"
                    )
                    continue

                bytes_total += int(br.bytes_wiped)
                labels.append(resolved.label)
                erased_ext.append(ext)
                if br.message:
                    messages.append(br.message)

                if self.drop_mappings:
                    self.registry.drop(collection, ext)

            except Exception as exc:  # noqa: BLE001 — batch isolation
                errors.append(f"{ext}: {exc}")

        # Compact: one rebuild per decision (always / coalesce / never).
        compacted = False
        deferred_compact = False
        if labels and self.backend_supports_compact:
            self.compact_policy.note_deletes(len(labels))

        should_compact = False
        compact_reason = ""
        if labels:
            if compact_mode == "never":
                should_compact = False
                compact_reason = "never"
            elif compact_mode == "always":
                should_compact = True
                compact_reason = "always"
            elif compact_mode == "auto" and self.backend_supports_compact:
                # Honor coalesce policy when mode is auto.
                if self.compact_policy.mode == "never":
                    should_compact = False
                    compact_reason = "policy_never"
                elif self.compact_policy.mode == "coalesce":
                    decision = self.compact_policy.decide()
                    should_compact = decision.should_compact
                    compact_reason = decision.reason
                    if not should_compact:
                        deferred_compact = True
                        messages.append(
                            f"compact:deferred({decision.reason})"
                        )
                else:
                    should_compact = True
                    compact_reason = "auto_always"
            elif compact_mode == "auto" and not self.backend_supports_compact:
                should_compact = False
                compact_reason = "no_compact_backend"

        if should_compact:
            if not self.backend_supports_compact:
                errors.append(
                    "compact requested but backend has no compact() method"
                )
            else:
                t0 = time.perf_counter()
                try:
                    n_live = self.backend.compact()  # type: ignore[attr-defined]
                    compacted = True
                    messages.append(
                        f"compact:live={n_live};reason={compact_reason}"
                    )
                    self.compact_policy.mark_compacted()
                    if self.metrics is not None:
                        self.metrics.observe_ms(
                            "compact",
                            (time.perf_counter() - t0) * 1000.0,
                        )
                        self.metrics.inc("compacts")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"compact failed: {exc}")
                    logger.exception("batch compact failed")
                    if self.metrics is not None:
                        self.metrics.inc("compact_failures")

        # Residual proofs (fail-closed when mode != off).
        t_proof = time.perf_counter()
        proof_summary = self._run_residual_proofs(
            collection=collection,
            labels=labels,
            erased_ext=erased_ext,
            originals=originals,
            mode=proof_mode,
            checkpoint_path=checkpoint_path,
        )
        if self.metrics is not None:
            self.metrics.observe_ms(
                "residual_proof",
                (time.perf_counter() - t_proof) * 1000.0,
            )
        if (
            proof_mode != "off"
            and proof_summary.passed is False
        ):
            errors.append(
                f"residual_proof failed: {proof_summary.details}"
            )
            if self.metrics is not None:
                self.metrics.inc("residual_proof_failures")

        # Deferred compact is still "complete" for live zeros; ANN residual
        # risk remains until compact — surface via backend_messages only.
        # For coalesce mode, treat as complete when wipe+proof OK (compact
        # is an operational optimization with pending tracked in policy).
        status = self._derive_status(
            requested=len(ext_ids),
            erased=len(labels),
            errors=errors,
            compacted=compacted,
            compact_mode=compact_mode,
            proof_mode=proof_mode,
            proof_passed=proof_summary.passed,
            deferred_compact=deferred_compact,
        )

        if labels:
            try:
                from integrations.delete_strategy import recommend_delete_strategy

                idx_size = max(len(labels), 1)
                try:
                    idx_size = max(
                        idx_size,
                        len(self.registry.labels(collection)) + len(labels),
                    )
                except Exception:  # noqa: BLE001
                    pass
                if hasattr(self.backend, "_vectors"):
                    idx_size = max(
                        idx_size,
                        int(getattr(self.backend, "_vectors").shape[0]),
                    )
                rec = recommend_delete_strategy(
                    delete_count=len(labels),
                    index_size=max(idx_size, len(labels)),
                    backend_supports_compact=self.backend_supports_compact,
                    backend_supports_heal=hasattr(self.backend, "_heal_mirror")
                    or self.persistence is not None,
                )
                if rec.prefer_full_rebuild and not compacted:
                    messages.append(
                        f"strategy_hint:{rec.action}({rec.reason})"
                    )
            except Exception:  # noqa: BLE001
                pass

        success = status == "complete"

        receipt = ErasureReceipt(
            request_id=req,
            collection=collection,
            external_ids=list(ext_ids),
            labels=labels,
            success=success,
            reason=reason,
            timestamp_unix_ns=ts,
            bytes_wiped_total=bytes_total,
            backend_messages=messages,
            transaction_ids=tx_ids,
            errors=errors,
            receipt_version=RECEIPT_VERSION,
            status=status,
            compacted=compacted,
            residual_proof=proof_summary.to_dict(),
        )
        receipt.signature = self._sign(receipt)

        if self.receipt_log is not None:
            try:
                self.receipt_log.append(receipt.to_dict())
            except OSError:
                logger.exception("receipt log append failed")

        if self.metrics is not None:
            self.metrics.inc("deletes_batches")
            self.metrics.inc("deletes_ids", float(len(labels)))
            if success:
                self.metrics.inc("deletes_complete")
            else:
                self.metrics.inc("deletes_incomplete")
            self.metrics.set_gauge(
                "compact_pending",
                float(self.compact_policy.pending_count()),
            )

        return receipt

    def force_compact(self) -> dict[str, Any]:
        """Run compact now if backend supports it (flush coalesced pending)."""
        if not self.backend_supports_compact:
            return {"compacted": False, "reason": "no_compact_backend"}
        t0 = time.perf_counter()
        n_live = self.backend.compact()  # type: ignore[attr-defined]
        self.compact_policy.mark_compacted()
        ms = (time.perf_counter() - t0) * 1000.0
        if self.metrics is not None:
            self.metrics.observe_ms("compact", ms)
            self.metrics.inc("compacts")
        return {"compacted": True, "live": n_live, "duration_ms": ms}

    def _run_residual_proofs(
        self,
        *,
        collection: str,
        labels: list[int],
        erased_ext: list[str],
        originals: dict[int, list[float]],
        mode: ResidualProofMode,
        checkpoint_path: str | None,
    ) -> ResidualProofSummary:
        if mode == "off" or not labels:
            return ResidualProofSummary(
                mode=mode,
                passed=None if mode == "off" or not labels else True,
                details="skipped" if mode == "off" else "no labels erased",
            )

        try:
            from compliance.residual import prove_vector_erased
        except ImportError:  # pragma: no cover
            return ResidualProofSummary(
                mode=mode,
                passed=False,
                details="compliance.residual not importable",
            )

        # sample: first + last + middle when large; full: all
        if mode == "sample" and len(labels) > 3:
            indices = sorted(
                {0, len(labels) // 2, len(labels) - 1}
            )
            sample_labels = [labels[i] for i in indices]
            sample_ext = [erased_ext[i] for i in indices]
        else:
            sample_labels = list(labels)
            sample_ext = list(erased_ext)

        proofs: list[dict[str, Any]] = []
        failed = 0
        for lab, ext in zip(sample_labels, sample_ext):
            try:
                live = self.backend.get_vector(lab)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: verify_zeroed only
                live_ok = self.backend.verify_zeroed(collection, lab)
                live = [0.0] if live_ok else [1.0]

            proof = prove_vector_erased(
                label_or_id=ext,
                live_vector=live,
                original_vector=originals.get(lab),
                checkpoint_path=checkpoint_path,
            )
            proofs.append(
                {
                    "label_or_id": proof.label_or_id,
                    "live_all_zeros": proof.live_all_zeros,
                    "file_pattern_absent": proof.file_pattern_absent,
                    "original_norm": proof.original_norm,
                    "details": proof.details,
                    "passed": proof.passed,
                }
            )
            if not proof.passed:
                failed += 1

        passed = failed == 0
        details = (
            f"checked={len(proofs)} failed={failed}"
            if proofs
            else "no proofs run"
        )
        return ResidualProofSummary(
            mode=mode,
            passed=passed,
            checked=len(proofs),
            failed=failed,
            proofs=proofs,
            details=details,
        )

    @staticmethod
    def _derive_status(
        *,
        requested: int,
        erased: int,
        errors: list[str],
        compacted: bool,
        compact_mode: CompactMode,
        proof_mode: ResidualProofMode,
        proof_passed: bool | None,
        deferred_compact: bool = False,
    ) -> ReceiptStatus:
        """
        complete — every requested id erased, compact policy satisfied,
                   residual proof passed (when enabled), no hard errors.
        partial — some ids erased and/or soft skips, or post-steps incomplete.
        failed  — nothing erased (except empty request → complete).

        Deferred coalesce compact does **not** block complete when wipe+proof
        succeeded (matrix zeros are durable; ANN rebuild is scheduled).
        """
        del compacted  # reflected via error strings / caller

        compact_failed = any(
            e.startswith("compact failed")
            or "compact requested but backend" in e
            for e in errors
        )
        proof_failed = (
            proof_mode != "off" and erased > 0 and proof_passed is not True
        )
        erase_failed = any(
            not e.startswith("skip ")
            and not e.startswith("residual_proof")
            and not e.startswith("compact")
            and "compact requested but backend" not in e
            for e in errors
        )
        only_skips = bool(errors) and all(e.startswith("skip ") for e in errors)

        if requested == 0:
            return "complete"

        if erased == 0:
            return "partial" if only_skips else "failed"

        all_requested_erased = erased == requested and not erase_failed
        if (
            all_requested_erased
            and not compact_failed
            and not proof_failed
            and not errors
        ):
            return "complete"

        # Coalesce deferred: no hard errors beyond optional skips
        if (
            all_requested_erased
            and not compact_failed
            and not proof_failed
            and deferred_compact
            and only_skips is False
            and not any(not e.startswith("skip ") for e in errors)
        ):
            return "complete"

        return "partial"

    def _sign(self, receipt: ErasureReceipt) -> str:
        payload = (
            f"{receipt.receipt_version}|{receipt.request_id}|"
            f"{receipt.collection}|{','.join(receipt.external_ids)}|"
            f"{receipt.timestamp_unix_ns}|{receipt.bytes_wiped_total}|"
            f"{receipt.success}|{receipt.status}|{int(receipt.compacted)}"
        ).encode("utf-8")
        return hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest()
