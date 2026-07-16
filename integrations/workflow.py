"""
Erasure workflow orchestration
==============================

Productizes the compliance runbook as a durable state machine:

  open → in_progress → complete | blocked

Steps cover live hard-delete, compact (via receipt), residual proof,
optional replica fan-out / crypto-shred / document-store / backup hooks.

Persist under ``HEALER_DATA_DIR/workflows/`` as JSON.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from integrations.erase_service import ErasureReceipt, ErasureService

WorkflowStatus = Literal["open", "in_progress", "complete", "blocked"]
StepStatus = Literal["pending", "running", "done", "failed", "skipped"]

STEP_ORDER = (
    "live_hard_delete",
    "residual_proof",  # included in receipt; recorded explicitly for export
    "replica_fanout",
    "crypto_shred",
    "document_store",
    "backup_policy",
)


@dataclass
class ErasureStep:
    name: str
    status: StepStatus = "pending"
    detail: str = ""
    artifact: dict[str, Any] | None = None
    updated_unix_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErasureStep:
        return cls(
            name=str(data["name"]),
            status=data.get("status", "pending"),  # type: ignore[arg-type]
            detail=str(data.get("detail", "")),
            artifact=data.get("artifact"),
            updated_unix_ns=int(data.get("updated_unix_ns", 0)),
        )


@dataclass
class ErasureWorkflow:
    """One GDPR-style erasure ticket spanning multiple control surfaces."""

    request_id: str
    collection: str
    external_ids: list[str]
    reason: str | None = None
    status: WorkflowStatus = "open"
    created_unix_ns: int = field(default_factory=time.time_ns)
    updated_unix_ns: int = field(default_factory=time.time_ns)
    steps: list[ErasureStep] = field(default_factory=list)
    receipt: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    # Policy flags: which optional steps are required for complete.
    require_replica: bool = False
    require_crypto_shred: bool = False
    require_document_store: bool = False
    require_backup_ack: bool = False

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = [ErasureStep(name=n) for n in STEP_ORDER]

    def step(self, name: str) -> ErasureStep:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "collection": self.collection,
            "external_ids": list(self.external_ids),
            "reason": self.reason,
            "status": self.status,
            "created_unix_ns": self.created_unix_ns,
            "updated_unix_ns": self.updated_unix_ns,
            "steps": [s.to_dict() for s in self.steps],
            "receipt": self.receipt,
            "errors": list(self.errors),
            "require_replica": self.require_replica,
            "require_crypto_shred": self.require_crypto_shred,
            "require_document_store": self.require_document_store,
            "require_backup_ack": self.require_backup_ack,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErasureWorkflow:
        wf = cls(
            request_id=str(data["request_id"]),
            collection=str(data["collection"]),
            external_ids=list(data.get("external_ids", [])),
            reason=data.get("reason"),
            status=data.get("status", "open"),  # type: ignore[arg-type]
            created_unix_ns=int(data.get("created_unix_ns", 0)),
            updated_unix_ns=int(data.get("updated_unix_ns", 0)),
            steps=[
                ErasureStep.from_dict(s) for s in data.get("steps", [])
            ]
            or [ErasureStep(name=n) for n in STEP_ORDER],
            receipt=data.get("receipt"),
            errors=list(data.get("errors", [])),
            require_replica=bool(data.get("require_replica", False)),
            require_crypto_shred=bool(data.get("require_crypto_shred", False)),
            require_document_store=bool(
                data.get("require_document_store", False)
            ),
            require_backup_ack=bool(data.get("require_backup_ack", False)),
        )
        return wf

    def export_package(self) -> dict[str, Any]:
        """Single artifact suitable for ticket attachment / audit export."""
        return {
            "kind": "hnsw_healer.erasure_export",
            "version": 1,
            "workflow": self.to_dict(),
            "closeable": self.status == "complete",
            "summary": {
                "request_id": self.request_id,
                "collection": self.collection,
                "ids": self.external_ids,
                "status": self.status,
                "steps": {
                    s.name: s.status for s in self.steps
                },
            },
        }


class ErasureWorkflowStore:
    """Filesystem persistence for workflows under ``root/workflows/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dir = self.root / "workflows"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, request_id: str) -> Path:
        safe = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in request_id
        )
        return self.dir / f"{safe}.json"

    def save(self, workflow: ErasureWorkflow) -> Path:
        workflow.updated_unix_ns = time.time_ns()
        path = self.path_for(workflow.request_id)
        path.write_text(
            json.dumps(workflow.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load(self, request_id: str) -> ErasureWorkflow:
        path = self.path_for(request_id)
        if not path.is_file():
            raise FileNotFoundError(f"workflow not found: {request_id}")
        return ErasureWorkflow.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))


# Optional hooks for non-live steps
DocumentStoreHook = Callable[[str, Sequence[str]], dict[str, Any]]
CryptoShredHook = Callable[[Sequence[str]], dict[str, Any]]
ReplicaFanoutHook = Callable[[str, Sequence[str], str], dict[str, Any]]
BackupAckHook = Callable[[str, Sequence[str]], dict[str, Any]]


class ErasureWorkflowRunner:
    """
    Advance a workflow: run live delete via ``ErasureService``, then optional hooks.
    """

    def __init__(
        self,
        store: ErasureWorkflowStore,
        erase_service: ErasureService,
        *,
        document_store: DocumentStoreHook | None = None,
        crypto_shred: CryptoShredHook | None = None,
        replica_fanout: ReplicaFanoutHook | None = None,
        backup_ack: BackupAckHook | None = None,
    ) -> None:
        self.store = store
        self.erase = erase_service
        self.document_store = document_store
        self.crypto_shred = crypto_shred
        self.replica_fanout = replica_fanout
        self.backup_ack = backup_ack

    def create(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        reason: str | None = None,
        request_id: str | None = None,
        require_replica: bool = False,
        require_crypto_shred: bool = False,
        require_document_store: bool = False,
        require_backup_ack: bool = False,
    ) -> ErasureWorkflow:
        rid = request_id or str(uuid.uuid4())
        wf = ErasureWorkflow(
            request_id=rid,
            collection=collection,
            external_ids=[str(i) for i in ids],
            reason=reason,
            require_replica=require_replica,
            require_crypto_shred=require_crypto_shred,
            require_document_store=require_document_store,
            require_backup_ack=require_backup_ack,
        )
        self.store.save(wf)
        return wf

    def advance(
        self,
        request_id: str,
        *,
        max_m: int = 16,
        skip_optional: bool = False,
    ) -> ErasureWorkflow:
        """Run pending required steps until blocked or complete."""
        wf = self.store.load(request_id)
        if wf.status == "complete":
            return wf

        wf.status = "in_progress"
        self._run_live_delete(wf, max_m=max_m)
        self._mirror_residual_from_receipt(wf)
        self._run_optional(
            wf,
            name="replica_fanout",
            required=wf.require_replica,
            hook=(
                (lambda: self.replica_fanout(
                    wf.collection, wf.external_ids, wf.request_id
                ))
                if self.replica_fanout
                else None
            ),
            skip_optional=skip_optional,
        )
        self._run_optional(
            wf,
            name="crypto_shred",
            required=wf.require_crypto_shred,
            hook=(
                (lambda: self.crypto_shred(wf.external_ids))
                if self.crypto_shred
                else None
            ),
            skip_optional=skip_optional,
        )
        self._run_optional(
            wf,
            name="document_store",
            required=wf.require_document_store,
            hook=(
                (lambda: self.document_store(wf.collection, wf.external_ids))
                if self.document_store
                else None
            ),
            skip_optional=skip_optional,
        )
        self._run_optional(
            wf,
            name="backup_policy",
            required=wf.require_backup_ack,
            hook=(
                (lambda: self.backup_ack(wf.collection, wf.external_ids))
                if self.backup_ack
                else None
            ),
            skip_optional=skip_optional,
        )

        wf.status = self._compute_status(wf)
        self.store.save(wf)
        try:
            from api.metrics import METRICS

            METRICS.inc("workflow_advances")
            METRICS.set_gauge(
                f"workflow_status_{wf.status}",
                METRICS.gauges.get(f"workflow_status_{wf.status}", 0) + 1,
            )
            if wf.status == "complete":
                METRICS.inc("workflow_complete")
            elif wf.status == "blocked":
                METRICS.inc("workflow_blocked")
        except Exception:  # noqa: BLE001
            pass
        return wf

    def _run_live_delete(self, wf: ErasureWorkflow, *, max_m: int) -> None:
        step = wf.step("live_hard_delete")
        if step.status == "done":
            return
        step.status = "running"
        step.updated_unix_ns = time.time_ns()
        try:
            receipt: ErasureReceipt = self.erase.delete(
                wf.collection,
                wf.external_ids,
                reason=wf.reason,
                request_id=wf.request_id,
                max_m=max_m,
            )
            wf.receipt = receipt.to_dict()
            if receipt.success:
                step.status = "done"
                step.detail = (
                    f"status={receipt.status} compacted={receipt.compacted}"
                )
                step.artifact = {
                    "success": receipt.success,
                    "labels": receipt.labels,
                    "bytes_wiped_total": receipt.bytes_wiped_total,
                }
            else:
                step.status = "failed"
                step.detail = "; ".join(receipt.errors) or receipt.status
                wf.errors.append(f"live_hard_delete: {step.detail}")
        except Exception as exc:  # noqa: BLE001
            step.status = "failed"
            step.detail = str(exc)
            wf.errors.append(f"live_hard_delete: {exc}")
        step.updated_unix_ns = time.time_ns()

    def _mirror_residual_from_receipt(self, wf: ErasureWorkflow) -> None:
        step = wf.step("residual_proof")
        if step.status == "done":
            return
        receipt = wf.receipt or {}
        proof = receipt.get("residual_proof") or {}
        mode = proof.get("mode", "off")
        if mode == "off":
            step.status = "skipped"
            step.detail = "residual proof disabled"
        elif proof.get("passed") is True:
            step.status = "done"
            step.detail = proof.get("details", "passed")
            step.artifact = proof
        elif proof.get("passed") is False:
            step.status = "failed"
            step.detail = proof.get("details", "failed")
            step.artifact = proof
            wf.errors.append(f"residual_proof: {step.detail}")
        else:
            step.status = "skipped"
            step.detail = "no residual proof data"
        step.updated_unix_ns = time.time_ns()

    def _run_optional(
        self,
        wf: ErasureWorkflow,
        *,
        name: str,
        required: bool,
        hook: Callable[[], dict[str, Any]] | None,
        skip_optional: bool,
    ) -> None:
        step = wf.step(name)
        if step.status == "done":
            return
        if not required:
            step.status = "skipped"
            step.detail = "not required by policy"
            step.updated_unix_ns = time.time_ns()
            return
        if hook is None:
            step.status = "failed" if not skip_optional else "pending"
            step.detail = "required but no hook configured"
            if not skip_optional:
                wf.errors.append(f"{name}: no hook configured")
            step.updated_unix_ns = time.time_ns()
            return
        step.status = "running"
        try:
            art = hook()
            step.artifact = art
            step.status = "done"
            step.detail = "ok"
        except Exception as exc:  # noqa: BLE001
            step.status = "failed"
            step.detail = str(exc)
            wf.errors.append(f"{name}: {exc}")
        step.updated_unix_ns = time.time_ns()

    @staticmethod
    def _compute_status(wf: ErasureWorkflow) -> WorkflowStatus:
        live = wf.step("live_hard_delete")
        if live.status != "done":
            return "blocked" if live.status == "failed" else "in_progress"

        for s in wf.steps:
            if s.name == "live_hard_delete":
                continue
            required = {
                "residual_proof": True,  # must not be failed
                "replica_fanout": wf.require_replica,
                "crypto_shred": wf.require_crypto_shred,
                "document_store": wf.require_document_store,
                "backup_policy": wf.require_backup_ack,
            }.get(s.name, False)

            if s.name == "residual_proof":
                if s.status == "failed":
                    return "blocked"
                continue

            if required and s.status not in ("done", "skipped"):
                if s.status == "failed":
                    return "blocked"
                return "in_progress"
            if required and s.status == "failed":
                return "blocked"

        return "complete"
