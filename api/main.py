"""
FastAPI application entrypoint for the Latent Space Erasure & Graph Healing
proxy API.

Includes: WAL hard-delete, enterprise ids, erasure workflows, vector ingest,
metrics, multi-tenant data dirs, append-only receipts, and hook wiring.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

try:
    import hnsw_healer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "hnsw_healer native module is not installed. "
        "Build it with: pip install -e ."
    ) from exc

from api.auth import (
    ApiKeyMiddleware,
    get_signing_key_bytes,
    validate_production_secrets,
)
from api.metrics import METRICS
from api.persistence import PersistenceEngine
from integrations.erase_service import ErasureService
from integrations.hooks_config import build_hooks_from_env
from integrations.id_registry import CollectionIdRegistry, IdMappingError
from integrations.native_backend import NativeHealerBackend
from integrations.outbox import FileOutbox, OutboxDispatcher
from integrations.receipt_log import AppendOnlyReceiptLog
from integrations.tenancy import TenantManager, multi_tenant_enabled
from integrations.workflow import ErasureWorkflowRunner, ErasureWorkflowStore

logger = logging.getLogger(__name__)

_SIGNING_KEY = get_signing_key_bytes()

_LOCK_MAX_ATTEMPTS = int(os.environ.get("HEALER_LOCK_MAX_ATTEMPTS", "5"))
_LOCK_BASE_DELAY_S = float(os.environ.get("HEALER_LOCK_BASE_DELAY_S", "0.01"))

T = TypeVar("T")

engine: PersistenceEngine | None = None
_bootstrap_info: dict[str, Any] = {}

id_registry = CollectionIdRegistry()
_erasure_service: ErasureService | None = None
_workflow_store: ErasureWorkflowStore | None = None
_workflow_runner: ErasureWorkflowRunner | None = None
_receipt_log: AppendOnlyReceiptLog | None = None
_tenant_manager: TenantManager | None = None
_file_outbox: FileOutbox | None = None


def with_lock_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int | None = None,
    base_delay_s: float | None = None,
) -> T:
    attempts = max_attempts if max_attempts is not None else _LOCK_MAX_ATTEMPTS
    delay = base_delay_s if base_delay_s is not None else _LOCK_BASE_DELAY_S
    if attempts < 1:
        attempts = 1

    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except hnsw_healer.LockContentionError as exc:
            last = exc
            METRICS.inc("lock_contention")
            if attempt + 1 >= attempts:
                break
            time.sleep(delay * (2**attempt))
    assert last is not None
    raise last


def get_engine() -> PersistenceEngine:
    global engine
    if engine is None:
        engine = PersistenceEngine(lock_retry=with_lock_retry)
    return engine


def get_receipt_log() -> AppendOnlyReceiptLog:
    global _receipt_log
    if _receipt_log is None:
        _receipt_log = AppendOnlyReceiptLog(
            get_engine().data_dir / "receipts.jsonl"
        )
    return _receipt_log


def get_tenant_manager() -> TenantManager:
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager(
            get_engine().data_dir, master_signing_key=_SIGNING_KEY
        )
    return _tenant_manager


def get_erasure_service() -> ErasureService:
    global _erasure_service
    if _erasure_service is None:
        _erasure_service = ErasureService(
            id_registry,
            NativeHealerBackend(),
            persistence=get_engine(),
            signing_key=_SIGNING_KEY,
            receipt_log=get_receipt_log(),
            metrics=METRICS,
        )
    return _erasure_service


def get_workflow_store() -> ErasureWorkflowStore:
    global _workflow_store
    if _workflow_store is None:
        _workflow_store = ErasureWorkflowStore(get_engine().data_dir)
    return _workflow_store


def get_workflow_runner() -> ErasureWorkflowRunner:
    global _workflow_runner
    if _workflow_runner is None:
        hooks = build_hooks_from_env(get_engine().data_dir)
        _workflow_runner = ErasureWorkflowRunner(
            get_workflow_store(),
            get_erasure_service(),
            **hooks,
        )
    return _workflow_runner


def get_outbox() -> FileOutbox:
    global _file_outbox
    if _file_outbox is None:
        _file_outbox = FileOutbox(get_engine().data_dir)
    return _file_outbox


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, _bootstrap_info, _erasure_service, _SIGNING_KEY
    global _workflow_store, _workflow_runner, _receipt_log, _tenant_manager
    global _file_outbox
    logging.basicConfig(level=logging.INFO)

    validate_production_secrets()
    _SIGNING_KEY = get_signing_key_bytes()

    engine = PersistenceEngine(lock_retry=with_lock_retry)
    _bootstrap_info = engine.bootstrap()
    _receipt_log = AppendOnlyReceiptLog(engine.data_dir / "receipts.jsonl")
    _tenant_manager = TenantManager(
        engine.data_dir, master_signing_key=_SIGNING_KEY
    )
    _file_outbox = FileOutbox(engine.data_dir)
    _erasure_service = ErasureService(
        id_registry,
        NativeHealerBackend(),
        persistence=engine,
        signing_key=_SIGNING_KEY,
        receipt_log=_receipt_log,
        metrics=METRICS,
    )
    hooks = build_hooks_from_env(engine.data_dir)
    _workflow_store = ErasureWorkflowStore(engine.data_dir)
    _workflow_runner = ErasureWorkflowRunner(
        _workflow_store, _erasure_service, **hooks
    )
    reg_path = engine.data_dir / "id_registry.json"
    if reg_path.is_file():
        id_registry.load(reg_path)
        logger.info("Loaded id registry from %s", reg_path)
    logger.info("Persistence bootstrap: %s", _bootstrap_info)
    logger.info(
        "multi_tenant=%s residual_proof=%s",
        multi_tenant_enabled(),
        os.environ.get("HEALER_RESIDUAL_PROOF", "sample"),
    )
    yield
    try:
        if engine is not None:
            id_registry.save(engine.data_dir / "id_registry.json")
    except OSError:
        logger.exception("failed to persist id registry")


app = FastAPI(
    title="Latent Space Erasure & Graph Healing API",
    description=(
        "Hard-delete middleware: soft-delete leaves invertible residual "
        "vectors; this service zeros embeddings, heals HNSW, proves residual "
        "absence, and emits signed audit receipts."
    ),
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(ApiKeyMiddleware)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DeleteRequest(BaseModel):
    node_id: int = Field(..., ge=0)
    max_m: int = Field(16, ge=1)


class DeleteResponse(BaseModel):
    status: str
    node_id: int
    signature: str
    message: str | None = None
    retries: int = 0
    transaction_id: int | None = None


class SearchRequest(BaseModel):
    query: list[float]
    k: int = Field(10, ge=1, le=1000)
    entry_node: int = Field(-1)


class SearchHitModel(BaseModel):
    node_id: int
    distance: float


class SearchResponse(BaseModel):
    hits: list[SearchHitModel]
    retries: int = 0


class EnterpriseDeleteRequest(BaseModel):
    collection: str = Field(..., min_length=1)
    ids: list[str] = Field(..., min_length=1)
    reason: str | None = None
    request_id: str | None = None
    max_m: int = Field(16, ge=1)
    compact: str | bool | None = None
    residual_proof: str | None = None


class EnterpriseDeleteResponse(BaseModel):
    success: bool
    request_id: str
    collection: str
    external_ids: list[str]
    labels: list[int]
    bytes_wiped_total: int
    signature: str
    reason: str | None = None
    errors: list[str] = []
    transaction_ids: list[int] = []
    receipt_version: int = 2
    status: str = "failed"
    compacted: bool = False
    residual_proof: dict[str, Any] | None = None


class RegisterIdsRequest(BaseModel):
    collection: str = Field(..., min_length=1)
    ids: list[str] = Field(..., min_length=1)
    labels: list[int] | None = None


class IngestVectorsRequest(BaseModel):
    """Register business ids and load float32 vectors into the native index."""

    collection: str = Field(..., min_length=1)
    ids: list[str] = Field(..., min_length=1)
    vectors: list[list[float]] = Field(..., min_length=1)
    labels: list[int] | None = None
    replace_index: bool = Field(
        False,
        description="If true, replace entire index with this batch; else append",
    )
    checkpoint: bool = Field(
        True, description="Write index.bin after successful load"
    )


class CreateErasureRequest(BaseModel):
    collection: str = Field(..., min_length=1)
    ids: list[str] = Field(..., min_length=1)
    reason: str | None = None
    request_id: str | None = None
    require_replica: bool = False
    require_crypto_shred: bool = False
    require_document_store: bool = False
    require_backup_ack: bool = False
    advance: bool = True
    max_m: int = Field(16, ge=1)


def _sign_erasure(node_id: int, status: str) -> str:
    payload = f"{status}:{node_id}".encode("utf-8")
    return hmac.new(_SIGNING_KEY, payload, hashlib.sha256).hexdigest()


def _receipt_response(receipt: Any) -> dict[str, Any]:
    return {
        "success": receipt.success,
        "request_id": receipt.request_id,
        "collection": receipt.collection,
        "external_ids": receipt.external_ids,
        "labels": receipt.labels,
        "bytes_wiped_total": receipt.bytes_wiped_total,
        "signature": receipt.signature,
        "reason": receipt.reason,
        "errors": receipt.errors,
        "transaction_ids": receipt.transaction_ids,
        "receipt_version": receipt.receipt_version,
        "status": receipt.status,
        "compacted": receipt.compacted,
        "residual_proof": receipt.residual_proof,
    }


def _tenant_from_headers(
    x_tenant_id: str | None = None,
) -> str | None:
    if not multi_tenant_enabled():
        return None
    if not x_tenant_id:
        raise HTTPException(
            status_code=400,
            detail="X-Tenant-ID required when HEALER_MULTI_TENANT=1",
        )
    return x_tenant_id.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    idx = hnsw_healer.default_index()
    eng = get_engine()
    return {
        "status": "ok",
        "index_loaded": bool(idx.is_loaded),
        "lock_pool_size": int(idx.lock_pool_size),
        "lock_timeout_ms": int(idx.lock_timeout_ms),
        "index_path": str(eng.index_path),
        "wal_path": str(eng.wal_path),
        "bootstrap": _bootstrap_info,
        "api_version": "0.3.0",
        "multi_tenant": multi_tenant_enabled(),
        "metrics": {
            "deletes_batches": METRICS.counters.get("deletes_batches", 0),
            "lock_contention": METRICS.counters.get("lock_contention", 0),
        },
    }


@app.get("/metrics")
async def metrics_prometheus() -> Response:
    """Prometheus text exposition (also useful for scrape configs)."""
    return Response(
        content=METRICS.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/metrics")
async def metrics_json() -> dict[str, Any]:
    return METRICS.snapshot()


@app.post("/delete", response_model=DeleteResponse)
async def delete_node(body: DeleteRequest) -> dict[str, Any]:
    eng = get_engine()
    if not hnsw_healer.default_index().is_loaded:
        raise HTTPException(
            status_code=409,
            detail="no index loaded; load or recover index.bin first",
        )

    attempts_used = 0

    def _tracked_retry(fn: Callable[[], Any]) -> Any:
        nonlocal attempts_used

        def wrapped() -> Any:
            nonlocal attempts_used
            attempts_used += 1
            return fn()

        return with_lock_retry(wrapped)

    eng._lock_retry = _tracked_retry  # noqa: SLF001

    t0 = time.perf_counter()
    try:
        result = eng.hard_delete_and_heal(body.node_id, body.max_m)
    except hnsw_healer.LockContentionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"lock contention after retries: {exc}",
            headers={"Retry-After": "1"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"durable erasure failed: {exc}",
        ) from exc
    finally:
        METRICS.observe_ms("heal", (time.perf_counter() - t0) * 1000.0)
        METRICS.inc("raw_deletes")

    status = "erased"
    return {
        "status": status,
        "node_id": body.node_id,
        "signature": _sign_erasure(body.node_id, status),
        "message": result.message,
        "retries": max(0, attempts_used - 1),
        "transaction_id": result.transaction_id,
    }


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest) -> dict[str, Any]:
    import numpy as np

    if not hnsw_healer.default_index().is_loaded:
        raise HTTPException(status_code=409, detail="no index loaded")

    attempts_used = 0
    query = np.asarray(body.query, dtype=np.float32)

    def _search() -> list[Any]:
        nonlocal attempts_used
        attempts_used += 1
        return list(
            hnsw_healer.search_knn(
                query, k=body.k, entry_node=body.entry_node
            )
        )

    t0 = time.perf_counter()
    try:
        hits = with_lock_retry(_search)
    except hnsw_healer.LockContentionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"lock contention after retries: {exc}",
            headers={"Retry-After": "1"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"native search failed: {exc}",
        ) from exc
    finally:
        METRICS.observe_ms("search", (time.perf_counter() - t0) * 1000.0)
        METRICS.inc("searches")

    return {
        "hits": [
            {"node_id": int(h.node_id), "distance": float(h.distance)}
            for h in hits
        ],
        "retries": max(0, attempts_used - 1),
    }


@app.post("/v1/ids/register")
async def register_ids(body: RegisterIdsRequest) -> dict[str, Any]:
    mapping: dict[str, int] = {}
    try:
        if body.labels is not None:
            if len(body.labels) != len(body.ids):
                raise HTTPException(
                    status_code=400, detail="ids and labels length mismatch"
                )
            for eid, lab in zip(body.ids, body.labels):
                mapping[eid] = id_registry.register(
                    body.collection, eid, label=lab
                )
        else:
            for eid in body.ids:
                mapping[eid] = id_registry.register(body.collection, eid)
    except (ValueError, IdMappingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    eng = get_engine()
    try:
        id_registry.save(eng.data_dir / "id_registry.json")
    except OSError:
        logger.exception("id registry save failed")

    return {"collection": body.collection, "mapping": mapping}


@app.post("/v1/vectors/ingest")
async def ingest_vectors(body: IngestVectorsRequest) -> dict[str, Any]:
    """
    Register business ids and load vectors into the process-local native index.

    This is the HTTP golden path for the sidecar: one call at ingest time so
    later ``/v1/collections/.../delete`` can resolve external ids without a
    separate register step in the app.
    """
    import numpy as np

    if len(body.ids) != len(body.vectors):
        raise HTTPException(
            status_code=400, detail="ids and vectors length mismatch"
        )
    if body.labels is not None and len(body.labels) != len(body.ids):
        raise HTTPException(
            status_code=400, detail="ids and labels length mismatch"
        )

    vecs = np.asarray(body.vectors, dtype=np.float32)
    if vecs.ndim != 2:
        raise HTTPException(status_code=400, detail="vectors must be 2-D")
    n, d = vecs.shape

    mapping: dict[str, int] = {}
    try:
        if body.labels is not None:
            for eid, lab in zip(body.ids, body.labels):
                mapping[str(eid)] = id_registry.register(
                    body.collection, str(eid), label=int(lab)
                )
        else:
            for eid in body.ids:
                mapping[str(eid)] = id_registry.register(
                    body.collection, str(eid)
                )
    except (ValueError, IdMappingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    labels_arr = np.array([mapping[str(i)] for i in body.ids], dtype=np.int64)

    # Order rows by label for dense native storage when replace_index.
    if body.replace_index:
        order = np.argsort(labels_arr)
        ordered = vecs[order]
        # Dense 0..n-1 expected by simple deployments: remap if labels not dense
        max_lab = int(labels_arr.max()) if n else -1
        if max_lab + 1 != n or set(labels_arr.tolist()) != set(range(n)):
            # Build dense matrix sized to max_label+1
            dense_n = max_lab + 1
            dense = np.zeros((dense_n, d), dtype=np.float32)
            for lab, row in zip(labels_arr, vecs):
                dense[int(lab)] = row
            ordered = dense
            n_load = dense_n
        else:
            n_load = n
            ordered = ordered
        hnsw_healer.load_index(ordered, d, n_load)
        idx = hnsw_healer.default_index()
        for i in range(n_load):
            idx.set_neighbors(
                i, 0, [(i - 1) % n_load, (i + 1) % n_load, (i + 2) % n_load]
            )
    else:
        # Append path: only supported when index empty or same dim
        idx = hnsw_healer.default_index()
        if not idx.is_loaded:
            # Load as dense from labels
            max_lab = int(labels_arr.max())
            dense = np.zeros((max_lab + 1, d), dtype=np.float32)
            for lab, row in zip(labels_arr, vecs):
                dense[int(lab)] = row
            hnsw_healer.load_index(dense, d, max_lab + 1)
            for i in range(max_lab + 1):
                hnsw_healer.default_index().set_neighbors(
                    i,
                    0,
                    [
                        (i - 1) % (max_lab + 1),
                        (i + 1) % (max_lab + 1),
                        (i + 2) % (max_lab + 1),
                    ],
                )
        else:
            raise HTTPException(
                status_code=409,
                detail=(
                    "append to non-empty native index not supported; "
                    "use replace_index=true or adapters"
                ),
            )

    if body.checkpoint:
        get_engine().save_initial_checkpoint()

    try:
        id_registry.save(get_engine().data_dir / "id_registry.json")
    except OSError:
        logger.exception("id registry save failed")

    METRICS.inc("ingest_batches")
    METRICS.inc("ingest_vectors", float(n))

    return {
        "collection": body.collection,
        "count": n,
        "dim": d,
        "mapping": mapping,
        "index_loaded": bool(hnsw_healer.default_index().is_loaded),
        "checkpointed": body.checkpoint,
    }


@app.post("/v1/internal/replica/delete")
async def replica_delete_intent(body: dict[str, Any]) -> dict[str, Any]:
    from integrations.replica_fanout import DeleteIntent

    try:
        intent = DeleteIntent.from_json(__import__("json").dumps(body))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad intent: {exc}") from exc

    if not hnsw_healer.default_index().is_loaded:
        raise HTTPException(status_code=409, detail="no index loaded on replica")

    svc = get_erasure_service()
    try:
        receipt = svc.delete(
            intent.collection,
            intent.external_ids,
            reason=intent.reason,
            request_id=intent.request_id,
            max_m=intent.max_m,
            idempotent=True,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "request_id": intent.request_id,
        }

    return {
        "success": receipt.success,
        "message": "; ".join(receipt.errors) if receipt.errors else "ok",
        "request_id": intent.request_id,
        "bytes_wiped_total": receipt.bytes_wiped_total,
        "status": receipt.status,
    }


@app.post(
    "/v1/collections/{collection}/delete",
    response_model=EnterpriseDeleteResponse,
)
async def enterprise_delete(
    collection: str, body: EnterpriseDeleteRequest
) -> dict[str, Any]:
    if body.collection != collection:
        raise HTTPException(
            status_code=400,
            detail="path collection must match body.collection",
        )
    if not hnsw_healer.default_index().is_loaded:
        raise HTTPException(
            status_code=409,
            detail="no index loaded; load or recover index.bin first",
        )

    svc = get_erasure_service()
    delete_kwargs: dict[str, Any] = {
        "reason": body.reason,
        "request_id": body.request_id,
        "max_m": body.max_m,
    }
    if body.compact is not None:
        delete_kwargs["compact"] = body.compact
    if body.residual_proof is not None:
        delete_kwargs["residual_proof"] = body.residual_proof

    t0 = time.perf_counter()
    try:
        receipt = svc.delete(collection, body.ids, **delete_kwargs)
    except IdMappingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except hnsw_healer.LockContentionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"lock contention: {exc}",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"enterprise erase failed: {exc}"
        ) from exc
    finally:
        METRICS.observe_ms(
            "enterprise_delete", (time.perf_counter() - t0) * 1000.0
        )

    try:
        id_registry.save(get_engine().data_dir / "id_registry.json")
    except OSError:
        logger.exception("id registry save failed")

    return _receipt_response(receipt)


@app.post("/v1/admin/compact")
async def admin_force_compact() -> dict[str, Any]:
    """Flush coalesced compact pending on the active backend."""
    return get_erasure_service().force_compact()


class StrategyRequest(BaseModel):
    delete_count: int = Field(..., ge=0)
    index_size: int = Field(..., ge=0)
    backend_supports_compact: bool = True
    backend_supports_heal: bool = True


@app.post("/v1/admin/delete-strategy")
async def recommend_strategy(body: StrategyRequest) -> dict[str, Any]:
    """
    Recommend wipe / heal / compact / full-rebuild given batch size.

    Use before large GDPR jobs to choose ``compact=always`` vs coalesce.
    """
    from integrations.delete_strategy import (
        compact_mode_for_recommendation,
        recommend_delete_strategy,
    )

    rec = recommend_delete_strategy(
        delete_count=body.delete_count,
        index_size=body.index_size,
        backend_supports_compact=body.backend_supports_compact,
        backend_supports_heal=body.backend_supports_heal,
    )
    out = rec.to_dict()
    out["suggested_compact_mode"] = compact_mode_for_recommendation(rec)
    return out


@app.get("/v1/receipts")
async def list_receipts(limit: int = 100) -> dict[str, Any]:
    """Tail of append-only receipt log (newest last among returned)."""
    rows = list(get_receipt_log().iter_receipts())
    if limit > 0:
        rows = rows[-limit:]
    return {"count": len(rows), "receipts": rows}


@app.get("/v1/receipts/{request_id}")
async def get_receipts_by_request(request_id: str) -> dict[str, Any]:
    found = get_receipt_log().find_by_request_id(request_id)
    if not found:
        raise HTTPException(status_code=404, detail="no receipts for request_id")
    return {"request_id": request_id, "receipts": found}


@app.post("/v1/outbox/dispatch")
async def dispatch_outbox(limit: int = 50) -> dict[str, Any]:
    """
    Dispatch pending replica outbox intents.

    Default dispatch records local success only unless peers are configured;
    for real fan-out, set workers via HEALER_OUTBOX_REPLICA + custom dispatch.
    """
    outbox = get_outbox()

    def _dispatch(env):
        # Best-effort: apply locally again (idempotent) and mark success.
        # Production should call ReplicaFanoutCoordinator here.
        svc = get_erasure_service()
        try:
            r = svc.delete(
                env.collection,
                env.external_ids,
                reason=env.reason,
                request_id=env.request_id,
                max_m=env.max_m,
                idempotent=True,
            )
            return {"success": r.success, "message": r.status}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "message": str(exc)}

    dispatcher = OutboxDispatcher(outbox, _dispatch)
    result = dispatcher.dispatch_pending(limit=limit)
    METRICS.inc("outbox_dispatch_runs")
    return result


@app.get("/v1/outbox/pending")
async def list_outbox_pending() -> dict[str, Any]:
    pending = get_outbox().list_pending()
    return {
        "count": len(pending),
        "envelopes": [e.to_dict() for e in pending],
    }


@app.get("/v1/tenants")
async def list_tenants() -> dict[str, Any]:
    return {
        "multi_tenant": multi_tenant_enabled(),
        "tenants": get_tenant_manager().list_tenants(),
    }


# ---------------------------------------------------------------------------
# Erasure workflow API
# ---------------------------------------------------------------------------


@app.post("/v1/erasure-requests")
async def create_erasure_request(body: CreateErasureRequest) -> dict[str, Any]:
    runner = get_workflow_runner()
    wf = runner.create(
        body.collection,
        body.ids,
        reason=body.reason,
        request_id=body.request_id,
        require_replica=body.require_replica,
        require_crypto_shred=body.require_crypto_shred,
        require_document_store=body.require_document_store,
        require_backup_ack=body.require_backup_ack,
    )
    if body.advance:
        if not hnsw_healer.default_index().is_loaded:
            raise HTTPException(
                status_code=409,
                detail="no index loaded; create with advance=false or load index",
            )
        wf = runner.advance(wf.request_id, max_m=body.max_m)
    return wf.to_dict()


@app.get("/v1/erasure-requests")
async def list_erasure_requests() -> dict[str, Any]:
    return {"request_ids": get_workflow_store().list_ids()}


@app.get("/v1/erasure-requests/{request_id}")
async def get_erasure_request(request_id: str) -> dict[str, Any]:
    try:
        wf = get_workflow_store().load(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return wf.to_dict()


@app.post("/v1/erasure-requests/{request_id}/advance")
async def advance_erasure_request(
    request_id: str, max_m: int = 16
) -> dict[str, Any]:
    try:
        get_workflow_store().load(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not hnsw_healer.default_index().is_loaded:
        raise HTTPException(status_code=409, detail="no index loaded")
    wf = get_workflow_runner().advance(request_id, max_m=max_m)
    return wf.to_dict()


@app.get("/v1/erasure-requests/{request_id}/export")
async def export_erasure_request(request_id: str) -> dict[str, Any]:
    try:
        wf = get_workflow_store().load(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    package = wf.export_package()
    # Attach append-only log rows for this request when present.
    package["receipt_log"] = get_receipt_log().find_by_request_id(request_id)
    return package
