# Golden path — hard-delete that actually erases

This is the recommended integration for self-hosted Python RAG stacks.

**Goal:** forget a business id in one call, with:

1. Physical zero of the embedding  
2. **One** ANN rebuild (`compact`) per delete batch  
3. Residual proof attached to a signed receipt  
4. Fail-closed metadata delete (Chroma)

You do **not** need a separate register step or manual `compact()` on this path.

---

## Prerequisites

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -e ".[chroma,dev]"
```

---

## Path A — Chroma + hnswlib (flagship)

```python
import numpy as np
import chromadb
from integrations import CollectionIdRegistry, ErasureService
from integrations.chroma_hook import ChromaHardDeleteCollection
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

reg = CollectionIdRegistry()
backend = HnswlibHardDeleteAdapter(
    dim=384, collection="docs", registry=reg, max_elements=50_000
)
svc = ErasureService(reg, backend)  # compact=auto, residual_proof=sample

client = chromadb.Client()
raw = client.get_or_create_collection("docs")
col = ChromaHardDeleteCollection(raw, svc, collection_name="docs")

# Ingest — ids are registered automatically
col.add(
    ids=["user-42-doc"],
    embeddings=[...],   # float lists, dim=384
    documents=["..."],
)

# Forget — wipe + compact once + residual proof + Chroma delete
receipt = col.delete(
    ids=["user-42-doc"],
    reason="gdpr_art_17",
    request_id="ticket-1001",
)
assert receipt.success
assert receipt.status == "complete"
assert receipt.compacted
assert receipt.residual_proof["passed"]
# Persist receipt.to_dict() in your audit log
```

Runnable copy: [`examples/chroma_forget/run.py`](../examples/chroma_forget/run.py).

```bash
python examples/chroma_forget/run.py
```

---

## Path B — hnswlib only (no Chroma)

```python
from integrations import CollectionIdRegistry, ErasureService
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

reg = CollectionIdRegistry()
store = HnswlibHardDeleteAdapter(dim=128, collection="users", registry=reg)
store.add(["user-1", "user-2"], embeddings_nxd)  # registers ids

svc = ErasureService(reg, store)
receipt = svc.delete("users", ["user-1"], reason="gdpr_art_17")
assert receipt.success and receipt.compacted
# no manual store.compact() required
```

---

## Path C — Native HTTP + workflow

1. Load vectors into `hnsw_healer` and checkpoint (`PersistenceEngine`).  
2. `POST /v1/ids/register` **once at ingest** (HTTP path has no dual-write helper yet).  
3. Either:
   - `POST /v1/collections/{collection}/delete`, or  
   - `POST /v1/erasure-requests` with `"advance": true` for a durable workflow JSON.

Set secrets before production:

```bash
export HEALER_SIGNING_KEY='long-random-secret'
export HEALER_API_KEY='long-random-api-key'
export HEALER_ENV=production
```

---

## Receipt schema v2 (what “done” means)

| Field | Meaning |
|-------|---------|
| `receipt_version` | `2` |
| `status` | `complete` \| `partial` \| `failed` |
| `success` | `true` only when `status == complete` |
| `compacted` | Backend ran `compact()` once for this batch |
| `residual_proof` | `{ mode, passed, checked, proofs, ... }` |
| `signature` | HMAC-SHA256 over versioned payload |

`success=true` means: all requested ids erased, compact policy satisfied, residual proof passed (when enabled).

---

## Fail-closed behavior

`ChromaHardDeleteCollection(fail_closed=True)` (default) **refuses** Chroma metadata delete when:

- hard erase fails  
- compact fails (`compact=always` / auto on compactable backends)  
- residual proof fails  

So a soft-delete-only state is not the easy outcome under failure.

---

## Configuration

| Variable | Default | Role |
|----------|---------|------|
| `HEALER_RESIDUAL_PROOF` | `sample` | `off` \| `sample` \| `full` |
| `HEALER_SIGNING_KEY` | dev placeholder | Receipt HMAC (required non-default in prod) |
| `HEALER_API_KEY` | unset | If set, API requires `X-API-Key` / Bearer |
| `HEALER_ENV` | `development` | `production` enforces API key + strong signing key |

---

## What this path does *not* cover

- Document store / object storage delete (use workflow hooks)  
- Replica fan-out quorum (workflow + `replica_fanout`)  
- Volume snapshots (crypto-shred + retention policy)  
- Hosted SaaS vector DBs without an adapter  

See [THREAT_MODEL.md](THREAT_MODEL.md) and [BACKUPS_AND_REPLICAS.md](BACKUPS_AND_REPLICAS.md).
