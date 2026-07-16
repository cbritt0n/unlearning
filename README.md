# HNSW Healer

### Hard-delete residual vectors in HNSW — wipe, rebuild, prove, receipt

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](src/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](CHANGELOG.md)

**Soft-delete leaves invertible float embeddings on disk.** HNSW Healer is middleware that **physically zeros** residual vectors, **rebuilds / compacts** the ANN structure so search stays usable (MN-RU heal is optional/experimental), **proves** residual absence, and emits a **signed audit receipt**—so \(\mathbf{v}\) is not waiting for a Vec2Text-class inversion.

**Headline evaluation (hnswlib, N=50k, 10% delete):** soft and wipe+rebuild both ~**0.35 recall@10**, but soft leaves **residual=YES** while rebuild is **residual=no** and still **usable**. Full pack: [docs/benchmarks/standard_hnswlib.md](docs/benchmarks/standard_hnswlib.md).

> **Start here:** [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) · [docs/INSTALL.md](docs/INSTALL.md) · [CONTRIBUTING.md](CONTRIBUTING.md)  
> **Windows + hnswlib:** [docs/HNSWLIB_AND_BENCHMARKS.md](docs/HNSWLIB_AND_BENCHMARKS.md)  
> **Upload to GitHub:** [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) · **Security:** [SECURITY.md](SECURITY.md)

| Layer | What you get |
|-------|----------------|
| **Native core** | C++17 / pybind11 — wipe, MN-RU heal, lock-coupled search, serialize |
| **Control plane** | FastAPI — durable delete, ingest, workflows, metrics, receipts |
| **Integrations** | hnswlib / FAISS / Chroma hooks, id registry, outbox fan-out |
| **Compliance** | Residual proofs, crypto-shred, webhooks, threat-model docs |

This is **not** a full GDPR product or a greenfield vector DB. It is an **index residual control** you wire onto the delete path.

---

## Table of contents

1. [The problem](#the-problem)
2. [What this project does](#what-this-project-does)
3. [What it does *not* claim](#what-it-does-not-claim)
4. [Architecture](#architecture)
5. [Repository layout](#repository-layout)
6. [Requirements](#requirements)
7. [Quick start](#quick-start)
8. [Usage paths](#usage-paths)
9. [HTTP API](#http-api)
10. [Native Python API](#native-python-api)
11. [Enterprise integration](#enterprise-integration)
12. [Durability & crash recovery](#durability--crash-recovery)
13. [Concurrency model](#concurrency-model)
14. [Configuration](#configuration)
15. [Testing & benchmarks](#testing--benchmarks)
16. [Docker](#docker)
17. [CI & packaging](#ci--packaging)
18. [Security & residual data](#security--residual-data)
19. [Troubleshooting](#troubleshooting)
20. [Roadmap & status](#roadmap--status)
21. [Further documentation](#further-documentation)
22. [Community](#community)
23. [License](#license)

---

## The problem

Modern RAG and semantic-search stacks store document embeddings in ANN indexes (often **HNSW**). When a business “deletes” a user or document, many vector databases perform a **metadata soft-delete**:

- Query filters skip the id.
- The raw embedding \(\mathbf{v}\) often **remains** in the in-memory or on-disk HNSW structure.

Security research has shown that residual vectors can be recovered from storage and inverted toward plaintext with models in the **Vec2Text** family—bypassing the intent of GDPR Art. 17–style erasure and similar contractual unlearning requirements.

Naïve hard delete (drop edges, leave holes) is rarely adopted because it **fragments** the graph and destroys recall. Operators therefore choose soft-delete for performance—and leave latent data on disk.

```text
  App:  DELETE user_42
           │
           ▼
  Vector DB soft-delete ──►  metadata tombstone
           │
           ▼
  HNSW binary still holds v_42  ──►  dump floats  ──►  inversion model  ──►  text
```

---

## What this project does

HNSW Healer treats delete as a **three-part unlearning operation**:

1. **Physical erasure** — write `0.0f` into the embedding region (volatile stores to resist compiler elision).
2. **Graph healing** — isolate the node, then reconnect orphaned neighbors with an **MN-RU** heuristic under degree cap \(M\), so the index remains navigable.
3. **Durable commit** — WAL `BEGIN` → mutate → `index.bin.tmp` → atomic `os.replace` → WAL `COMMIT`, with **startup replay** of incomplete transactions.

| Concern | Soft delete | Naïve hard delete | **HNSW Healer** |
|---------|-------------|-------------------|-----------------|
| Residual \(\mathbf{v}\) in index | Remains | Often zeroed, not always durable | Zeroed in process + rewritten checkpoint |
| Search quality after delete | Intact | Orphans / fragmentation | MN-RU rewire under max-\(M\) |
| Crash mid-delete | N/A | Old checkpoint can resurrect \(\mathbf{v}\) | WAL + atomic rename + recovery |
| Concurrent queries during rewire | Usually fine | Race risk | Neighborhood-striped R/W locks |
| Business ids (`user_id`) | App-level | App-level | `ErasureService` + registry + HTTP `/v1` |
| hnswlib / Chroma | Soft-delete native | Manual | Adapters: wipe + `mark_deleted` + `compact` / hook |

---

## What it does *not* claim

Be explicit for security and legal readers:

- **Not** a drop-in binary-compatible replacement for every FAISS/Chroma segment format (adapters and dual-write are the integration path).
- **Not** a wipe of OS swap, core dumps, CPU caches, or offline volume snapshots.
- **Not** a complete GDPR determination by itself—pair with document-store delete, backup policy, and (optionally) crypto-shred.
- **Not** a formal proof that MN-RU preserves optimal HNSW invariants under all delete patterns; benchmarks compare soft / unhealed / healed empirically.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the residual matrix.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Clients / apps / ETL                             │
│         GDPR job  ·  admin API  ·  Chroma wrapper  ·  scripts            │
└─────────────┬───────────────────────────────┬────────────────────────────┘
              │ HTTP                          │ Python library
              ▼                               ▼
┌─────────────────────────────┐   ┌────────────────────────────────────────┐
│  FastAPI  (api/main.py)     │   │  integrations/                         │
│  GET  /health               │   │  ErasureService.delete(collection,ids) │
│  POST /delete               │   │  CollectionIdRegistry                  │
│  POST /search               │   │  HnswlibHardDeleteAdapter              │
│  POST /v1/ids/register      │   │  ChromaHardDeleteCollection            │
│  POST /v1/collections/...   │   └──────────────────┬─────────────────────┘
└─────────────┬───────────────┘                      │
              │                                      │
              ▼                                      ▼
┌─────────────────────────────┐   ┌────────────────────────────────────────┐
│  PersistenceEngine          │   │  compliance/                           │
│  WAL BEGIN → erase → flush  │   │  residual proofs · crypto-shred vault  │
│  → atomic index.bin → COMMIT│   └────────────────────────────────────────┘
└─────────────┬───────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  hnsw_healer  (C++17 / pybind11)                                         │
│  HNSWIndexProxy · secure wipe · heal_graph_structure · search_knn        │
│  neighborhood striped shared_timed_mutex · save/load binary              │
└──────────────────────────────────────────────────────────────────────────┘
              │
              ▼
         HEALER_DATA_DIR/
           index.bin   index.bin.tmp   index.wal   id_registry.json
```

**Hot path (delete):** intent is fsynced to the WAL *before* mutation. After a successful heal, a full snapshot is written to a temp file and published with an atomic rename so a crash never leaves a half-written production index.

---

## Repository layout

```text
unlearning/
├── api/                      # FastAPI control plane
│   ├── main.py               # Routes, lifespan recovery, enterprise delete
│   ├── persistence.py        # WAL orchestration + atomic index flush
│   └── wal.py                # Append-only binary WAL (SHA-256 records)
├── integrations/             # Business vector DB glue
│   ├── erase_service.py      # delete(collection, ids) → ErasureReceipt
│   ├── id_registry.py        # external id ↔ HNSW label
│   ├── hnswlib_adapter.py    # wipe + mark_deleted + compact()
│   ├── chroma_hook.py        # hard-erase then Chroma metadata delete
│   └── native_backend.py     # HardDeleteBackend over default proxy
├── compliance/
│   ├── residual.py           # Live zero + checkpoint pattern proofs
│   └── crypto_shred.py       # Per-entity AES-GCM DEK destroy
├── src/                      # Native extension sources
│   ├── healer.cpp            # pybind11 module
│   ├── hnsw_helper.hpp       # Index, heal, serialize
│   └── neighborhood_locks.hpp
├── docs/                     # Threat model, integration, backups, testing
├── tests/                    # Unit + residual + optional adapter tests
│   └── benchmark.py          # Soft vs unhealed vs healed evaluation
├── CMakeLists.txt
├── setup.py / pyproject.toml
├── Dockerfile
├── requirements.txt
└── LICENSE
```

---

## Requirements

| Dependency | Notes |
|------------|--------|
| **Python 3.10+** | 3.11 recommended for Docker image parity |
| **CMake ≥ 3.15** | Used by `setup.py` / cibuildwheel |
| **C++17 compiler** | Linux: `g++` / `build-essential`; macOS: Xcode CLT; Windows: **MSVC** *or* **LLVM-MinGW** (`clang++`) |
| **Ninja** (recommended) | `pip install ninja` — auto-detected |
| **Optional** | Docker; `hnswlib`; `chromadb` |

---

## Quick start

### 1. Clone and virtualenv

```bash
git clone https://github.com/<your-org>/unlearning.git
cd unlearning

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
```

### 2. Build the native module

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` runs CMake, compiles `hnsw_healer`, and installs Python packages (`api`, `integrations`, `compliance`) in editable mode.

Verify:

```bash
python -c "import hnsw_healer; print(hnsw_healer.__version__)"
```

**Windows + LLVM-MinGW:** set `CXX=clang++` before install if CMake does not find a compiler. Runtime DLLs (`libc++.dll`, `libunwind.dll`) are copied next to the `.pyd` when discoverable.

### 3. Run the API

```bash
# Optional data directory (WAL + checkpoints); default ./data
export HEALER_DATA_DIR=./data          # bash
# $env:HEALER_DATA_DIR="./data"       # PowerShell

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Load vectors (required before search/delete)

The HTTP layer serves whatever is loaded into the process-local native index (or recovered from `index.bin` on startup).

```python
import numpy as np
import hnsw_healer
from api.persistence import PersistenceEngine

n, d = 1000, 64
data = np.random.randn(n, d).astype(np.float32)

hnsw_healer.load_index(data, dimensions=d, num_elements=n)

# Minimal navigable layer-0 graph (replace with real HNSW adjacency in production)
idx = hnsw_healer.default_index()
for i in range(n):
    idx.set_neighbors(i, 0, [(i - 1) % n, (i + 1) % n, (i + 2) % n])

PersistenceEngine().save_initial_checkpoint()
print("checkpoint written; API can hard-delete and recover across restarts")
```

### 5. Hard-delete and search

```bash
# Raw label delete (durable WAL path)
curl -s -X POST http://127.0.0.1:8000/delete \
  -H "Content-Type: application/json" \
  -d '{"node_id": 42, "max_m": 16}'

# k-NN
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.0, 0.0 /* … d floats */], "k": 10}'
```

---

## Usage paths

Choose the path that matches how you deploy:

| Path | When to use | Entry point |
|------|-------------|-------------|
| **HTTP sidecar** | Separate unlearning service next to your app | `uvicorn api.main:app` |
| **Native library** | You own the float matrix / custom HNSW | `import hnsw_healer` |
| **ErasureService** | Business ids + audit receipts | `integrations.erase_service` |
| **hnswlib adapter** | Python HNSW stacks built on hnswlib | `integrations.hnswlib_adapter` |
| **Chroma hook** | Chroma collections must not soft-delete alone | `integrations.chroma_hook` |
| **Docker** | Production container, non-root | `Dockerfile` |

---

## HTTP API

Base URL: `http://127.0.0.1:8000` (default). Full schema: `/docs`, `/openapi.json`.

| Method | Path | Body (summary) | Description |
|--------|------|----------------|-------------|
| `GET` | `/health` | — | Liveness, lock config, bootstrap/recovery info |
| `GET` | `/metrics` | — | Prometheus text metrics |
| `GET` | `/v1/metrics` | — | JSON metrics snapshot |
| `POST` | `/delete` | `{ "node_id": int, "max_m"?: int }` | Durable hard-delete by **internal label** |
| `POST` | `/search` | `{ "query": float[], "k"?: int, "entry_node"?: int }` | Concurrent k-NN |
| `POST` | `/v1/vectors/ingest` | `{ "collection", "ids", "vectors", "replace_index"? }` | Register ids + load native index |
| `POST` | `/v1/ids/register` | `{ "collection", "ids", "labels"? }` | Map business ids → labels (legacy) |
| `POST` | `/v1/collections/{collection}/delete` | `{ "collection", "ids", … }` | Enterprise hard-delete by **external ids** |
| `POST` | `/v1/erasure-requests` | workflow create / advance | Durable ticket + optional hooks |
| `GET` | `/v1/erasure-requests/{id}/export` | — | Compliance export package + receipt log |
| `GET` | `/v1/receipts` | — | Append-only receipt log tail |
| `POST` | `/v1/outbox/dispatch` | — | Drain durable replica outbox |
| `POST` | `/v1/admin/compact` | — | Force coalesced compact flush |

### Example: enterprise delete (collection + user id)

```bash
# 1) Register ids when you ingest vectors (labels must match HNSW node ids)
curl -s -X POST http://127.0.0.1:8000/v1/ids/register \
  -H "Content-Type: application/json" \
  -d '{"collection":"users","ids":["user-42"],"labels":[42]}'

# 2) Hard-delete by business id (WAL + zero + heal + signed receipt)
curl -s -X POST http://127.0.0.1:8000/v1/collections/users/delete \
  -H "Content-Type: application/json" \
  -d '{
        "collection": "users",
        "ids": ["user-42"],
        "reason": "gdpr_art_17",
        "request_id": "ticket-1001",
        "max_m": 16
      }'
```

Example receipt shape (schema **v2**):

```json
{
  "receipt_version": 2,
  "success": true,
  "status": "complete",
  "request_id": "ticket-1001",
  "collection": "users",
  "external_ids": ["user-42"],
  "labels": [42],
  "bytes_wiped_total": 256,
  "compacted": true,
  "residual_proof": { "mode": "sample", "passed": true, "checked": 1 },
  "signature": "<hmac-sha256 hex>",
  "reason": "gdpr_art_17",
  "errors": [],
  "transaction_ids": [1]
}
```

**HTTP status notes**

| Code | Meaning |
|------|---------|
| `200` | Success (check `success` on enterprise delete for partial batch issues) |
| `400` | Validation / collection mismatch |
| `404` | Unknown external id (non-idempotent mapping errors) |
| `409` | No index loaded |
| `422` | Pydantic validation (e.g. negative `node_id`) |
| `503` | Lock contention after retries (`Retry-After: 1`) |

---

## Native Python API

```python
import numpy as np
import hnsw_healer

# Load N x D float32 (or flat length N*D)
hnsw_healer.load_index(data, dimensions=d, num_elements=n)

idx = hnsw_healer.default_index()
idx.set_neighbors(0, 0, [1, 2, 3])

# Physical wipe only
idx.overwrite_vector(7)

# MN-RU heal after isolation (or full pipeline)
metrics = idx.heal_graph_structure(7, max_m=16)
# metrics.success, .edges_removed, .edges_added, .repair_duration_ms

# Zero + heal
result = hnsw_healer.erase_node(7, max_m=16)

# Search (raises LockContentionError on stripe timeout — retry)
hits = hnsw_healer.search_knn(query_vec, k=10)

# Persistence helpers
hnsw_healer.save_index("data/index.bin.tmp")
hnsw_healer.load_index_file("data/index.bin")
```

| Symbol | Role |
|--------|------|
| `load_index` / `HNSWIndexProxy.load_index` | Copy vectors into owned RAM |
| `set_neighbors` / `load_adjacency` | Install layered neighbor lists |
| `overwrite_vector` | Zero one embedding |
| `heal_graph_structure` | Isolate + MN-RU rewire |
| `erase_node` | Full hard delete |
| `search_knn` | Lock-coupled k-NN |
| `save_index` / `load_index_file` | Binary snapshot |
| `LockContentionError` | Timed stripe lock failure (retryable) |
| `ValueError` | Out-of-range node/layer (from C++ `out_of_range`) |

### MN-RU healing (short)

For deleted node \(q\) on each layer \(L\):

1. Snapshot \(N_L(q)\) (orphans).
2. Sever edges into/out of \(q\) under an exclusive lock on the **2-hop** neighborhood.
3. Rank orphan pairs by Euclidean distance; insert edges under degree cap \(M\).
4. If at capacity, swap the furthest neighbor when the candidate is closer **and** the navigability score \(\sum 1/(1+d)\) improves.

---

## Enterprise integration

Soft-delete in **hnswlib** / **Chroma** is the common production footgun. This repo provides adapters so hard-erase can sit **on the write path** of those stacks.

| Component | Purpose |
|-----------|---------|
| [`integrations/erase_service.py`](integrations/erase_service.py) | `delete(collection, ids, reason=…)` → signed `ErasureReceipt` |
| [`integrations/id_registry.py`](integrations/id_registry.py) | Per-collection external id ↔ dense label |
| [`integrations/hnswlib_adapter.py`](integrations/hnswlib_adapter.py) | Zero matrix + `mark_deleted` + **`compact()`** rebuild |
| [`integrations/faiss_adapter.py`](integrations/faiss_adapter.py) | FAISS `IndexHNSW` + IDMap hard-delete backend |
| [`integrations/replica_fanout.py`](integrations/replica_fanout.py) | Multi-replica intent fan-out + quorum ACKs |
| [`integrations/vendor_attach.py`](integrations/vendor_attach.py) | Zero-copy `attach_index` / shared-memory wipe |
| [`integrations/chroma_hook.py`](integrations/chroma_hook.py) | Hard-erase **before** Chroma metadata delete (fail-closed) |
| [`compliance/residual.py`](compliance/residual.py) | Prove live zeros + absence of float32 pattern in checkpoint |
| [`compliance/crypto_shred.py`](compliance/crypto_shred.py) | In-process DEK vault |
| [`compliance/kms_backends.py`](compliance/kms_backends.py) | KMS envelope shred (Local / AWS / GCP / Vault) |
| [`compliance/recall_bounds.py`](compliance/recall_bounds.py) | Formal fragmentation & heal dominance claims |

### hnswlib (library)

```bash
pip install -e ".[hnswlib]"
```

```python
from integrations import CollectionIdRegistry, ErasureService
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

reg = CollectionIdRegistry()
store = HnswlibHardDeleteAdapter(dim=128, collection="users", registry=reg)
store.add(["user-1", "user-2"], embeddings_nxd)  # float32 ndarray

svc = ErasureService(reg, store)
receipt = svc.delete("users", ["user-1"], reason="gdpr_art_17")
assert receipt.success and receipt.status == "complete"
assert receipt.compacted  # one compact per batch (no manual compact needed)
assert receipt.residual_proof and receipt.residual_proof["passed"]
```

### Chroma (hook)

```bash
pip install -e ".[chroma]"   # or .[enterprise]
```

```python
import chromadb
from integrations import CollectionIdRegistry, ErasureService
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter
from integrations.chroma_hook import ChromaHardDeleteCollection

reg = CollectionIdRegistry()
backend = HnswlibHardDeleteAdapter(dim=384, collection="docs", registry=reg)
svc = ErasureService(reg, backend)

client = chromadb.Client()
raw = client.get_or_create_collection("docs")
col = ChromaHardDeleteCollection(raw, svc, collection_name="docs")

# register-on-add: ids land in the registry automatically
col.add(ids=["d1"], embeddings=[...], documents=["..."])
receipt = col.delete(ids=["d1"], reason="user_request")  # wipe+compact+proof+Chroma
assert receipt.success and receipt.compacted
```

**Golden path (recommended):** [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) · [`examples/chroma_forget/`](examples/chroma_forget/)

Deep wiring notes: [docs/INTEGRATION.md](docs/INTEGRATION.md).

### Optional extras

```bash
pip install -e ".[hnswlib]"      # hnswlib only
pip install -e ".[chroma]"       # chromadb + hnswlib
pip install -e ".[enterprise]"   # both
pip install -e ".[dev]"          # pytest, httpx, matplotlib
```

---

## Durability & crash recovery

Hard-delete pipeline:

```text
1. Append BEGIN_DELETE to index.wal  (fsync)     ← intent durable
2. C++ erase_node / heal                         ← RAM mutation
3. save → index.bin.tmp + fsync
4. os.replace(tmp, index.bin)                    ← atomic publish
5. Append COMMIT to index.wal  (fsync)           ← intent retired
```

On process start, `PersistenceEngine.bootstrap()`:

1. Loads `index.bin` if present.
2. Scans the WAL for `BEGIN` without matching `COMMIT`.
3. Replays hard-delete + flush for each orphaned transaction.

| Crash window | Outcome |
|--------------|---------|
| During `index.bin.tmp` write | Old `index.bin` intact; BEGIN open → replay on boot |
| After replace, before COMMIT | Index already correct; recovery re-applies (idempotent zeros) + COMMIT |
| After COMMIT | Clean |

**Data directory** (`HEALER_DATA_DIR`, default `./data`):

| File | Role |
|------|------|
| `index.bin` | Production atomic checkpoint |
| `index.bin.tmp` | In-progress snapshot (discarded if crash mid-write) |
| `index.wal` | Append-only intent log (`WDEL` / `WCMT` + SHA-256) |
| `id_registry.json` | Optional persisted collection id map |

WAL record formats are documented in [`api/wal.py`](api/wal.py).

---

## Concurrency model

| Operation | Locking |
|-----------|---------|
| `search_knn` | Shared stripe locks + **hand-over-hand** coupling along the path |
| `heal_graph_structure` / erase | Exclusive locks on \(\{q\} \cup N(q) \cup N(N(q))\) only |
| `load_index` / structure changes | Structure-level exclusive mutex |

- Stripe: `node_id % pool_size` → `std::shared_timed_mutex` (timed try-lock).
- Timeout → C++ `LockTimeoutError` → Python **`LockContentionError`**.
- FastAPI wraps native calls in **`with_lock_retry`** (exponential backoff); exhaustion → **HTTP 503**.

---

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `HEALER_DATA_DIR` | `./data` | WAL, checkpoints, id registry, workflows, receipts, outbox |
| `HEALER_SIGNING_KEY` | dev placeholder | HMAC for delete / receipt signatures — **set in production** |
| `HEALER_API_KEY` | unset | If set, API requires `X-API-Key` or Bearer |
| `HEALER_ENV` | `development` | `production` requires API key + non-default signing key |
| `HEALER_RESIDUAL_PROOF` | `sample` | `off` \| `sample` \| `full` on enterprise delete |
| `HEALER_COMPACT_POLICY` | `always` | `always` \| `never` \| `coalesce` (every-N / max-age) |
| `HEALER_COMPACT_EVERY_N` | `32` | Coalesce: compact after N pending deletes |
| `HEALER_COMPACT_MAX_AGE_S` | `60` | Coalesce: compact if pending older than T seconds |
| `HEALER_ADAPTIVE_COMPACT` | `1` | Map delete fraction → compact=always for large jobs |
| `HEALER_ALLOW_HEAL` | off | Opt-in experimental MN-RU heal suggestions |
| `HEALER_MULTI_TENANT` | off | Per-tenant data dirs under `tenants/{id}/` |
| `HEALER_WEBHOOK_DOCUMENT_STORE` | unset | Workflow document-store webhook URL |
| `HEALER_WEBHOOK_BACKUP_ACK` | unset | Workflow backup-ack webhook URL |
| `HEALER_CRYPTO_SHRED` | off | Enable in-process crypto-shred workflow hook |
| `HEALER_OUTBOX_REPLICA` | off | File outbox for replica fan-out intents |
| `HEALER_LOCK_MAX_ATTEMPTS` | `5` | Retries on lock contention |
| `HEALER_LOCK_BASE_DELAY_S` | `0.01` | Backoff base (seconds, exponential) |
| `HEALER_CRYPTO_MASTER_KEY` | empty | Optional marker for crypto-shred vault demos |
| `CXX` / `CC` | (system) | Compiler for native build (e.g. `clang++`) |

---

## Testing & benchmarks

Full guide: **[docs/TESTING.md](docs/TESTING.md)**.

```bash
# Core unit suite (requires compiled hnsw_healer)
pip install -r requirements.txt
pip install -e .
pytest tests/ -v --ignore=tests/benchmark.py

# Optional adapters
pip install -e ".[hnswlib]"
pytest tests/test_hnswlib_adapter.py -v

# Evaluation suite (see docs/BENCHMARKS.md) — residual x quality x cost
python tests/benchmark.py --profile quick --backend hnswlib
python tests/benchmark.py --profile gdpr_light --backend hnswlib
python tests/benchmark.py --profile gdpr_batch --backend hnswlib
python tests/benchmark.py --profile standard --backend native   # stress only
```

| Suite | Validates |
|-------|-----------|
| `test_api.py` | Health, delete, search, `/v1` register & enterprise delete, lock retry |
| `test_wal.py` | WAL checksums, durable delete, crash recovery replay |
| `test_id_registry.py` | Multi-collection id map + JSON persist |
| `test_erase_service.py` | External-id erase via native backend |
| `test_residual_proof.py` | Live zeros + post-delete checkpoint pattern absence |
| `test_crypto_shred.py` | Encrypt / shred / fail-closed decrypt |
| `test_hnswlib_adapter.py` | Zero + mark_deleted + compact *(skip if no hnswlib)* |
| `test_chroma_hook.py` | Chroma hook *(skip if no chromadb)* |
| `benchmark.py` | Soft vs unhealed vs healed: recall@k, latency, unreachable % |

**Benchmark scenarios**

- **A — Soft-delete:** tombstones only (topology control).
- **B — Unhealed hard-delete:** zero + sever, no MN-RU.
- **C — Healed hard-delete:** full `erase_node` pipeline.

Artifacts: `benchmark_results/benchmark_report.json`, optional `pareto_frontier.png`.

Typical green core run: **`25 passed`**, optional adapter files **skipped** when deps missing.

---

## Docker

Multi-stage image: build wheel with full toolchain → run as unprivileged user on `python:3.11-slim`.

```bash
docker build -t hnsw-healer:latest .

docker run --rm -p 8000:8000 \
  -e HEALER_SIGNING_KEY='replace-with-long-random-secret' \
  -e HEALER_DATA_DIR=/app/data \
  -v healer-data:/app/data \
  hnsw-healer:latest
```

- Process user: UID **10001** (`healer`)
- Volume: `/app/data` for WAL + `index.bin`
- Entrypoint: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

---

## CI & packaging

[`.github/workflows/build.yml`](.github/workflows/build.yml) on PRs/pushes to `main`:

| Job | Purpose |
|-----|---------|
| **cibuildwheel** | Linux x86_64, Linux aarch64, macOS arm64; CPython 3.10–3.12 |
| **pytest** | Unit suite on each wheel (`benchmark.py` ignored) |
| **Docker Buildx** | Validates production `Dockerfile` (no push) |
| **ci-success** | Gate requiring all of the above |

Local wheel:

```bash
pip install build
python -m build
# → dist/*.whl
```

---

## Security & residual data

### Mitigations implemented

- Physical zero of live embeddings in the controlled index.
- Atomic durable rewrite of this project’s `index.bin`.
- hnswlib **`compact()`** to rebuild without deleted float rows.
- Signed **erasure receipts** for audit logs.
- Optional **crypto-shred** so cold ciphertext is useless without DEKs.
- Automated **residual proofs** for live zeros and checkpoint pattern scans.

### Residual matrix (honest)

| Surface | Cleared by hard delete? |
|---------|-------------------------|
| Native `HNSWIndexProxy` RAM | Yes (if erase succeeded) |
| Project `index.bin` after COMMIT | Yes |
| hnswlib after `mark_deleted` only | **No** — call `compact()` |
| hnswlib after `compact()` | Yes (rebuilt without row) |
| Chroma metadata delete alone | **No** — use `ChromaHardDeleteCollection` |
| Volume snapshots / object backups | **No** — retention + crypto-shred |
| Replica that never received delete | **No** — fan-out required |
| OS swap / core dumps | **No** |

Production checklist: wipe + heal → compact/rewrite → crypto-shred if used → fan-out to replicas → backup policy → sample residual proofs → store receipts.

Details: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md), [docs/BACKUPS_AND_REPLICAS.md](docs/BACKUPS_AND_REPLICAS.md).

### Operational hygiene

- Set a strong `HEALER_SIGNING_KEY` (never ship the default).
- Run the container as non-root (default).
- Prefer encrypted volumes for `HEALER_DATA_DIR`.
- Treat WAL and receipts as potentially sensitive (who was deleted, when).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError: hnsw_healer native module is not installed` | Extension not built | `pip install -e .` with CMake + C++17 toolchain |
| Windows: `DLL load failed` | Missing MinGW C++ runtime | Ensure `libc++.dll` / `libunwind.dll` beside the `.pyd`, or add LLVM-MinGW `bin` to `PATH` |
| CMake: generator does not support `-A x64` | NMake/Ninja + platform flag | Use current `setup.py` (Ninja preferred); avoid forcing `-A` with Ninja |
| HTTP `409 no index loaded` | No vectors / no `index.bin` | Load via Python or restore checkpoint |
| HTTP `503 lock contention` | Heal vs search hotspot | Retry; raise `HEALER_LOCK_MAX_ATTEMPTS` / timeout; reduce concurrent heal fan-out |
| Enterprise delete errors for unknown ids | Id not registered | `POST /v1/ids/register` at ingest time |
| Soft-delete still leaves floats in hnswlib file | Bypassed `ErasureService` or `compact=never` | Use `ErasureService.delete` (auto-compact) or call `store.compact()` |
| HTTP `401` on delete | `HEALER_API_KEY` set | Send `X-API-Key` or `Authorization: Bearer` |
| Startup fails in production | Default signing key / missing API key | Set `HEALER_SIGNING_KEY` + `HEALER_API_KEY` |
| `hnswlib` pip build fails on Python 3.14 | No wheel / needs MSVC | Use a CPython version with wheels, or install MSVC Build Tools |

Rebuild after C++ changes:

```bash
pip install -e . --force-reinstall --no-deps
```

---

## Roadmap & status

**Current: v0.3.2 (Alpha)** — residual-first eval + wipe/rebuild product path; stay on 0.x until a production pilot + published hnswlib GDPR packs.

**Implemented**

- [x] Physical wipe + MN-RU heal in C++
- [x] Neighborhood locks + Python retry
- [x] WAL + atomic checkpoint + recovery
- [x] FastAPI + enterprise id delete + **vector ingest** (`POST /v1/vectors/ingest`)
- [x] hnswlib adapter + Chroma hook
- [x] Residual proofs + crypto-shred helpers
- [x] Docker + cibuildwheel CI + evaluation harness
- [x] First-class FAISS `IndexHNSW` backend (`FaissHNSWHardDeleteAdapter`)
- [x] Multi-replica delete fan-out + **durable file outbox**
- [x] KMS-backed crypto-shred (LocalFile / AWS / GCP / Vault interfaces)
- [x] Deeper in-process vendor attach (`attach_index`, shared memory, `InPlaceVendorSession`)
- [x] Formal recall bounds under adversarial deletes
- [x] Receipt schema v2 + **append-only receipt log** (`receipts.jsonl`)
- [x] Auto / **coalesced compact** policy + residual proof fail-closed
- [x] Golden path, attack demo, pilot checklist, hooks docs
- [x] `ErasureWorkflow` + HTTP webhooks / crypto-shred / backup hooks
- [x] API key middleware + production signing-key guard
- [x] **Metrics** (`GET /metrics`, `GET /v1/metrics`)
- [x] Multi-tenant data-dir + key derivation helpers
- [x] Benchmark residual × quality × cost; scenarios A–E; hnswlib backend; `gdpr_*` profiles
- [x] **Qdrant** + **Weaviate** rebuild-based adapters (in-memory clients for tests)
- [x] Delete strategy: wipe+compact default; heal opt-in (`HEALER_ALLOW_HEAL`); adaptive compact
- [x] Queue transports: file / Redis / SQS for replica intents

**Planned / welcome contributions**

- [ ] Milvus first-class adapter
- [ ] Production KMS grant automation (per-tenant CMK lifecycle)
- [ ] Distributed consensus for delete quorum (Raft) beyond outbox/HTTP
- [x] Published **standard/hnswlib** number pack ([docs/benchmarks/standard_hnswlib.md](docs/benchmarks/standard_hnswlib.md))
- [ ] More `gdpr_*` / `publish` packs on dedicated release hardware
- [ ] PyPI wheel release automation for all platforms

Production deployments typically **ingest** vectors via adapters (or load adjacency from an upstream HNSW builder) and use this stack for **erasure, heal, concurrent search, and durable commit**—not as a full greenfield vector database.

---

## Further documentation

| Document | Contents |
|----------|----------|
| [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) | Recommended Chroma/hnswlib forget path |
| [docs/INSTALL.md](docs/INSTALL.md) | 30-minute install + production secrets |
| [docs/PILOT.md](docs/PILOT.md) | Design-partner pilot checklist |
| [docs/HOOKS.md](docs/HOOKS.md) | Workflow webhooks, crypto-shred, outbox |
| [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) | Adversaries, residual matrix, non-claims |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | FAISS / Chroma / replica / KMS / attach wiring |
| [docs/BACKUPS_AND_REPLICAS.md](docs/BACKUPS_AND_REPLICAS.md) | Snapshots, fan-out, crypto-shred runbook |
| [docs/RECALL_BOUNDS.md](docs/RECALL_BOUNDS.md) | Theorems on recall under deletes |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Benchmark profiles and how to read residual × quality |
| [docs/benchmarks/standard_hnswlib.md](docs/benchmarks/standard_hnswlib.md) | **Published** N=50k hnswlib pack |
| [docs/ENGINES.md](docs/ENGINES.md) | Qdrant / Weaviate / strategy / queues |
| [docs/TESTING.md](docs/TESTING.md) | Full test matrix |
| [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) | First push / org settings checklist |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Community

| Resource | Link |
|----------|------|
| Contributing guide | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Code of conduct | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Security reports | [SECURITY.md](SECURITY.md) |
| Support expectations | [SUPPORT.md](SUPPORT.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

**PR basics:** fork → branch → `pytest tests/ -v --ignore=tests/benchmark.py` → open PR with residual note if you touch the delete path. Details in CONTRIBUTING.

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).

---

<p align="center">
  <sub>
    Hard delete means zeros in the index—not a tombstone that still holds the latent vector.
  </sub>
</p>
