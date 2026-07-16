# Integrating HNSW Healer into business vector databases

## Recommended paths

### 1. Library mode (best for Python apps)

```python
from integrations import CollectionIdRegistry, ErasureService
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

registry = CollectionIdRegistry()
backend = HnswlibHardDeleteAdapter(dim=128, collection="users", registry=registry)
backend.add(["u1", "u2"], embeddings_nxd)

svc = ErasureService(registry, backend)
receipt = svc.delete("users", ["u1"], reason="gdpr_art_17")
assert receipt.success and receipt.compacted  # auto-compact once per batch
assert receipt.residual_proof["passed"]
```

### 2. Chroma hook

```python
import chromadb
from integrations import ErasureService, CollectionIdRegistry
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter
from integrations.chroma_hook import ChromaHardDeleteCollection

reg = CollectionIdRegistry()
backend = HnswlibHardDeleteAdapter(dim=384, collection="docs", registry=reg)
svc = ErasureService(reg, backend)

client = chromadb.Client()
raw = client.get_or_create_collection("docs")
col = ChromaHardDeleteCollection(raw, svc, collection_name="docs")

col.add(ids=["d1"], embeddings=[[...]], documents=["..."])  # register-on-add
receipt = col.delete(ids=["d1"], reason="user_request")
assert receipt.success and receipt.compacted
```

See also [GOLDEN_PATH.md](GOLDEN_PATH.md).

### 3. Native proxy + WAL API

Load vectors into `hnsw_healer`, checkpoint with `PersistenceEngine`, call
`POST /delete` or `ErasureService` with `NativeHealerBackend` + persistence.

### 4. Sidecar

Run the FastAPI container next to your app; dual-write embeddings into the
healer-backed store; on delete, call the sidecar **and** your primary DB.

## ID model

Always use **collection + external_id** at the business edge:

| Layer | Identifier |
|-------|------------|
| App / GDPR request | `user_id` string |
| `CollectionIdRegistry` | maps to dense `label: int` |
| HNSW / hnswlib | integer label |
| WAL | integer `node_id` (= label) |

## FAISS `IndexHNSW`

```bash
pip install -e ".[faiss]"
```

```python
from integrations import CollectionIdRegistry, ErasureService
from integrations.faiss_adapter import FaissHNSWHardDeleteAdapter

reg = CollectionIdRegistry()
store = FaissHNSWHardDeleteAdapter(dim=128, collection="users", registry=reg)
store.add(["u1", "u2"], embeddings)
svc = ErasureService(reg, store)
receipt = svc.delete("users", ["u1"], reason="gdpr_art_17")
assert receipt.compacted  # rebuild FAISS HNSW without residual rows
```

## Multi-replica fan-out

```python
from integrations.replica_fanout import (
    InProcessTransport, ReplicaFanoutCoordinator, ReplicaWorker,
)

# Each worker wraps local ErasureService.delete
workers = {rid: ReplicaWorker(rid, apply_fn) for rid in ("a", "b", "c")}
coord = ReplicaFanoutCoordinator(
    ["a", "b", "c"], InProcessTransport(workers), quorum=2
)
# After primary local success:
result = coord.publish("users", ["u1"], reason="gdpr", request_id="req-1")
assert result.quorum_met
```

HTTP peers: `HttpReplicaTransport` → `POST /v1/internal/replica/delete`.

## KMS crypto-shred

```python
from compliance.kms_backends import LocalFileKMS, KmsCryptoShredVault
# AWS: AwsKmsBackend(key_id=...)   requires boto3
# GCP: GcpKmsBackend(key_name=...) requires google-cloud-kms
# Vault: VaultTransitBackend(key_name=...) requires hvac

vault = KmsCryptoShredVault(LocalFileKMS("data/kms.json"))
ct = vault.encrypt_vector("user-1", embedding)
vault.shred("user-1")  # DEK unwrap fails forever
```

## Zero-copy vendor attach

```python
from integrations.vendor_attach import InPlaceVendorSession
session = InPlaceVendorSession.from_numpy(app_owned_float32_matrix, adjacency=adj)
session.hard_delete_row(label)  # zeros app buffer in place + MN-RU heal
```

## Checklist for production cutover

- [ ] All writes register ids in `CollectionIdRegistry`
- [ ] All deletes go through `ErasureService` (not raw soft-delete alone)
- [x] hnswlib/Chroma: `ErasureService` auto-compacts once per batch (override with `compact=never`)
- [ ] Residual proof job on a sample of deletes
- [ ] Crypto-shred keys for backup-sensitive tenants
- [ ] Delete fan-out to every replica / region
- [ ] Audit store for `ErasureReceipt` JSON

## Automated tests covering integration

See [TESTING.md](TESTING.md). Core commands:

```bash
pip install -e .
pytest tests/ -v --ignore=tests/benchmark.py
pip install -e ".[hnswlib]" && pytest tests/test_hnswlib_adapter.py -v
```

HTTP enterprise path under test: `POST /v1/ids/register` then
`POST /v1/collections/{collection}/delete` (see `tests/test_api.py`).
