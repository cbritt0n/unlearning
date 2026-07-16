# Vector engine adapters

HNSW Healer is **not** a drop-in replacement for Qdrant/Weaviate/Milvus segment
codecs. Adapters dual-write an authoritative float matrix and hard-delete via:

1. **Physical zero** of the matrix row (residual mitigation)  
2. Vendor delete / mark-deleted  
3. **`compact()`** rebuild so the serving structure drops residual floats  

| Engine | Module | Strategy |
|--------|--------|----------|
| hnswlib | `integrations/hnswlib_adapter.py` | mark_deleted + compact rebuild |
| FAISS HNSW | `integrations/faiss_adapter.py` | remove_ids / rebuild |
| Chroma | `integrations/chroma_hook.py` | hard-erase then metadata delete |
| **Qdrant** | `integrations/qdrant_adapter.py` | point delete + collection recreate on compact |
| **Weaviate** | `integrations/weaviate_adapter.py` | object delete + clear/reinsert on compact |
| Native proxy | `integrations/native_backend.py` | MN-RU erase + WAL checkpoint |

## Delete strategy (quality)

See `integrations/delete_strategy.py`. Rule of thumb:

| Delete fraction | Recommendation |
|-----------------|----------------|
| &lt; 1% | wipe + auto compact (coalesce OK) |
| 1–10% | wipe + compact; heal optional |
| 10–25% | wipe + **always compact** |
| ≥ 25% | **full rebuild** (compact); do not rely on MN-RU alone |

Benchmarks on synthetic ring graphs showed healed recall can collapse; treat
compact/rebuild as the production quality path.

## Qdrant example

```python
from integrations import CollectionIdRegistry, ErasureService
from integrations.qdrant_adapter import QdrantHardDeleteAdapter, InMemoryQdrantClient

reg = CollectionIdRegistry()
# Tests / offline:
store = QdrantHardDeleteAdapter(dim=128, collection="docs", registry=reg,
                                client=InMemoryQdrantClient())
# Production:
# store = QdrantHardDeleteAdapter(dim=128, collection="docs", registry=reg,
#                                 url="http://localhost:6333")

store.add(["d1", "d2"], embeddings)
svc = ErasureService(reg, store)
receipt = svc.delete("docs", ["d1"], reason="gdpr_art_17")
assert receipt.success and receipt.compacted
```

```bash
pip install -e ".[qdrant]"
```

## Weaviate example

```python
from integrations.weaviate_adapter import WeaviateHardDeleteAdapter, InMemoryWeaviateClient

store = WeaviateHardDeleteAdapter(
    dim=384, collection="Document", registry=reg, client=InMemoryWeaviateClient()
)
```

```bash
pip install -e ".[weaviate]"
```

## Milvus

Not yet first-class. Pattern to copy: authoritative matrix +
`collection.delete(expr=…)` + flush/compact via pymilvus, then residual proof
on matrix zeros. Contributions welcome under `integrations/milvus_adapter.py`.

## Queue transports (replicas)

```python
from integrations.queue_transports import FileQueueTransport, enqueue_delete_intent

q = FileQueueTransport("./data")
enqueue_delete_intent(q, collection="docs", external_ids=["u1"], request_id="t1")
```

Redis / SQS: `RedisListTransport`, `SqsTransport` in `integrations/queue_transports.py`.
