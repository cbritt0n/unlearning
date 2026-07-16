# Pilot checklist

Use this with one design partner on **self-hosted** Chroma / hnswlib / FAISS RAG.

## Goals

1. Close **one** real erasure ticket with a signed export package.  
2. Prove residual absence on live storage + checkpoint.  
3. Measure delete cost vs soft-delete on their data scale.  
4. Confirm ops can run the stack without a C++ expert (wheels).

## Pre-flight

- [ ] Python 3.10–3.12 with wheel install **or** working CMake toolchain  
- [ ] `pip install hnsw-healer[chroma]` (or `pip install -e ".[chroma,dev]"` from source)  
- [ ] `HEALER_SIGNING_KEY` set (non-default)  
- [ ] `HEALER_API_KEY` set if exposing the API  
- [ ] Data dir on encrypted volume  

## Integration day (≤ 1 day)

1. Follow [GOLDEN_PATH.md](GOLDEN_PATH.md) on a **staging** collection.  
2. Wire document-store delete webhook ([HOOKS.md](HOOKS.md)).  
3. Enable `HEALER_CRYPTO_SHRED=1` if cold embeddings are envelope-encrypted.  
4. Run `python examples/attack_demo/run.py` for stakeholder education.  
5. Create workflow: `POST /v1/erasure-requests` with real ticket id.  
6. Export: `GET /v1/erasure-requests/{id}/export` → attach to ticket.  
7. Confirm `GET /v1/receipts/{id}` shows append-only log rows.  

## Success criteria

| Criterion | Pass |
|-----------|------|
| Golden path &lt; 30 min for a new engineer | |
| `receipt.status == complete` + residual proof passed | |
| Export JSON filed on a real ticket | |
| Soft-delete contrast demo understood by security | |
| No default signing key in staging/prod | |
| Metrics scraped (`/metrics` or `/v1/metrics`) | |

## Load / quality

```bash
python tests/benchmark.py --profile quick   # smoke on their box
python tests/benchmark.py --profile standard  # if time allows
```

Record numbers in [BENCHMARKS.md](BENCHMARKS.md).

## Explicit non-goals for pilot

- Full multi-region Raft  
- Hosted Pinecone/Weaviate without adapter  
- Legal determination of GDPR completeness  

## After pilot

File issues for only what the partner blocked on. Prefer second engine only if they need it.
