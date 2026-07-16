# Benchmark pack: standard + hnswlib

This is the run we treat as the main public number set (not the synthetic
native-graph stress tests).

| | |
|-|-|
| Profile | `standard` |
| Backend | hnswlib |
| Size | 50k vectors × 128 dims |
| Queries | 1k, recall@10 |
| Deletes | 5 waves of 10% each (up to half the index) |
| When / where | 2026-07-16, Windows, Python 3.12 venv |

```bash
python tests/benchmark.py --profile standard --backend hnswlib \
  --out-dir benchmark_results/standard_hnswlib
```

Raw JSON lives under `benchmark_results/` (gitignored). This markdown file is
what we keep in git for the community.

## After 50% of the index is gone

| Scenario | Recall@10 | vs baseline | Residual floats? | Still usable? | Time spent deleting |
|----------|-----------|-------------|------------------|---------------|---------------------|
| baseline (nothing deleted) | 0.331 | 1.00× | — | yes | — |
| A soft | 0.470 | 1.42× | **still there** | no (privacy) | ~0.02s |
| B unhealed | 0.475 | 1.43× | gone | yes | ~0.02s |
| C “healed”* | 0.473 | 1.43× | gone | yes | ~0.03s |
| D rebuild | 0.447 | 1.35× | gone | yes | ~2.4s |
| E adaptive | 0.456 | 1.38× | gone | yes | ~2.4s |

\* On hnswlib this isn’t real MN-RU — it’s basically zero + `mark_deleted`.

### How to read it

Soft-delete wins or ties on pure recall, but the vectors are still on disk.
Rebuild is a few hundred ms to ~0.7s per wave, yet search quality stays close
to soft and residual checks pass. That’s the trade we care about.

## After the first 10% (more like a real batch job)

| Scenario | Recall@10 | Residual | Usable | delete_s |
|----------|-----------|----------|--------|----------|
| A soft | 0.349 | yes | no | 0.003 |
| D rebuild | 0.351 | no | yes | 0.675 |
| E adaptive | 0.346 | no | yes | 0.665 |

Here soft and rebuild are basically tied on quality; only rebuild clears residual.

## Caveats

- Absolute recall depends on ef/M in the harness; bump ef if you want flashier numbers.
- Data is random unit Gaussians, not your production embeddings.
- Deleting half the index in waves is a stress test. Also look at `gdpr_light` / `gdpr_batch`.
