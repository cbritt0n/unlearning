# Standard profile — hnswlib backend (published pack)

| Field | Value |
|-------|--------|
| Profile | `standard` |
| Backend | **hnswlib** |
| N / d | 50 000 / 128 |
| Queries / k | 1 000 / 10 |
| Waves | 5 × 10% (up to **50%** deleted) |
| M / ef | 16 / construction defaults in harness |
| Date | 2026-07-16 |
| Machine | Windows (local), Python 3.12 venv |

Reproduce:

```bash
python tests/benchmark.py --profile standard --backend hnswlib \
  --out-dir benchmark_results/standard_hnswlib
```

Raw JSON is gitignored under `benchmark_results/`; this summary is the
**committed** community-facing record.

## Final wave (50% deleted)

| Scenario | Recall@10 | Retention (÷ baseline) | Residual | Usable | Total delete wall (s) |
|----------|-----------|------------------------|----------|--------|------------------------|
| baseline (0%) | **0.331** | 1.00 | — | Y | — |
| A soft | 0.470 | 1.42 | **YES** | **N** | ~0.02 |
| B unhealed | 0.475 | 1.43 | no | Y | ~0.02 |
| C healed* | 0.473 | 1.43 | no | Y | ~0.03 |
| **D rebuild** | **0.447** | **1.35** | **no** | **Y** | ~2.4 |
| **E adaptive** | **0.456** | **1.38** | **no** | **Y** | ~2.4 |

\* On hnswlib, “heal” ≈ zero + `mark_deleted` (no native MN-RU).

## Product reading

1. **Privacy:** A always **residual=YES**; D/E **residual=no** at every wave.  
2. **Quality:** D/E stay within ~0.02–0.03 absolute recall of soft at 50% delete; retention &gt; 1.0 vs baseline.  
3. **Cost:** Rebuild is slower to apply (~0.3–0.7 s per wave) but search p95 **improves** (smaller live index).  
4. Soft is **not** privacy-usable (`usable=False`) despite high recall.

**Headline:** wipe + rebuild matches soft-class quality closely while eliminating residual floats.

## Wave-1 (10% delete) — GDPR-like stress step

| Scenario | Recall@10 | Residual | Usable | delete_s |
|----------|-----------|----------|--------|----------|
| A soft | 0.349 | YES | N | 0.003 |
| D rebuild | 0.351 | no | Y | 0.675 |
| E adaptive | 0.346 | no | Y | 0.665 |

At 10% deleted, **D ≈ A on recall** with residual cleared.

## Caveats

* Baseline recall@10 ≈ 0.33 depends on harness ef/M; raise ef for higher absolute recall in your own packs.  
* This is a **synthetic Gaussian** corpus, not production embeddings.  
* Multi-wave 50% is a **torture** schedule; also publish `gdpr_light` / `gdpr_batch`.  
