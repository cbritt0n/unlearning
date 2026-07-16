# Benchmarks

We care about three things at once: **are residual floats gone?**, **does
search still work?**, and **how long did the delete take?** Soft-delete will
often win pure recall — that’s fine. Soft fails the residual check, which is
why this project exists.

MN-RU heal (scenario C on the native backend) is experimental. On weak
synthetic graphs it can trash recall. Don’t lead demos with it.

## Scenarios

| ID | Behavior | Residual | Role |
|----|----------|----------|------|
| **A soft** | Tombstones only | **YES** | Quality upper bound (unsafe) |
| **B unhealed** | Zero + sever | no | Naive hard delete |
| **C healed** | MN-RU `erase_node` | no | Experimental |
| **D rebuild** | Wipe + rebuild live graph | no | **Recommended hard path** |
| **E adaptive** | Heal only below frac threshold, else rebuild | no | **Product default behavior** |

## Backends

| Backend | Flag | Notes |
|---------|------|--------|
| **hnswlib** | `--backend hnswlib` | Real HNSW; preferred for absolute recall |
| **native** | `--backend native` | C++ proxy + synthetic adjacency (research stress) |

```bash
# Prefer hnswlib when installed (Windows: see docs/HNSWLIB_AND_BENCHMARKS.md)
.\.venv\Scripts\Activate.ps1

python tests/benchmark.py --profile quick --backend hnswlib
python tests/benchmark.py --profile gdpr_light --backend hnswlib
python tests/benchmark.py --profile gdpr_batch --backend hnswlib
python tests/benchmark.py --profile standard --backend native   # stress / torture
```

## Profiles

| Profile | Intent | Default backend |
|---------|--------|-----------------|
| `quick` | Smoke | auto |
| `standard` | Multi-wave stress (up to 50%) | auto |
| `publish` | Larger N/d | **hnswlib** |
| `gdpr_light` | 3×1% waves | **hnswlib** |
| `gdpr_batch` | Single 5% job | **hnswlib** |

## Metrics in JSON

Each wave includes:

- `recall_at_k`, `recall_retention` (÷ baseline)
- `residual_present`, `residual_checked`, `residual_hits`
- `usable` (residual-safe + retention floor)
- `delete_wall_s` (cost)
- latency p50/p95/p99

Report `summary` block has headline strings and final-wave comparisons.

## How to read a “good” result for this project

1. **A residual = true**, **D/E residual = false** → privacy win.  
2. **D recall retention** competitive with baseline (hnswlib runs).  
3. **D − C recall > 0** on stress runs → rebuild beats heal.  
4. Soft may win pure recall — that is expected; soft fails residual.

## Recorded runs

### quick / native (historical)

See prior notes: C heal → 0 recall; D rebuild recovers. Used to demote heal.

### standard / native (N=50k, user run)

| Scenario (50% del) | Recall@10 | Residual (expected) |
|--------------------|-----------|---------------------|
| A soft | ~0.135 | YES |
| B unhealed | ~0.096 | no |
| C healed | **0.000** | no |
| D rebuild | ~0.054 | no |

Baseline ANN quality was weak (~0.08) on synthetic graphs → prefer **hnswlib** profiles for external numbers.

### quick / hnswlib (recorded, N=2k, 20% deleted)

| Scenario | Recall@10 | Retention | Residual | Usable |
|----------|-----------|-----------|----------|--------|
| baseline | **0.916** | 1.00 | no | Y |
| A soft | 0.953 | 1.04 | **YES** | **N** |
| B unhealed | 0.949 | 1.04 | no | Y |
| C healed* | 0.953 | 1.04 | no | Y |
| **D rebuild** | **0.950** | **1.04** | **no** | **Y** |
| **E adaptive** | **0.955** | **1.04** | **no** | **Y** |

\* On hnswlib, “heal” is zero+mark_deleted (no MN-RU); native C is where heal collapse was observed.

**Privacy win:** A residual=YES vs D residual=no at nearly equal recall (~0.95).

### standard / hnswlib (recorded, N=50k, up to 50% deleted) — **headline pack**

Full write-up: **[docs/benchmarks/standard_hnswlib.md](benchmarks/standard_hnswlib.md)**

| Scenario (50% del) | Recall@10 | Retention | Residual | Usable | Σ delete_s |
|--------------------|-----------|-----------|----------|--------|------------|
| baseline (0%) | 0.331 | 1.00 | — | Y | — |
| A soft | 0.470 | 1.42 | **YES** | **N** | ~0.02 |
| B unhealed | 0.475 | 1.43 | no | Y | ~0.02 |
| C healed* | 0.473 | 1.43 | no | Y | ~0.03 |
| **D rebuild** | **0.447** | **1.35** | **no** | **Y** | ~2.4 |
| **E adaptive** | **0.456** | **1.38** | **no** | **Y** | ~2.4 |

**At 10% delete:** D recall **0.351** vs A **0.349** — match quality, clear residual.  
**Torture (50%):** D/E stay residual-safe and within ~0.02–0.03 recall of soft.

### Re-run pack for release notes

```bash
python tests/benchmark.py --profile gdpr_light --backend hnswlib --out-dir benchmark_results/gdpr_light
python tests/benchmark.py --profile gdpr_batch --backend hnswlib --out-dir benchmark_results/gdpr_batch
python tests/benchmark.py --profile quick --backend hnswlib --out-dir benchmark_results/quick_hnswlib
```

## Related

- [HNSWLIB_AND_BENCHMARKS.md](HNSWLIB_AND_BENCHMARKS.md) — Windows setup  
- [ENGINES.md](ENGINES.md) — adapters + strategy  
- `integrations/delete_strategy.py` — adaptive compact defaults  
