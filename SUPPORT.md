# Support

HNSW Healer is **alpha** open-source middleware. Support is **best effort** from
maintainers and the community — there is no commercial SLA unless you arrange
one separately.

## Where to get help

| Need | Where |
|------|--------|
| Install / how-to | [docs/INSTALL.md](docs/INSTALL.md), [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) |
| Benchmarks | [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/HNSWLIB_AND_BENCHMARKS.md](docs/HNSWLIB_AND_BENCHMARKS.md) |
| Bug | [GitHub Issues](../../issues) using the bug template |
| Feature idea | Feature-request issue template |
| Security residual leak | [SECURITY.md](SECURITY.md) — **private** report |
| Design discussion | GitHub Discussions (if enabled) |

## Before opening an issue

1. Search existing issues.  
2. Confirm you are on a supported Python (3.10–3.12 recommended).  
3. Include OS, Python version, and whether `hnsw_healer` / `hnswlib` import.  
4. Prefer a **minimal reproduction** (local fixtures only).

## What we do not support here

* Legal advice on GDPR / CCPA completeness  
* Hardening of third-party SaaS vector DBs without an adapter  
* Guaranteed wipe of OS swap, core dumps, or cloud snapshots  
  (see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md))

## Response times

No guaranteed response time. Security reports are prioritized per SECURITY.md.
