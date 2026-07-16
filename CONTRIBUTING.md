# Contributing to HNSW Healer

Thanks for helping improve residual-safe hard delete for HNSW / ANN stacks.

## Code of conduct

Be respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Ways to contribute

| Area | Examples |
|------|----------|
| **Bugs** | Failing residual proof, compact footguns, Windows build issues |
| **Adapters** | Milvus, deeper Qdrant/Weaviate, pgvector |
| **Eval** | hnswlib benchmark packs, residual metrics, cost tables |
| **Docs** | Golden path, threat model honesty, install for new platforms |
| **CI** | Wheels, optional-dep matrices |

## Development setup

### Recommended: Python 3.11 or 3.12

```bash
git clone https://github.com/<your-org>/unlearning.git
cd unlearning
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

**Windows + hnswlib:** see [docs/HNSWLIB_AND_BENCHMARKS.md](docs/HNSWLIB_AND_BENCHMARKS.md)  
(or `scripts/setup_hnswlib_env.ps1` + `scripts/install_hnswlib_msvc.bat`).

Verify:

```bash
python -c "import hnsw_healer; print('ok')"
pytest tests/ -v --ignore=tests/benchmark.py
```

Optional extras:

```bash
pip install -e ".[hnswlib]"     # adapter tests
pip install -e ".[chroma]"      # golden path example
pip install -e ".[faiss]"       # FAISS adapter
pip install -e ".[qdrant]"      # Qdrant client
pip install -e ".[enterprise]"  # common production extras
```

## Project layout (where to put changes)

| Path | Role |
|------|------|
| `src/` | C++ hot path (wipe, heal, locks, serialize) |
| `api/` | FastAPI control plane, auth, metrics, WAL glue |
| `integrations/` | ErasureService, adapters, workflow, strategy |
| `compliance/` | Residual proofs, crypto-shred, bounds |
| `tests/` | Unit tests; `tests/benchmark.py` is separate / heavy |
| `docs/` | Threat model, golden path, benchmarks, engines |
| `examples/` | Runnable demos (chroma forget, attack contrast) |

## Coding guidelines

1. **Security honesty** — do not claim full GDPR or wipe of swap/backups. Update [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) residual matrix when adding surfaces.
2. **Fail-closed deletes** — hard-erase failure must not silently fall back to soft-only metadata delete.
3. **Product default is wipe + compact/rebuild** — MN-RU heal is experimental (`HEALER_ALLOW_HEAL`). Prefer residual-safe usable search over heal-only.
4. **Tests** — add/adjust tests under `tests/`; skip optional deps with `pytest.importorskip`.
5. **Small PRs** — focused changes with a short residual/threat note if security-sensitive.

## Running tests

```bash
# Core unit suite (CI-like)
pytest tests/ -v --ignore=tests/benchmark.py

# Adapter (if installed)
pytest tests/test_hnswlib_adapter.py -v
```

## Benchmarks (optional; long)

```bash
# Credible absolute numbers
python tests/benchmark.py --profile quick --backend hnswlib
python tests/benchmark.py --profile gdpr_light --backend hnswlib

# Stress (can take a long time at N=50k)
python tests/benchmark.py --profile standard --backend hnswlib
```

Do **not** commit raw `benchmark_results/` trees (gitignored).  
Summaries belong in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) or `docs/benchmarks/*.md`.

## Pull requests

1. Fork and branch from `main` (`feature/…` or `fix/…`).
2. Ensure unit tests pass locally.
3. PR description: **what / why**, residual impact if any, how to test.
4. Link related issues.
5. Maintainers may ask for a short threat-model note on delete-path changes.

## Reporting security issues

Do **not** open a public issue for exploitable residual leaks in production deployments.  
See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
