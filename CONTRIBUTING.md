# Contributing

Thanks for taking a look. This project is small and alpha; thoughtful PRs and
honest bug reports help a lot.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## What kinds of help we need

- Bugs in wipe / compact / residual proofs / Windows builds
- Adapters (Milvus is an obvious gap; deeper Qdrant/Weaviate is welcome)
- Benchmark packs on real machines
- Docs that save the next person an afternoon of cmake pain
- CI / packaging improvements

## Setup

Use **Python 3.11 or 3.12** if you can. 3.14 often has no hnswlib wheels.

```bash
git clone https://github.com/YOUR_ORG/unlearning.git   # after you fork
cd unlearning
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
python -c "import hnsw_healer; print('ok')"
pytest tests/ -v --ignore=tests/benchmark.py
```

On Windows, building **hnswlib** needs MSVC. Short path:

- [docs/HNSWLIB_AND_BENCHMARKS.md](docs/HNSWLIB_AND_BENCHMARKS.md)
- or `scripts/setup_hnswlib_env.ps1` and `scripts/install_hnswlib_msvc.bat`

Optional extras: `.[hnswlib]`, `.[chroma]`, `.[faiss]`, `.[qdrant]`, `.[enterprise]`.

## Where code lives

| Path | Stuff |
|------|--------|
| `src/` | C++ wipe, heal, locks, serialize |
| `api/` | FastAPI, auth, metrics, WAL glue |
| `integrations/` | ErasureService, adapters, workflows |
| `compliance/` | Residual proofs, crypto-shred |
| `tests/` | Unit tests (benchmarks are separate and slow) |
| `docs/`, `examples/` | Guides and runnable demos |

## A few ground rules

1. **Don’t oversell.** We don’t claim full GDPR, or wiping swap / offline snapshots. If you add a new storage surface, update [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
2. **Fail closed.** If hard erase fails, don’t silently soft-delete and call it success.
3. **Default is wipe + compact/rebuild.** MN-RU heal is optional and needs measurement (`HEALER_ALLOW_HEAL`).
4. **Tests.** Add them when you can. Skip optional deps with `pytest.importorskip`.
5. **Keep PRs small** enough to review in one sitting.

## Tests and benchmarks

```bash
pytest tests/ -v --ignore=tests/benchmark.py

# optional, needs hnswlib
pytest tests/test_hnswlib_adapter.py -v

# optional, can be slow
python tests/benchmark.py --profile quick --backend hnswlib
```

Please don’t commit `benchmark_results/` (gitignored). If you publish numbers, write a short summary under `docs/benchmarks/` or update [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Pull requests

1. Branch off `main`.
2. Make sure the unit suite is green.
3. In the PR, say what changed, why, and how you tested it.
4. If the delete path or residual story changes, call that out explicitly.

## Security

If you found a way to keep residual floats while claiming success, or to bypass auth on delete endpoints, **don’t** open a public issue. See [SECURITY.md](SECURITY.md).

## License

Contributions are under the same [Apache 2.0](LICENSE) license as the rest of the project.
