# Publishing this repo to GitHub

Checklist for maintainers before the first public push.

## 1. Create the empty GitHub repository

1. On GitHub: **New repository** (public or private).  
2. Do **not** initialize with a README if you will push an existing tree.  
3. Note the URL: `https://github.com/<org-or-user>/<repo>.git`

Replace placeholders in:

* `setup.py` → `url=`
* `.github/ISSUE_TEMPLATE/config.yml` → security / discussions URLs
* `README.md` → clone URL / badges (if any)

## 2. Local git (first time)

```bash
cd unlearning
git init
git branch -M main
git add .
git status   # confirm no .venv, data/, benchmark_results/, *.pyd, secrets
git commit -m "Initial public release: HNSW Healer 0.3.2"
git remote add origin https://github.com/<org-or-user>/<repo>.git
git push -u origin main
```

## 3. Must not be committed

Already covered by `.gitignore` (verify with `git status`):

* `.venv/`, `data/`, `benchmark_results/`, `*.pyd`, `*.dll`, wheels, caches  
* Live indexes, WAL, receipt logs, API keys  

## 4. GitHub repository settings (recommended)

| Setting | Action |
|---------|--------|
| **About** | Description: “Hard-delete residual vectors in HNSW — wipe, rebuild, prove, receipt” |
| **Topics** | `machine-unlearning`, `hnsw`, `gdpr`, `vector-database`, `rag`, `privacy` |
| **Security** | Enable private vulnerability reporting |
| **Actions** | Allow GitHub Actions (CI builds wheels) |
| **Branch protection** (optional) | Require `CI green` on `main` |
| **Discussions** | Enable if you want Q&A |

## 5. After first push

1. Confirm Actions run is green (or note known Windows-only local builds).  
2. Edit Security advisory URL if org/repo name differs.  
3. Optional: create release tag `v0.3.2` with notes from [CHANGELOG.md](../CHANGELOG.md).  
4. Optional: publish wheels to PyPI when ready (`python -m build` + trusted publishing).

## 6. Community onboarding links

Point newcomers to:

1. [README.md](../README.md)  
2. [docs/GOLDEN_PATH.md](GOLDEN_PATH.md)  
3. [docs/INSTALL.md](INSTALL.md) / [HNSWLIB_AND_BENCHMARKS.md](HNSWLIB_AND_BENCHMARKS.md)  
4. [CONTRIBUTING.md](../CONTRIBUTING.md)  
5. [docs/THREAT_MODEL.md](THREAT_MODEL.md) (honesty)  
6. [docs/benchmarks/standard_hnswlib.md](benchmarks/standard_hnswlib.md) (numbers)

## 7. Support expectations (alpha)

Documented in [SUPPORT.md](../SUPPORT.md): best-effort issues, no SLA, security via private channel.
