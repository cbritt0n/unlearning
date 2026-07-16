# Putting this on GitHub

Rough checklist if you’re publishing the tree for the first time.

## Create the empty repo

On GitHub, create a new repo. Skip the “add a README” option if you’re pushing
this directory as-is. Copy the clone URL.

Then replace `YOUR_ORG` (or similar placeholders) in:

- `setup.py` (`url` and `project_urls`)
- `.github/ISSUE_TEMPLATE/config.yml` (security / discussions links)

## Push (this tree already has git history)

```bash
cd unlearning
git remote add origin https://github.com/YOUR_ORG/YOUR_REPO.git
git push -u origin main
```

If you started from a clean machine without commits, `git init`, `git add .`,
and commit first — then push.

Before you push, run `git status` and make sure you’re **not** shipping:

- `.venv/`, `data/`, `benchmark_results/`
- `*.pyd`, runtime DLLs, wheels
- real `.env` files or signing keys

`.gitignore` should already cover those.

## Repo settings that help

- **Description:** something like “Hard-delete residual vectors in HNSW indexes”
- **Topics:** `machine-unlearning`, `hnsw`, `privacy`, `rag`, `vector-search`
- Turn on **private vulnerability reporting**
- Allow **Actions** so the existing wheel CI can run
- Optional: protect `main` and require the CI green check
- Optional: enable Discussions for Q&A

## After the first push

1. Check that Actions either passes or fails for a reason you understand.
2. Fix the security advisory URL in issue templates if the org/repo name changed.
3. Tag a release (`v0.3.2`) using notes from [CHANGELOG.md](../CHANGELOG.md) if you want.
4. PyPI can wait until you’re ready for a real package publish.

## What to link for newcomers

1. README  
2. [GOLDEN_PATH.md](GOLDEN_PATH.md)  
3. [INSTALL.md](INSTALL.md) / [HNSWLIB_AND_BENCHMARKS.md](HNSWLIB_AND_BENCHMARKS.md)  
4. [CONTRIBUTING.md](../CONTRIBUTING.md)  
5. [THREAT_MODEL.md](THREAT_MODEL.md)  
6. [benchmarks/standard_hnswlib.md](benchmarks/standard_hnswlib.md)  

Support expectations: [SUPPORT.md](../SUPPORT.md).
