# 30-minute install check

## Preferred: prebuilt wheel (when published)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install "hnsw-healer[chroma,dev]"
python -c "import hnsw_healer; print(hnsw_healer.__version__)"
python examples/chroma_forget/run.py   # if package ships examples
```

If the package is installed from this monorepo:

```bash
pip install -e ".[chroma,dev]"
python examples/chroma_forget/run.py
python examples/attack_demo/run.py
```

## From source (native build)

Requirements: CMake ≥ 3.15, C++17 compiler, Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
python -c "import hnsw_healer; print('ok', hnsw_healer.__version__)"
pytest tests/ -v --ignore=tests/benchmark.py
```

### Windows notes

- Prefer **MSVC Build Tools** or set `CXX=clang++` (LLVM-MinGW).  
- Python **3.14** may lack hnswlib/faiss wheels — use 3.11/3.12 for adapters.  

### Docker

```bash
docker build -t hnsw-healer:latest .
docker run --rm -p 8000:8000 \
  -e HEALER_SIGNING_KEY='replace-me' \
  -e HEALER_API_KEY='replace-me' \
  -e HEALER_ENV=production \
  -v healer-data:/app/data \
  hnsw-healer:latest
curl -s http://127.0.0.1:8000/health
```

## Smoke API (sidecar)

```bash
export HEALER_DATA_DIR=./data
export HEALER_SIGNING_KEY=dev-local-not-for-prod
uvicorn api.main:app --port 8000

# ingest + forget
curl -s -X POST http://127.0.0.1:8000/v1/vectors/ingest \
  -H 'Content-Type: application/json' \
  -d '{"collection":"docs","ids":["a","b"],"vectors":[[0.1,0.2],[0.3,0.4]],"replace_index":true}'

curl -s -X POST http://127.0.0.1:8000/v1/collections/docs/delete \
  -H 'Content-Type: application/json' \
  -d '{"collection":"docs","ids":["a"],"reason":"test"}'
```

## Production secrets

```bash
export HEALER_ENV=production
export HEALER_SIGNING_KEY='long-random'
export HEALER_API_KEY='long-random'
# optional
export HEALER_RESIDUAL_PROOF=sample
export HEALER_COMPACT_POLICY=coalesce
export HEALER_COMPACT_EVERY_N=32
export HEALER_COMPACT_MAX_AGE_S=60
export HEALER_MULTI_TENANT=0
```

Process **refuses to start** in production with the default signing key or without an API key.
