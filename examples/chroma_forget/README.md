# Chroma forget (golden path)

Minimal end-to-end: **ingest → search → forget → signed receipt with residual proof**.

No separate id registration or `compact()` calls — those run automatically.

## Setup

```bash
# from repo root
pip install -e ".[chroma,dev]"
```

## Run

```bash
python examples/chroma_forget/run.py
```

Expected: prints a receipt with `status=complete`, `compacted=true`, and
`residual_proof.passed=true`, then confirms the id is gone from Chroma and
the backend matrix row is zeros.
