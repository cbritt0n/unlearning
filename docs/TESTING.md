# Testing guide

## Prerequisites

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# Native build needs a C++17 toolchain + CMake (+ Ninja recommended)
# Windows examples: Visual Studio Build Tools, or LLVM-MinGW (clang++)
#   winget install MartinStorsjo.LLVM-MinGW.UCRT
#   set CXX=clang++  (PowerShell: $env:CXX="clang++")

pip install -e .

# Optional adapters
pip install -e ".[hnswlib]"   # hnswlib adapter tests
# pip install -e ".[chroma]"  # Chroma hook tests (heavy deps)
```

The native module **must** import:

```bash
python -c "import hnsw_healer; print(hnsw_healer.__version__)"
```

On Windows with MinGW/LLVM, `setup.py` copies `libc++.dll` / `libunwind.dll`
next to the `.pyd` when those DLLs are discoverable. If import fails with
“DLL load failed”, ensure the toolchain `bin` directory is on `PATH` or
copy those DLLs beside `hnsw_healer*.pyd`.

## Run all unit tests

```bash
pytest tests/ -v --ignore=tests/benchmark.py
```

Expected on a normal install (no optional adapters):

```text
25 passed, 2 skipped
```

(`test_hnswlib_adapter.py` and `test_chroma_hook.py` are collected only when
those packages import; if the modules load but deps are missing they skip at
importorskip. Without the packages installed, pytest reports those files as
skipped collections.)

| File | Coverage |
|------|----------|
| `test_api.py` | `/health`, `/delete`, `/search`, `/v1/ids/register`, `/v1/collections/.../delete`, lock retry |
| `test_wal.py` | WAL begin/commit, checksum, durable hard-delete, crash recovery |
| `test_id_registry.py` | Collection id map, isolation, JSON persist |
| `test_erase_service.py` | Enterprise erase by external id + native backend |
| `test_residual_proof.py` | Live zeros + post-delete checkpoint pattern absence |
| `test_crypto_shred.py` | AES-GCM encrypt/decrypt + shred fail-closed |
| `test_hnswlib_adapter.py` | Zero + mark_deleted + auto-compact (**skipped** if no `hnswlib`) |
| `test_faiss_adapter.py` | FAISS HNSW hard-delete + auto-compact (**skipped** if no `faiss`) |
| `test_chroma_hook.py` | Register-on-add + fail-closed erase/compact/proof matrix |
| `test_workflow.py` | ErasureWorkflow persist / advance / export |
| `test_auth.py` | API key middleware + production signing-key guard |
| `test_erase_service.py` | Receipt v2, batch compact-once, residual fail-closed |
| `test_chroma_hook.py` | Hard-erase before Chroma delete (**skipped** if no `chromadb`) |
| `test_replica_fanout.py` | Quorum fan-out + idempotent replica workers |
| `test_kms_crypto.py` | LocalFileKMS + KmsCryptoShredVault |
| `test_vendor_attach.py` | Zero-copy attach / shared-memory wipe |
| `test_recall_bounds.py` | Theorem-style fragmentation & heal dominance |
| `benchmark.py` | Long precision/recall suite — **not** part of default CI |

**Note:** `hnswlib` often needs MSVC Build Tools (or a prebuilt wheel for your
Python version). On Python 3.14 / MinGW-only hosts the adapter tests may stay
skipped until a wheel is available.

## CI

GitHub Actions (`.github/workflows/build.yml`) runs the same unit suite via
cibuildwheel after building wheels (benchmark ignored).

## Writing new tests

- Use `tmp_path` + `HEALER_DATA_DIR` so WAL/index files never touch `./data`.
- Reset `api.main.engine`, `id_registry`, and `_erasure_service` in fixtures
  when hitting HTTP endpoints.
- Prefer `pytest.importorskip("hnswlib")` / `chromadb` for optional stacks.
