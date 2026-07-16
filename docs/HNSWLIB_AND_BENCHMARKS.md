# Setup guide: hnswlib + benchmarks (Windows)

**Already set up on this machine (2026-07-16):** Python 3.12 venv at `.venv`,
`hnsw_healer` editable install, `hnswlib` built with VS 2022 Build Tools,
quick benchmark run OK, `tests/test_hnswlib_adapter.py` green.

Day-to-day:

```powershell
cd C:\Users\britt\Documents\unlearning
.\.venv\Scripts\Activate.ps1
python tests/benchmark.py --profile quick
```

---

Two different tools in this repo often get mixed up:

| What | Package | What it does |
|------|---------|----------------|
| **Native healer** | `hnsw_healer` (this repo’s C++ module) | Wipe + MN-RU heal + WAL; used by `tests/benchmark.py` **today** |
| **hnswlib** | PyPI `hnswlib` | Popular HNSW library; used by **adapters** (`HnswlibHardDeleteAdapter`, Chroma path) |

You can run **native benchmarks without hnswlib**.  
You need **hnswlib** for adapter tests, Chroma golden path, and production-like stacks.

---

## Your machine (typical blocker)

- **Python 3.14** → often **no hnswlib wheel** → pip tries to compile → needs **MSVC Build Tools**.  
- **Fix:** use **Python 3.11 or 3.12** for a project venv (wheels available).

---

## Path A — Recommended (Python 3.12 venv)

### 1. Install Python 3.12

```powershell
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
```

Close and reopen the terminal, then check:

```powershell
py -3.12 --version
```

### 2. Create a venv in the repo

```powershell
cd C:\Users\britt\Documents\unlearning
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # should show 3.12.x
python -m pip install -U pip
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Install this project + hnswlib + dev tools

```powershell
# Native module + API + tests
pip install -r requirements.txt
pip install -e ".[dev,hnswlib]"

# Optional: Chroma golden path
# pip install -e ".[chroma,dev]"
```

### 4. Build hnswlib (Windows needs MSVC — no official wheel)

```powershell
# One-time: C++ build tools (large download)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Then compile hnswlib with the compiler on PATH:
.\scripts\install_hnswlib_msvc.bat
# or: pip install hnswlib   from an "x64 Native Tools" VS prompt
```

### 5. Verify

```powershell
python -c "import hnsw_healer; print('healer OK')"
python -c "import hnswlib; print('hnswlib OK', hnswlib.__file__)"
pytest tests/test_hnswlib_adapter.py -v
```

### 6. Run benchmarks

```powershell
# Smoke (~seconds)
python tests/benchmark.py --profile quick

# Mid-size (can take several minutes; N=50k)
python tests/benchmark.py --profile standard

# With plot
python tests/benchmark.py --profile quick --out-dir benchmark_results
```

Outputs:

- `benchmark_results/benchmark_report.json`
- `benchmark_results/pareto_frontier.png` (if matplotlib works)

Scenarios: **A soft**, **B unhealed**, **C MN-RU heal**, **D wipe+rebuild**.  
See [BENCHMARKS.md](BENCHMARKS.md) for how to read them.

---

## Path B — Native benchmarks only (stay on 3.14)

If you only care about the C++ proxy evaluation:

```powershell
# already have hnsw_healer working:
python tests/benchmark.py --profile quick
```

Skip hnswlib until you install 3.12. Adapter tests will stay **skipped**.

---

## Path C — Build hnswlib from source on 3.14 (harder)

Only if you must stay on 3.14:

1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)  
   Workload: **Desktop development with C++**
2. Open **“x64 Native Tools Command Prompt for VS”**
3. `pip install hnswlib`

Prefer Path A unless you have a reason to stay on 3.14.

---

## One-shot setup script

From the repo root (after Python 3.12 is installed):

```powershell
.\scripts\setup_hnswlib_env.ps1
```

---

## Common failures

| Symptom | Fix |
|---------|-----|
| `No module named hnswlib` | Activate `.venv` and `pip install -e ".[hnswlib]"` |
| `Microsoft Visual C++ 14.0 required` | Use Python 3.12 (wheel) or install MSVC |
| `hnsw_healer` import fails | `pip install -e .` with CMake + C++17 |
| Benchmark “C healed” recall ~0 | Expected on weak synthetic graphs; compare to **D rebuild** — see BENCHMARKS.md |
| PowerShell can’t activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Suggested first afternoon

1. Path A (3.12 venv + hnswlib)  
2. `python tests/benchmark.py --profile quick`  
3. Open `benchmark_results/benchmark_report.json`  
4. `pytest tests/test_hnswlib_adapter.py -v`  
5. Optional: `python examples/chroma_forget/run.py` after `pip install -e ".[chroma]"`  
