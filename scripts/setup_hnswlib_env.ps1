# Setup Python 3.12 venv + project + optional hnswlib for Windows.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_hnswlib_env.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Repo: $Root"

# Prefer py -3.12 launcher
$py = $null
try {
    $ver = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver) { $py = $ver.Trim() }
} catch {}

if (-not $py) {
    Write-Host "Python 3.12 not found. Install with:"
    Write-Host "  winget install Python.Python.3.12"
    exit 1
}

Write-Host "==> Using $py"
if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating .venv"
    & $py -m venv .venv
}

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $venvPy -m pip install -U pip
& $venvPy -m pip install -r requirements.txt
& $venvPy -m pip install -e .

Write-Host "==> Trying hnswlib (needs MSVC on Windows if no wheel)..."
& $venvPy -m pip install "hnswlib>=0.8.0"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "hnswlib install FAILED (common on Windows without C++ Build Tools)."
    Write-Host "Install tools, then re-run this script:"
    Write-Host "  winget install Microsoft.VisualStudio.2022.BuildTools --override `"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`""
    Write-Host ""
    Write-Host "You can still run NATIVE benchmarks without hnswlib:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python tests/benchmark.py --profile quick"
    exit 0
}

Write-Host "==> Verify"
& $venvPy -c "import hnsw_healer, hnswlib; print('healer OK'); print('hnswlib', hnswlib.__version__)"
& $venvPy -m pytest tests/test_hnswlib_adapter.py -v --tb=short

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python tests/benchmark.py --profile quick"
Write-Host "  python tests/benchmark.py --profile standard"
