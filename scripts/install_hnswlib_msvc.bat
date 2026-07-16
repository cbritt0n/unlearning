@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m pip install "hnswlib>=0.8.0"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -c "import hnswlib; print('hnswlib OK', hnswlib.__version__)"
".venv\Scripts\python.exe" -m pytest tests/test_hnswlib_adapter.py -v --tb=short
exit /b %ERRORLEVEL%
