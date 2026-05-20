@echo off
setlocal

cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run:
    echo python -m venv .venv
    echo .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\python.exe" -m pytest
