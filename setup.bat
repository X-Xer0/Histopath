@echo off
echo ========================================================
echo  Setting up HistologyAI Ecosystem (Windows)
echo ========================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python could not be found. Please install Python 3.10+.
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

echo [3/3] Running backend test suite...
set PYTHONPATH=%cd%
python backend\tests\test_api.py

echo ========================================================
echo  Setup Complete! To start the local server:
echo    venv\Scripts\activate.bat
echo    python backend\app\main.py
echo  Open http://localhost:8000 in your browser.
echo ========================================================
