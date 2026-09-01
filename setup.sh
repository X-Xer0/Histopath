#!/bin/bash
echo "========================================================"
echo " Setting up HistologyAI Ecosystem (Linux / macOS)"
echo "========================================================"

# Check Python version
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtualenv
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "[2/3] Installing dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Run Unit Tests
echo "[3/3] Running backend test suite..."
export PYTHONPATH=$(pwd)
python3 backend/tests/test_api.py

echo "========================================================"
echo " Setup Complete! To start the local server:"
echo "   source venv/bin/activate"
echo "   python3 backend/app/main.py"
echo " Open http://localhost:8000 in your browser."
echo "========================================================"
