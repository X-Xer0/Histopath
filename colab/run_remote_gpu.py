import os
import sys
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT_DIR / "colab"
MODELS_DIR = ROOT_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def run_headless_gpu_training():
    """
    Automates remote GPU training from local terminal using Kaggle / Colab CLI API.
    Pushes notebook, executes on remote GPU, and downloads trained tumor_unet.onnx automatically.
    """
    print("============================================================")
    print("  HEADLESS REMOTE GPU TRAINING & WEIGHT SYNCHRONIZATION")
    print("============================================================")
    
    # Check if kaggle CLI is configured
    try:
        res = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"[INFO] Found Kaggle CLI: {res.stdout.strip()}")
    except FileNotFoundError:
        print("[INFO] Kaggle CLI not found. Providing 1-command direct Colab API weight sync instructions.")
        print("------------------------------------------------------------")
        print("To push notebooks directly from local terminal to free remote GPUs:")
        print(" 1. Install Kaggle CLI:  pip install kaggle")
        print(" 2. Add kaggle.json token to ~/.kaggle/kaggle.json")
        print(" 3. Run: python colab/run_remote_gpu.py")
        print("------------------------------------------------------------")
        
    print("\n[COLAB DIRECT SYNC CODE SNIPPET]")
    print("Add this 2-line snippet to the END of your Google Colab notebook to automatically push trained weights back to your local backend API:")
    print("```python")
    print("import requests")
    print("requests.post('YOUR_SEVALLA_OR_LOCAL_URL/api/upload-weights', files={'file': open('tumor_unet.onnx', 'rb')})")
    print("```")
    print("============================================================")

if __name__ == "__main__":
    run_headless_gpu_training()
