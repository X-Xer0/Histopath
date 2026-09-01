import sys
import argparse
import urllib.request
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "backend" / "models"
ONNX_PATH = MODELS_DIR / "tumor_unet.onnx"

def sync_weights_to_server(target_url: str):
    """
    POSTs trained ONNX model weights to the target backend API (/api/upload-weights)
    so the server automatically reloads the trained model live.
    """
    if not ONNX_PATH.exists():
        print(f"[ERROR] No trained model weights found at {ONNX_PATH}")
        print("Please train your model in Colab/Kaggle first and place tumor_unet.onnx in backend/models/")
        return
        
    upload_endpoint = target_url.rstrip("/") + "/api/upload-weights"
    print(f"[INFO] Uploading ONNX model weights to {upload_endpoint}...")
    
    try:
        import requests
        with open(ONNX_PATH, "rb") as f:
            files = {"file": ("tumor_unet.onnx", f, "application/octet-stream")}
            resp = requests.post(upload_endpoint, files=files)
            
        if resp.status_code == 200:
            print(f"[SUCCESS] Server updated! Response: {resp.json()}")
        else:
            print(f"[ERROR] Failed uploading weights. Status: {resp.status_code}, Body: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Sync failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Colab Remote Sync & Model Weight Manager")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Target API server URL (e.g. http://localhost:8000 or Sevalla URL)")
    parser.add_argument("--sync-weights", action="store_true", help="Automatically POST trained ONNX weights to the target server")
    
    args = parser.parse_args()
    
    if args.sync_weights:
        sync_weights_to_server(args.server)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
