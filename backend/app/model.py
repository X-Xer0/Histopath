import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.stain_norm import extract_hematoxylin_channel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tumor_unet.onnx"

class HistopathologyInferenceEngine:
    def __init__(self):
        self.onnx_session = None
        self._load_model()
        
    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(str(MODEL_PATH))
                print(f"[INFO] Successfully loaded ONNX model from {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Failed loading ONNX session ({e}). Using Computer Vision Fallback engine.")
                self.onnx_session = None
        else:
            print(f"[INFO] No ONNX weights found at {MODEL_PATH}. Active fallback: Adaptive Stain Deconvolution Engine.")
            
    def _onnx_tile_predict(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Runs ONNX model on a single 256x256 BGR patch."""
        patch_resized = cv2.resize(patch_bgr, (256, 256))
        img_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]
        
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        raw_pred = self.onnx_session.run([output_name], {input_name: input_tensor})[0]
        
        pred_256 = raw_pred.squeeze()
        if patch_bgr.shape[:2] != (256, 256):
            pred_256 = cv2.resize(pred_256, (patch_bgr.shape[1], patch_bgr.shape[0]))
        return pred_256

    def predict(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs segmentation on input H&E image using Sliding Window Tiling
        to preserve cell magnification scale.
        """
        # Ensure 3-channel BGR
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            
        h, w = img_bgr.shape[:2]
        
        if self.onnx_session is not None:
            # If image is small (<= 512x512), run direct inference
            if h <= 512 and w <= 512:
                prob_map = self._onnx_tile_predict(img_bgr)
            else:
                # Sliding-window tiling for high-res biopsy slides
                prob_map = np.zeros((h, w), dtype=np.float32)
                counts = np.zeros((h, w), dtype=np.float32)
                
                tile_size = 256
                stride = 192  # 25% overlap blending
                
                for y in range(0, max(1, h - tile_size + 1), stride):
                    for x in range(0, max(1, w - tile_size + 1), stride):
                        y_end = min(y + tile_size, h)
                        x_end = min(x + tile_size, w)
                        
                        patch = img_bgr[y:y_end, x:x_end]
                        if patch.shape[0] < 32 or patch.shape[1] < 32:
                            continue
                            
                        pred_patch = self._onnx_tile_predict(patch)
                        prob_map[y:y_end, x:x_end] += pred_patch
                        counts[y:y_end, x:x_end] += 1.0
                        
                counts[counts == 0] = 1.0
                prob_map /= counts
                
            binary_mask = (prob_map > 0.5).astype(np.uint8)
        else:
            # Fallback Engine: Hematoxylin Deconvolution + Morphological Thresholding
            h_channel = extract_hematoxylin_channel(img_bgr)
            blurred = cv2.GaussianBlur(h_channel, (5, 5), 0)
            _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological noise cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
        # Tissue mask filtering (zero out non-tissue blank background)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        background = (gray >= 238) | (gray <= 15)
        binary_mask[background] = 0
        
        # Create Colored Overlay (Red mask over tumor region)
        overlay_bgr = img_bgr.copy()
        if np.any(binary_mask > 0):
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[binary_mask > 0] = (0, 0, 220)  # Red BGR
            blended = cv2.addWeighted(img_bgr, 0.5, colored_mask, 0.5, 0)
            overlay_bgr[binary_mask > 0] = blended[binary_mask > 0]
            
            # Draw Contour Boundaries
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
        
        return binary_mask, overlay_bgr

# Singleton engine instance
engine = HistopathologyInferenceEngine()
