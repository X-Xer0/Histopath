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
            
    def predict(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs segmentation on input H&E image.
        Returns tuple of (binary_mask, colored_overlay_bgr).
        """
        h, w = img_bgr.shape[:2]
        
        if self.onnx_session is not None:
            # ONNX Inference
            img_resized = cv2.resize(img_bgr, (256, 256))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]
            
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            raw_pred = self.onnx_session.run([output_name], {input_name: input_tensor})[0]
            
            pred_mask_256 = (raw_pred.squeeze() > 0.5).astype(np.uint8)
            binary_mask = cv2.resize(pred_mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback Computer Vision Engine: Hematoxylin Deconvolution + Morphological Thresholding
            h_channel = extract_hematoxylin_channel(img_bgr)
            blurred = cv2.GaussianBlur(h_channel, (5, 5), 0)
            _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological Operations (clean noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
        # Create Colored Overlay (Red mask over tumor region)
        overlay_bgr = img_bgr.copy()
        overlay_bgr[binary_mask > 0] = cv2.addWeighted(
            img_bgr[binary_mask > 0], 0.5, 
            np.full_like(img_bgr[binary_mask > 0], (0, 0, 220)), 0.5, 0
        )
        
        # Draw Contour Boundaries
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
        
        return binary_mask, overlay_bgr

# Singleton engine instance
engine = HistopathologyInferenceEngine()
