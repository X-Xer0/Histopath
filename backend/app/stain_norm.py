import numpy as np
import cv2

def normalize_he_stain(img_bgr: np.ndarray) -> np.ndarray:
    """
    Performs Macenko stain normalization / intensity scaling on H&E tissue images.
    Converts RGB/BGR image to Optical Density (OD) space.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid input image")
        
    # Ensure 3-channel BGR
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = np.clip(img_rgb, 1.0, 255.0)
    
    # Calculate Optical Density (OD)
    OD = -np.log10(img_rgb / 255.0)
    
    # Normalize OD values safely
    od_min, od_max = np.min(OD), np.max(OD)
    if od_max - od_min < 1e-5:
        return img_bgr
        
    OD_norm = (OD - od_min) / (od_max - od_min + 1e-5)
    
    # Reconvert back to RGB scale [0, 255]
    RGB_norm = (10 ** (-OD_norm)) * 255.0
    RGB_norm = np.clip(RGB_norm, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(RGB_norm, cv2.COLOR_RGB2BGR)

def extract_hematoxylin_channel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts Hematoxylin (purple cell nuclei) stain intensity via color deconvolution.
    Includes background tissue masking to ignore slide margin borders.
    """
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) + 1.0
    OD = -np.log(img_rgb / 255.0)
    
    # Hematoxylin vector weighting
    H_OD = OD[:, :, 0] * 0.65 + OD[:, :, 1] * 0.70 + OD[:, :, 2] * 0.29
    H_norm = cv2.normalize(H_OD, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Suppress non-tissue blank background (bright white background or dark outer margins)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tissue_mask = (gray < 235) & (gray > 20)
    H_norm[~tissue_mask] = 0
    
    return H_norm
