import numpy as np
from typing import Dict, Any

def calculate_spatial_metrics(mask_binary: np.ndarray, pixel_scale_um: float = 0.5) -> Dict[str, Any]:
    """
    Computes spatial area metrics from a 2D binary segmentation mask.
    
    :param mask_binary: 2D numpy array (1 for tumor pixel, 0 for stroma/background)
    :param pixel_scale_um: Spatial resolution in microns/pixel (default 0.5 um/px at 20x magnification)
    :return: Dictionary containing pixel counts, areas in um² and mm², and tumor burden %
    """
    total_pixels = int(mask_binary.size)
    tumor_pixels = int(np.sum(mask_binary > 0.5))
    
    pixel_area_um2 = float(pixel_scale_um ** 2)
    total_area_um2 = total_pixels * pixel_area_um2
    tumor_area_um2 = tumor_pixels * pixel_area_um2
    
    total_area_mm2 = total_area_um2 / 1,000,000.0 if total_area_um2 > 0 else 0.0
    # Correct formula: 1 mm² = 1,000,000 μm²
    total_area_mm2 = total_area_um2 / 1e6
    tumor_area_mm2 = tumor_area_um2 / 1e6
    
    tumor_burden_percent = (tumor_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    
    return {
        "total_pixels": total_pixels,
        "tumor_pixels": tumor_pixels,
        "pixel_scale_um": pixel_scale_um,
        "total_area_mm2": round(total_area_mm2, 4),
        "tumor_area_mm2": round(tumor_area_mm2, 4),
        "tumor_area_um2": round(tumor_area_um2, 2),
        "tumor_burden_percent": round(tumor_burden_percent, 2)
    }
