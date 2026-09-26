"""
Quantitative nucleus morphometry.

Everything here is a direct measurement of the segmented nuclei mask:
how many nuclei there are, how much of the tissue they cover, how large they
are, and how variable that size is. Nothing in this module makes a diagnosis -
it reports measurements and a descriptive cellularity band.

Spatial calibration
-------------------
The MoNuSeg 2018 slides were scanned on a 40x microscope, but the released
1000x1000 crops are downsampled to an effective ~20x, i.e. 0.50 um/pixel.
This was verified against the dataset itself: the median annotated nucleus
measures 19.1 px across, which is 9.57 um at 0.50 um/px (the expected size of
an epithelial tumour nucleus) and only 4.78 um at 0.25 um/px (smaller than a
lymphocyte, which is impossible across seven carcinoma types).

    area per pixel = 0.50^2 = 0.25 um^2
    nuclear area (mm^2) = nuclear pixels x 0.25 / 1,000,000
"""

import numpy as np
from typing import Any, Dict, Optional

# Histogram used for the nuclear size distribution shown in the UI and report.
SIZE_BIN_EDGES_UM = np.arange(0.0, 26.0, 2.0)   # 12 bins, 0-24 um


def _cellularity_band(density_percent: float) -> Dict[str, str]:
    """
    Descriptive label for measured nuclear density.

    The cut points are the terciles of the nuclear density this pipeline
    measures across the 37 MoNuSeg 2018 training slides
    (mean 28.3 %, median 29.1 %, range 11.7 - 44.8 %; 33rd percentile 23.0 %,
    67th percentile 30.5 %). They describe the pipeline's own output
    distribution, not a clinical reference range.

    Note that the pipeline measures slightly above the annotation density
    (about 28 % versus 24 % on the same slides) because it over-covers nuclei in
    places while missing others. The bands are therefore calibrated on the
    pipeline's own output, which is what the app compares against.

    This is a summary of a measurement, not a diagnosis.
    """
    if density_percent < 23.0:
        label = "Low"
        detail = "below the lower third of the reference distribution"
    elif density_percent <= 30.5:
        label = "Moderate"
        detail = "within the middle third of the reference distribution"
    else:
        label = "High"
        detail = "above the upper third of the reference distribution"

    return {
        "category": label,
        "detail": detail,
        "basis": "terciles of measured nuclear density across the 37 MoNuSeg 2018 training slides",
    }


def calculate_nucleus_metrics(
    binary_mask: np.ndarray,
    instance_labels: Optional[np.ndarray] = None,
    pixel_scale_um: float = 0.5,
    tissue_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Measure the segmented nuclei mask.

    :param binary_mask:   2D array, non-zero = nucleus pixel
    :param instance_labels: optional int label map from separate_nuclei();
                            when omitted only pixel-level metrics are returned
    :param pixel_scale_um: um per pixel (0.50 for the distributed MoNuSeg crops)
    :param tissue_mask:   optional boolean mask of tissue (non-glass) pixels,
                          used as the denominator for nuclear density
    :return: dictionary of measurements
    """
    pixel_area_um2 = float(pixel_scale_um) ** 2

    mask = (binary_mask > 0)
    total_pixels = int(mask.size)
    nuclei_pixels = int(mask.sum())

    if tissue_mask is not None:
        tissue_pixels = int(np.count_nonzero(tissue_mask))
    else:
        tissue_pixels = total_pixels
    if tissue_pixels <= 0:
        tissue_pixels = total_pixels

    tissue_area_mm2 = tissue_pixels * pixel_area_um2 / 1e6
    nuclear_area_um2 = nuclei_pixels * pixel_area_um2
    nuclear_area_mm2 = nuclear_area_um2 / 1e6
    density_percent = (nuclei_pixels / tissue_pixels * 100.0) if tissue_pixels else 0.0

    metrics: Dict[str, Any] = {
        "pixel_scale_um": float(pixel_scale_um),
        "total_image_pixels": total_pixels,
        "tissue_pixels": tissue_pixels,
        "tissue_area_mm2": round(tissue_area_mm2, 6),
        "nuclei_pixels": nuclei_pixels,
        "nuclear_area_um2": round(nuclear_area_um2, 2),
        "nuclear_area_mm2": round(nuclear_area_mm2, 6),
        "nuclear_density_percent": round(density_percent, 2),
        "nuclei_count": 0,
        "nuclei_per_mm2": 0.0,
        "mean_nuclear_area_um2": 0.0,
        "median_nuclear_area_um2": 0.0,
        "mean_equivalent_diameter_um": 0.0,
        "min_equivalent_diameter_um": 0.0,
        "max_equivalent_diameter_um": 0.0,
        "std_equivalent_diameter_um": 0.0,
        "size_variability_cv_percent": 0.0,
        "size_distribution": {
            "bin_edges_um": [round(float(e), 2) for e in SIZE_BIN_EDGES_UM],
            "counts": [0] * (len(SIZE_BIN_EDGES_UM) - 1),
        },
        "cellularity": _cellularity_band(round(density_percent, 2)),
    }

    # ---- per-nucleus measurements (only when instance labels are available) ----
    if instance_labels is not None and instance_labels.max() > 0:
        areas_px = np.bincount(instance_labels.ravel())
        areas_px = areas_px[1:]                      # drop background label 0
        areas_px = areas_px[areas_px > 0]

        if areas_px.size:
            areas_um2 = areas_px.astype(np.float64) * pixel_area_um2
            diameters_um = 2.0 * np.sqrt(areas_um2 / np.pi)   # equivalent circular diameter

            metrics["nuclei_count"] = int(areas_px.size)
            metrics["mean_nuclear_area_um2"] = round(float(areas_um2.mean()), 2)
            metrics["median_nuclear_area_um2"] = round(float(np.median(areas_um2)), 2)
            metrics["mean_equivalent_diameter_um"] = round(float(diameters_um.mean()), 2)
            metrics["min_equivalent_diameter_um"] = round(float(diameters_um.min()), 2)
            metrics["max_equivalent_diameter_um"] = round(float(diameters_um.max()), 2)
            metrics["std_equivalent_diameter_um"] = round(float(diameters_um.std()), 2)

            mean_d = float(diameters_um.mean())
            if mean_d > 0:
                metrics["size_variability_cv_percent"] = round(
                    float(diameters_um.std()) / mean_d * 100.0, 2
                )

            counts, _ = np.histogram(diameters_um, bins=SIZE_BIN_EDGES_UM)
            metrics["size_distribution"]["counts"] = [int(c) for c in counts]

            if tissue_area_mm2 > 0:
                metrics["nuclei_per_mm2"] = round(int(areas_px.size) / tissue_area_mm2, 1)

    return metrics


def per_nucleus_rows(instance_labels: np.ndarray, pixel_scale_um: float = 0.5):
    """
    Yield one row of measurements per nucleus, for CSV export.

    Columns: nucleus_id, area_px, area_um2, equivalent_diameter_um,
             centroid_x_px, centroid_y_px
    """
    pixel_area_um2 = float(pixel_scale_um) ** 2
    rows = []
    if instance_labels is None or instance_labels.max() == 0:
        return rows

    num, _, stats, centroids = __import__("cv2").connectedComponentsWithStats(
        (instance_labels > 0).astype(np.uint8), connectivity=8
    )
    nid = 0
    for i in range(1, num):
        area_px = int(stats[i, 4])
        if area_px <= 0:
            continue
        nid += 1
        area_um2 = area_px * pixel_area_um2
        diameter_um = 2.0 * float(np.sqrt(area_um2 / np.pi))
        rows.append({
            "nucleus_id": nid,
            "area_px": area_px,
            "area_um2": round(area_um2, 3),
            "equivalent_diameter_um": round(diameter_um, 3),
            "centroid_x_px": round(float(centroids[i][0]), 1),
            "centroid_y_px": round(float(centroids[i][1]), 1),
        })
    return rows
