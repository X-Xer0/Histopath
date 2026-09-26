"""
Benchmark of the deployed ONNX model.

Two scopes:

  default      the six sample slides in samples/ (fast, regression check)
  --official   all fourteen slides of the official MoNuSeg 2018 test set, with
               their XML annotations - the same scope the training notebook
               reports, so the two can be compared directly

Reports pixel-level quality (Dice, IoU, precision, recall, specificity),
instance-level quality (Aggregated Jaccard Index) and the nucleus detection
rate against the annotated count.

Usage:
    python -m backend.tests.evaluate_model                 # 6 sample slides
    python -m backend.tests.evaluate_model --precise       # 6 samples, 8x TTA
    python -m backend.tests.evaluate_model --official      # all 14 official slides
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.model import engine
from backend.app.metrics import calculate_nucleus_metrics

SAMPLES_DIR = ROOT_DIR / "samples"
OFFICIAL_DIR = ROOT_DIR / "MoNuSegTestData"
REPORT_PATH = ROOT_DIR / "backend" / "tests" / "evaluation_report.json"
OFFICIAL_REPORT_PATH = ROOT_DIR / "backend" / "tests" / "evaluation_report_official.json"

TEST_ORGAN_MAP = {
    "TCGA-2Z-A9J9": "Testis", "TCGA-44-2665": "Lung", "TCGA-69-7764": "Lung",
    "TCGA-A6-6782": "Colon", "TCGA-AC-A2FO": "Breast", "TCGA-AO-A0J2": "Breast",
    "TCGA-CU-A0YN": "Prostate", "TCGA-EJ-A46H": "Bladder", "TCGA-FG-A4MU": "Kidney",
    "TCGA-GL-6846": "Brain", "TCGA-HC-7209": "Kidney", "TCGA-HT-8564": "Brain",
    "TCGA-IZ-8196": "Thyroid", "TCGA-ZF-A9R5": "Bladder",
}


def _patient_key(stem: str) -> str:
    return "-".join(stem.split("-")[:3])


def _parse_xml_mask(xml_path: Path, height: int, width: int) -> np.ndarray:
    """MoNuSeg XML polygon annotations -> binary mask (port of the official MATLAB)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in ET.parse(xml_path).getroot().findall(".//Region"):
        pts = [[int(round(float(v.attrib['X']))), int(round(float(v.attrib['Y'])))]
               for v in region.findall(".//Vertex")]
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def _px_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    smooth = 1e-6
    tp = float(np.sum((pred == 1) & (gt == 1)))
    fp = float(np.sum((pred == 1) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt == 1)))
    tn = float(np.sum((pred == 0) & (gt == 0)))
    return {
        "dice": round((2 * tp + smooth) / (2 * tp + fp + fn + smooth), 4),
        "iou": round((tp + smooth) / (tp + fp + fn + smooth), 4),
        "precision": round((tp + smooth) / (tp + fp + smooth), 4),
        "recall": round((tp + smooth) / (tp + fn + smooth), 4),
        "specificity": round((tn + smooth) / (tn + fp + smooth), 4),
        "pixel_accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "fp_px": int(fp),
        "fn_px": int(fn),
    }


def _aji(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Aggregated Jaccard Index, ported from the official MoNuSeg MATLAB reference:
    each ground-truth nucleus matches the predicted nucleus with the highest IoU,
    a prediction may be matched by more than one nucleus, and both missed nuclei
    and unmatched predictions contribute their area to the denominator.
    """
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    n_gt, gt_lab = cv2.connectedComponents(gt, connectivity=8)
    n_pr, pr_lab = cv2.connectedComponents(pred, connectivity=8)
    if n_gt <= 1 and n_pr <= 1:
        return 1.0
    if n_gt <= 1 or n_pr <= 1:
        return 0.0

    gt_area = np.bincount(gt_lab.ravel(), minlength=n_gt)
    pr_area = np.bincount(pr_lab.ravel(), minlength=n_pr)
    pair = gt_lab.astype(np.int64) * n_pr + pr_lab
    inter = np.bincount(pair.ravel(), minlength=n_gt * n_pr).reshape(n_gt, n_pr)

    correct = 0.0
    union = 0.0
    matched = np.zeros(n_pr, dtype=bool)
    for g in range(1, n_gt):
        row = inter[g, 1:]
        if row.max() == 0:
            union += gt_area[g]
            continue
        union_gp = gt_area[g] + pr_area[1:] - row
        p = int(np.argmax(row / np.maximum(union_gp, 1))) + 1
        correct += row[p - 1]
        union += union_gp[p - 1]
        matched[p] = True
    union += pr_area[1:][~matched[1:]].sum()
    return float(correct / union) if union > 0 else 0.0


def _collect_official() -> List[Dict[str, Any]]:
    """Every .tif in MoNuSegTestData, with its XML annotation as ground truth."""
    items = []
    for tif in sorted(OFFICIAL_DIR.glob("*.tif")):
        xml = tif.with_suffix(".xml")
        if not xml.exists():
            continue
        img = cv2.imread(str(tif))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = (_parse_xml_mask(xml, h, w) > 127).astype(np.uint8)
        n_ann = len(ET.parse(xml).getroot().findall(".//Region"))
        items.append({"name": tif.stem, "img": img, "gt": gt, "annotated": n_ann,
                      "organ": TEST_ORGAN_MAP.get(_patient_key(tif.stem), "Other")})
    return items


def _collect_samples() -> List[Dict[str, Any]]:
    """The six slides in samples/, each with a pre-rendered reference mask."""
    items = []
    for slide in sorted(p for p in SAMPLES_DIR.glob("*.png") if not p.stem.endswith("_mask")):
        mask_path = slide.with_name(f"{slide.stem}_mask.png")
        if not mask_path.exists():
            continue
        img = cv2.imread(str(slide))
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if img is None:
            continue
        tcga = slide.stem.replace("monuseg_test_", "")
        xml = OFFICIAL_DIR / f"{tcga}.xml"
        n_ann = (len(ET.parse(xml).getroot().findall(".//Region")) if xml.exists()
                 else max(0, cv2.connectedComponents(gt, connectivity=8)[0] - 1))
        items.append({"name": slide.stem, "img": img, "gt": gt, "annotated": n_ann,
                      "organ": TEST_ORGAN_MAP.get(_patient_key(tcga), "Other")})
    return items


def run_evaluation(precise: bool = False, pixel_scale_um: float = 0.5,
                   official: bool = False) -> Dict[str, Any]:
    items = _collect_official() if official else _collect_samples()
    scope = "official MoNuSeg 2018 test set" if official else "sample slides"

    if not items:
        print(f"[WARN] no slides found for scope '{scope}'")
        return {}

    print("=" * 96)
    print("  NUCLEUS SEGMENTATION BENCHMARK")
    print("=" * 96)
    print(f"  Engine       : {engine.engine_name}")
    print(f"  Scope        : {scope} - {len(items)} slides")
    print(f"  Mode         : {'precise (8x TTA)' if precise else 'standard'}")
    print(f"  Calibration  : {pixel_scale_um} um/pixel")
    print("-" * 96)
    print(f"  {'slide':34s} {'organ':9s} {'dice':>6s} {'iou':>6s} {'AJI':>6s} "
          f"{'prec':>6s} {'rec':>6s} {'detected':>9s} {'annot':>7s}")
    print("-" * 96)

    records: List[Dict[str, Any]] = []
    for it in items:
        result = engine.predict(it["img"], precise=precise)
        pred = (result["mask"] > 0).astype(np.uint8)
        gt = it["gt"]
        px = _px_metrics(pred, gt)
        aji = _aji(pred, gt)
        m = calculate_nucleus_metrics(pred, result["instances"], pixel_scale_um,
                                      result["tissue_mask"])
        detected, gt_count = m["nuclei_count"], it["annotated"]
        rate = detected / gt_count * 100 if gt_count else 0.0

        records.append({
            "slide": it["name"], "organ": it["organ"],
            "dice": px["dice"], "iou": px["iou"], "aji": round(aji, 4),
            "precision": px["precision"], "recall": px["recall"],
            "specificity": px["specificity"], "pixel_accuracy": px["pixel_accuracy"],
            "fp_px": px["fp_px"], "fn_px": px["fn_px"],
            "nuclei_detected": detected, "nuclei_annotated": gt_count,
            "detection_rate_percent": round(rate, 1),
            "nuclear_density_percent": m["nuclear_density_percent"],
        })
        print(f"  {it['name'][:34]:34s} {it['organ']:9s} {px['dice']:6.3f} {px['iou']:6.3f} "
              f"{aji:6.3f} {px['precision']:6.3f} {px['recall']:6.3f} "
              f"{detected:9,d} {gt_count:7,d}")

    def mean(key):
        return round(float(np.mean([r[key] for r in records])), 4)

    summary = {
        "engine": engine.engine_name,
        "model_file": engine.model_file,
        "scope": scope,
        "precise_mode": precise,
        "pixel_scale_um": pixel_scale_um,
        "slides_evaluated": len(records),
        "mean_dice": mean("dice"),
        "mean_iou": mean("iou"),
        "mean_aji": mean("aji"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "mean_specificity": mean("specificity"),
        "mean_pixel_accuracy": mean("pixel_accuracy"),
        "total_fp_px": int(sum(r["fp_px"] for r in records)),
        "total_fn_px": int(sum(r["fn_px"] for r in records)),
        "total_nuclei_detected": int(sum(r["nuclei_detected"] for r in records)),
        "total_nuclei_annotated": int(sum(r["nuclei_annotated"] for r in records)),
        "mean_detection_rate_percent": mean("detection_rate_percent"),
        "slides": records,
    }

    print("-" * 96)
    print(f"  Mean Dice            : {summary['mean_dice']:.4f}")
    print(f"  Mean IoU             : {summary['mean_iou']:.4f}")
    print(f"  Mean AJI (instance)  : {summary['mean_aji']:.4f}")
    print(f"  Mean Precision       : {summary['mean_precision']:.4f}")
    print(f"  Mean Recall          : {summary['mean_recall']:.4f}")
    print(f"  Mean Specificity     : {summary['mean_specificity']:.4f}")
    print(f"  FP / FN pixels       : {summary['total_fp_px']:,} / {summary['total_fn_px']:,}")
    print(f"  Nuclei detected      : {summary['total_nuclei_detected']:,} of "
          f"{summary['total_nuclei_annotated']:,} annotated "
          f"({summary['mean_detection_rate_percent']:.1f}%)")
    print("-" * 96)
    print("  PER-ORGAN:")
    for organ in sorted({r["organ"] for r in records}):
        rows = [r for r in records if r["organ"] == organ]
        print(f"    {organ:10s} n={len(rows)}  dice {np.mean([r['dice'] for r in rows]):.4f}  "
              f"AJI {np.mean([r['aji'] for r in rows]):.4f}  "
              f"precision {np.mean([r['precision'] for r in rows]):.4f}  "
              f"recall {np.mean([r['recall'] for r in rows]):.4f}")
    print("=" * 96)

    report_path = OFFICIAL_REPORT_PATH if official else REPORT_PATH
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {report_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precise", action="store_true", help="8x test-time augmentation")
    parser.add_argument("--official", action="store_true",
                        help="evaluate all 14 slides of the official test set")
    parser.add_argument("--scale", type=float, default=0.5, help="micrometres per pixel")
    args = parser.parse_args()
    run_evaluation(precise=args.precise, pixel_scale_um=args.scale, official=args.official)
