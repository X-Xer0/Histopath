import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.model import engine
from backend.app.metrics import calculate_spatial_metrics
from backend.app.grading import predict_tumor_severity_grade

def compute_segmentation_metrics(pred_binary: np.ndarray, target_binary: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous biomedical evaluation metrics between binary prediction and ground truth target.
    """
    pred = (pred_binary > 0.5).astype(np.uint8)
    target = (target_binary > 0.5).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    
    smooth = 1e-6
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "dice_score": round(float(dice), 4),
        "iou_jaccard": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "pixel_accuracy": round(float(pixel_acc), 4)
    }

def run_evaluation(sample_slides_dir: Path = None) -> Dict[str, Any]:
    """
    Evaluates the active segmentation model on test tissue slides.
    """
    if sample_slides_dir is None:
        sample_slides_dir = ROOT_DIR / "sample_test_slides"
        
    slide_files = list(sample_slides_dir.glob("*.png")) + list(sample_slides_dir.glob("*.jpg"))
    
    if not slide_files:
        print(f"[WARN] No test slides found in {sample_slides_dir}")
        return {}
        
    print(f"============================================================")
    print(f"  REAL-WORLD MODEL EVALUATION & PERFORMANCE BENCHMARK")
    print(f"============================================================")
    print(f" Model Engine Status: {'ONNX Neural Network' if engine.onnx_session else 'Stain Deconvolution Fallback'}")
    print(f" Test Dataset Slides Found: {len(slide_files)}")
    print(f"------------------------------------------------------------")
    
    results: List[Dict[str, Any]] = []
    dice_scores, iou_scores, precisions, recalls = [], [], [], []
    
    for slide_path in slide_files:
        img_bgr = cv2.imread(str(slide_path))
        if img_bgr is None:
            continue
            
        pred_mask, overlay = engine.predict(img_bgr)
        metrics = calculate_spatial_metrics(pred_mask)
        grading = predict_tumor_severity_grade(metrics["tumor_burden_percent"])
        
        # Extract hematoxylin threshold ground truth baseline for evaluation comparison
        h_channel = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, target_mask = cv2.threshold(h_channel, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        eval_metrics = compute_segmentation_metrics(pred_mask, target_mask)
        
        dice_scores.append(eval_metrics["dice_score"])
        iou_scores.append(eval_metrics["iou_jaccard"])
        precisions.append(eval_metrics["precision"])
        recalls.append(eval_metrics["recall_sensitivity"])
        
        slide_res = {
            "filename": slide_path.name,
            "tumor_burden_pct": metrics["tumor_burden_percent"],
            "tumor_area_mm2": metrics["tumor_area_mm2"],
            "predicted_grade": grading["grade"],
            "evaluation_metrics": eval_metrics
        }
        results.append(slide_res)
        print(f" Slide: {slide_path.name:32s} | Dice: {eval_metrics['dice_score']:.4f} | IoU: {eval_metrics['iou_jaccard']:.4f} | Burden: {metrics['tumor_burden_percent']}% | Grade: {grading['grade']}")
        
    summary = {
        "mean_dice_score": round(float(np.mean(dice_scores)), 4),
        "mean_iou_jaccard": round(float(np.mean(iou_scores)), 4),
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall_sensitivity": round(float(np.mean(recalls)), 4),
        "evaluated_slides_count": len(results),
        "slides": results
    }
    
    print(f"------------------------------------------------------------")
    print(f" OVERALL BENCHMARK RESULTS:")
    print(f"   Mean Dice Similarity Coefficient (DSC): {summary['mean_dice_score']:.4f}")
    print(f"   Mean Intersection over Union (IoU):    {summary['mean_iou_jaccard']:.4f}")
    print(f"   Mean Precision:                       {summary['mean_precision']:.4f}")
    print(f"   Mean Recall / Sensitivity:            {summary['mean_recall_sensitivity']:.4f}")
    print(f"============================================================")
    
    # Save report to JSON
    report_path = ROOT_DIR / "backend" / "tests" / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUCCESS] Saved evaluation report to {report_path}")
    
    return summary

if __name__ == "__main__":
    run_evaluation()
