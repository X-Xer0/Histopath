"""
FastAPI service for automated nucleus segmentation and morphometry in
H&E stained histopathology slides.

Endpoints
---------
GET  /api/health            service and model status
POST /api/predict           segment a slide, return measurements + overlays
POST /api/export-csv        per-nucleus measurements as CSV
POST /api/generate-report   nucleus morphometry report as PDF
POST /api/upload-weights    replace the ONNX model without a restart

The frontend in ../frontend is served by this same process, so production is a
single container and a single origin.
"""

import base64
import csv
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.model import engine
from backend.app.metrics import calculate_nucleus_metrics, per_nucleus_rows
from backend.app.report_generator import generate_pdf_report_bytes

app = FastAPI(
    title="Histopathology Nucleus Segmentation & Morphometry API",
    description=(
        "Segments cell nuclei in H&E stained histopathology slides and reports "
        "quantitative morphometry. Research and quantitative analysis only."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT_DIR / "frontend"

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def decode_image_bytes(contents: bytes) -> np.ndarray:
    """Decode upload bytes to a 3-channel BGR array. OpenCV first, Pillow as a fallback."""
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    if img_bgr is not None and len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    return img_bgr


def png_data_uri(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def analyse_slide(contents: bytes, pixel_scale_um: float, precise: bool):
    """Shared pipeline: decode -> segment -> instances -> metrics."""
    img_bgr = decode_image_bytes(contents)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported or unreadable image. Use PNG, JPG, JPEG, TIFF, BMP or WEBP.",
        )
    if pixel_scale_um <= 0 or pixel_scale_um > 10:
        raise HTTPException(status_code=400, detail="pixel_scale_um must be between 0 and 10")

    result = engine.predict(img_bgr, precise=precise)
    metrics = calculate_nucleus_metrics(
        result["mask"],
        result["instances"],
        pixel_scale_um=pixel_scale_um,
        tissue_mask=result["tissue_mask"],
    )
    return img_bgr, result, metrics


@app.get("/api/health")
def health_check():
    """Service status, active inference engine, model precision and calibration default."""
    return {
        "status": "online",
        "service": "Histopathology Nucleus Segmentation & Morphometry API",
        "version": "2.0.0",
        "engine": engine.engine_name,
        "model_file": engine.model_file or None,
        "default_pixel_scale_um": 0.5,
        "precise_mode_available": engine.onnx_session is not None,
        "scope": "quantitative image analysis only - not a diagnostic service",
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5),
    precise_mode: bool = Form(False),
):
    """
    Segment nuclei and return quantitative morphometry plus overlay images.
    `precise_mode` enables 8x test-time augmentation (slower, higher accuracy).
    """
    try:
        contents = await file.read()
        # Inference is CPU-bound; run it off the event loop so /api/health and the
        # static frontend stay responsive while a slide is being processed.
        img_bgr, result, metrics = await run_in_threadpool(
            analyse_slide, contents, pixel_scale_um, precise_mode
        )

        return {
            "status": "success",
            "filename": file.filename,
            "engine": result["engine"],
            "precise_mode": result["precise_mode"],
            "metrics": metrics,
            "original_base64": png_data_uri(img_bgr),
            "mask_base64": png_data_uri((result["mask"] * 255).astype(np.uint8)),
            "overlay_base64": png_data_uri(result["overlay"]),
            "instance_base64": png_data_uri(result["instance_overlay"]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/api/export-csv")
async def export_csv(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5),
    precise_mode: bool = Form(False),
):
    """Per-nucleus measurements (id, area, equivalent diameter, centroid) as CSV."""
    try:
        contents = await file.read()
        _, result, _ = await run_in_threadpool(
            analyse_slide, contents, pixel_scale_um, precise_mode
        )
        rows = per_nucleus_rows(result["instances"], pixel_scale_um)

        out = io.StringIO()
        fieldnames = ["nucleus_id", "area_px", "area_um2",
                      "equivalent_diameter_um", "centroid_x_px", "centroid_y_px"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        stem = Path(file.filename or "slide").stem
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=nuclei_{stem}.csv"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV export error: {exc}")


@app.post("/api/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5),
    precise_mode: bool = Form(False),
    sample_id: str = Form(""),
):
    """Stream the nucleus morphometry report as a PDF."""
    try:
        contents = await file.read()
        img_bgr, result, metrics = await run_in_threadpool(
            analyse_slide, contents, pixel_scale_um, precise_mode
        )

        pdf_bytes = generate_pdf_report_bytes(
            img_bgr=img_bgr,
            overlay_bgr=result["overlay"],
            instance_overlay_bgr=result["instance_overlay"],
            metrics=metrics,
            engine_name=result["engine"],
            precise_mode=result["precise_mode"],
            sample_id=sample_id or (file.filename or ""),
        )

        stem = Path(file.filename or "slide").stem
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Nuclei_Report_{stem}.pdf"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation error: {exc}")


@app.post("/api/upload-weights")
async def upload_model_weights(file: UploadFile = File(...)):
    """
    Replace the ONNX model file and hot-reload the inference engine.
    Used to push freshly trained weights from Colab without a restart.
    """
    try:
        if not (file.filename or "").endswith(".onnx"):
            raise HTTPException(status_code=400, detail="Only .onnx model files are accepted")

        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        save_path = models_dir / file.filename

        with open(save_path, "wb") as fh:
            fh.write(await file.read())

        engine._load_model()
        return {
            "status": "success",
            "message": f"Updated model weights: {file.filename}",
            "engine": engine.engine_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {exc}")


# Static frontend is mounted last so it never shadows the API routes.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
