"""
FastAPI service for automated nucleus segmentation and morphometry in
H&E stained histopathology slides.

Endpoints
---------
GET  /api/health                service and model status
POST /api/analyse               queue an analysis, returns a job id immediately
GET  /api/job/{id}              poll a job; returns the full result when done
GET  /api/job/{id}/csv          per-nucleus measurements from a finished job
GET  /api/job/{id}/report       morphometry PDF from a finished job
POST /api/predict               synchronous segmentation (small images / local use)
POST /api/export-csv            synchronous CSV
POST /api/generate-report       synchronous PDF
POST /api/upload-weights        replace the ONNX model without a restart

Why there is a job queue
------------------------
Segmenting a 1000x1000 slide takes 25-80 seconds on container CPUs, and the
reverse proxy in front of the app returns HTTP 504 after about 15 seconds.
Measured on the deployed host: a 256x256 image (1 tile, ~5 s) succeeded while a
512x512 image (9 tiles) was cut off at 15.4 s. So the browser must not hold the
request open. `/api/analyse` returns in milliseconds and the client polls.

The synchronous endpoints are kept for small images and local development; they
expose exactly the same measurements.

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
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.model import engine
from backend.app.metrics import calculate_nucleus_metrics, per_nucleus_rows
from backend.app.report_generator import generate_pdf_report_bytes
from backend.app.jobs import JobStore

app = FastAPI(
    title="Histopathology Nucleus Segmentation & Morphometry API",
    description=(
        "Segments cell nuclei in H&E stained histopathology slides and reports "
        "quantitative morphometry. Research and quantitative analysis only."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT_DIR / "frontend"
jobs = JobStore()

# How often the browser should poll. The analysis takes tens of seconds, so
# there is nothing to gain from polling faster than this.
POLL_INTERVAL_MS = 1500

# Returned images are downscaled to this longest side before being base64-encoded.
# At full resolution the four PNG layers of a 1000x1000 slide are ~8 MB, which
# becomes ~11 MB of base64 in the JSON response - slow on a demo connection and
# pointless, because the canvas never displays them that large. Measurements are
# still computed at full resolution; only the displayed pictures are scaled.
MAX_DISPLAY_PX = 900

CSV_FIELDS = ["nucleus_id", "area_px", "area_um2",
              "equivalent_diameter_um", "centroid_x_px", "centroid_y_px"]


# --------------------------------------------------------------------- helpers
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


def encode_display(img: np.ndarray, fmt: str = "jpeg",
                   max_px: int = MAX_DISPLAY_PX) -> bytes:
    """
    Encode an image layer for the browser, downscaled so the longest side is at
    most max_px.

    Photographic layers use JPEG: the four PNG layers of a 1000x1000 slide come
    to ~6.4 MB of base64 JSON, which is slow over a demo connection, while JPEG
    at quality 88 brings the same four down to well under 2 MB with no visible
    difference at viewer size. The binary mask stays PNG because PNG compresses
    a two-tone image far better than JPEG does.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_px:
        scale = max_px / longest
        img = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)

    if fmt == "png":
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Could not encode result image")
    return buf.tobytes()


def data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")


def rows_to_csv(rows) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def validate_scale(pixel_scale_um: float) -> None:
    if pixel_scale_um <= 0 or pixel_scale_um > 10:
        raise HTTPException(status_code=400, detail="pixel_scale_um must be between 0 and 10")


def segment(img_bgr: np.ndarray, pixel_scale_um: float, precise: bool):
    """Run the engine and the measurements at full resolution."""
    result = engine.predict(img_bgr, precise=precise)
    metrics = calculate_nucleus_metrics(
        result["mask"],
        result["instances"],
        pixel_scale_um=pixel_scale_um,
        tissue_mask=result["tissue_mask"],
    )
    return result, metrics


def build_analysis_payload(img_bgr: np.ndarray, result, metrics, filename: str,
                           pixel_scale_um: float) -> dict:
    """
    Everything a finished job needs: display images, measurements and the
    per-nucleus table. Built once, so the CSV and the PDF do not re-run inference.
    """
    rows = per_nucleus_rows(result["instances"], pixel_scale_um)
    return {
        "filename": filename,
        "engine": result["engine"],
        "precise_mode": result["precise_mode"],
        "metrics": metrics,
        "png": {
            "original": encode_display(img_bgr),
            "mask": encode_display((result["mask"] * 255).astype(np.uint8), fmt="png"),
            "overlay": encode_display(result["overlay"]),
            "instance": encode_display(result["instance_overlay"]),
        },
        "csv": rows_to_csv(rows),
        "nucleus_rows": len(rows),
    }


def job_result_response(job, payload: dict) -> dict:
    img = payload["png"]
    return {
        # job-level status: the client polls until this says "done"
        "status": "done",
        "job_id": job.id,
        "filename": payload["filename"],
        "engine": payload["engine"],
        "precise_mode": payload["precise_mode"],
        "elapsed_seconds": round(job.elapsed(), 1),
        "metrics": payload["metrics"],
        "nucleus_rows": payload["nucleus_rows"],
        "original_base64": data_uri(img["original"], "image/jpeg"),
        "mask_base64": data_uri(img["mask"], "image/png"),
        "overlay_base64": data_uri(img["overlay"], "image/jpeg"),
        "instance_base64": data_uri(img["instance"], "image/jpeg"),
    }


# ------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health_check():
    """Service status, active inference engine and calibration default."""
    return {
        "status": "online",
        "service": "Histopathology Nucleus Segmentation & Morphometry API",
        "version": "2.1.0",
        "engine": engine.engine_name,
        "model_file": engine.model_file or None,
        "onnx_threads": engine.ort_num_threads,
        "default_pixel_scale_um": 0.5,
        "precise_mode_available": engine.onnx_session is not None,
        "async_analysis": True,
        "poll_interval_ms": POLL_INTERVAL_MS,
        "scope": "quantitative image analysis only - not a diagnostic service",
    }


# ---- asynchronous flow used by the web UI -----------------------------------
@app.post("/api/analyse", status_code=202)
async def analyse(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5),
    precise_mode: bool = Form(False),
):
    """
    Queue a segmentation and return a job id straight away.

    The request finishes in milliseconds, which is what keeps it inside the
    reverse proxy's timeout. Poll `GET /api/job/{job_id}` for the result.
    """
    validate_scale(pixel_scale_um)
    contents = await file.read()

    # Decode up front: a bad upload should fail immediately rather than burn a
    # job slot. Decoding is milliseconds, inference is tens of seconds.
    img_bgr = await run_in_threadpool(decode_image_bytes, contents)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported or unreadable image. Use PNG, JPG, JPEG, TIFF, BMP or WEBP.",
        )

    filename = file.filename or "slide"

    def work() -> dict:
        result, metrics = segment(img_bgr, pixel_scale_um, precise_mode)
        return build_analysis_payload(img_bgr, result, metrics, filename, pixel_scale_um)

    job = jobs.submit(work)
    return {
        "job_id": job.id,
        "status": job.status,
        "poll_interval_ms": POLL_INTERVAL_MS,
        "image_size": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
    }


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    """Poll a queued analysis. Returns the full result once the job is done."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job id")
    if job.status == "done" and job.result:
        return job_result_response(job, job.result)
    return job.summary()


@app.get("/api/job/{job_id}/csv")
def job_csv(job_id: str):
    """Per-nucleus measurements from a finished job, computed once at job time."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job id")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail=f"Job is not finished (status: {job.status})")

    stem = Path(job.result["filename"]).stem
    return Response(
        content=job.result["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=nuclei_{stem}.csv"},
    )


@app.get("/api/job/{job_id}/report")
def job_report(job_id: str, sample_id: str = ""):
    """Morphometry PDF from a finished job, reusing the stored images and metrics."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job id")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail=f"Job is not finished (status: {job.status})")

    payload = job.result
    png = payload["png"]

    def to_bgr(data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    try:
        pdf_bytes = generate_pdf_report_bytes(
            img_bgr=to_bgr(png["original"]),
            overlay_bgr=to_bgr(png["overlay"]),
            instance_overlay_bgr=to_bgr(png["instance"]),
            metrics=payload["metrics"],
            engine_name=payload["engine"],
            precise_mode=payload["precise_mode"],
            sample_id=sample_id or payload["filename"],
        )
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Report generation error: {exc}")

    stem = Path(payload["filename"]).stem
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Nuclei_Report_{stem}.pdf"},
    )


# ---- synchronous flow (small images, local use, tests) ----------------------
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5),
    precise_mode: bool = Form(False),
):
    """
    Segment nuclei and return measurements plus overlay images in one response.
    Use this for small images; the browser uses the job queue instead because a
    full slide exceeds the proxy timeout.
    """
    try:
        validate_scale(pixel_scale_um)
        contents = await file.read()
        img_bgr = await run_in_threadpool(decode_image_bytes, contents)
        if img_bgr is None:
            raise HTTPException(
                status_code=400,
                detail="Unsupported or unreadable image. Use PNG, JPG, JPEG, TIFF, BMP or WEBP.",
            )

        result, metrics = await run_in_threadpool(
            segment, img_bgr, pixel_scale_um, precise_mode
        )
        payload = build_analysis_payload(img_bgr, result, metrics,
                                         file.filename or "slide", pixel_scale_um)

        return {
            "status": "success",
            "filename": payload["filename"],
            "engine": payload["engine"],
            "precise_mode": payload["precise_mode"],
            "metrics": payload["metrics"],
            "original_base64": data_uri(payload["png"]["original"], "image/jpeg"),
            "mask_base64": data_uri(payload["png"]["mask"], "image/png"),
            "overlay_base64": data_uri(payload["png"]["overlay"], "image/jpeg"),
            "instance_base64": data_uri(payload["png"]["instance"], "image/jpeg"),
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
    """Per-nucleus measurements as CSV (synchronous)."""
    try:
        validate_scale(pixel_scale_um)
        contents = await file.read()
        img_bgr = await run_in_threadpool(decode_image_bytes, contents)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Unsupported or unreadable image.")

        result, _ = await run_in_threadpool(segment, img_bgr, pixel_scale_um, precise_mode)
        text = rows_to_csv(per_nucleus_rows(result["instances"], pixel_scale_um))

        stem = Path(file.filename or "slide").stem
        return Response(
            content=text,
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
    """Stream the nucleus morphometry report as a PDF (synchronous)."""
    try:
        validate_scale(pixel_scale_um)
        contents = await file.read()
        img_bgr = await run_in_threadpool(decode_image_bytes, contents)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Unsupported or unreadable image.")

        result, metrics = await run_in_threadpool(
            segment, img_bgr, pixel_scale_um, precise_mode
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
