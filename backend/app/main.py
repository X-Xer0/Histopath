import os
import sys
import base64
import cv2
import numpy as np
from pathlib import Path

# Add project root directory to sys.path dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, UploadFile, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.model import engine
from backend.app.metrics import calculate_spatial_metrics
from backend.app.grading import predict_tumor_severity_grade
from backend.app.report_generator import generate_pdf_report_bytes

app = FastAPI(
    title="Automated Histopathology Segmentation & Grading API",
    description="Microservice API for slide segmentation, area metrics, tumor grading, and pathology PDF reports.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

import io
from PIL import Image

def decode_image_bytes(contents: bytes) -> np.ndarray:
    """Robustly decodes image bytes (PNG, JPG, TIFF, WEBP, RGBA) to 3-channel BGR numpy array."""
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        try:
            pil_img = Image.open(io.BytesIO(contents))
            pil_img = pil_img.convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return None
            
    if img_bgr is not None and len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        
    return img_bgr

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "Histopathology Segmentation & Grading API",
        "engine": "ONNX Model" if engine.onnx_session else "Deconvolution Fallback Active",
        "version": "1.0.0"
    }

@app.post("/api/predict")
async def predict_tissue_slide(
    file: UploadFile = File(...),
    pixel_scale_um: float = Form(0.5)
):
    try:
        contents = await file.read()
        img_bgr = decode_image_bytes(contents)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, TIFF, WEBP")
            
        # Run segmentation engine
        binary_mask, overlay_bgr = engine.predict(img_bgr)
        
        # Calculate quantitative metrics
        metrics = calculate_spatial_metrics(binary_mask, pixel_scale_um=pixel_scale_um)
        
        # Predict tumor grade
        grading = predict_tumor_severity_grade(metrics["tumor_burden_percent"])
        
        # Encode original image, overlay, and mask to Base64 PNGs
        _, orig_png = cv2.imencode(".png", img_bgr)
        _, mask_png = cv2.imencode(".png", (binary_mask * 255).astype(np.uint8))
        _, overlay_png = cv2.imencode(".png", overlay_bgr)
        
        orig_base64 = base64.b64encode(orig_png).decode("utf-8")
        mask_base64 = base64.b64encode(mask_png).decode("utf-8")
        overlay_base64 = base64.b64encode(overlay_png).decode("utf-8")
        
        return {
            "status": "success",
            "filename": file.filename,
            "metrics": metrics,
            "grading": grading,
            "original_base64": f"data:image/png;base64,{orig_base64}",
            "mask_base64": f"data:image/png;base64,{mask_base64}",
            "overlay_base64": f"data:image/png;base64,{overlay_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/generate-report")
async def generate_pdf_report(
    file: UploadFile = File(...),
    patient_id: str = Form("PAT-89210"),
    biopsy_site: str = Form("Breast Tissue / Lymph Node"),
    pixel_scale_um: float = Form(0.5)
):
    try:
        contents = await file.read()
        img_bgr = decode_image_bytes(contents)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, TIFF, WEBP")
            
        binary_mask, overlay_bgr = engine.predict(img_bgr)
        metrics = calculate_spatial_metrics(binary_mask, pixel_scale_um=pixel_scale_um)
        grading = predict_tumor_severity_grade(metrics["tumor_burden_percent"])
        
        pdf_bytes = generate_pdf_report_bytes(
            img_bgr=img_bgr,
            overlay_bgr=overlay_bgr,
            metrics=metrics,
            grading_info=grading,
            patient_id=patient_id,
            biopsy_site=biopsy_site
        )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Pathology_Report_{patient_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

@app.post("/api/upload-weights")
async def upload_model_weights(file: UploadFile = File(...)):
    """
    Accepts uploaded ONNX model weights from Colab/Kaggle remote scripts and updates local backend.
    """
    try:
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        save_path = models_dir / file.filename
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
            
        # Trigger dynamic engine reload
        engine._load_model()
        
        return {
            "status": "success",
            "message": f"Successfully updated model weight: {file.filename}",
            "engine_status": "ONNX Active" if engine.onnx_session else "Deconvolution Fallback"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed updating model weights: {str(e)}")

# Mount static frontend directory at root AFTER all API routes
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=True)
