# AGENTS.md - Governance & Operational Protocols

This document defines the strict operational rules and behavioral constraints for Antigravity AI agents managing the **Automated Histopathology Tumor Segmentation & Grading System** inside `/home/xer0/histopath_app/`.

---

## 1. Primary Workspace Constraint

- **Workspace Path Isolation**: ALL project files, scripts, models, frontend assets, backend services, and configs MUST reside inside `/home/xer0/histopath_app/`.
- NO code or output files should be saved outside this directory.

---

## 2. Cost & Deployment Principles (Free & Low-Cost Policy)

1. **Sevalla Deployment Compatibility**:
   - The application MUST be fully containerized using a unified, multi-stage `Dockerfile` and `docker-compose.yml` ready for instant deployment on **Sevalla Application Hosting**.
   - The backend MUST expose a `PORT` environment variable (default `8000` or dynamic `$PORT`) as required by Sevalla.
2. **Zero-Cost Tech Stack**:
   - Use open-source libraries only (PyTorch, FastAPI, OpenCV, ReportLab, Albumentations, ONNX Runtime).
   - Use free Google Colab T4 GPUs for training.
   - Use free public datasets (MoNuSeg, CAMELYON16, GlaS, PANDA).

---

## 3. Engineering & Code Quality Protocols

- **Cross-Platform Handling**: Use `pathlib.Path` or `os.path` for all file operations (compatible with Windows, macOS, Linux).
- **Graceful Fallbacks**:
  - If trained ONNX/PyTorch model weights (`backend/models/tumor_unet.onnx`) are missing during initial deployment, the backend MUST fall back to a deterministic computer vision segmentation model (Color Deconvolution + Morphological Thresholding) so the application remains 100% functional out-of-the-box on Sevalla.
- **Unified FastAPI Deployment**:
  - Mount the `frontend/` directory directly within FastAPI using `StaticFiles` so the app can run either as a unified single-container app on Sevalla or as decoupled microservices.

---

## 4. API Data Contract & Metrics Standard

- **Spatial Calibration**: $0.50 \mu m/\text{pixel}$ (20x magnification).
  - $\text{Tumor Area (\mu m^2)} = \text{Tumor Pixels} \times 0.25$.
  - $\text{Tumor Area (mm^2)} = \frac{\text{Tumor Area (\mu m^2)}}{1,000,000}$.
  - $\text{Tumor Burden \%} = \frac{\text{Tumor Pixels}}{\text{Total Tissue Pixels}} \times 100$.
- **Severity Grading Scale**:
  - **Grade I (Low)**: $< 15.0\%$ tumor area.
  - **Grade II (Moderate)**: $15.0\% - 35.0\%$ tumor area.
  - **Grade III (High / Invasive)**: $> 35.0\%$ tumor area.
- **REST Endpoints**:
  - `GET /api/health`: Status check.
  - `POST /api/predict`: Analyzes uploaded slide, returns JSON metrics & Base64 mask overlays.
  - `POST /api/generate-report`: Streams dynamic PDF report byte stream.

---

## 5. Security & Verification Rules

- Test all endpoints locally using `python -m unittest` or `pytest` before declaring any task complete.
- Maintain credentials in `.env` files (never commit secrets).
