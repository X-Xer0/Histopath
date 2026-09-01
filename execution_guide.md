# Self-Execution Guide: Step-by-Step Implementation Strategy

> **Agent Execution Protocol**: This document outlines the exact execution sequence I (Antigravity AI Agent) will follow to construct, verify, and package the full-stack system inside `/home/xer0/histopath_app/`.

---

## Step 1: Workspace & Directory Structuring
1. Create all subdirectories:
   - `backend/app/`
   - `backend/models/`
   - `backend/tests/`
   - `frontend/`
   - `colab/`
2. Create environment setup scripts: `setup.sh` (Linux/Mac) and `setup.bat` (Windows).

---

## Step 2: Backend Core Implementation
1. **`backend/app/stain_norm.py`**: Implement H&E stain decomposition and Macenko Optical Density color normalization.
2. **`backend/app/metrics.py`**: Implement spatial area formulas ($\mu m^2$, $mm^2$, Tumor Burden %).
3. **`backend/app/grading.py`**: Implement Grade I, II, III severity predictor.
4. **`backend/app/model.py`**: Implement ONNX model loader + Computer Vision Deconvolution fallback engine.
5. **`backend/app/report_generator.py`**: Implement ReportLab PDF generator microservice.
6. **`backend/app/main.py`**: Create FastAPI app with CORS middleware, `/api/health`, `/api/predict`, `/api/generate-report`, and `StaticFiles` mounting for `frontend/`.

---

## Step 3: Frontend Web Application Implementation
1. **`frontend/index.html`**: Build semantic HTML5 layout with upload dropzone, dual canvas viewer, opacity slider, metric cards, and report download trigger.
2. **`frontend/style.css`**: Define CSS tokens, glassmorphism backdrop filters, responsive grid layout, status badges, and loading spinner animations.
3. **`frontend/app.js`**: Implement drag-and-drop file upload, dual-layer canvas rendering, opacity slider event listeners, API fetch client, and PDF binary blob download handler.

---

## Step 4: Sevalla Deployment Packaging & Verification
1. **`Dockerfile`**: Write multi-stage build exposing `$PORT` for Sevalla deployment.
2. **`docker-compose.yml`**: Write local docker-compose configuration.
3. **`backend/tests/test_api.py`**: Write test suite testing health check, prediction output shape, metric accuracy, and PDF byte generation.
4. **Execution Verification**: Execute test suite via `pytest` / `python3 -m unittest` to confirm 100% test pass before completion.
