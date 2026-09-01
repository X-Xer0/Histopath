# Master Roadmap: Automated Histopathology Tumor Segmentation & Grading System

> **Target Directory**: `/home/xer0/histopath_app/`  
> **Deployment Target**: **Sevalla Application Hosting** (Free/Low-Cost Containerized Docker Deployment)  
> **Training Platform**: **Google Colab** (Free T4 GPU + Direct Dataset Ingestion)

---

## Ecosystem Architecture Overview

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                           GOOGLE COLAB GPU                                │
 │ - Direct Download: MoNuSeg, CAMELYON16, GlaS, PANDA Datasets             │
 │ - Preprocessing: Macenko Optical Density H&E Stain Normalization          │
 │ - Model Training: PyTorch U-Net with BCE + Dice Loss                      │
 │ - Export: Export lightweight tumor_unet.onnx weight file                  │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │ Sync Model (.onnx)
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                      SEVALLA LIVE DEPLOYMENT ECOSYSTEM                    │
 │                                                                           │
 │ ┌───────────────────────────────────────────────────────────────────────┐ │
 │ │                   UNIFIED DOCKER CONTAINER (PORT $PORT)               │ │
 │ │                                                                       │ │
 │ │  ┌─────────────────────────────────────────────────────────────────┐  │ │
 │ │  │                 FASTAPI BACKEND SERVICE                         │  │ │
 │ │  │ - GET /api/health                                               │  │ │
 │ │  │ - POST /api/predict (ONNX Inference + Fallback CV Engine)       │  │ │
 │ │  │ - POST /api/generate-report (ReportLab PDF Builder)             │  │ │
 │ │  └────────────────────────────────┬────────────────────────────────┘  │ │
 │ │                                   │ Mounts Static Files               │ │
 │ │  ┌────────────────────────────────▼────────────────────────────────┐  │ │
 │ │  │                 GLASSMORPHISM FRONTEND CLIENT                   │  │ │
 │ │  │ - Dark Glassmorphism UI (HTML5 / CSS3 / Vanilla JS)              │  │ │
 │ │  │ - Biopsy Slide Drag & Drop Uploader                             │  │ │
 │ │  │ - Interactive Dual-Canvas Overlay with Opacity Slider            │  │ │
 │ │  │ - Quantitative Metrics Dashboard & Grade Badge                   │  │ │
 │ │  │ - 1-Click Diagnostic PDF Report Downloader                       │  │ │
 │ │  └─────────────────────────────────────────────────────────────────┘  │ │
 │ └───────────────────────────────────────────────────────────────────────┘ │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Execution Phases & Rationale

### Phase 1: Local Machine Workspace Setup (`/home/xer0/histopath_app/`)
- **What**: Create modular directory structure (`backend/`, `frontend/`, `colab/`, `models/`).
- **Why**: Keeps all codebase components cleanly separated and isolated within `/home/xer0/histopath_app/`.

### Phase 2: Google Colab Training & Dataset Synchronization
- **What**: Develop `colab/Histopathology_Tumor_Segmentation_and_Grading.ipynb` to download public histopathology datasets directly into Colab, normalize colors using Macenko OD transform, train U-Net, and export `tumor_unet.onnx`.
- **Why**: Keeps multi-gigabyte dataset processing in Google Cloud for 100% free compute without using local disk space.

### Phase 3: FastAPI Backend & Inference Engine
- **What**: Build Python backend with PyTorch/ONNX inference, spatial area metrics calculator ($\mu m^2 / mm^2$), tumor severity predictor (Grade I-III), and ReportLab clinical PDF generator.
- **Why**: Delivers fast REST endpoints with an automatic fallback computer vision pipeline if model weights are pending export from Colab.

### Phase 4: Glassmorphism Web Frontend
- **What**: Build a responsive dark-themed dashboard with dual-layer canvas slide rendering, opacity slider, real-time metrics cards, and PDF downloader.
- **Why**: Gives pathologists an intuitive visual interface with zero heavy client framework overhead.

### Phase 5: Sevalla Live Deployment & Verification
- **What**: Package application into a Sevalla-optimized `Dockerfile` (listening on `$PORT`), write `docker-compose.yml`, `setup.sh`, and `setup.bat`, and run local test suite (`backend/tests/test_api.py`).
- **Why**: Ensures 1-click live deployment on Sevalla and cross-platform execution on Windows, macOS, and Linux.
