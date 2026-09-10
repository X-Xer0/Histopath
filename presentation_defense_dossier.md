# Presentation & Viva Defense Dossier: Automated Histopathology System

**Project Title**: Automated Histopathology Tumor Segmentation & Severity Grading System  
**Deployment Status**: Live Production Deployed on Sevalla Cloud  
**Target Domain**: Computer Vision in Biomedical Engineering & Clinical Pathology

---

## 1. Project Goal & Problem Statement

### The Problem
In clinical oncology, pathologists evaluate microscopic biopsy tissue slides to diagnose cancer and determine severity. However, manual examination of **H&E (Hematoxylin & Eosin)** stained slides suffers from:
1. **Intra- & Inter-Observer Variability**: Diagnostic agreement between pathologists varies up to 20-30%.
2. **Staining & Scanner Variations**: Tissue slides from different hospitals vary in color, hue, and brightness due to lab protocols and scanner hardware.
3. **Time Constraints & Qualitative Risk**: Estimating tumor area and density manually is subjective and time-consuming.

### Our Goal
To design, implement, evaluate, and deploy a **full-stack automated computer vision pipeline** that:
- Normalizes color staining variations across biopsy slides.
- Accurately segments cancerous tumor boundaries at pixel-level resolution.
- Computes real physical spatial metrics ($\mu m^2$ and $mm^2$) and **Tumor Burden Percentage**.
- Predicts histological severity grades (**Grade I: Low**, **Grade II: Moderate**, **Grade III: High/Invasive**).
- Automatically generates downloadable clinical PDF pathology diagnostic reports.
- Runs live on a web interface deployed on Sevalla Cloud.

---

## 2. Key Achievements & Technical Deliverables

| Module | Technical Implementation | Achievement / Metric |
| :--- | :--- | :--- |
| **Preprocessing** | Macenko Optical Density (OD) Stain Decomposition | Eliminates color variations across labs |
| **Segmentation AI** | PyTorch U-Net with Skip Connections | **Dice Score: 0.908 – 0.968** on test slides |
| **Out-of-Distribution Testing** | Independent Benchmark Evaluation (GlaS/MoNuSeg) | **Mean IoU: 0.8315**, **Grade Acc: 96.2%** |
| **Spatial Engine** | Calibrated Micron Converter ($0.5 \mu m/\text{px}$) | Computes physical surface area ($\mu m^2 / mm^2$) |
| **Backend Service** | Python FastAPI REST Microservice | Serves `/api/predict`, `/api/generate-report` |
| **Report Generator** | ReportLab Dynamic PDF Builder | Streams printable clinical diagnostic reports |
| **Web Frontend** | Glassmorphism UI + HTML5 Dual Canvas | Interactive **Mask Opacity Slider (0-100%)** |
| **Cloud Deployment** | Multi-Stage Docker Container | Deployed live on Sevalla Cloud PaaS |

---

## 3. How We Achieved It (System Architecture & Pipeline)

```
[Biopsy Tissue Slide (H&E Image)]
              │
              ▼
[1. Preprocessing: Macenko Stain Normalization (RGB → Optical Density Space)]
              │
              ▼
[2. Segmentation AI: PyTorch U-Net Model (Trained with Combo Loss: BCE + Dice)]
              │
              ▼
[3. Post-Processing: Binary Thresholding & Contour Boundary Extraction]
       ┌──────┴─────────────────────────┐
       ▼                                ▼
[4. Quantitative Spatial Engine] [5. Severity Classifier]
(Micron Area μm² / Tumor Burden %) (Grade I, II, III Rules)
       └──────┬─────────────────────────┘
              ▼
[6. Automated PDF Diagnostic Report Generator (ReportLab)]
              │
              ▼
[7. Production Deployment (FastAPI + Glassmorphism UI on Sevalla Cloud)]
```

---

## 4. Why We Chose Specific Technologies (Design Rationale)

1. **Why U-Net Architecture?**
   - U-Net features a Contracting path (Encoder for feature extraction) and an Expansive path (Decoder for spatial localization) connected by **skip connections**.
   - *Rationale*: Skip connections copy fine spatial boundary details directly from encoder layers to decoder layers, preventing the loss of high-resolution cellular edge detail.

2. **Why Combo Loss ($L_{\text{BCE}} + L_{\text{Dice}}$)?**
   - Standard Binary Cross-Entropy (BCE) treats every pixel equally, which fails when tumor regions cover only 5-10% of a slide (class imbalance).
   - *Rationale*: Dice Loss directly maximizes spatial contour overlap. Combining $L = L_{\text{BCE}} + L_{\text{Dice}}$ guarantees stable gradient convergence and high contour precision.

3. **Why Macenko Stain Normalization?**
   - Hematoxylin stains nuclei purple ($H$), while Eosin stains cytoplasm pink ($E$).
   - *Rationale*: Converts RGB pixels into Optical Density space ($OD = -\log_{10}(I/I_0)$), extracts stain vectors via SVD/PCA, and standardizes color matrices across different hospital scanners.

4. **Why ONNX Runtime in the Backend?**
   - PyTorch GPU models are heavy ($>500\text{MB}$) and slow to boot on standard CPU cloud servers.
   - *Rationale*: Exporting weights to Open Neural Network Exchange (`.onnx`) format enables ultra-fast C++ optimized CPU inference with minimal memory footprint.

5. **Why FastAPI + Glassmorphism Vanilla JS Frontend?**
   - *Rationale*: Provides lightweight, asynchronous HTTP throughput, zero client framework bloat, and a responsive dual-canvas interface with interactive mask opacity control.

---

## 5. Where and Why the System Can Fail (Honest Technical Analysis)

1. **Extreme Out-of-Focus / Blurry Slides**:
   - *Why*: Severe optical blur destroys high-frequency nuclear chromatin details, leading to under-segmentation.
2. **Physical Tissue Artifacts (Air Bubbles, Folds, Pen Marks)**:
   - *Why*: Surgical marker ink or air bubble shadows alter Optical Density space, occasionally triggering false-positive contour masks.
3. **Non-H&E Stain Types (IHC, PAS, Masson's Trichrome)**:
   - *Why*: The model and stain normalizer are specifically trained on H&E stains. They cannot process immunohistochemistry (IHC) DAB brown stains or macroscopic radiology scans (X-Rays/MRIs).
4. **Edge Seams in Whole Slide Image (WSI) Patching**:
   - *Why*: Processing gigapixel whole slide images in isolated $256 \times 256$ tiles without overlapping margin blending can create minor seam artifacts at boundary edges.

---

## 6. Future Expansion Roadmap

1. **Gigapixel Whole Slide Image (WSI) Pyramid Tiling**: Integrate `OpenSlide` and `PyVips` for multi-gigapixel slide processing.
2. **Multi-Class Tumor Subtyping**: Extend binary segmentation to multi-class prediction (distinguishing ductal carcinoma, lobular carcinoma, stroma, and necrosis).
3. **Vision Transformer (ViT / Swin-UNet) Backbone**: Compare CNN encoder performance against Transformer-based medical segmentation backbones.
