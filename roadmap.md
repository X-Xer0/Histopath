# Master Roadmap — Automated Nucleus Segmentation & Morphometry

> **Target directory**: `/home/xer0/histopath_app/`
> **Deployment**: Sevalla application hosting (free-tier container, CPU only)
> **Training**: Google Colab, free T4 GPU
> **Benchmark**: official MoNuSeg 2018 test set (14 unseen slides)

---

## 1. What the system is

A full-stack pipeline that takes an H&E stained histopathology slide and returns
quantitative nuclear morphometry: how many nuclei are present, how much of the
tissue they cover, how large they are, and how variable that size is.

It does **not** diagnose disease. The segmentation model detects nuclei; it has
no way to tell malignant from benign, so tumour burden, histological grading,
risk categories and treatment recommendations are deliberately out of scope.

---

## 2. Architecture

```
   GOOGLE COLAB (free T4)
   ├── MoNuSeg 2018 training slides + XML polygon annotations
   ├── 1000x1000 slides -> 256x256 patches (stride 186, 27% overlap)
   ├── U-Net (base 64, 7.7 M parameters) trained from scratch
   └── export -> single-file tumor_unet.onnx (opset 11, 29.4 MB)
                    |
                    v
   SEVALLA / DOCKER (single container, $PORT)
   ┌──────────────────────────────────────────────────────────────┐
   │  FastAPI service                                             │
   │   GET  /api/health            engine + calibration status    │
   │   POST /api/predict           metrics + 4 Base64 PNG layers  │
   │   POST /api/export-csv        per-nucleus measurements       │
   │   POST /api/generate-report   PDF morphometry report         │
   │   POST /api/upload-weights    hot-swap weights, no restart   │
   │                                                              │
   │  Inference engine (CPU, ONNX Runtime)                        │
   │   256x256 tiles, stride 192, overlap-averaged probability map │
   │   threshold 0.50 -> binary mask                              │
   │   distance-transform seeds -> nearest-seed instance partition │
   │   fallback: hematoxylin deconvolution + Otsu if no weights    │
   │                                                              │
   │  StaticFiles mount -> frontend (index.html, style.css, app.js)│
   │   viewer with original/mask/overlay/per-nucleus views,        │
   │   opacity control, morphometry dashboard, size histogram,     │
   │   CSV + PDF export, precise-mode toggle                       │
   └──────────────────────────────────────────────────────────────┘
```

---

## 3. Phases completed

### Phase 1 — Workspace and backend core
Modular layout (`backend/app`, `backend/models`, `backend/tests`, `frontend`,
`colab`, `samples`), `setup.sh` / `setup.bat`, and the FastAPI service with
`/api/health`, `/api/predict`, `/api/generate-report` and `/api/upload-weights`.

### Phase 2 — Data understanding and training in Colab
- Local MoNuSeg 2018 data (37 training slides, 14 test slides) uploaded rather
  than cloned, so the pipeline trains on the real, complete annotations.
- XML polygon annotations converted to masks with a Python port of the official
  MATLAB routine.
- Exploratory analysis: 37 slides, 24,140 annotated nuclei, mean nuclear density
  24.5 %, density varying threefold across organs (Kidney 15.7 % → Stomach 34.5 %).
- Microscope calibration validated from the annotations themselves (0.50 µm/px).

### Phase 3 — Model and training
U-Net with skip connections, balanced BCE + Dice loss, AdamW at 3e-4, cosine
learning-rate decay, gradient clipping, 50 epochs, patient-wise organ-stratified
validation split. Best validation Dice **0.8534**.

### Phase 4 — Benchmark and hardening
Official test-set evaluation over all 14 unseen slides with a decision-threshold
sweep: Dice **0.8146 ± 0.0497**, IoU 0.6900, AJI 0.5849, precision 0.7802,
recall 0.8589, specificity 0.9341, detecting 7,938 nuclei against 6,697
annotated. Validation Dice agreed within 0.001 across three independent runs.

An inference bug found late in the project had been costing 0.13 Dice: the tile
grid stopped 168 px short of the slide edge, so 31 % of every image was never
processed. See §5.

### Phase 5 — Nucleus application
Backend and frontend rewritten around nuclei. Removed everything the model cannot
support (tumour burden, grading, risk, recommendations) and added instance
separation, nuclear morphometry, size distribution, per-nucleus CSV export and a
PDF morphometry report.

---

## 4. State of the art assessment

| Aspect | Status |
|---|---|
| Segmentation quality | Dice 0.815 — in line with, and for a from-scratch 7.7 M-parameter model above, typical published U-Net baselines |
| Cross-organ generalisation | **Strong.** Unseen organs scored 0.789 vs 0.825 for organs present in training — a two-point gap |
| Reproducibility | **Strong.** Validation Dice within 0.1 % across three independent runs |
| Recall | **Good.** 0.86 — about 14 % of annotated nuclear pixels are missed |
| Instance metrics | **Moderate.** AJI 0.585; nuclei are separated from a binary mask rather than predicted as instances |
| Calibration | Verified from the data, documented, and exposed as an API parameter |

---

## 5. Known limitations

1. **Residual recall.** Pixel recall is 0.86; the remaining misses are mostly
   small or faintly stained nuclei, which the Dice loss under-weights. Threshold
   tuning cannot recover them — the sweep is flat within 0.4 % — so it needs a
   boundary-aware loss or more patches.
2. **Detection bias.** Detected counts run about 20 % above the annotated count
   (per-slide 103 %–167 %): partly genuine separation of touching nuclei, partly
   speckle. Counts and density are relative measurements, not absolute.
3. **Tumour vs benign.** The model segments all nuclei. Any claim about
   malignancy would require tumour-region annotations, which MoNuSeg does not have.
4. **Scale.** Slides are processed whole, not as gigapixel pyramids.
5. **Single stain.** H&E only; IHC and special stains are out of scope.
6. **Breast tissue** is consistently the weakest organ (0.659 across all runs);
   the cause has not been isolated.

---

## 6. Future work, ranked by expected value

1. **Boundary-aware loss.** Add a term that emphasises nucleus boundaries to
   directly attack the 2.7:1 false-negative-to-false-positive ratio.
2. **More patches.** Reduce the tiling stride from 186 to 128, roughly 1,300
   patches instead of 925 — nearly free at training time.
3. **Instance segmentation head.** HoVer-Net or a Mask R-CNN style model would
   raise AJI substantially, because nuclei would be predicted as instances rather
   than separated after the fact.
4. **Whole-slide support.** OpenSlide / PyVips pyramid tiling for gigapixel slides.
5. **Stain-robust augmentation** in optical-density space to narrow the gap
   between validation and unseen-patient performance.
6. **Pretrained encoder comparison** (ResNet34 via segmentation-models-pytorch)
   against the from-scratch U-Net, reported honestly as a different setup.

---

## 7. Operating economics

| Item | Cost |
|---|---|
| Training | Free — Colab T4 GPU |
| Dataset | Free — MoNuSeg 2018, CC BY-NC-SA 4.0 (attribution required) |
| Runtime | Container on Sevalla. **No GPU exists there**; CPU inference only. Cheapest pod that can serve the app is Standard S1 (0.5 CPU / 1 GB) at $10/month — the $5 Hobby pod (0.3 GB) is OOM-killed because inference peaks at 447 MB |
| Libraries | Open source only — PyTorch, ONNX Runtime, OpenCV, FastAPI, ReportLab |

Inference cost: about 20–30 s per 1000×1000 slide on four unloaded CPU cores in
standard mode (16 tiles), and about 8× that in precise mode. A free container
host with 1–2 vCPUs is 2–4× slower again.

Measured dead ends, so they are not retried blind: INT8 dynamic quantization made
inference 4.5× slower (the integer convolution path needs VNNI; without it ONNX
Runtime falls back to a scalar kernel), and larger tiles were slower overall
because the coarser grid covers 44 % more pixels for identical accuracy. The
free wins were dropping the tile overlap from 64 px to 8 px — half the tiles, no
loss in Dice — and skipping bare-glass tiles outright. The remaining options all
cost accuracy: a distilled or smaller model, or downscaled input.
