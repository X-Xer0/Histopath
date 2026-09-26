# HistologyAI — Automated Nucleus Segmentation & Morphometry

A full-stack system that detects **cell nuclei** in H&E stained histopathology
slides and reports quantitative morphometry: how much of the tissue is covered by
nuclei, how many nuclei there are, how large they are, and how variable that size is.

Trained on the **MoNuSeg 2018** multi-organ dataset and benchmarked on its
official held-out test set.

> **Scope.** This is a quantitative image-analysis tool. The segmentation model
> detects nuclei and does **not** classify tissue as benign or malignant. Nothing
> this system produces is a diagnosis or a treatment recommendation.

---

## What it does

| Capability | Detail |
|---|---|
| Nucleus segmentation | U-Net (7.7 M parameters) exported to ONNX, tiled inference with overlap blending |
| Individual nuclei | Distance-transform seeding + nearest-seed partition, so touching nuclei are counted separately |
| Nuclear density | Share of tissue area covered by nuclei |
| Nucleus count | Objects detected, plus nuclei per mm² |
| Nuclear morphometry | Mean/median area, equivalent diameter, min–max range, size variability (CV) |
| Size distribution | Histogram of equivalent nuclear diameters |
| Cellularity band | Descriptive Low / Moderate / High, from terciles of the measured reference distribution |
| Exports | Per-nucleus CSV and a PDF morphometry report |
| Accuracy modes | Standard (one pass) and Precise (8× test-time augmentation) |
| Graceful fallback | Stain-deconvolution + Otsu engine if the ONNX weights are missing |

**Not supported** (by design — the model cannot do these): tumour vs benign
classification, tumour burden, histological grading, risk categories, treatment
recommendations, Ki-67 or nuclear-to-cytoplasm ratio.

---

## Results

Benchmark on the **official MoNuSeg 2018 test set** — 14 slides from patients
never seen in training, evaluated at a decision threshold of 0.50:

| Metric | Value |
|---|---|
| Mean Dice | **0.8146 ± 0.0497** |
| Mean IoU (Jaccard) | 0.6900 |
| Mean AJI (instance-level) | 0.5849 |
| Mean Precision | 0.7802 |
| Mean Recall / Sensitivity | 0.8589 |
| Mean Specificity | 0.9341 |
| Best validation Dice (patient-wise split) | 0.8534 |

Reproducibility: three independent 50-epoch runs produced validation Dice of
0.8525 / 0.8529 / 0.8534.

**Generalisation across organs.** Unseen organs — Thyroid 0.855, Testis 0.800,
Brain 0.751 — averaged 0.789 against 0.825 for organs present in training
(Bladder 0.847, Kidney 0.846, Lung 0.838, Prostate 0.810, Colon 0.803,
Breast 0.787). A two-point gap: the model learned general nuclear morphology
rather than organ-specific appearance.

**Known limitations.** Recall is 0.86, so about 14 % of annotated nuclear pixels
are still missed, and the pipeline detects roughly 20 % more nuclei than were
annotated (partly genuine separation of touching nuclei, partly speckle). Breast
and brain are the weakest organs; brain is dragged down by one slide
(TCGA-HT-8564, Dice 0.668) where precision falls to 0.55. Instance metrics are
below a true instance-segmentation method because nuclei are separated from a
binary mask rather than predicted as instances. See
[`presentation_defense_dossier.md`](presentation_defense_dossier.md) for the
full analysis.

> **Fixed during development.** An earlier revision of the tiled inference built
> its tile grid with `range(0, h - tile + 1, stride)`, which stops at the last
> stride multiple. On a 1000 px slide that left the final 168 px band — 31 % of
> the image — never processed, and those pixels were silently reported as
> background. It accounted for 70 % of all missed nuclei and held the score at
> Dice 0.688. Fixing the grid moved Dice to 0.815 with no retraining.

---

## Running it

### Docker (recommended)

```bash
docker compose up --build
# open http://localhost:8000
```

### Deploying to Sevalla

See [`SEVALLA_DEPLOYMENT.md`](SEVALLA_DEPLOYMENT.md). Three things to know up
front:

* **No GPU exists on Sevalla** — CPU containers only.
* The app peaks at **447 MB of RAM** during a 1000×1000 inference, so the $5
  Hobby pod (0.3 GB) is killed on the first request. Use Standard S1 (1 GB) or
  larger, and expect roughly 1–2 minutes per slide there.
* **Check `version` in `/api/health` after deploying.** A container that is still
  serving an older build looks exactly like an application bug; comparing that
  field against `backend/app/main.py` takes two seconds and has already saved a
  wasted debugging session.

Thread count is derived from the container's cgroup quota so ONNX Runtime does
not oversubscribe a fractional CPU. `ORT_NUM_THREADS` overrides it; changing it
does not affect the measurements.

### Directly

```bash
./setup.sh            # or setup.bat on Windows
source venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The frontend is served by the same FastAPI process, so there is a single
service and a single origin.

### Verify

```bash
python -m pytest backend/tests/test_api.py -q       # 20 unit + API tests
python backend/tests/evaluate_model.py              # benchmark on samples/
python backend/tests/evaluate_model.py --precise    # benchmark with 8x TTA
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service status, active engine, calibration default |
| **POST** | **`/api/analyse`** | **Queue an analysis, returns a `job_id` in milliseconds** |
| **GET** | **`/api/job/{id}`** | **Poll a job; returns the measurements and overlay images when done** |
| **GET** | **`/api/job/{id}/csv`** | Per-nucleus CSV from a finished job |
| **GET** | **`/api/job/{id}/report`** | Morphometry PDF from a finished job |
| POST | `/api/predict` | Synchronous segmentation (small images, local use) |
| POST | `/api/export-csv` | Per-nucleus measurements as CSV (synchronous) |
| POST | `/api/generate-report` | Nucleus morphometry report as a PDF stream (synchronous) |
| POST | `/api/upload-weights` | Replace the ONNX model and hot-reload, no restart |

**Why the UI polls rather than waits.** A full slide takes 25–80 s and the hosting
proxy returns 504 after about 15 s, so a synchronous request cannot survive in
production. The browser queues the work and polls. Because the finished job holds
its result, the CSV and PDF downloads reuse that same analysis instead of running
inference again.

`/api/predict` accepts `file`, `pixel_scale_um` (default `0.5`) and
`precise_mode` (default `false`), and returns:

```json
{
  "status": "success",
  "engine": "ONNX U-Net",
  "precise_mode": false,
  "metrics": {
    "nuclei_count": 690,
    "nuclear_density_percent": 21.4,
    "nuclear_area_um2": 53500.0,
    "nuclear_area_mm2": 0.0535,
    "tissue_area_mm2": 0.25,
    "nuclei_per_mm2": 2760.0,
    "mean_nuclear_area_um2": 77.5,
    "median_nuclear_area_um2": 71.2,
    "mean_equivalent_diameter_um": 9.6,
    "min_equivalent_diameter_um": 3.8,
    "max_equivalent_diameter_um": 22.1,
    "size_variability_cv_percent": 34.2,
    "size_distribution": { "bin_edges_um": [0, 2, 4], "counts": [12, 40, 61] },
    "cellularity": { "category": "Moderate", "detail": "...", "basis": "..." }
  },
  "original_base64": "data:image/png;base64,...",
  "mask_base64": "data:image/png;base64,...",
  "overlay_base64": "data:image/png;base64,...",
  "instance_base64": "data:image/png;base64,..."
}
```

---

## Spatial calibration

**0.50 µm per pixel**, i.e. 0.25 µm² per pixel.

The MoNuSeg slides were captured on a 40× microscope, but the distributed
1000×1000 crops are downsampled to an effective ~20×. This was verified from the
dataset's own annotations: the median annotated nucleus measures 19.1 px across,
which is **9.57 µm at 0.50 µm/px** — the expected size of an epithelial tumour
nucleus — and only 4.78 µm at 0.25 µm/px, which would be smaller than a
lymphocyte and is impossible across seven carcinoma types.

Nuclear density, counts and cellularity bands are scale-independent; only the
µm²/mm² figures depend on it. The API accepts `pixel_scale_um` so a different
scanner can supply its own calibration.

---

## Repository layout

```
backend/
  app/
    main.py             FastAPI service and endpoints
    model.py            ONNX tiled inference, precise mode, fallback engine
    instances.py        nucleus instance separation
    metrics.py          nucleus morphometry and calibration
    report_generator.py PDF report
    stain_norm.py       stain deconvolution (fallback engine)
  models/               tumor_unet.onnx
  tests/                test_api.py, evaluate_model.py
frontend/               index.html, style.css, app.js
colab/                  training notebook, smoke test, CLI runner
samples/                MoNuSeg test slides + reference masks
```

---

## Training

The model is trained in `colab/Histopathology_Tumor_Segmentation_and_Grading.ipynb`
on a free Colab T4 GPU. `colab/run_on_colab.sh` drives the whole cycle from a
local terminal using the official `google-colab-cli`, and `colab/smoke_test.py`
validates the pipeline end-to-end in about five minutes before any long run.
See [`execution_guide.md`](execution_guide.md) for the step-by-step procedure and
[`roadmap.md`](roadmap.md) for the architecture and operating constraints.
