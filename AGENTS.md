# AGENTS.md — Governance & Operational Protocols

Project: **Automated Histopathology Nucleus Segmentation & Morphometry**
Location: `/home/xer0/histopath_app/`

This document is the contract for anyone (human or agent) changing this
codebase. It describes what the system is, what it must never claim, and the
engineering rules that keep it working.

---

## 1. What this system is — and what it must never claim

The system segments **cell nuclei** in H&E stained histopathology slides and
reports quantitative morphometry.

**In scope:** nucleus segmentation, individual nucleus separation, nuclear
density, nucleus count, nuclear size statistics, size distribution, descriptive
cellularity band, CSV and PDF export.

**Out of scope — do not add these back:**

| Removed feature | Why it must not return |
|---|---|
| Tumour burden % | The model detects *all* nuclei; it cannot separate malignant from benign |
| Grade I / II / III | Not supported by the training signal or the annotations |
| Risk category | Would be fabricated |
| Clinical recommendation | Would be fabricated and is a patient-safety risk |
| Ki-67, N/C ratio, mitotic index | Require IHC stains or cytoplasm segmentation |

Any output that could be read as a diagnosis must carry the disclaimer that this
is quantitative image analysis only. Both the PDF report and the web UI already
do this; keep it that way.

---

## 2. Workspace & deployment constraints

1. **Path isolation** — all project files stay inside `/home/xer0/histopath_app/`.
2. **Containerised** — a multi-stage `Dockerfile` plus `docker-compose.yml`, ready
   for Sevalla. The service must listen on `$PORT` (default 8000).

   **Hosting requirement, measured:** Sevalla has no GPU. The process peaks at
   **447 MB RSS** during a 1000×1000 inference (189 MB idle, model loaded), so the
   0.3 GB Hobby pod is killed on the first request. **Standard S1 (0.5 CPU / 1 GB)
   is the minimum**, and expect roughly 1–2 minutes per slide there. Full notes and
   the deployment procedure are in `SEVALLA_DEPLOYMENT.md`.
3. **Zero-cost stack** — open-source libraries only (PyTorch, FastAPI, OpenCV,
   ReportLab, ONNX Runtime). Training on free Colab T4 GPUs. Public datasets.
4. **Cross-platform** — `pathlib.Path` or `os.path` everywhere; no shell-specific
   commands in application code; `setup.sh` and `setup.bat` both maintained.
5. **Unified service** — FastAPI mounts `frontend/` with `StaticFiles`. One
   process, one origin, no CORS hop in production.
6. **Graceful fallback** — if `backend/models/tumor_unet.onnx` is missing, the
   engine switches to hematoxylin colour deconvolution + Otsu thresholding so the
   service still answers. `/api/health` reports which engine is live.

---

## 3. API contract

| Method | Endpoint | Payload | Returns |
|---|---|---|---|
| GET | `/api/health` | — | status, engine, calibration default, scope note |
| POST | `/api/predict` | `file`, `pixel_scale_um=0.5`, `precise_mode=false` | metrics + 4 Base64 PNGs |
| POST | `/api/export-csv` | `file`, `pixel_scale_um`, `precise_mode` | per-nucleus CSV |
| POST | `/api/generate-report` | `file`, `pixel_scale_um`, `precise_mode`, `sample_id` | PDF byte stream |
| POST | `/api/upload-weights` | `.onnx` file | hot-reloads the engine |

`POST /api/predict` returns:

```json
{
  "status": "success",
  "filename": "slide.png",
  "engine": "ONNX U-Net",
  "precise_mode": false,
  "metrics": {
    "pixel_scale_um": 0.5,
    "total_image_pixels": 1000000,
    "tissue_pixels": 1000000,
    "tissue_area_mm2": 0.25,
    "nuclei_pixels": 214000,
    "nuclear_area_um2": 53500.0,
    "nuclear_area_mm2": 0.0535,
    "nuclear_density_percent": 21.4,
    "nuclei_count": 690,
    "nuclei_per_mm2": 2760.0,
    "mean_nuclear_area_um2": 77.5,
    "median_nuclear_area_um2": 71.2,
    "mean_equivalent_diameter_um": 9.6,
    "min_equivalent_diameter_um": 3.8,
    "max_equivalent_diameter_um": 22.1,
    "std_equivalent_diameter_um": 3.3,
    "size_variability_cv_percent": 34.2,
    "size_distribution": { "bin_edges_um": [0, 2, "..."], "counts": [12, 40, "..."] },
    "cellularity": { "category": "Moderate", "detail": "...", "basis": "..." }
  },
  "original_base64": "data:image/png;base64,...",
  "mask_base64": "data:image/png;base64,...",
  "overlay_base64": "data:image/png;base64,...",
  "instance_base64": "data:image/png;base64,..."
}
```

Do not rename `nuclei_count`, `nuclear_density_percent` or the four Base64 image
keys without updating the frontend and the tests together.

---

## 4. Measurement standards

### Spatial calibration — 0.50 µm/pixel (0.25 µm² per pixel)

The MoNuSeg slides were captured at 40× and distributed downsampled to an
effective ~20×. Verified from the annotations: the median annotated nucleus is
19.1 px across, which is 9.57 µm at 0.50 µm/px (correct for an epithelial tumour
nucleus) and 4.78 µm at 0.25 µm/px (smaller than a lymphocyte — impossible across
seven carcinoma types). Do not "correct" this to 0.25 without re-deriving the
evidence.

Formulas:

```
nuclear area (µm²)  = nucleus pixels × 0.25
nuclear area (mm²)  = nuclear area (µm²) / 1,000,000
nuclear density (%) = nucleus pixels / tissue pixels × 100
nuclei per mm²      = nuclei count / tissue area (mm²)
equivalent diameter = 2 × sqrt(area / π)
```

**Tissue pixels**, not total image pixels, is the density denominator — glass
background must not dilute the measurement.

### Cellularity bands (descriptive only)

Low `< 23.0 %`, Moderate `23.0 – 30.5 %`, High `> 30.5 %`.

These are the terciles of the density this pipeline measures across the 37
MoNuSeg 2018 training slides (mean 28.3 %, median 29.1 %, range 11.7 – 44.8 %).
They describe the pipeline's own output distribution, **not** a clinical
reference range. Never attach risk or recommendation language to them.

### Detection bias — always disclosed

Measured against the official MoNuSeg 2018 test set:

* ~86 % of annotated nuclear **pixels** are recovered.
* Detected **counts** run roughly 20 % above the annotated count (per-slide range
  103 % – 167 %). Part of that is genuinely separating touching nuclei; part is
  segmentation speckle, and it varies with how densely the tissue is packed.

The UI and the PDF both state this. Keep it there.

---

## 5. Model & training rules

Current model: U-Net, 4 resolution levels, base 64 filters (32/64/128/256 then
mirrored), 7,700,161 parameters, 256×256 RGB input, sigmoid output,
single-file ONNX (opset 11, ~29.4 MB).

> **A note on the filename.** The weight file is still called `tumor_unet.onnx`
> because the training notebook in `colab/` exports it under that name and the two
> must stay in sync. The model does not detect tumours — it detects nuclei. If the
> file is ever renamed, update `MODEL_PATH` in `backend/app/model.py`, the export
> cell in the notebook, and the `/api/health` response together.

Training configuration that produced the deployed weights:

| Setting | Value | Why |
|---|---|---|
| Loss | BCE + Dice (Tversky α=β=0.5) | Balanced and stable |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 | Stable for a 7.7 M-parameter model |
| Schedule | CosineAnnealingLR, T_max = epochs, eta_min 1e-6 | Decays smoothly; the plateau scheduler never fired |
| Gradient clipping | max-norm 1.0 | A single bad batch otherwise destroys the model |
| Epochs | 50 | Best checkpoint landed at epoch 40 |
| Batch size | 8 | Fits the T4 comfortably |
| Mixed precision | **OFF** | fp16 autocast produced NaN gradients |
| Validation split | **patient-wise**, organ-stratified | Patch-level splits leak and inflate the score |

**Four hard-won rules — do not undo them:**

0. **Never build the tile grid with `range(0, length - tile + 1, stride)`.** It
   stops at the last stride multiple and leaves the far edge unprocessed. For a
   1000 px slide with tile 256 and stride 192 it produced origins
   0/192/384/576, ending at 832, so the final 168 px band — 31 % of the image —
   was never seen and was silently reported as background. That single line
   accounted for 70 % of all missed nuclei and held the reported Dice at 0.688
   instead of 0.815. Use `_tile_origins()` in `backend/app/model.py`, which
   appends a final tile flush with the edge; `test_api.py::TestTilingCoverage`
   guards it.

1. **Never re-enable AMP** without re-testing. With autocast on, the gradient norm
   became NaN from the second epoch, the GradScaler decayed 65536 → 0 and training
   collapsed. With AMP off the gradient norm stays ~0.65–1.1 and Dice climbs
   monotonically. fp32 is ~2× slower and actually works.
2. **Never use Tversky β > 0.5 for the deployed loss.** β = 0.7 rewards
   over-prediction so strongly that the model collapses into an "almost everything
   is foreground" state it never escapes. Precision/recall is tuned at the decision
   threshold instead.
3. **Never split train/validation by patch.** Tiles overlap by 27 % and share
   patients; a patch-level split reports ~0.86 validation Dice that does not
   survive contact with unseen patients.

Additional training facts: 925 patches (25 per slide × 37 slides, 256², stride
186), 29 train / 8 validation patients, 725 / 200 patches.

---

## 6. Verification requirements

Run before declaring any task complete:

```bash
python -m pytest backend/tests/test_api.py -q     # 22 tests, must all pass
python backend/tests/evaluate_model.py            # benchmark on samples/
```

Also required when relevant:

* Re-build `docker-compose.yml` after any dependency or path change.
* Re-run the benchmark after any change to `model.py`, `instances.py` or
  `metrics.py`, and record the new numbers — never claim an improvement without
  the before/after figures.
* Never state that a component works without having executed it.

---

## 7. Data provenance

* **Training** — MoNuSeg 2018 training set, 37 slides (30 patients are documented
  in the official organ-information PDF; **7 slides are not covered by that
  metadata** and are reported as "Unlisted" in the EDA).
* **Benchmark** — official MoNuSeg 2018 test set, 14 slides, organs include
  Testis, Brain and Thyroid which do not appear in the training set.
* **License** — MoNuSeg is CC BY-NC-SA 4.0. Attribution required; non-commercial;
  share-alike. Keep the attribution in any redistribution.
* **Reference implementations used** — the official MATLAB
  `he_to_binary_mask_final.m` (XML polygon → mask, ported to Python) and
  `Aggregated_Jaccard_Index_v1_0.m` (AJI, ported to Python in
  `backend/tests/evaluate_model.py`).

Note the case-sensitivity trap that caused a real bug: the repository's training
`Tissue Images` are pre-sliced patches named `0.png…`, while `Annotations` are
named `TCGA-*.xml`. Deriving annotation paths from image stems silently fails.
Always resolve annotations from the annotations directory.

---

## 8. Security

* Credentials live in `.env` files and are never committed.
* The Colab CLI token lives at `~/.config/colab-cli/token.json`; never copy it
  into the repository.
* Uploaded slides are processed in memory and are not persisted server-side.
