# Execution Guide — Build, Run, Train and Verify

Step-by-step procedure for the nucleus segmentation and morphometry system.
Every command below has been executed; the numbers quoted come from the files
they produce.

---

## Step 1 — Local environment

```bash
cd /home/xer0/histopath_app
./setup.sh                 # Linux / macOS      (setup.bat on Windows)
source venv/bin/activate
```

This creates `venv/` and installs `backend/requirements.txt`: FastAPI, Uvicorn,
NumPy, OpenCV (headless), ReportLab, Pydantic, ONNX Runtime, Pillow, pytest.

---

## Step 2 — Verify the backend

```bash
python -m pytest backend/tests/test_api.py -q
# 22 passed
```

The suite covers the measurement maths (area, density, nuclei per mm², size
distribution), the instance separator (split, merge, speck removal, compact
labelling), tile-grid coverage, and every endpoint (`/api/health`,
`/api/predict` schema, `precise_mode`, invalid-image rejection, CSV, PDF,
weight-upload validation).

Then benchmark the deployed ONNX model against the reference masks in `samples/`:

```bash
python backend/tests/evaluate_model.py             # standard mode
python backend/tests/evaluate_model.py --precise   # 8x test-time augmentation
```

Latest result on the six sample slides: Dice 0.8151, IoU 0.6890, AJI 0.5685,
precision 0.7930, recall 0.8432, 3,576 nuclei detected against 3,146 annotated.
The authoritative run is the full 14-slide official test set:

```bash
python backend/tests/evaluate_model.py --official   # ~10 min on four cores
```

which gives **Dice 0.8146 ± 0.0497, IoU 0.6900, AJI 0.5849, precision 0.7802,
recall 0.8589, specificity 0.9341**, and detects 7,938 nuclei against 6,697
annotated (118 %).

---

## Step 3 — Run the service

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

The FastAPI process serves both the API and `frontend/`, so production is a
single container and a single origin.

---

## Step 4 — Backend module map

| File | Responsibility |
|---|---|
| `backend/app/main.py` | FastAPI app, endpoints, image decoding, shared `analyse_slide()` pipeline |
| `backend/app/model.py` | ONNX session, tiled overlap-averaged inference, 8× TTA precise mode, stain-deconvolution fallback |
| `backend/app/instances.py` | Distance-transform seeding + nearest-seed partition to separate touching nuclei |
| `backend/app/metrics.py` | Nuclear morphometry, cellularity bands, per-nucleus rows for CSV |
| `backend/app/report_generator.py` | ReportLab PDF morphometry report |
| `backend/app/stain_norm.py` | Hematoxylin colour deconvolution used by the fallback engine |

Inference is CPU-only by design: the ONNX graph loads in seconds on a container
with no CUDA. A 1000×1000 slide needs 16 tiles at tile 256 / stride 248, which
is roughly 20–30 s on four unloaded cores (1.3–2 s per tile) and about 8× that
in precise mode. A 1–2 vCPU container host is 2–4× slower again, so budget a
couple of minutes per slide on a free tier.

Three things measured rather than assumed:

* **INT8 dynamic quantization is not worth it here.** It shrank the model from
  29.4 MB to 9.4 MB but inference went from 2.4 s to 10.5 s per tile, because
  ONNX Runtime falls back to a scalar ConvInteger kernel on CPUs without VNNI
  (Ice Lake or newer). With a 30 MB model there is nothing to gain, so it is not
  shipped — the attempt is recorded rather than the artefact.
* **Larger tiles are slower overall.** The network is fully convolutional so
  512 px tiles score identically, but the coarser grid processes 44 % more pixels.
* **Less overlap is free.** Stride 248 (8 px overlap) ties stride 192 (64 px) on
  Dice while needing 16 tiles instead of 25 — half the time. Bare-glass tiles are
  skipped entirely, which matters most on real slides with wide glass margins.

**Tuning found by measurement:** ONNX Runtime's default thread selection was the
fastest option (1275 ms/tile); forcing 4 or 8 intra-op threads was slower
(1899 / 2155 ms) because the machine has 4 cores and oversubscription costs more
than it gains. Do not "optimise" this without measuring.

---

## Step 5 — Docker deployment (Sevalla)

```bash
docker compose up --build      # local
```

The multi-stage `Dockerfile` installs dependencies in a builder layer and copies
only `backend/` and `frontend/` into the runtime. The service reads `$PORT`
(default 8000) which is what Sevalla provides.

`docker-compose.yml` bind-mounts `./backend/models` so weights can be replaced
without rebuilding the image.

---

## Step 6 — Training on Colab (free T4)

Training runs in `colab/Histopathology_Tumor_Segmentation_and_Grading.ipynb`.
The notebook is self-contained: it unpacks the dataset, profiles it, builds
patches, trains, benchmarks and exports the ONNX model.

### Option A — drive it from this machine with the official Colab CLI

```bash
./colab/run_on_colab.sh auth        # one-time Google sign-in (prints a URL to paste a code into)
./colab/run_on_colab.sh up          # allocate a T4 runtime
./colab/run_on_colab.sh push-data   # upload colab/_data_zip/monuseg_local.zip and unpack
./colab/run_on_colab.sh deps        # install python packages on the VM
./colab/run_on_colab.sh smoke       # ~5 min end-to-end pipeline check — always run this first
./colab/run_on_colab.sh train       # the full 50-epoch run (hours)
./colab/run_on_colab.sh fetch       # download the model and logs
./colab/run_on_colab.sh down        # release the VM
```

**Upload caveat.** Colab's web uploader truncates files above roughly 100 MB. The
160 MB dataset archive is therefore split into four 40 MB parts
(`_data_zip/monuseg_local.zip.part-00 … -03`); concatenate them on the VM before
extracting, and verify with

```
size : 167,664,881 bytes
sha256: 316113257924189913c1bc075c60148118dd98b2501212188c9679a03d10ddd4
```

If the size differs, the upload was truncated — re-upload that part.

**Session caveat.** On the free tier the VM is released as soon as a CLI command
finishes, so download the model immediately in the same command chain, or use
`colab run --keep` which holds the VM open afterwards.

### Option B — run it in the Colab web UI

1. Open the notebook in Colab, set Runtime → T4 GPU.
2. Upload the dataset zip (or its four parts, then concatenate) to `/content/`.
3. Runtime → Run all. Download `tumor_unet.onnx` from the Files panel when it
   finishes.

The browser tab keeps the VM alive, so nothing is lost.

### What the notebook does

| Step | Contents |
|---|---|
| 1 | Environment setup, seed, GPU check (mixed precision reported as disabled) |
| 2 | Unpack the uploaded archive, parse MoNuSeg XML polygons into masks, slice 1000² slides into 256² patches at stride 186 |
| 3 | EDA + microscope-calibration evidence |
| 4 | Augmentation, patient-wise organ-stratified split, DataLoaders |
| 5 | U-Net (base 64) and the balanced BCE + Dice loss |
| 6 | 50 epochs, AdamW 3e-4, cosine schedule, gradient clipping, early stopping |
| 7 | Benchmark on the official test set: threshold sweep, 8× TTA, per-slide and per-organ tables, FP/FN counts, AJI |
| 8 | Single-file ONNX export (legacy exporter pinned, opset 11) |

### Training configuration that works

| Setting | Value |
|---|---|
| Split | patient-wise, organ-stratified — 29 train / 8 validation patients |
| Patches | 925 total (25 per slide), 725 train / 200 validation |
| Epochs | 50, best checkpoint at epoch 40 |
| Loss | BCE + Dice (Tversky α = β = 0.5) |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | CosineAnnealingLR to 1e-6 |
| Gradient clipping | max-norm 1.0 |
| Mixed precision | **off** — fp16 autocast gives NaN gradients |
| Best validation Dice | 0.8534 |
| Test Dice (14 unseen slides, corrected) | 0.8146 |

### Results

| Metric | Value |
|---|---|
| Test Dice (14 unseen slides) | 0.8146 ± 0.0497 |
| Test IoU | 0.6900 |
| Test AJI | 0.5849 |
| Precision / Recall | 0.7802 / 0.8589 |
| Specificity | 0.9341 |
| Nuclei detected | 7,938 of 6,697 annotated (118 %) |
| Best threshold | 0.50 (sweep 0.30–0.70) |

Validation Dice across three independent 50-epoch runs: 0.8525 / 0.8529 /
0.8534.

---

## Step 7 — Deploying new weights

```bash
# from the machine holding the freshly exported model
curl -F "file=@tumor_unet.onnx" http://<host>:8000/api/upload-weights
```

The engine reloads in place — no restart. `/api/health` then reports
`"engine": "ONNX U-Net"`.

---

## Step 8 — Things that were measured, and the conclusions

These were all found the hard way; they are recorded so they are not repeated.

1. **The tile grid must reach the far edge.** `range(0, h - tile + 1, stride)`
   stops at the last stride multiple: for a 1000 px slide with tile 256 and
   stride 192 it produced 0/192/384/576, ending at 832, so the final 168 px band
   — 31 % of the image — was never processed and was silently reported as
   background. That band held **70 % of all missed nuclei**. Fixing it moved the
   official test score from Dice 0.688 to 0.815 with no retraining. Use
   `_tile_origins()` and keep the `TestTilingCoverage` regression tests.
2. **Dropping the overlap is free accuracy-wise.** Stride 248 (8 px overlap)
   ties stride 192 (64 px overlap) on Dice — 0.8236 vs 0.8237 over three slides —
   while needing 16 tiles instead of 25.
3. **INT8 quantization is slower here, not faster.** Dynamic quantization shrank
   the model from 29.4 MB to 9.4 MB but inference went from 2.4 s to 10.5 s per
   tile, because the integer convolution path needs VNNI to be fast and falls
   back to a scalar kernel without it. Measure before shipping a quantized graph.
4. **Larger tiles do not help.** The network is fully convolutional, so 512 px
   tiles work (and score identically), but the coarser grid processes 44 % more
   pixels and ends up slower than 256 px tiling.
5. **Mixed precision destroys this model.** With autocast enabled the gradient
   norm is NaN from the second epoch and the GradScaler decays 65536 → 0. With it
   off the gradient norm stays ~0.65–1.1 and Dice improves monotonically.
6. **Tversky β = 0.7 collapses training.** Weighting false negatives that heavily
   rewards predicting everything; the model settles into an "almost all
   foreground" state. Balanced Dice is stable, and precision/recall is tuned at
   the decision threshold instead.
7. **The LR schedule matters more than the LR.** The first 25-epoch run never
   decayed the learning rate and was still improving at the final epoch. Cosine
   decay over 50 epochs converged properly.
8. **Patch-level validation splits lie.** Tiles overlap 27 % and share patients;
   a patch-level split reported 0.86 validation Dice that did not survive unseen
   patients. Patient-wise splits are mandatory.
9. **Instance separation needs a Voronoi partition, not watershed.** OpenCV's
   watershed on a synthetic distance surface misassigns symmetric touching blobs
   (two overlapping circles came out as one region of 2593 px and one of 9 px).
   `distanceTransformWithLabels` with `DIST_LABEL_CCOMP` gives the correct
   nearest-seed partition and is deterministic.
10. **Calibration is 0.50 µm/px, not 0.25.** The median annotated nucleus is
   19.1 px across; at 0.50 µm/px that is 9.57 µm (correct for epithelial tumour
   nuclei), at 0.25 µm/px 4.78 µm (smaller than a lymphocyte — impossible across
   seven carcinoma types). The "40×" in the challenge description refers to the
   original scan; the released crops are downsampled.
11. **Seven training slides have no organ metadata.** The official organ PDF
   documents 30 patients; the dataset contains 37 slides. The EDA reports the
   extra seven as "Unlisted" rather than guessing.
