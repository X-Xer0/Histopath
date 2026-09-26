# Deploying to Sevalla

Everything below was measured on this project, not assumed. Where a number is an
extrapolation from a local measurement it says so.

---

## 1. The short answer on GPUs

**Sevalla has no GPU option at any price.** Application hosting is CPU-only
containers, and no GPU hardware appears anywhere in their product or pricing
pages. The system was built for that — inference runs on ONNX Runtime on the CPU —
but it means you cannot make inference dramatically faster by throwing hardware at
it on this host.

Pod sizes available (CPU / RAM / price per month):

| Pod | CPU | RAM | Price | Verdict for this app |
|---|---|---|---|---|
| Hobby **H1** | 0.3 | 0.3 GB | $5 | ✗ **will crash** — see below |
| Standard **S1** | 0.5 | 1 GB | $10 | ✓ minimum viable |
| Standard **S2** | 1 | 2 GB | $40 | ✓ comfortable |
| Standard **S3** | 2 | 4 GB | $80 | fastest CPU option |
| Standard S4/S5 | 4 / 8 | 8 / 16 GB | $160 / $320 | unnecessary |

**Why H1 will not work.** I ran the application in a clean environment built from
`backend/requirements.txt` and sampled its memory:

| State | Resident memory |
|---|---|
| Idle, model loaded, serving | 189 MB |
| **Peak during one 1000×1000 inference** | **447 MB** |

H1 gives you 0.3 GB. The process peaks at 447 MB, so it will be killed. **S1
(1 GB) is the smallest pod that can serve this app**; S2 leaves proper headroom.

Sevalla does offer a **free trial**, but there is no permanent free tier for
applications — signup requires a payment method. Once the trial ends the cheapest
workable pod is S1 at $10/month.

---

## 2. Speed you should expect

Measured on this machine (4 cores):

| Configuration | Per 1000×1000 slide |
|---|---|
| 4 threads (idle machine) | ~23–29 s |
| 1 thread | ~47 s |
| 8× TTA precise mode | ~8× the above |

Extrapolating to Sevalla pods (fractional vCPU, so treat these as estimates, not
measurements): **S1 roughly 1–2 minutes per slide, S2 about a minute, S3 under a
minute.** The UI shows a spinner for the duration, so it is usable for a demo, but
do not promise instant results.

Things already done to keep it as fast as it can be on a CPU host:

* tile overlap cut from 64 px to 8 px — 16 tiles per slide instead of 25, same Dice
* tiles that are bare glass are skipped entirely
* the tile grid covers the whole slide (see §5 — this was a bug worth 0.13 Dice)
* CPU work runs in a thread pool, so `/api/health` and the page stay responsive
  while a slide is being processed

Things that do **not** help, all measured: model quantization (4.5× slower),
bigger tiles (slower), and batching (the exported graph has a fixed batch size).
A 30 MB model simply takes this long to convolve on a shared CPU core.

---

## 3. Before you deploy: push the code

The application must exist in the Git repository Sevalla will clone. Two things
were missing:

* **`backend/app/instances.py` was untracked.** Without it the container crashes
  on import — it is the nucleus instance separator, a core module.
* **The datasets were not gitignored.** A plain `git add .` would have committed
  ~344 MB of MoNuSeg data, and Sevalla would clone all of it on every deploy. The
  `.gitignore` now excludes them.

From `/home/xer0/histopath_app`:

```bash
git add .
git status --short          # sanity check: the two MoNuSeg folders must NOT appear
git commit -m "Nucleus segmentation app: correct tiling, instance separation, morphometry"
git push origin main        # or whichever branch you deploy from
```

The repository should be roughly 60 MB after this (the ONNX model is 29 MB of it).
If `git status` shows a MoNuSeg folder, stop — the ignore rule is not applying.

---

## 4. Create the application on Sevalla

1. Sign up at <https://app.sevalla.com> and complete billing details (a payment
   method is required even for the trial).
2. **Applications → Create → Application**.
3. Choose **Git repository** and connect GitHub. Select `X-Xer0/Histopath` and the
   branch you pushed to.
   * If the repository selector is empty, Sevalla requires the connected Git
     account to be more than 30 days old; contact their support if that is the
     blocker.
4. **Name:** anything, e.g. `histology-nuclei`.
5. **Location:** pick the region closest to you (the app has no external
   dependencies, so any region works).
6. **Resource: choose Standard S1 (0.5 CPU / 1 GB) or larger.** Do not pick Hobby
   H1 — it will be OOM-killed (see §1).
7. Review the cost estimate and **Create application**.
8. Open **Settings → Environment variables** and add:

   | Name | Value | Why |
   |---|---|---|
   | `PORT` | `8000` | Sevalla normally injects this; setting it explicitly removes any ambiguity between the container port and the routing port |

   Nothing else is required. There are no secrets, no database and no external
   services — the model ships inside the image.
9. Deploy from the **Overview** page.

Sevalla builds with a dedicated build pod (4 CPU / 8 GB, $0.02 per minute). The
Docker build installs 39 Python packages and takes roughly 3–6 minutes, so expect
about **$0.10 per deploy** in build time.

### What the image does

```
Stage 1 (builder)  python:3.12-slim + build-essential
                   pip install -r backend/requirements.txt into /opt/venv
Stage 2 (runtime)  python:3.12-slim
                   copies /opt/venv, backend/, frontend/
                   runs as an unprivileged user (uid 10001)
                   HEALTHCHECK polls /api/health every 30 s
                   CMD uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

One worker is deliberate: inference is CPU-bound, so extra processes would only
compete for the same cores. Concurrency is handled by the thread pool instead.

---

## 5. Verify the deployment

Open the application URL (the **View** button on the Overview page) and check:

```bash
# 1. the service is up and the model loaded
curl https://<your-app>.sevalla.app/api/health
```

Expected:

```json
{
  "status": "online",
  "engine": "ONNX U-Net",
  "model_file": "tumor_unet.onnx",
  "scope": "quantitative image analysis only - not a diagnostic service"
}
```

If `engine` says **"Stain Deconvolution Fallback"** instead, the ONNX file did not
make it into the image — check that `backend/models/tumor_unet.onnx` is committed
and that `.dockerignore` does not exclude it.

Then open the UI in a browser, upload one of the slides from `samples/`, and run
the segmentation. First inference will be slower than the numbers in §2 because
the machine is cold.

**Expect the first upload to take 1–2 minutes on S1.** If the browser shows an
error, check the application logs inside Sevalla — an OOM kill appears there as
the container restarting.

---

## 6. Keeping the cost down

* **Hibernation** — Sevalla can pause a runtime pod when it is not in use. For a
  capstone demo this is the difference between $10 and a few dollars a month.
  Enable it in the application settings.
* **Scale to zero between demos** rather than leaving S2 or S3 running.
* **Do not upgrade for speed before measuring.** Deploy on S1 first, time one
  slide, and only move up if it is genuinely too slow for your demo.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Container restarts under load | OOM — peak RSS 447 MB | use S1 or larger, never H1 |
| `ModuleNotFoundError: backend.app.instances` | module not committed | `git add backend/app/instances.py && git push` |
| `engine: Stain Deconvolution Fallback` | model absent from the image | confirm `backend/models/tumor_unet.onnx` is tracked, redeploy |
| Build fails on `pip install` | a dependency lost its wheel for 3.12 | all current deps have cp312 manylinux wheels; check the build log for which package |
| Health check fails but the app runs | port mismatch | set `PORT=8000` explicitly and confirm Sevalla routes to the same port |
| Slow first request | cold container | expected; subsequent requests reuse the loaded model |

---

## 8. If free hosting matters more than Sevalla does

Worth knowing: **Hugging Face Docker Spaces now require a paid plan** (PRO), so
that is no longer a free alternative despite the free CPU hardware table. Render's
free tier gives 0.1 CPU and 512 MB, which is below the 447 MB peak here. Google
Cloud Run's free allowance would cover this app comfortably but needs a GCP
account.

Sevalla's free trial is the practical way to demonstrate the deployment; the S1
pod at $10/month is the cheapest configuration that genuinely runs it.
