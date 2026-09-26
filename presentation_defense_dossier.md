# Presentation & Viva Defence Dossier

**Project**: Automated Histopathology Nucleus Segmentation & Morphometry
**Deployment**: containerised, live on Sevalla (CPU-only)
**Domain**: computer vision in biomedical engineering and computational pathology

---

## 1. Problem statement

Histopathology — examining stained tissue under a microscope — is how cancer is
diagnosed. The most common stain, hematoxylin and eosin (H&E), colours nuclei
purple and cytoplasm pink. Three practical problems make manual assessment hard:

1. **Observer variability.** Grading is a judgement call; the same slide can be
   read differently by two pathologists, or by the same pathologist on different
   days.
2. **Colour inconsistency.** Stain batch, lab protocol and scanner hardware all
   change how the same tissue appears. A method tuned on one scanner may not
   transfer.
3. **Eyeball estimation.** Nuclear density and cell counts are usually judged
   visually rather than measured.

## 2. Goal

Build, evaluate and deploy a system that segments **cell nuclei** in H&E slides
and reports **quantitative morphometry** — how much of the tissue is nuclei, how
many nuclei there are, how large they are, and how variable that size is — with
every measurement traceable to a verified spatial calibration.

## 3. What was built

| Module | Implementation | Outcome |
|---|---|---|
| Data handling | MoNuSeg 2018, 37 slides, XML polygon annotations parsed to masks with a Python port of the official MATLAB routine | 24,140 annotated nuclei ingested |
| Preprocessing | 256×256 patches, stride 186 (27 % overlap), blank-tile rejection, geometric + elastic + stain augmentation | 925 training patches |
| Segmentation | U-Net, 4 levels, base 64 filters, 7.7 M parameters, trained from scratch | Best validation Dice **0.8534** |
| Loss | Balanced BCE + Dice (Tversky α = β = 0.5) | Stable; earlier recall-weighted variants collapsed |
| Inference | Single-file ONNX (opset 11, 29.4 MB), tiled with overlap averaging, threshold 0.50 | CPU-only, no GPU in production |
| Instances | Distance-transform seeding + nearest-seed partition | Touching nuclei counted separately |
| Morphometry | Nuclear density, count, nuclei/mm², area, equivalent diameter, CV, size distribution | 20 unit/API tests green |
| Reporting | Per-nucleus CSV + ReportLab PDF morphometry report | No clinical claims |
| Fallback | Hematoxylin colour deconvolution + Otsu | Service never goes dark |
| Deployment | Multi-stage Docker, `$PORT`-aware, Sevalla | Single container, single origin |

## 4. Results

Benchmark on the **official MoNuSeg test set** — 14 slides from patients never
seen in training, threshold 0.50, 8× test-time augmentation:

| Metric | Value |
|---|---|
| Mean Dice | **0.8146 ± 0.0497** |
| Mean IoU | 0.6900 |
| Mean AJI (instance) | 0.5849 |
| Precision | 0.7802 |
| Recall | 0.8589 |
| Specificity | 0.9341 |
| Nuclei detected | 7,938 of 6,697 annotated |
| Validation Dice (patient-wise) | 0.8534 |

**Reproducibility** — three independent 50-epoch runs gave validation Dice of
0.8525 / 0.8529 / 0.8534 (within 0.1 %).

---

## 5. The finding worth leading with

**The model generalises to organs it has never seen.**

| Organs present in training | Mean Dice | Organs never seen | Mean Dice |
|---|---|---|---|
| Bladder, Kidney, Lung, Colon, Prostate, Breast | **0.8247** | Thyroid 0.855, Testis 0.800, Brain 0.751 | **0.7894** |

A two-point gap, and **Thyroid was the single best organ of all nine**. MoNuSeg
exists to test exactly this: it means the model learned general nuclear
morphology rather than organ-specific texture.

Per-organ breakdown:

| Organ | Dice | AJI | Precision | Recall | Seen in training? |
|---|---|---|---|---|---|
| Thyroid | 0.8548 | 0.6489 | 0.8305 | 0.8806 | **no** |
| Bladder | 0.8470 | 0.6392 | 0.7877 | 0.9159 | yes |
| Kidney | 0.8455 | 0.6010 | 0.8222 | 0.8702 | yes |
| Lung | 0.8376 | 0.6039 | 0.8399 | 0.8353 | yes |
| Prostate | 0.8098 | 0.6088 | 0.7351 | 0.9014 | yes |
| Colon | 0.8031 | 0.5993 | 0.7888 | 0.8179 | yes |
| Testis | 0.8003 | 0.4695 | 0.7157 | 0.9076 | **no** |
| Breast | 0.7869 | 0.5222 | 0.7752 | 0.8061 | yes |
| Brain | 0.7513 | 0.5645 | 0.7016 | 0.8312 | **no** |

---

## 6. Design rationale

**Why U-Net.** The encoder shrinks the image to learn context; the decoder
rebuilds resolution; skip connections copy fine detail straight across so nucleus
boundaries stay sharp rather than blurring on the way back up.

**Why BCE + Dice and not something more exotic.** Tumour tissue is a small
fraction of each tile, so BCE alone converges to "predict background everywhere"
and still scores acceptably. Dice directly rewards overlap. We tried weighting
false negatives more heavily with Tversky β = 0.7 and it **destroyed training** —
the model collapsed into an "almost everything is foreground" state and never
escaped. Precision/recall is therefore tuned at the decision threshold, which is
both safer and measurable.

**Why ONNX on CPU.** A PyTorch GPU runtime is heavy and slow to boot on a
CPU container. The exported graph is 29.4 MB, loads in seconds, and needs no CUDA,
which is what makes free-tier hosting possible.

**Why stain handling.** Colour differences between scanners are a real failure
mode, so inference can normalise in optical-density space and the fallback engine
works directly on the hematoxylin channel.

**Why the calibration is 0.50 µm/pixel.** The challenge states the slides were
captured at 40×, which would suggest 0.25 µm/px. That is wrong for the distributed
files, and the dataset proves it: the median annotated nucleus measures 19.1 px
across, which is **9.57 µm at 0.50 µm/px** — a textbook epithelial tumour nucleus
— and only 4.78 µm at 0.25 µm/px, smaller than a lymphocyte, which is impossible
across seven carcinoma types. The released 1000×1000 crops are the 40× scan
downsampled to an effective ~20×. Everything is reported as "40× source,
effective 20×", and the API accepts a `pixel_scale_um` parameter so another
scanner can supply its own.

---

## 7. Honest limitations

1. **Residual recall gap.** Recall is 0.86, so about 14 % of annotated nuclear
   pixels are still missed — 428 k false negatives against 718 k false positives on
   the test set. The remaining misses are small or faintly stained nuclei, which
   the Dice loss under-weights. Threshold tuning cannot recover them: the
   Dice-versus-threshold curve is flat within 0.4 % between 0.30 and 0.60.
2. **Counts are relative, not absolute.** The pipeline detects about 20 % more
   nuclei than were annotated (7,938 vs 6,697; per-slide 103 %–167 %). Part of that
   is genuinely separating touching nuclei, part is segmentation speckle, and the
   balance shifts with how densely the tissue is packed. The UI and the PDF both
   disclose this.
3. **Nuclei are not tumours.** MoNuSeg annotates *all* nuclei. The model cannot
   distinguish malignant from benign, which is why tumour burden and grading were
   removed rather than approximated.
4. **Instance metrics are limited by method.** AJI 0.585 because nuclei are
   separated from a binary mask after the fact, not predicted as instances. The
   MoNuSeg leaderboard's 0.13–0.69 AJI range comes from true instance-segmentation
   entries, so the numbers are not directly comparable — worth stating before you
   are asked.
5. **Brain is the weakest organ** (0.751), dragged down by one slide
   (TCGA-HT-8564, Dice 0.668) where precision collapses to 0.55's worth of
   spurious detections. The cause is not yet isolated.
6. **Not clinically usable.** Research use only; the tool reports measurements and
   does not diagnose.

---

## 8. Anticipated questions

**"Your Dice moved from 0.69 to 0.81 late in the project. What happened?"**
We found an inference bug, not a model problem. The tiled sliding window built its
tile origins with `range(0, h - tile + 1, stride)`, which stops at the last
multiple of the stride. On a 1000 px slide that produced origins 0/192/384/576, so
the grid ended at 832 px and **the final 168 px band — 31 % of every image — was
never processed**. Those pixels came back as probability zero and were silently
reported as background. When we measured where the errors actually were, 70 % of
all missed pixels sat inside that band. Fixing the grid — one helper function and
two loop lines — took Dice from 0.688 to 0.815 with **no retraining**. The model
was always better than the pipeline around it.

**"How did you not notice sooner?"**
Because we were reading the validation number and the test number separately and
attributing the gap to domain shift. Validation is computed on 256×256 patches, so
it never touched the tiling code; it was reporting 0.853 the whole time. The honest
lesson is the one we now follow: when two numbers disagree, measure where the error
physically is before theorising about why.

**"Is 0.81 good?"**
It is in line with — and for a from-scratch 7.7 M-parameter model trained on 925
patches, above — typical published U-Net baselines on this dataset, which sit in
the mid-0.70s to low-0.80s. Those setups usually have pretrained encoders, far more
patches and test-time ensembling. We report the untouched official test set.

**"Why not grade the tumour — that was the original idea?"**
Because we tested it and it would have been fabricated. The annotations are
nuclear boundaries, not tumour regions; there is no signal in the data that
separates malignant from benign. The original app produced Grade I/II/III from
nuclear density, which is not a grading criterion. We removed it rather than
dress up a number the model cannot support.

**"How do you know 0.50 µm/px is right?"**
From the dataset's own annotations: 19.1 px median nuclear diameter is 9.57 µm at
0.50 µm/px and 4.78 µm at 0.25. The second is smaller than a lymphocyte, which
cannot be true across breast, kidney, lung, prostate, bladder, colon and stomach
carcinomas. The "40×" refers to the original scan.

**"How do you know your nucleus counts are right?"**
We measured them against the annotated count on all 14 reference slides. The
pipeline detects about 20 % more objects than were annotated — 7,938 against
6,697 — and the per-slide range runs from 103 % to 167 %. Some of that excess is
real: the reference mask merges touching nuclei into a single connected component
while our separator deliberately splits them. The rest is segmentation speckle.
That figure is disclosed in the UI and the report rather than hidden, and it is why
we present counts as relative measurements rather than absolutes.

**"What happens if the model file is missing?"**
The engine falls back to hematoxylin colour deconvolution + Otsu thresholding, the
API keeps answering, and `/api/health` reports which engine is live. This is a
deliberate design decision so a deployment never goes dark.

**"What did you learn that a textbook would not have taught you?"**
Six things, all recorded in the execution guide: mixed precision silently destroys
this model (NaN gradients from epoch 2); recall-weighted Tversky collapses training
into all-foreground, so balanced Dice plus a tuned threshold is the safer lever;
patch-level validation splits leak and produce numbers that do not survive unseen
patients; OpenCV's watershed on a synthetic distance surface misassigns symmetric
touching nuclei, so we use `distanceTransformWithLabels`; **a tile grid built with
`range(0, h - tile + 1, stride)` can silently leave a third of the image
unprocessed**; and INT8 quantization made this model 4.5× *slower*, not faster —
the integer convolution path needs VNNI and falls back to a scalar kernel without
it. Every one of those was found by measuring, not by reasoning.

**"Why not just quantize the model to make it faster on the server?"**
We tried it and rejected it on measurement. Dynamic INT8 quantization shrank the
model from 29.4 MB to 9.4 MB but inference went from 2.4 s to 10.5 s per tile —
on CPUs without VNNI acceleration the integer convolution path is slower than
float32, and a 30 MB model has nothing to gain in the first place. What did work:
reducing the tile overlap from 64 px to 8 px halved the tile count with no loss in
Dice, and skipping bare-glass tiles. That is the honest answer to a question you
will probably get.

**"What would you do next?"**
Add a boundary-aware loss to attack the false negatives, halve the tiling stride
to roughly 1,300 patches, and move to an instance-segmentation head (HoVer-Net
style) to lift AJI. Then whole-slide pyramid support for real clinical slides.
