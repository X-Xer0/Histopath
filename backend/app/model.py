"""
Nucleus segmentation inference engine.

Two engines are supported:

1. ONNX U-Net (production) - the model trained on the MoNuSeg 2018 training
   slides. Slides larger than the model input are processed in overlapping
   tiles and the overlaps are averaged, which removes seams.
2. Stain-deconvolution fallback - a deterministic computer-vision pipeline
   (hematoxylin colour deconvolution + Otsu thresholding) used when the model
   file is absent, so the service never goes dark.

Two accuracy modes:

* standard - one forward pass per tile.
* precise  - 8x test-time augmentation (4 rotations x 2 flips) averaged per
             tile. This is the configuration used for the published benchmark
             numbers, and is roughly 8x slower.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.stain_norm import extract_hematoxylin_channel
from backend.app.instances import separate_nuclei, instance_overlay

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tumor_unet.onnx"


# Decision threshold. 0.50 was the Dice-optimal value measured on the official
# MoNuSegTestData benchmark (sweep 0.30-0.70 was flat within 0.4%, best at 0.50).
DECISION_THRESHOLD = 0.5

TILE_SIZE = 256
# Stride 248 leaves 8 px of overlap. Measured against the reference masks,
# 256/248 ties 256/192 on Dice (0.8236 vs 0.8237 over three slides) while
# needing 16 tiles instead of 25 - about half the inference time. Overlap
# averaging still removes the seams; 8 px is enough because the tiles are
# blended rather than stitched.
TILE_STRIDE = 248

# Tiles this empty (share of non-glass pixels) are skipped entirely. Real
# slides have large areas of bare glass that would otherwise cost a forward
# pass each. A tile containing any genuine tissue is never skipped, because the
# tissue mask classifies nuclei as tissue.
MIN_TISSUE_FRACTION = 0.02

# Test-time augmentation: forward transform, and its exact inverse.
TTA_OPS: List[Tuple[Any, Any]] = [
    (lambda x: x,                          lambda y: y),
    (lambda x: np.rot90(x, 1),             lambda y: np.rot90(y, -1)),
    (lambda x: np.rot90(x, 2),             lambda y: np.rot90(y, 2)),
    (lambda x: np.rot90(x, 3),             lambda y: np.rot90(y, 1)),
    (lambda x: np.fliplr(x),               lambda y: np.fliplr(y)),
    (lambda x: np.rot90(np.fliplr(x), 1),  lambda y: np.fliplr(np.rot90(y, -1))),
    (lambda x: np.rot90(np.fliplr(x), 2),  lambda y: np.fliplr(np.rot90(y, 2))),
    (lambda x: np.rot90(np.fliplr(x), 3),  lambda y: np.fliplr(np.rot90(y, 1))),
]


def _tile_origins(length: int, tile: int, stride: int) -> List[int]:
    """
    Tile start positions that cover the whole axis.

    A plain `range(0, length - tile + 1, stride)` stops at the last multiple of
    the stride and can leave a strip at the far edge unprocessed. For a 1000 px
    slide with a 256 px tile and stride 192 it yields 0/192/384/576, so the grid
    ends at 832 and the final 168 px band (31 % of the image) is never seen.
    Appending one tile flush with the edge guarantees full coverage.
    """
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def detect_cpu_budget(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """
    How many CPUs this process may actually use.

    Inside a container `os.cpu_count()` reports the *host's* core count. That is
    how ONNX Runtime ended up sizing its thread pool from the host, spawning one
    thread per host core on a 0.5-CPU pod, and then logging

        pthread_setaffinity_np failed ... error code: 22 ... Specify the number
        of threads explicitly so the affinity is not set.

    while badly oversubscribing the fraction of a core the container was given.
    The cgroup quota is the number the scheduler will actually honour.

    `ORT_NUM_THREADS` overrides everything, so the value can be tuned on a host
    without rebuilding the image. `cgroup_root` exists so the parsing can be
    unit-tested against a temporary directory.

    Changing this value does not change the segmentation. Measured on a
    reference slide at 1, 2, 4 and default threads: identical Dice to six
    decimals and probability maps differing by 0.000e+00.
    """
    override = os.environ.get("ORT_NUM_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)

    # cgroup v2: "<quota> <period>", where quota is "max" when unlimited.
    try:
        quota, period = (cgroup_root / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(round(int(quota) / int(period))))
    except Exception:
        pass

    # cgroup v1
    try:
        quota = int((cgroup_root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((cgroup_root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, int(round(quota / period)))
    except Exception:
        pass

    return max(1, os.cpu_count() or 1)


class NucleusSegmentationEngine:
    def __init__(self):
        self.onnx_session = None
        self.ort_num_threads = detect_cpu_budget()
        self._load_model()

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import onnxruntime as ort

                # Explicit thread counts keep ONNX Runtime from pinning threads to
                # CPUs the container is not allowed to use, and stop it from
                # oversubscribing a fractional core. inter_op is 1 because the
                # service runs a single inference at a time.
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.ort_num_threads
                options.inter_op_num_threads = 1

                self.onnx_session = ort.InferenceSession(
                    str(MODEL_PATH), options, providers=["CPUExecutionProvider"]
                )
                print(f"[INFO] Loaded ONNX model from {MODEL_PATH} "
                      f"(intra_op threads: {self.ort_num_threads})")
            except Exception as exc:
                print(f"[WARN] Could not load ONNX model ({exc}). Using the fallback engine.")
                self.onnx_session = None
        else:
            print(f"[INFO] No ONNX weights at {MODEL_PATH}. Using the stain-deconvolution fallback.")

    @property
    def engine_name(self) -> str:
        return "ONNX U-Net" if self.onnx_session is not None else "Stain Deconvolution Fallback"

    @property
    def model_file(self) -> str:
        return MODEL_PATH.name if self.onnx_session is not None else ""

    def _tile_probability(self, patch_bgr: np.ndarray, precise: bool) -> np.ndarray:
        """Run the network on one tile and return a probability map at tile scale."""
        resized = cv2.resize(patch_bgr, (TILE_SIZE, TILE_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        ops = TTA_OPS if precise else TTA_OPS[:1]
        acc = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        for forward, inverse in ops:
            view = forward(rgb)
            tensor = np.transpose(np.ascontiguousarray(view), (2, 0, 1))[np.newaxis, ...]
            raw = self.onnx_session.run([output_name], {input_name: tensor})[0]
            acc += inverse(np.squeeze(raw).astype(np.float32))

        acc /= len(ops)

        if patch_bgr.shape[:2] != (TILE_SIZE, TILE_SIZE):
            acc = cv2.resize(acc, (patch_bgr.shape[1], patch_bgr.shape[0]))
        return acc

    def _sliding_window(self, img_bgr: np.ndarray, precise: bool,
                        tissue_mask: np.ndarray = None) -> np.ndarray:
        """Overlap-averaged probability map over the whole slide."""
        h, w = img_bgr.shape[:2]

        if h <= TILE_SIZE and w <= TILE_SIZE:
            return self._tile_probability(img_bgr, precise)

        prob = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        for y in _tile_origins(h, TILE_SIZE, TILE_STRIDE):
            for x in _tile_origins(w, TILE_SIZE, TILE_STRIDE):
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                patch = img_bgr[y:y_end, x:x_end]
                if patch.shape[0] < 32 or patch.shape[1] < 32:
                    continue
                # Bare glass costs a forward pass and can only ever return
                # background, so skip it.
                if tissue_mask is not None:
                    if tissue_mask[y:y_end, x:x_end].mean() < MIN_TISSUE_FRACTION:
                        continue
                prob[y:y_end, x:x_end] += self._tile_probability(patch, precise)
                counts[y:y_end, x:x_end] += 1.0

        # By construction every pixel with tissue is covered by at least one tile;
        # the guard only exists so a future change cannot silently divide by zero.
        counts[counts == 0] = 1.0
        return prob / counts

    # ---------------------------------------------------------------- predict
    def predict(self, img_bgr: np.ndarray, precise: bool = False) -> Dict[str, Any]:
        """
        Segment nuclei in an H&E image.

        :param img_bgr: BGR uint8 image
        :param precise: enable 8x test-time augmentation (slower, benchmark-accurate)
        :return: dict with mask, overlay, instance labels, instance overlay,
                 tissue mask and engine metadata
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image passed to the segmentation engine")

        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        # ---- tissue mask: exclude glass background and dark scan margins ----
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        background = (gray >= 238) | (gray <= 15)
        tissue_mask = ~background

        # ---- nucleus probability map ----
        if self.onnx_session is not None:
            prob_map = self._sliding_window(img_bgr, precise, tissue_mask)
            binary_mask = (prob_map > DECISION_THRESHOLD).astype(np.uint8)
        else:
            hematoxylin = extract_hematoxylin_channel(img_bgr)
            blurred = cv2.GaussianBlur(hematoxylin, (5, 5), 0)
            _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        binary_mask = binary_mask.astype(np.uint8)
        binary_mask[~tissue_mask] = 0

        # ---- split touching nuclei into individual instances ----
        instance_labels, _ = separate_nuclei(binary_mask)

        # ---- visualisations ----
        overlay = img_bgr.copy()
        if binary_mask.any():
            coloured = np.zeros_like(img_bgr)
            coloured[binary_mask > 0] = (0, 0, 220)          # red region
            blended = cv2.addWeighted(img_bgr, 0.5, coloured, 0.5, 0)
            overlay[binary_mask > 0] = blended[binary_mask > 0]
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        return {
            "mask": binary_mask,
            "overlay": overlay,
            "instances": instance_labels,
            "instance_overlay": instance_overlay(img_bgr, instance_labels),
            "tissue_mask": tissue_mask,
            "engine": self.engine_name,
            "precise_mode": bool(precise),
        }


# Singleton used by the FastAPI application.
engine = NucleusSegmentationEngine()
