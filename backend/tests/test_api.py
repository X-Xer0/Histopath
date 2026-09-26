"""
Test suite for the nucleus segmentation service.

Covers the measurement maths, the instance separator, and every API endpoint.
Run with:  python -m pytest backend/tests/test_api.py -q
"""

import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.main import app
from backend.app.metrics import calculate_nucleus_metrics, per_nucleus_rows
from backend.app.instances import separate_nuclei

client = TestClient(app)


def synthetic_slide(size: int = 256) -> np.ndarray:
    """A pink background with a few purple nucleus-like blobs."""
    img = np.full((size, size, 3), (200, 180, 220), dtype=np.uint8)   # BGR pale pink
    for cx, cy, r in [(70, 70, 16), (150, 90, 20), (110, 180, 14), (200, 200, 18)]:
        cv2.circle(img, (cx, cy), r, (90, 40, 110), -1)
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestInstanceSeparation(unittest.TestCase):
    """The separator must split touching nuclei into individual objects."""

    def _circles(self, centers, radius, size=200):
        mask = np.zeros((size, size), np.uint8)
        for c in centers:
            cv2.circle(mask, c, radius, 1, -1)
        return mask

    def test_separated_circles_are_counted(self):
        mask = self._circles([(40, 40), (40, 140), (140, 40)], 18)
        _, count = separate_nuclei(mask)
        self.assertEqual(count, 3)

    def test_overlapping_circles_are_split(self):
        mask = self._circles([(80, 100), (115, 100)], 22)
        _, count = separate_nuclei(mask)
        self.assertEqual(count, 2)

    def test_cluster_is_split(self):
        mask = self._circles([(85, 85), (115, 85), (85, 115), (115, 115)], 20)
        _, count = separate_nuclei(mask)
        self.assertEqual(count, 4)

    def test_single_nucleus_stays_single(self):
        mask = self._circles([(50, 50)], 15, size=100)
        _, count = separate_nuclei(mask)
        self.assertEqual(count, 1)

    def test_empty_mask(self):
        _, count = separate_nuclei(np.zeros((50, 50), np.uint8))
        self.assertEqual(count, 0)

    def test_speck_is_removed(self):
        mask = self._circles([(50, 50)], 15, size=100)
        mask[10, 10] = mask[11, 10] = 1
        _, count = separate_nuclei(mask, min_area_px=12)
        self.assertEqual(count, 1)

    def test_labels_are_compact_and_contiguous(self):
        mask = self._circles([(40, 40), (40, 140)], 18)
        labels, count = separate_nuclei(mask)
        self.assertEqual(sorted(int(v) for v in np.unique(labels)), [0] + list(range(1, count + 1)))


class TestNucleusMetrics(unittest.TestCase):
    """The measurement maths must be exact and correctly calibrated."""

    def test_area_and_density_maths(self):
        # 100x100 field with a 20x20 (400 px) nucleus block.
        mask = np.zeros((100, 100), np.uint8)
        mask[40:60, 40:60] = 1
        labels, count = separate_nuclei(mask)

        m = calculate_nucleus_metrics(mask, labels, pixel_scale_um=0.5)
        self.assertEqual(m["nuclei_pixels"], 400)
        self.assertEqual(m["nuclei_count"], 1)
        # 400 px x 0.25 um^2 = 100 um^2 = 0.0001 mm^2
        self.assertAlmostEqual(m["nuclear_area_um2"], 100.0, places=2)
        self.assertAlmostEqual(m["nuclear_area_mm2"], 0.0001, places=6)
        # 400 / 10000 = 4 %
        self.assertAlmostEqual(m["nuclear_density_percent"], 4.0, places=2)
        # one 100 um^2 nucleus -> equivalent diameter 2*sqrt(100/pi) = 11.28 um
        self.assertAlmostEqual(m["mean_equivalent_diameter_um"], 11.28, places=1)

    def test_tissue_mask_is_the_density_denominator(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:20, 10:20] = 1                      # 100 nucleus pixels
        tissue = np.zeros((100, 100), bool)
        tissue[0:50, :] = True                      # half the image is tissue
        labels, _ = separate_nuclei(mask)

        m = calculate_nucleus_metrics(mask, labels, 0.5, tissue)
        # 100 nucleus px over 5000 tissue px = 2 %
        self.assertAlmostEqual(m["nuclear_density_percent"], 2.0, places=2)
        self.assertEqual(m["tissue_pixels"], 5000)
        # 5000 px x 0.25 um^2 = 1250 um^2 = 0.00125 mm^2
        self.assertAlmostEqual(m["tissue_area_mm2"], 0.00125, places=6)

    def test_nuclei_per_mm2(self):
        mask = np.zeros((200, 200), np.uint8)
        for cx, cy in [(50, 50), (150, 50), (50, 150), (150, 150)]:
            cv2.circle(mask, (cx, cy), 12, 1, -1)
        labels, count = separate_nuclei(mask)
        m = calculate_nucleus_metrics(mask, labels, 0.5)
        self.assertEqual(m["nuclei_count"], 4)
        # field = 200x200 px x 0.25 um^2 = 10000 um^2 = 0.01 mm^2 -> 400 nuclei/mm^2
        self.assertAlmostEqual(m["nuclei_per_mm2"], 400.0, places=1)

    def test_size_distribution_sums_to_count(self):
        mask = np.zeros((200, 200), np.uint8)
        for cx, cy, r in [(50, 50, 8), (100, 50, 14), (150, 50, 20), (100, 150, 11)]:
            cv2.circle(mask, (cx, cy), r, 1, -1)
        labels, count = separate_nuclei(mask)
        m = calculate_nucleus_metrics(mask, labels, 0.5)
        self.assertEqual(sum(m["size_distribution"]["counts"]), count)

    def test_cellularity_band_follows_density(self):
        def band_for(pixels):
            mask = np.zeros((100, 100), np.uint8)
            mask.ravel()[:pixels] = 1
            m = calculate_nucleus_metrics(mask, None, 0.5)
            return m["cellularity"]["category"]

        self.assertEqual(band_for(1500), "Low")        # 15 %
        self.assertEqual(band_for(2600), "Moderate")   # 26 %
        self.assertEqual(band_for(4000), "High")       # 40 %

    def test_per_nucleus_rows(self):
        mask = np.zeros((120, 120), np.uint8)
        cv2.circle(mask, (40, 40), 12, 1, -1)
        cv2.circle(mask, (90, 90), 12, 1, -1)
        labels, count = separate_nuclei(mask)
        rows = per_nucleus_rows(labels, 0.5)
        self.assertEqual(len(rows), count)
        self.assertEqual([r["nucleus_id"] for r in rows], list(range(1, count + 1)))
        for r in rows:
            self.assertGreater(r["area_um2"], 0)
            self.assertGreater(r["equivalent_diameter_um"], 0)


class TestTilingCoverage(unittest.TestCase):
    """
    Regression tests for the tiled inference grid.

    A previous version built the tile origins with
    `range(0, length - tile + 1, stride)`, which stops at the last multiple of
    the stride. For a 1000 px slide with a 256 px tile and stride 192 it yielded
    0/192/384/576, so the grid ended at 832 and the final 168 px band - 31 % of
    the image - was never processed and silently reported as background.
    """

    def test_tile_origins_cover_every_axis_length(self):
        from backend.app.model import _tile_origins, TILE_SIZE, TILE_STRIDE

        for length in (256, 257, 512, 600, 744, 800, 850, 900, 1000, 1024, 1400, 2048, 3333):
            origins = _tile_origins(length, TILE_SIZE, TILE_STRIDE)
            self.assertEqual(origins[0], 0, f"length {length}")
            self.assertEqual(origins, sorted(set(origins)), f"length {length}: not monotonic")
            self.assertEqual(
                origins[-1] + TILE_SIZE, max(length, TILE_SIZE),
                f"length {length}: grid ends at {origins[-1] + TILE_SIZE}, "
                f"leaving {length - origins[-1] - TILE_SIZE} px uncovered",
            )
            for a, b in zip(origins, origins[1:]):
                self.assertLessEqual(b - a, TILE_STRIDE, f"length {length}: gap in stride")

    def test_sliding_window_touches_every_pixel(self):
        """Stub the network with an all-ones map: any uncovered pixel stays 0."""
        from backend.app.model import engine

        if engine.onnx_session is None:
            self.skipTest("ONNX model not available")

        original = engine._tile_probability
        engine._tile_probability = lambda patch, precise: np.ones(patch.shape[:2], np.float32)
        try:
            # 1000 px is the size that used to lose its bottom-right band.
            prob = engine._sliding_window(np.zeros((1000, 1000, 3), np.uint8), precise=False)
        finally:
            engine._tile_probability = original

        uncovered = int(np.count_nonzero(prob == 0.0))
        self.assertEqual(uncovered, 0, f"{uncovered} pixels were never covered by a tile")
        self.assertAlmostEqual(float(prob.min()), 1.0, places=5)


class TestAPI(unittest.TestCase):

    def test_health(self):
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["status"], "online")
        self.assertIn("engine", d)
        self.assertIn("not a diagnostic", d["scope"])
        self.assertEqual(d["default_pixel_scale_um"], 0.5)

    def test_predict_schema(self):
        files = {"file": ("slide.png", encode_png(synthetic_slide()), "image/png")}
        r = client.post("/api/predict", files=files, data={"pixel_scale_um": 0.5, "precise_mode": False})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["status"], "success")

        for key in ("original_base64", "mask_base64", "overlay_base64", "instance_base64"):
            self.assertTrue(d[key].startswith("data:image/png;base64,"))

        m = d["metrics"]
        for key in ("nuclei_count", "nuclear_density_percent", "mean_equivalent_diameter_um",
                    "nuclei_per_mm2", "tissue_area_mm2", "nuclear_area_mm2",
                    "size_variability_cv_percent", "size_distribution", "cellularity"):
            self.assertIn(key, m)
        self.assertIn("bin_edges_um", m["size_distribution"])
        self.assertIn("counts", m["size_distribution"])
        self.assertIn("category", m["cellularity"])

        # the old tumour-grading contract must be gone
        self.assertNotIn("tumor_burden_percent", m)
        self.assertNotIn("grading", d)

    def test_precise_mode_runs(self):
        files = {"file": ("slide.png", encode_png(synthetic_slide(128)), "image/png")}
        r = client.post("/api/predict", files=files, data={"pixel_scale_um": 0.5, "precise_mode": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["precise_mode"])

    def test_invalid_image_rejected(self):
        files = {"file": ("bad.png", b"this is not an image", "image/png")}
        r = client.post("/api/predict", files=files)
        self.assertEqual(r.status_code, 400)

    def test_csv_export(self):
        files = {"file": ("slide.png", encode_png(synthetic_slide()), "image/png")}
        r = client.post("/api/export-csv", files=files, data={"pixel_scale_um": 0.5})
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/csv", r.headers["content-type"])
        text = r.content.decode("utf-8")
        header = text.splitlines()[0]
        self.assertEqual(
            header,
            "nucleus_id,area_px,area_um2,equivalent_diameter_um,centroid_x_px,centroid_y_px",
        )

    def test_pdf_report(self):
        files = {"file": ("slide.png", encode_png(synthetic_slide()), "image/png")}
        r = client.post("/api/generate-report", files=files,
                        data={"pixel_scale_um": 0.5, "precise_mode": False, "sample_id": "TEST-1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/pdf")
        self.assertGreater(len(r.content), 2000)
        self.assertTrue(r.content.startswith(b"%PDF"))

    def test_upload_weights_rejects_non_onnx(self):
        files = {"file": ("weights.txt", b"nope", "text/plain")}
        r = client.post("/api/upload-weights", files=files)
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
