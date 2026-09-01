import os
import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.metrics import calculate_spatial_metrics
from backend.app.grading import predict_tumor_severity_grade

client = TestClient(app)

class TestHistopathologySystem(unittest.TestCase):
    
    def test_health_endpoint(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")

    def test_spatial_metrics_calculation(self):
        # Create 100x100 mask with 2500 active tumor pixels
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1
        
        metrics = calculate_spatial_metrics(mask, pixel_scale_um=0.5)
        self.assertEqual(metrics["total_pixels"], 10000)
        self.assertEqual(metrics["tumor_pixels"], 2500)
        self.assertEqual(metrics["tumor_burden_percent"], 25.0)
        
    def test_tumor_grading_brackets(self):
        g1 = predict_tumor_severity_grade(10.0)
        self.assertEqual(g1["grade"], "Grade I")
        
        g2 = predict_tumor_severity_grade(25.0)
        self.assertEqual(g2["grade"], "Grade II")
        
        g3 = predict_tumor_severity_grade(45.0)
        self.assertEqual(g3["grade"], "Grade III")

    def test_prediction_api_endpoint(self):
        # Create synthetic H&E image
        img = np.full((128, 128, 3), 200, dtype=np.uint8)
        cv2.circle(img, (64, 64), 30, (80, 20, 120), -1) # purple nucleus cluster
        
        _, img_png = cv2.imencode(".png", img)
        files = {"file": ("test_slide.png", img_png.tobytes(), "image/png")}
        
        response = client.post("/api/predict", files=files, data={"pixel_scale_um": 0.5})
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "success")
        self.assertIn("metrics", json_data)
        self.assertIn("grading", json_data)
        self.assertIn("mask_base64", json_data)

    def test_pdf_report_generation(self):
        img = np.full((128, 128, 3), 200, dtype=np.uint8)
        cv2.circle(img, (64, 64), 30, (80, 20, 120), -1)
        _, img_png = cv2.imencode(".png", img)
        
        files = {"file": ("test_slide.png", img_png.tobytes(), "image/png")}
        data = {"patient_id": "TEST-100", "biopsy_site": "Prostate Biopsy", "pixel_scale_um": 0.5}
        
        response = client.post("/api/generate-report", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertGreater(len(response.content), 1000)

if __name__ == "__main__":
    unittest.main()
