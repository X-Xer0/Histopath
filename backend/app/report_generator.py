"""
PDF report builder.

Produces a nucleus segmentation and morphometry report: the slide, the
segmentation, the measurements and the nuclear size distribution.

This is a quantitative analysis report. It deliberately contains no diagnosis,
no severity grade and no treatment recommendation, because the segmentation
model detects nuclei and cannot distinguish malignant from benign tissue.
"""

import io
from typing import Any, Dict

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#1A2A4A")
ACCENT = colors.HexColor("#2B6CB0")
MUTED = colors.HexColor("#5A6472")
LINE = colors.HexColor("#CBD5E0")

DISCLAIMER = (
    "This report contains quantitative image analysis only. The segmentation model detects "
    "cell nuclei in H&E stained tissue and does not classify tissue as benign or malignant. "
    "Nothing in this document is a diagnosis or a treatment recommendation, and it must not "
    "be used as the basis for clinical decisions."
)

DETECTION_NOTE = (
    "Detection bias: measured against the official MoNuSeg 2018 test set, this pipeline "
    "recovers about 86% of annotated nuclear pixels and detects roughly 20% more nuclei "
    "than were annotated - part of that is genuinely separating touching nuclei, part is "
    "segmentation speckle. Treat every figure here as a relative measurement: compare "
    "slides processed by this same pipeline rather than against absolute reference values."
)


def _size_histogram(metrics: Dict[str, Any], width: float = 460, height: float = 120) -> Drawing:
    """Bar chart of the nuclear size distribution, drawn with ReportLab primitives."""
    dist = metrics.get("size_distribution", {})
    edges = dist.get("bin_edges_um") or []
    counts = dist.get("counts") or []
    drawing = Drawing(width, height)

    plot_h = height - 26
    plot_w = width - 34
    drawing.add(Line(30, 18, 30 + plot_w, 18, strokeColor=LINE, strokeWidth=0.6))

    if not counts or max(counts) == 0:
        drawing.add(String(30 + plot_w / 2, plot_h / 2, "no nuclei detected",
                           fontSize=8, fillColor=MUTED, textAnchor="middle"))
        return drawing

    peak = max(counts)
    n = len(counts)
    bar_w = plot_w / n
    for i, c in enumerate(counts):
        h = (c / peak) * plot_h
        x = 30 + i * bar_w + bar_w * 0.15
        drawing.add(Rect(x, 18, bar_w * 0.7, h, fillColor=ACCENT, strokeColor=None))
        drawing.add(String(x + bar_w * 0.35, 8, f"{edges[i]:.0f}", fontSize=6,
                           fillColor=MUTED, textAnchor="middle"))
    drawing.add(String(30 + plot_w / 2, height - 10,
                       "nuclear equivalent diameter (µm)", fontSize=7.5,
                       fillColor=MUTED, textAnchor="middle"))
    return drawing


def generate_pdf_report_bytes(
    img_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    instance_overlay_bgr: np.ndarray,
    metrics: Dict[str, Any],
    engine_name: str = "ONNX U-Net",
    precise_mode: bool = False,
    sample_id: str = "",
) -> bytes:
    """Build the PDF report and return it as a byte stream."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=16,
                                 textColor=INK, spaceAfter=2)
    sub_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=8.5,
                               textColor=MUTED, spaceAfter=10)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11,
                              textColor=ACCENT, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle('B', parent=styles['Normal'], fontSize=8.5, leading=11.5,
                          textColor=colors.HexColor("#333A45"))
    small = ParagraphStyle('SM', parent=styles['Normal'], fontSize=7.5, leading=10,
                           textColor=MUTED)

    # ---------------- header ----------------
    story.append(Paragraph("NUCLEI SEGMENTATION &amp; MORPHOMETRY REPORT", title_style))
    story.append(Paragraph(
        "Automated histopathology image analysis &nbsp;·&nbsp; quantitative measurements only",
        sub_style))

    meta = [
        ["Sample", sample_id or "uploaded slide", "Analysis engine", engine_name],
        ["Stain", "H&E (hematoxylin & eosin)",
         "Mode", "precise (8x TTA)" if precise_mode else "standard"],
        ["Spatial calibration",
         f"{metrics.get('pixel_scale_um', 0.5)} µm/pixel",
         "Calibration basis", "40x source scan, distributed at effective 20x"],
    ]
    t_meta = Table(meta, colWidths=[80, 165, 95, 190])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F7FA')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#2D3748")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#E2E8F0')),
    ]))
    story.append(t_meta)

    # ---------------- images ----------------
    def encode(img: np.ndarray) -> io.BytesIO:
        ok, png = cv2.imencode(".png", img)
        return io.BytesIO(png.tobytes())

    panel = np.hstack([
        cv2.resize(img_bgr, (215, 215)),
        cv2.resize(overlay_bgr, (215, 215)),
        cv2.resize(instance_overlay_bgr, (215, 215)),
    ])
    story.append(Paragraph("1. Slide, detected nuclei and individual nucleus map", h2_style))
    story.append(RLImage(encode(panel), width=460, height=153))
    story.append(Paragraph(
        "Left: uploaded slide. Centre: detected nuclear regions. "
        "Right: each detected nucleus coloured separately.", small))

    # ---------------- measurements ----------------
    story.append(Paragraph("2. Quantitative measurements", h2_style))
    cellularity = metrics.get("cellularity", {})
    rows = [
        ["Measurement", "Value", "Measurement", "Value"],
        ["Nuclei detected", f"{metrics.get('nuclei_count', 0):,}",
         "Nuclear density", f"{metrics.get('nuclear_density_percent', 0):.2f} %"],
        ["Mean nuclear area", f"{metrics.get('mean_nuclear_area_um2', 0):.2f} µm²",
         "Median nuclear area", f"{metrics.get('median_nuclear_area_um2', 0):.2f} µm²"],
        ["Mean nuclear diameter", f"{metrics.get('mean_equivalent_diameter_um', 0):.2f} µm",
         "Size variability (CV)", f"{metrics.get('size_variability_cv_percent', 0):.2f} %"],
        ["Nuclei per mm²", f"{metrics.get('nuclei_per_mm2', 0):,.1f}",
         "Cellularity", f"{cellularity.get('category', '-')}"],
        ["Total nuclear area", f"{metrics.get('nuclear_area_mm2', 0):.6f} mm²",
         "Tissue area analysed", f"{metrics.get('tissue_area_mm2', 0):.6f} mm²"],
    ]
    t = Table(rows, colWidths=[120, 110, 120, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F7FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Paragraph(
        f"Cellularity is a descriptive summary of measured nuclear density: "
        f"{cellularity.get('detail', '')}. Its bands are {cellularity.get('basis', '')}, "
        f"not a clinical reference range.", small))

    # ---------------- size distribution ----------------
    story.append(Paragraph("3. Nuclear size distribution", h2_style))
    story.append(_size_histogram(metrics))

    # ---------------- notes ----------------
    story.append(Paragraph("4. Interpretation notes and limitations", h2_style))
    story.append(Paragraph(DETECTION_NOTE, body))
    story.append(Spacer(1, 5))
    story.append(Paragraph(DISCLAIMER, body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
