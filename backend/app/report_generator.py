import io
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report_bytes(
    img_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    metrics: Dict[str, Any],
    grading_info: Dict[str, str],
    patient_id: str = "PAT-89210",
    biopsy_site: str = "Breast Tissue / Lymph Node"
) -> bytes:
    """
    Generates a clinical pathology PDF report as a binary byte stream.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    
    # Save visual panel to buffer
    panel_img = np.hstack([cv2.resize(img_bgr, (250, 250)), cv2.resize(overlay_bgr, (250, 250))])
    _, panel_png = cv2.imencode(".png", panel_img)
    panel_bytes = io.BytesIO(panel_png.tobytes())
    
    # PDF Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A365D'), spaceAfter=10)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=5)
    
    # 1. Header
    story.append(Paragraph("AUTOMATED HISTOPATHOLOGY DIAGNOSTIC REPORT", title_style))
    story.append(Spacer(1, 5))
    
    # Metadata Table
    meta_data = [
        ["Patient ID:", patient_id, "Report Date:", "2026-08-27"],
        ["Biopsy Site:", biopsy_site, "Magnification Scale:", f"20x ({metrics.get('pixel_scale_um', 0.5)} um/px)"],
        ["Stain Type:", "H&E (Hematoxylin & Eosin)", "Pipeline Model:", "PyTorch U-Net / ONNX Engine"]
    ]
    t_meta = Table(meta_data, colWidths=[90, 160, 110, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#2D3748')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))
    
    # 2. Visual Panel
    story.append(Paragraph("<b>1. Microscopic Slide & Tumor Segmentation Overlay</b>", h2_style))
    story.append(RLImage(panel_bytes, width=480, height=240))
    story.append(Spacer(1, 10))
    
    # 3. Quantitative Summary Table
    story.append(Paragraph("<b>2. Quantitative Tissue Metrics & Severity Grading</b>", h2_style))
    
    metrics_table_data = [
        ["Diagnostic Parameter", "Measured Value"],
        ["Total Tissue Surface Area", f"{metrics.get('total_area_mm2', 0)} mm²"],
        ["Segmented Tumor Area", f"{metrics.get('tumor_area_mm2', 0)} mm² ({metrics.get('tumor_area_um2', 0)} μm²)"],
        ["Tumor Burden Ratio (%)", f"{metrics.get('tumor_burden_percent', 0):.2f} %"],
        ["Histological Severity Grade", f"{grading_info.get('grade', 'N/A')} - {grading_info.get('description', '')}"],
        ["Assessed Risk Category", f"{grading_info.get('risk_category', 'N/A')}"]
    ]
    t_metrics = Table(metrics_table_data, colWidths=[240, 300])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#EDF2F7')]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))
    
    # 4. Clinical Impression & Recommendation
    story.append(Paragraph("<b>3. Pathologist Impression & Clinical Protocol</b>", h2_style))
    impression_text = f"The automated computer vision pipeline identified region(s) consistent with malignant tumor tissue comprising <b>{metrics.get('tumor_burden_percent', 0):.2f}%</b> of the biopsy field. Classification indicates <b>{grading_info.get('grade')} ({grading_info.get('description')})</b>. Clinical recommendation: <i>{grading_info.get('clinical_recommendation')}</i>"
    story.append(Paragraph(impression_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
