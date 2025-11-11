from __future__ import annotations
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import os, re
from PIL import Image as PILImage  # for aspect-ratio sizing

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ---------- helpers ----------
_FIELD_PATTERNS = {
    "name": r"\bname\s*:\s*(?P<val>.+)",
    "age": r"\bage\s*:\s*(?P<val>[\d]+)",
    "gender": r"\bgender\s*:\s*(?P<val>[A-Za-z]+)",
    "mrn": r"\bmrn\s*:\s*(?P<val>[\w\-]+)",
    "medical_history": r"\bmedical\s*history\s*:\s*(?P<val>.+)",
    "symptoms": r"\b(symptoms|indications)\s*:\s*(?P<val>.+)",
}

def _parse_patient_details_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = (text or "").strip()
    for key, pat in _FIELD_PATTERNS.items():
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = re.sub(r"[,\s]+$", "", m.group("val").strip())
            if key == "gender": val = val.capitalize()
            if key == "age":
                try:
                    out[key] = int(val)
                    continue
                except:
                    pass
            out[key] = val
    return out

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _md_to_rl(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

def _split_report_to_flowables(final_report: str, H2, BODY, SECTION) -> List:
    flows: List = []
    bullets: List[str] = []
    def flush_bullets():
        nonlocal bullets
        if not bullets: return
        items = [ListItem(Paragraph(_md_to_rl(b.strip()), BODY), leftIndent=8, value='•') for b in bullets]
        flows.append(ListFlowable(items, bulletType='bullet', start='bullet', leftIndent=14, spaceBefore=1, spaceAfter=1))
        bullets = []

    SECTION_KEYS = ("TECHNIQUE:", "FINDINGS:", "IMPRESSION:", "RECOMMENDATIONS:")

    for raw in (final_report or "").splitlines():
        line = raw.strip()
        if not line:
            flush_bullets(); continue
        if line.startswith("## "):
            flush_bullets()
            flows.append(Paragraph(_md_to_rl(line[3:].strip()), H2))
            continue
        if any(line.upper().startswith(k) for k in SECTION_KEYS):
            flush_bullets()
            flows.append(Paragraph(_md_to_rl(line), SECTION))
            continue
        if line.startswith("* "):
            bullets.append(line[2:].strip()); continue
        flush_bullets()
        flows.append(Paragraph(_md_to_rl(line), BODY))
    flush_bullets()
    return flows

# ---------- main ----------
def generate_pdf_report(
    patient_details: Union[Dict[str, Any], str],
    symptoms: str,
    final_report: str,
    image_path: str,
    output_path: Optional[str] = None,
) -> str:
    # normalize patient details
    if isinstance(patient_details, dict):
        pdict = {
            "name": patient_details.get("name", "N/A"),
            "age": patient_details.get("age", "N/A"),
            "gender": patient_details.get("gender", "N/A"),
            "mrn": patient_details.get("mrn", "N/A"),
            "medical_history": patient_details.get("medical_history", "None reported"),
        }
        parsed_symptoms = None
    else:
        parsed = _parse_patient_details_text(patient_details or "")
        pdict = {
            "name": parsed.get("name", "N/A"),
            "age": parsed.get("age", "N/A"),
            "gender": parsed.get("gender", "N/A"),
            "mrn": parsed.get("mrn", "N/A"),
            "medical_history": parsed.get("medical_history", "None reported"),
        }
        parsed_symptoms = parsed.get("symptoms")

    symptoms_text = (symptoms or "").strip() or (parsed_symptoms or "")

    if output_path is None:
        output_path = f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    _ensure_dir(output_path)

    # compact styles
    base = getSampleStyleSheet()
    Title = ParagraphStyle(
        "TitleCompact",
        parent=base["Heading1"],
        fontSize=18, leading=20, textColor=colors.HexColor("#1f4788"),
        spaceAfter=6, alignment=TA_CENTER,
    )
    H1 = ParagraphStyle(
        "H1Compact",
        parent=base["Heading2"],
        fontSize=12.5, leading=14, textColor=colors.HexColor("#1f4788"),
        spaceAfter=4, spaceBefore=4,
    )
    H2 = ParagraphStyle(
        "H2Compact",
        parent=base["Heading3"],
        fontSize=11, leading=12.5, textColor=colors.HexColor("#1f4788"),
        spaceAfter=2, spaceBefore=4,
    )
    SECTION = ParagraphStyle(
        "SectionCompact",
        parent=base["Heading3"],
        fontSize=11, leading=12.5, textColor=colors.HexColor("#0f3057"),
        spaceAfter=2, spaceBefore=6,
    )
    BODY = ParagraphStyle(
        "BodyCompact",
        parent=base["BodyText"],
        fontSize=9.2, leading=11.2, alignment=TA_LEFT,
        spaceAfter=1,
    )
    FOOT = ParagraphStyle(
        "FooterCompact",
        parent=base["Normal"],
        fontSize=7.8, leading=9, textColor=colors.grey, alignment=TA_CENTER,
    )

    # compact page layout
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    story: List = []

    # title
    story.append(Paragraph("CHEST X-RAY RADIOLOGY REPORT", Title))

    # patient info (tight table)
    story.append(Paragraph("PATIENT INFORMATION", H1))
    table = Table([
        ["Patient Name:", str(pdict.get("name", "N/A"))],
        ["Age:", str(pdict.get("age", "N/A"))],
        ["Gender:", str(pdict.get("gender", "N/A"))],
        ["MRN:", str(pdict.get("mrn", "N/A"))],
        ["Date:", datetime.now().strftime("%Y-%m-%d")],
        ["Medical History:", str(pdict.get("medical_history", "None reported"))],
    ], colWidths=[1.6 * inch, 3.9 * inch])  # narrower
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#edf6fb")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.8),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
        ("TOPPADDING", (0,0), (-1,-1), 2.5),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
    ]))
    story.append(table)

    # indication
    story.append(Paragraph("CLINICAL INDICATION", H1))
    story.append(Paragraph(symptoms_text or "Not provided.", BODY))

    # image heading + smaller image (preserve aspect)
    story.append(Paragraph("CHEST X-RAY", H1))
    try:
        pil_im = PILImage.open(image_path)
        target_w = 4.0 * inch
        aspect = pil_im.height / float(pil_im.width or 1)
        target_h = target_w * aspect
        # cap height if too tall
        max_h = 4.0 * inch
        if target_h > max_h:
            target_h = max_h
            target_w = target_h / aspect
        img = RLImage(image_path, width=target_w, height=target_h)
        img.hAlign = "CENTER"
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"[Image not available: {e}]", BODY))

    # report (no page break; compact)
    story.append(Paragraph("RADIOLOGY REPORT", H1))
    story.extend(_split_report_to_flowables(final_report, H2, BODY, SECTION))

    # footer
    story.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", FOOT))
    story.append(Paragraph("This report is electronically generated and verified.", FOOT))

    doc.build(story)
    return output_path
