import base64
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from docx import Document as DocxDocument
from docx.shared import Inches
import pdfplumber


DPI = 150  # Render resolution for page images
JPEG_QUALITY = 90


def pdf_to_images(pdf_path: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Convert PDF pages to base64 PNG images.
    Returns (list_of_base64_strings, list_of_(width, height) tuples).
    """
    doc = fitz.open(pdf_path)
    images_b64 = []
    sizes = []
    mat = fitz.Matrix(DPI / 72, DPI / 72)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        images_b64.append(base64.b64encode(img_data).decode())
        sizes.append((pix.width, pix.height))

    doc.close()
    return images_b64, sizes


def docx_to_pdf(docx_path: str, output_path: str) -> str:
    """
    Convert DOCX to PDF using LibreOffice if available,
    otherwise generate a simple PDF via reportlab.
    Returns path to the PDF file.
    """
    import subprocess
    import shutil
    import os

    output_dir = str(Path(output_path).parent)

    # Try LibreOffice first
    lo_binary = shutil.which("libreoffice") or shutil.which("soffice")
    if lo_binary:
        result = subprocess.run(
            [lo_binary, "--headless", "--convert-to", "pdf", "--outdir", output_dir, docx_path],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            # LibreOffice names the output based on input filename
            base_name = Path(docx_path).stem + ".pdf"
            lo_output = Path(output_dir) / base_name
            if lo_output.exists():
                lo_output.rename(output_path)
                return output_path

    # Fallback: extract text and create simple PDF with reportlab
    return _docx_to_pdf_fallback(docx_path, output_path)


def _docx_to_pdf_fallback(docx_path: str, output_path: str) -> str:
    """Fallback DOCX to PDF conversion using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT

    doc_obj = DocxDocument(docx_path)
    pdf_doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
    )

    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        "Normal",
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        spaceAfter=4,
        alignment=TA_LEFT,
    )

    story = []
    for para in doc_obj.paragraphs:
        text = para.text.strip()
        if text:
            # Escape HTML characters
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(text, normal_style))
        else:
            story.append(Spacer(1, 6))

    pdf_doc.build(story)
    return output_path


def process_upload(file_path: str, file_type: str, session_id: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Process uploaded file and return page images as base64.
    """
    if file_type == "application/pdf" or file_path.lower().endswith(".pdf"):
        return pdf_to_images(file_path)
    elif file_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ) or file_path.lower().endswith(".docx"):
        pdf_path = str(Path(file_path).with_suffix(".pdf"))
        pdf_path = docx_to_pdf(file_path, pdf_path)
        return pdf_to_images(pdf_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def get_page_sizes_from_b64(pages_b64: list[str]) -> list[tuple[int, int]]:
    """Extract (width, height) from list of base64 page images."""
    sizes = []
    for b64 in pages_b64:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        sizes.append(img.size)
    return sizes
