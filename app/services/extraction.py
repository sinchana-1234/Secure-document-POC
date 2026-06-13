"""
Content extraction = turn an uploaded file into plain text + light metadata.
  .txt   -> decode bytes
  .docx  -> python-docx (paragraphs + tables)
  .pdf   -> pdfplumber text layer
  image  -> OCR (Tesseract) — no text layer exists
  scanned .pdf -> looks like a PDF but is pictures of pages; if almost no text comes
                  out, we rasterise each page and OCR it. Auto-detected, not asked.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from app.config import settings


@dataclass
class ExtractionResult:
    text: str
    doc_type: str
    page_count: Optional[int] = None
    ocr_used: bool = False
    mime_type: Optional[str] = None
    meta: dict = field(default_factory=dict)


_EXT_TYPE = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "docx", ".txt": "txt", ".md": "txt",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tif": "image",
    ".tiff": "image", ".bmp": "image", ".webp": "image",
}
_SCANNED_PDF_TEXT_THRESHOLD = 40


def detect_doc_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_TYPE.get(ext, "unknown")


def extract(path: str, original_filename: str) -> ExtractionResult:
    doc_type = detect_doc_type(original_filename)
    if doc_type == "txt":
        return _extract_txt(path)
    if doc_type == "docx":
        return _extract_docx(path)
    if doc_type == "image":
        return _extract_image(path)
    if doc_type == "pdf":
        return _extract_pdf(path)
    raise ValueError(f"Unsupported file type for '{original_filename}'")


def _extract_txt(path: str) -> ExtractionResult:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return ExtractionResult(text=text, doc_type="txt", mime_type="text/plain")


def _extract_docx(path: str) -> ExtractionResult:
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return ExtractionResult(
        text=text, doc_type="docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def _extract_image(path: str) -> ExtractionResult:
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang=settings.OCR_LANG)
    return ExtractionResult(text=text, doc_type="image", ocr_used=True, mime_type="image/*")


def _extract_pdf(path: str) -> ExtractionResult:
    text_parts = []
    page_count = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    text = "\n".join(text_parts).strip()

    if len(text) < _SCANNED_PDF_TEXT_THRESHOLD:
        return _ocr_scanned_pdf(path, page_count)

    return ExtractionResult(text=text, doc_type="pdf", page_count=page_count, mime_type="application/pdf")


def _ocr_scanned_pdf(path: str, page_count: int) -> ExtractionResult:
    images = convert_from_path(path)
    ocr_parts = [pytesseract.image_to_string(img, lang=settings.OCR_LANG) for img in images]
    text = "\n".join(ocr_parts).strip()
    return ExtractionResult(
        text=text, doc_type="scanned_pdf",
        page_count=page_count or len(images), ocr_used=True, mime_type="application/pdf",
    )