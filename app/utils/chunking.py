"""
Chunking: split a long document into OVERLAPPING windows before embedding.
Smaller chunks = sharper retrieval + precise citations. Overlap means a sentence
sitting on a boundary appears in both neighbours, so we never lose it to a cut.
"""
from typing import List
from app.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)   # guard against overlap >= chunk_size
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks