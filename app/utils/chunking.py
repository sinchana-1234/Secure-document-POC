"""
Chunking: split a long document into overlapping windows before embedding.

WHY chunk: an embedding model squeezes a whole input into ONE vector. Embed a 40-page
PDF as one vector and retrieval can't point to the relevant paragraph. Smaller chunks =
sharper retrieval and precise citations.

WHY OVERLAP: a sentence answering the question might sit on a chunk boundary. Overlap
means that sentence appears in BOTH neighbours, so an unlucky cut never loses it.

WHY SENTENCE-AWARE: cutting mid-sentence mangles meaning and produces noisy embeddings.
We pack whole sentences up to the limit and carry whole sentences as overlap.
"""
import re
from typing import List

from app.config import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    units: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENTENCE_SPLIT.split(para):
            s = s.strip()
            if s:
                units.append(s)
    return units


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping, sentence-aligned chunks. [] for empty input."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    overlap = min(overlap, chunk_size // 2)

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text[:chunk_size]]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for s in sentences:
        s_len = len(s) + 1

        if s_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, len(s), chunk_size):
                chunks.append(s[i:i + chunk_size])
            continue

        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            tail: List[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) + 1 > overlap:
                    if not tail and len(prev) + 1 <= chunk_size // 2:
                        tail.insert(0, prev)
                        tail_len += len(prev) + 1
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current = tail
            current_len = tail_len

        current.append(s)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return chunks