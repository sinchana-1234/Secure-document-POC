"""
dedup.py — Duplicate detection, cheap-first.

TWO LAYERS
──────────
LAYER 1  EXACT  (free, instant)
    SHA-256 of the raw file bytes.  If it matches any existing document's
    file_hash, it IS the same file — reject before touching OpenAI or Pinecone.
    Covers 99 % of real-world re-uploads (same PDF, same name, same bytes).

LAYER 2  NEAR-DUPLICATE  (semantic, only runs if layer 1 passes)
    A report re-exported as a new PDF, or a document with only a timestamp
    changed, will have different bytes but almost identical meaning.  We embed
    a representative sample of the document's chunks and query Pinecone for the
    nearest existing vectors.  If the average cosine similarity of the sample
    exceeds NEAR_DUP_THRESHOLD the document is flagged as a near-duplicate.

    Sampling strategy: we take the FIRST, LAST, and up to (SAMPLE_SIZE - 2)
    evenly-spaced middle chunks.  This gives a better cross-document picture
    than using only the first chunk, which is often a cover page or boilerplate.

WHY NOT JUST ONE CHECK?
    Exact match is O(1) with a DB index and costs nothing.
    Near-dup requires an OpenAI embedding call + a Pinecone query, so we only
    pay that cost when the file is provably new bytes.

THREAD SAFETY
    Both functions are stateless and safe to call concurrently.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document
from app.services import vectorstore
from app.services.vectorstore import VectorStoreAPIError

logger = logging.getLogger("doc-poc.dedup")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# How many chunk vectors to average when probing for near-duplicates.
# More samples = more accurate, but more Pinecone queries.
# 5 is a good balance for most document sizes.
_SAMPLE_SIZE = 5


# ---------------------------------------------------------------------------
# Layer 1 — Exact duplicate (hash)
# ---------------------------------------------------------------------------

def find_exact_duplicate(db: Session, file_hash: str) -> Optional[Document]:
    """
    Return the first Document whose SHA-256 hash matches, or None.

    Uses the indexed file_hash column — O(log n) lookup, no full table scan.
    We intentionally return the first match only; duplicates of duplicates are
    still duplicates of the original.
    """
    match = db.query(Document).filter(Document.file_hash == file_hash).first()
    if match:
        logger.info(
            "Exact duplicate detected: new file matches document id=%d (hash=%s…)",
            match.id,
            file_hash[:12],
        )
    return match


# ---------------------------------------------------------------------------
# Layer 2 — Near-duplicate (semantic / vector similarity)
# ---------------------------------------------------------------------------

def _select_sample_indices(total: int, sample_size: int) -> List[int]:
    """
    Pick up to `sample_size` representative chunk indices from a document.

    Strategy: always include first and last chunk; fill the middle with
    evenly-spaced indices.  Works correctly for documents with fewer chunks
    than the sample size.

    Examples (total=10, sample_size=5) → [0, 2, 4, 7, 9]
             (total=3,  sample_size=5) → [0, 1, 2]
             (total=1,  sample_size=5) → [0]
    """
    if total <= sample_size:
        return list(range(total))

    indices = {0, total - 1}  # always include first and last
    if sample_size > 2:
        # Distribute the remaining slots evenly across the interior
        step = (total - 1) / (sample_size - 1)
        for i in range(1, sample_size - 1):
            indices.add(round(i * step))

    return sorted(indices)


def _average_vector(vectors: List[List[float]]) -> List[float]:
    """
    Component-wise average of a list of vectors.
    Returns a unit-length representative vector for the sample set.
    """
    if not vectors:
        raise ValueError("Cannot average an empty list of vectors.")

    dim = len(vectors[0])
    avg = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            avg[i] += x

    n = len(vectors)
    avg = [x / n for x in avg]

    # L2-normalise so cosine similarity == dot product in Pinecone
    magnitude = sum(x * x for x in avg) ** 0.5
    if magnitude > 0:
        avg = [x / magnitude for x in avg]

    return avg


def find_near_duplicate(
    vectors: List[List[float]],
) -> Tuple[Optional[int], float]:
    """
    Check whether a new document is semantically near-duplicate of an existing one.

    Parameters
    ----------
    vectors:
        All chunk embeddings for the new document (from embeddings.embed_texts).

    Returns
    -------
    (existing_document_id, similarity_score)
        existing_document_id is None if no near-duplicate is found.
        similarity_score is the cosine similarity of the sample probe (0–1).

    Raises
    ------
    Does NOT raise on Pinecone errors — near-dup failure is non-fatal.
    A warning is logged and (None, 0.0) is returned so the upload continues.
    This avoids blocking every upload when Pinecone is temporarily unavailable.
    """
    if not vectors:
        logger.debug("No vectors to probe; skipping near-duplicate check.")
        return None, 0.0

    # 1. Select a representative sample of chunk indices
    sample_indices = _select_sample_indices(len(vectors), _SAMPLE_SIZE)
    sample_vectors = [vectors[i] for i in sample_indices]

    logger.debug(
        "Near-dup probe: %d total chunks, sampling indices %s",
        len(vectors),
        sample_indices,
    )

    # 2. Average the sample into a single representative vector
    probe_vector = _average_vector(sample_vectors)

    # 3. Query Pinecone for the closest existing vector
    try:
        nearest = vectorstore.find_nearest(probe_vector)
    except VectorStoreAPIError as exc:
        # Non-fatal: log and allow the upload to proceed
        logger.warning(
            "Near-duplicate check skipped — Pinecone query failed: %s", exc
        )
        return None, 0.0

    if nearest is None:
        # Index is empty — first document, definitely not a duplicate
        logger.debug("Pinecone index is empty; no near-duplicate possible.")
        return None, 0.0

    score = float(nearest.get("score", 0.0))
    doc_id_raw = nearest.get("metadata", {}).get("document_id")

    logger.debug(
        "Nearest Pinecone match: score=%.4f, document_id=%s",
        score,
        doc_id_raw,
    )

    if score < settings.NEAR_DUP_THRESHOLD:
        logger.debug(
            "Score %.4f < threshold %.4f — not a near-duplicate.",
            score,
            settings.NEAR_DUP_THRESHOLD,
        )
        return None, score

    # Score meets or exceeds threshold
    if doc_id_raw is None:
        logger.warning(
            "Near-duplicate threshold exceeded (score=%.4f) but metadata "
            "missing document_id — cannot identify the original. Allowing upload.",
            score,
        )
        return None, score

    existing_id = int(doc_id_raw)
    logger.info(
        "Near-duplicate detected: similarity=%.4f >= threshold=%.4f, "
        "matches document id=%d",
        score,
        settings.NEAR_DUP_THRESHOLD,
        existing_id,
    )
    return existing_id, score