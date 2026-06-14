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
    changed, will have different bytes but almost identical meaning.

    Strategy: we select a representative sample of chunk indices (first, last,
    and evenly-spaced middle chunks), then query Pinecone INDEPENDENTLY for
    each sampled vector — top-3 matches per probe.  Results are grouped by
    document_id and averaged.  A document is flagged as a near-duplicate only
    when the SAME existing document_id appears in at least MIN_CHUNK_HITS
    probes AND its average score meets or exceeds NEAR_DUP_THRESHOLD.

    Why independent queries instead of an averaged centroid?
    Averaging chunk vectors before querying produces a centroid that points
    toward the mean topic space of the document rather than any specific
    content.  This centroid matches the most "average" chunk of any large
    document in the index, not a near-duplicate specifically.  Independent
    per-chunk queries then aggregated by document_id are both more precise
    (fewer false positives from boilerplate) and more robust (a near-dup
    with slightly shifted wording still accumulates multiple hits).

WHY NOT JUST ONE CHECK?
    Exact match is O(1) with a DB index and costs nothing.
    Near-dup requires an OpenAI embedding call + Pinecone queries, so we only
    pay that cost when the file is provably new bytes.

THREAD SAFETY
    Both functions are stateless and safe to call concurrently.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document
from app.services import vectorstore
from app.services.vectorstore import VectorStoreAPIError

logger = logging.getLogger("doc-poc.dedup")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# How many chunk vectors to sample when probing for near-duplicates.
# Each sampled chunk becomes one independent Pinecone query (top-3).
# More samples = more accurate, but more API calls.
# 5 is a good balance for most document sizes.
_SAMPLE_SIZE = 5

# How many of the sampled chunks must match the SAME existing document
# before we consider it a near-duplicate.  Requiring >= 2 hits eliminates
# false positives from boilerplate chunks (cover pages, footers, disclaimers)
# that appear identically across many unrelated documents.
_MIN_CHUNK_HITS = 2

# How many nearest neighbours to fetch per probe query.
# top-3 gives us enough signal to find the right document_id even when the
# closest chunk belongs to a different doc (e.g. shared boilerplate).
_TOP_K_PER_PROBE = 3


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
    match = db.query(Document).filter(
        Document.file_hash == file_hash,
        Document.deleted_at.is_(None),
    ).first()
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


def find_near_duplicate(
    vectors: List[List[float]],
    owner_id: int,
    exclude_doc_id: Optional[int] = None,
) -> Tuple[Optional[int], float]:
    """
    Check whether a new document is semantically near-duplicate of an existing one.

    Each sampled chunk vector is queried against Pinecone independently.
    Results are grouped by document_id.  A near-duplicate is declared only
    when the same document_id accumulates at least _MIN_CHUNK_HITS probe
    matches AND its average similarity meets NEAR_DUP_THRESHOLD.

    Parameters
    ----------
    vectors:
        All chunk embeddings for the new document (from embeddings.embed_texts).
    owner_id:
        The uploading user's ID.  Pinecone queries are scoped to this owner so
        users cannot inadvertently learn about each other's documents.
    exclude_doc_id:
        Optional document ID to exclude from results.  Pass the new document's
        own ID when re-processing to prevent self-matches.

    Returns
    -------
    (existing_document_id, similarity_score)
        existing_document_id is None if no near-duplicate is found.
        similarity_score is the average cosine similarity of matched probes (0–1).

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

    logger.debug(
        "Near-dup probe: %d total chunks, sampling indices %s (owner_id=%d, exclude=%s)",
        len(vectors),
        sample_indices,
        owner_id,
        exclude_doc_id,
    )

    # 2. Query each sampled chunk vector independently and accumulate scores
    #    grouped by document_id.
    scores_by_doc: Dict[int, List[float]] = defaultdict(list)

    for idx in sample_indices:
        try:
            matches = vectorstore.query_for_dedup(
                vector=vectors[idx],
                owner_id=owner_id,
                top_k=_TOP_K_PER_PROBE,
                exclude_doc_id=exclude_doc_id,
            )
        except VectorStoreAPIError as exc:
            # Non-fatal: a single probe failure is logged and skipped.
            # If ALL probes fail the loop exits and we return (None, 0.0) below.
            logger.warning(
                "Near-dup probe %d skipped — Pinecone query failed: %s", idx, exc
            )
            continue

        for match in matches:
            doc_id_raw = match.get("metadata", {}).get("document_id")
            score = float(match.get("score", 0.0))
            if doc_id_raw is not None:
                scores_by_doc[int(doc_id_raw)].append(score)

    if not scores_by_doc:
        # Index is empty or every probe failed
        logger.debug("No Pinecone matches across all probes; no near-duplicate.")
        return None, 0.0

    # 3. Find the best candidate: must have enough hits AND meet the threshold
    best_doc_id: Optional[int] = None
    best_avg_score: float = 0.0

    for doc_id, scores in scores_by_doc.items():
        if len(scores) < _MIN_CHUNK_HITS:
            logger.debug(
                "doc_id=%d: only %d/%d required chunk hits — skipping.",
                doc_id,
                len(scores),
                _MIN_CHUNK_HITS,
            )
            continue

        avg_score = sum(scores) / len(scores)

        logger.debug(
            "doc_id=%d: %d chunk hits, avg_score=%.4f (threshold=%.4f)",
            doc_id,
            len(scores),
            avg_score,
            settings.NEAR_DUP_THRESHOLD,
        )

        if avg_score >= settings.NEAR_DUP_THRESHOLD and avg_score > best_avg_score:
            best_doc_id = doc_id
            best_avg_score = avg_score

    if best_doc_id is None:
        logger.debug(
            "No document met both the hit count (%d) and threshold (%.4f) criteria.",
            _MIN_CHUNK_HITS,
            settings.NEAR_DUP_THRESHOLD,
        )
        return None, best_avg_score

    logger.info(
        "Near-duplicate detected: avg_similarity=%.4f >= threshold=%.4f, "
        "matches document id=%d",
        best_avg_score,
        settings.NEAR_DUP_THRESHOLD,
        best_doc_id,
    )
    return best_doc_id, best_avg_score