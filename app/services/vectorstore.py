"""
vectorstore.py — Pinecone vector storage + search.

Postgres holds a document's facts; Pinecone holds its MEANING — one vector per chunk,
searched by closeness. This module owns creating the index, putting vectors in, finding
similar ones (with metadata filters), and removing a document's vectors.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logger = logging.getLogger("doc-poc.vectorstore")


class VectorStoreError(Exception):
    """Base class for vector-store failures."""


class VectorStoreConfigError(VectorStoreError):
    """Missing key or a mismatched/misconfigured index — operator must fix."""


class VectorStoreAPIError(VectorStoreError):
    """Upstream Pinecone failure (network, server, quota)."""


_UPSERT_BATCH = 100
_INDEX_READY_TIMEOUT_S = 120

# Module-level singletons — guarded by _lock to prevent double-initialisation
# under concurrent FastAPI requests (multiple threads can all see None at once
# without the lock, each calling ensure_index() and create_index() in parallel).
_lock: threading.Lock = threading.Lock()
_pc: Optional[Pinecone] = None
_index = None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field whether the SDK returns dicts or objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except Exception:  # noqa: BLE001
        return default


def _client() -> Pinecone:
    global _pc
    if not settings.PINECONE_API_KEY:
        raise VectorStoreConfigError("PINECONE_API_KEY is not set. Add it to backend/.env.")
    # Fast path — client already built, no lock needed (read of a reference is atomic in CPython)
    if _pc is not None:
        return _pc
    with _lock:
        # Re-check inside the lock: another thread may have built it while we waited
        if _pc is None:
            _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def _wait_until_ready(pc: Pinecone, name: str) -> None:
    """Serverless indexes provision asynchronously; block until ready (or time out)."""
    start = time.time()
    while time.time() - start < _INDEX_READY_TIMEOUT_S:
        desc = pc.describe_index(name)
        if _get(_get(desc, "status", {}), "ready", False):
            logger.info("Pinecone index '%s' is ready.", name)
            return
        time.sleep(2)
    raise VectorStoreAPIError(f"Index '{name}' did not become ready within {_INDEX_READY_TIMEOUT_S}s.")


def ensure_index() -> None:
    """Create the index if missing (matching our embeddings), else verify it matches."""
    pc = _client()
    try:
        existing = pc.list_indexes().names()
    except Exception as e:  # noqa: BLE001
        raise VectorStoreAPIError(f"Could not list Pinecone indexes: {e}") from e

    if settings.PINECONE_INDEX not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%s, metric=%s, %s/%s)",
            settings.PINECONE_INDEX,
            settings.EMBEDDING_DIM,
            settings.PINECONE_METRIC,
            settings.PINECONE_CLOUD,
            settings.PINECONE_REGION,
        )
        try:
            pc.create_index(
                name=settings.PINECONE_INDEX,
                dimension=settings.EMBEDDING_DIM,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
            )
        except Exception as e:  # noqa: BLE001
            raise VectorStoreAPIError(f"Failed to create Pinecone index: {e}") from e
        _wait_until_ready(pc, settings.PINECONE_INDEX)
    else:
        desc = pc.describe_index(settings.PINECONE_INDEX)
        dim = _get(desc, "dimension")
        if dim is not None and int(dim) != settings.EMBEDDING_DIM:
            raise VectorStoreConfigError(
                f"Pinecone index '{settings.PINECONE_INDEX}' has dimension {dim}, but "
                f"EMBEDDING_DIM is {settings.EMBEDDING_DIM}. Recreate the index or fix config."
            )
        logger.info("Pinecone index '%s' present (dim=%s).", settings.PINECONE_INDEX, dim)


def get_index():
    global _index
    # Fast path — index already initialised
    if _index is not None:
        return _index
    with _lock:
        # Re-check inside the lock: another thread may have initialised it while we waited
        if _index is None:
            ensure_index()
            _index = _client().Index(settings.PINECONE_INDEX)
    return _index


def upsert_chunks(
    document_id: int,
    chunks: List[str],
    vectors: List[List[float]],
    base_metadata: Dict[str, Any],
) -> int:
    """Store one vector per chunk. Returns how many were upserted."""
    if len(chunks) != len(vectors):
        raise VectorStoreError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch.")
    if not chunks:
        return 0
    if len(vectors[0]) != settings.EMBEDDING_DIM:
        raise VectorStoreError(
            f"Vector dimension {len(vectors[0])} != index dimension {settings.EMBEDDING_DIM}."
        )

    index = get_index()
    items = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        md = dict(base_metadata)
        md["text"] = chunk[:3000]
        md["chunk_index"] = i
        md["document_id"] = document_id
        items.append({"id": f"{document_id}:{i}", "values": vec, "metadata": md})

    upserted = 0
    try:
        for b in range(0, len(items), _UPSERT_BATCH):
            batch = items[b:b + _UPSERT_BATCH]
            res = index.upsert(vectors=batch)
            upserted += int(_get(res, "upserted_count", len(batch)))
    except Exception as e:  # noqa: BLE001
        raise VectorStoreAPIError(f"Pinecone upsert failed: {e}") from e

    logger.info("Upserted %d vectors for document %d.", upserted, document_id)
    return upserted


def query(
    vector: List[float],
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Semantic search + optional metadata filter (the 'hybrid' part)."""
    index = get_index()
    try:
        res = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or None,
        )
    except Exception as e:  # noqa: BLE001
        raise VectorStoreAPIError(f"Pinecone query failed: {e}") from e

    out: List[Dict[str, Any]] = []
    for m in (_get(res, "matches", []) or []):
        out.append({
            "id": _get(m, "id"),
            "score": float(_get(m, "score", 0.0)),
            "metadata": _get(m, "metadata", {}) or {},
        })
    return out


def query_for_dedup(
    vector: List[float],
    owner_id: int,
    top_k: int,
    exclude_doc_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scoped near-duplicate probe — used exclusively by dedup.find_near_duplicate.

    Differences from the general query():
    - Always filters by owner_id so users cannot match against each other's documents.
    - Optionally excludes a specific document_id (to avoid self-match during re-processing).
    - Returns top_k matches so the caller can aggregate scores across probes by document_id.

    Parameters
    ----------
    vector:
        A single chunk embedding to probe with.
    owner_id:
        Restrict results to documents owned by this user.
    top_k:
        Number of nearest neighbours to return per probe.
    exclude_doc_id:
        If provided, results from this document_id are filtered out.
        Pass the new document's own ID when re-processing an existing record.

    Returns
    -------
    List of match dicts with keys: id, score, metadata.
    Empty list if the index is empty or no matches are found.
    """
    # Build the filter — always scope to owner, optionally exclude self
    pinecone_filter: Dict[str, Any] = {"owner_id": {"$eq": owner_id}}
    if exclude_doc_id is not None:
        pinecone_filter["document_id"] = {"$ne": exclude_doc_id}

    logger.debug(
        "query_for_dedup: owner_id=%d, exclude_doc_id=%s, top_k=%d",
        owner_id,
        exclude_doc_id,
        top_k,
    )

    return query(vector=vector, top_k=top_k, metadata_filter=pinecone_filter)


def find_nearest(vector: List[float]) -> Optional[Dict[str, Any]]:
    """
    Top-1 match across everything — used by RAG search, not dedup.

    For near-duplicate detection use query_for_dedup() instead, which
    applies owner scoping and self-exclusion.
    """
    matches = query(vector, top_k=1, metadata_filter=None)
    return matches[0] if matches else None


def delete_document(document_id: int) -> None:
    """
    Remove every chunk of a document. Serverless can't delete by metadata filter, so we
    list the vector IDs by their "{document_id}:" prefix and delete those. Falls back to
    a filter-delete for pod-based indexes (which don't support prefix listing).
    """
    index = get_index()
    try:
        ids: Optional[List[str]] = []
        try:
            for page in index.list(prefix=f"{document_id}:"):
                if isinstance(page, (list, tuple)):
                    ids.extend(page)
                else:
                    ids.append(page)
        except Exception:  # noqa: BLE001  — pod-based index without list(): use fallback
            ids = None

        if ids:
            for b in range(0, len(ids), _UPSERT_BATCH):
                index.delete(ids=ids[b:b + _UPSERT_BATCH])
            logger.info("Deleted %d vectors for document %d.", len(ids), document_id)
        elif ids is None:
            index.delete(filter={"document_id": document_id})
            logger.info("Deleted vectors for document %d via metadata filter.", document_id)
    except Exception as e:  # noqa: BLE001
        raise VectorStoreAPIError(f"Failed to delete document {document_id}: {e}") from e