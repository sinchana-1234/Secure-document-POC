"""
Pinecone wrapper = where vectors live and how we search them. Metadata is stored ON
each vector so Pinecone can FILTER server-side during search — that's the "hybrid"
in hybrid retrieval (semantic similarity + structured filters in one query).
Each vector id is "{document_id}:{chunk_index}" so we can delete a doc's chunks.
"""
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from app.config import settings

_pc: Pinecone | None = None
_index = None


def _get_index():
    global _pc, _index
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to backend/.env to enable vector storage.")
    if _index is not None:
        return _index

    _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing = [ix["name"] for ix in _pc.list_indexes()]
    if settings.PINECONE_INDEX not in existing:
        _pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=settings.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )
    _index = _pc.Index(settings.PINECONE_INDEX)
    return _index


def upsert_chunks(document_id: int, chunks: List[str], vectors: List[List[float]],
                  base_metadata: Dict[str, Any]) -> int:
    index = _get_index()
    items = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        md = dict(base_metadata)
        md["text"] = chunk[:3000]
        md["chunk_index"] = i
        md["document_id"] = document_id
        items.append({"id": f"{document_id}:{i}", "values": vec, "metadata": md})
    for b in range(0, len(items), 100):
        index.upsert(vectors=items[b:b + 100])
    return len(items)


def query(vector: List[float], top_k: int, metadata_filter: Optional[Dict[str, Any]] = None):
    index = _get_index()
    res = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=metadata_filter or None)
    out = []
    for m in res.get("matches", []):
        out.append({"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})})
    return out


def delete_document(document_id: int) -> None:
    index = _get_index()
    index.delete(filter={"document_id": document_id})


def find_nearest(vector: List[float]) -> Optional[Dict[str, Any]]:
    matches = query(vector, top_k=1, metadata_filter=None)
    return matches[0] if matches else None