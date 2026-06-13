"""
RAG = Retrieve relevant chunks -> Augment the prompt with them as the ONLY source of
truth -> Generate an answer that must cite source numbers. If the context lacks the
answer, the model is told to say so rather than hallucinate. That auditability is the
whole point of "secure document intelligence".
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI

from app.config import settings
from app.services.embeddings import embed_query
from app.services import vectorstore

_client: OpenAI | None = None


def _client_or_raise() -> OpenAI:
    global _client
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to backend/.env to enable RAG answers.")
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def build_metadata_filter(*, doc_type=None, tags=None, department=None,
                          owner_ids=None, date_from_ts=None, date_to_ts=None) -> Optional[Dict[str, Any]]:
    """Translate user filters into Pinecone's Mongo-style operators ($eq/$in/$gte/$lte)."""
    f: Dict[str, Any] = {}
    if doc_type:
        f["doc_type"] = {"$eq": doc_type}
    if department:
        f["department"] = {"$eq": department}
    if tags:
        f["tags"] = {"$in": tags if isinstance(tags, list) else [tags]}
    if owner_ids:
        f["owner_id"] = {"$in": owner_ids}
    if date_from_ts is not None or date_to_ts is not None:
        rng = {}
        if date_from_ts is not None:
            rng["$gte"] = date_from_ts
        if date_to_ts is not None:
            rng["$lte"] = date_to_ts
        f["upload_ts"] = rng
    return f or None


def retrieve(question: str, top_k: int = None, metadata_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.RETRIEVAL_TOP_K
    qvec = embed_query(question)
    return vectorstore.query(qvec, top_k=top_k, metadata_filter=metadata_filter)


_SYSTEM_PROMPT = (
    "You are a document analyst. Answer the user's question using ONLY the numbered "
    "context sources provided. If the answer is not in the sources, say you cannot find "
    "it in the provided documents. After each claim, cite the source number(s) in "
    "square brackets like [1] or [2]. Do not use outside knowledge."
)


def answer(question: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not matches:
        return {"answer": "I couldn't find anything relevant in the documents you have access to.", "sources": []}

    context_lines = []
    sources = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {})
        text = md.get("text", "")
        context_lines.append(f"[{i}] (from document_id={md.get('document_id')}, title={md.get('title')}): {text}")
        sources.append({
            "ref": i, "document_id": md.get("document_id"), "title": md.get("title"),
            "doc_type": md.get("doc_type"), "chunk_index": md.get("chunk_index"),
            "score": round(float(m.get("score", 0.0)), 4), "snippet": text[:300],
        })

    context = "\n\n".join(context_lines)
    client = _client_or_raise()
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context sources:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": resp.choices[0].message.content, "sources": sources}