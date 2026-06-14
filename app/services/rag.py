"""
rag.py — Retrieval-Augmented Generation: the question-answering engine.

THE PRINCIPLE: never let the model answer from its own memory. We RETRIEVE the most
relevant chunks from the user's documents, hand them to the model as the ONLY allowed
source, and require it to cite which chunk each claim came from. If the answer isn't in
the retrieved context, the model must say so rather than invent one.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError

from app.config import settings
from app.services.embeddings import embed_query, EmbeddingError, EmbeddingConfigError
from app.services import vectorstore

logger = logging.getLogger("doc-poc.rag")
# Common greetings / chitchat that aren't document questions. Handled before any
# embedding or vector search so we don't waste an API round-trip — and so the user
# gets a friendly reply instead of "I couldn't find anything in the documents".
_GREETINGS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "yo", "hola",
    "thanks", "thank you", "thankyou", "thx", "ty", "ok", "okay", "k",
    "bye", "goodbye", "good morning", "good afternoon", "good evening",
    "how are you", "sup", "what's up", "whats up",
}

_GREETING_REPLY = (
    "Hello! I'm your document assistant. Ask me anything about your uploaded "
    "documents — for example, what a document covers, or to summarize a section."
)


def is_greeting(question: str) -> bool:
    """True if the message is a bare greeting / chitchat, not a document question."""
    cleaned = (question or "").strip().lower().rstrip("!.?")
    return cleaned in _GREETINGS


def greeting_response() -> Dict[str, Any]:
    """Friendly canned reply for greetings — same shape as answer(), no sources."""
    return {"answer": _GREETING_REPLY, "sources": []}

# ── File-retrieval intent ────────────────────────────────────────────────────
# Some messages ask for a FILE ("get me the firewall guide"), not an answer about
# its contents. We detect that intent and return the matching document for download,
# instead of running Q&A (which would confusingly say "I can't find an answer").
_FILE_VERBS = ("get", "download", "give me", "show me", "fetch", "find", "send me", "open", "pull up")
_FILE_NOUNS = ("file", "document", "doc", "pdf", "docx", "attachment")


def is_file_request(question: str) -> bool:
    """True if the user is asking to retrieve a file, not asking about its contents."""
    q = (question or "").lower()
    has_verb = any(v in q for v in _FILE_VERBS)
    has_noun = any(n in q for n in _FILE_NOUNS)
    return has_verb and has_noun


def file_response(doc) -> Dict[str, Any]:
    """Return a matched document as a downloadable result (renders as a download chip)."""
    title = doc.title or doc.original_filename
    return {
        "answer": f"Here's the document you asked for: \"{title}\". You can download it below.",
        "sources": [{
            "ref": 1,
            "document_id": doc.id,
            "title": title,
            "doc_type": doc.doc_type,
            "chunk_index": None,
            "score": 1.0,
            "snippet": doc.original_filename,
        }],
    }


def file_not_found_response() -> Dict[str, Any]:
    return {
        "answer": "I couldn't find a document matching that name. Check the Files tab to see all your documents.",
        "sources": [],
    }

class RagError(Exception):
    """Base class for RAG failures."""


class RagInputError(RagError):
    """Bad caller input (e.g. empty question) — maps to a 400."""


class RagConfigError(RagError):
    """Missing/invalid OpenAI key — operator must fix; maps to a 503."""


class RagAPIError(RagError):
    """Upstream failure (embeddings, vector search, or chat) — maps to a 502/503."""


_MIN_TOP_K = 1
_MAX_TOP_K = 20

_client: Optional[OpenAI] = None


def _client_or_raise() -> OpenAI:
    global _client
    if not settings.OPENAI_API_KEY:
        raise RagConfigError("OPENAI_API_KEY is not set. Add it to backend/.env to enable answers.")
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3, timeout=30.0)
    return _client


def _clamp_top_k(top_k: Optional[int]) -> int:
    if not top_k or top_k < _MIN_TOP_K:
        return settings.RETRIEVAL_TOP_K
    return min(top_k, _MAX_TOP_K)


def build_metadata_filter(*, doc_type: Optional[str] = None, tags=None,
                          owner_ids=None, date_from_ts: Optional[int] = None,
                          date_to_ts: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Translate UI filters into Pinecone's Mongo-style operators. owner_ids is the RBAC
    enforcement point: a non-admin's search is restricted to their own docs.
    """
    f: Dict[str, Any] = {}
    if doc_type:
        f["doc_type"] = {"$eq": doc_type}
    if tags:
        f["tags"] = {"$in": tags if isinstance(tags, list) else [tags]}
    if owner_ids:
        f["owner_id"] = {"$in": owner_ids}
    if date_from_ts is not None or date_to_ts is not None:
        rng: Dict[str, Any] = {}
        if date_from_ts is not None:
            rng["$gte"] = date_from_ts
        if date_to_ts is not None:
            rng["$lte"] = date_to_ts
        f["upload_ts"] = rng
    return f or None


def retrieve(question: str, top_k: Optional[int] = None,
             metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Embed the question and pull the most similar chunks (within the filter)."""
    if not question or not question.strip():
        raise RagInputError("Question cannot be empty.")

    k = _clamp_top_k(top_k)

    try:
        qvec = embed_query(question)
    except EmbeddingConfigError as e:
        raise RagConfigError(str(e)) from e
    except EmbeddingError as e:
        raise RagAPIError(f"Could not embed the question: {e}") from e

    try:
        return vectorstore.query(qvec, top_k=k, metadata_filter=metadata_filter)
    except Exception as e:  # noqa: BLE001 — vectorstore raises its own typed errors
        raise RagAPIError(f"Vector search failed: {e}") from e


_SYSTEM_PROMPT = (
    "You are a document analyst. Answer the user's question using ONLY the numbered "
    "context sources provided. If the answer is not contained in the sources, say you "
    "cannot find it in the provided documents — do not use outside knowledge. After each "
    "claim, cite the supporting source number(s) in square brackets like [1] or [2]."
)


def answer(question: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn retrieved chunks into a cited answer. No matches => honest 'not found'."""
    if not matches:
        return {
            "answer": "I couldn't find anything relevant in the documents you have access to.",
            "sources": [],
        }

    context_lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        text = md.get("text", "")
        context_lines.append(
            f"[{i}] (document_id={md.get('document_id')}, title={md.get('title')}): {text}"
        )
        sources.append({
            "ref": i,
            "document_id": md.get("document_id"),
            "title": md.get("title"),
            "doc_type": md.get("doc_type"),
            "chunk_index": md.get("chunk_index"),
            "score": round(float(m.get("score", 0.0)), 4),
            "snippet": text[:300],
        })

    context = "\n\n".join(context_lines)
    client = _client_or_raise()

    try:
        resp = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context sources:\n{context}\n\nQuestion: {question}"},
            ],
        )
    except AuthenticationError as e:
        raise RagConfigError("OpenAI rejected the API key (check OPENAI_API_KEY).") from e
    except RateLimitError as e:
        raise RagAPIError("OpenAI rate limit or quota exceeded — try again later.") from e
    except OpenAIError as e:
        raise RagAPIError(f"OpenAI chat request failed: {e}") from e

    answer_text = resp.choices[0].message.content or ""
    logger.info("Answered question with %d source(s).", len(sources))
    return {"answer": answer_text, "sources": sources}