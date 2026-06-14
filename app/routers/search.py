"""
AI search = source-grounded Q&A. SEMANTIC + METADATA FILTER + RBAC FILTER (a non-admin's
search is silently restricted to owner_id == self). The router enforces RBAC, translates
filters, calls rag, and maps rag's typed errors to the right HTTP status.
"""
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User, Role, Document
from app.schemas import SearchRequest, SearchResponse
from app.services import rag
from app.services.rag import RagInputError, RagConfigError, RagAPIError

router = APIRouter(prefix="/api/search", tags=["search"])


# Words to strip from a file request so we're left with the likely filename words.
# e.g. "get me the firewall installation guide file" -> ["firewall","installation","guide"]
_FILE_STOPWORDS = {
    "get", "download", "give", "me", "show", "fetch", "find", "send", "open", "pull",
    "up", "the", "a", "an", "file", "document", "doc", "pdf", "docx", "please", "my",
    "for", "of", "to", "and", "want", "need", "can", "you",
}


def _find_document_by_name(db: Session, question: str, user: User):
    """
    Find a document whose title/filename best matches the request, RBAC-scoped.
    Scores each candidate by how many request words appear in its title/filename.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower())
             if w not in _FILE_STOPWORDS and len(w) > 2]
    if not words:
        return None

    q = db.query(Document).filter(Document.status == "indexed")
    if user.role != Role.admin:
        q = q.filter(Document.owner_id == user.id)   # RBAC: users see only their own docs

    best, best_score = None, 0
    for d in q.all():
        haystack = f"{d.title or ''} {d.original_filename}".lower()
        score = sum(1 for w in words if w in haystack)
        if score > best_score:
            best, best_score = d, score
    return best if best_score > 0 else None


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db),
           user: User = Depends(get_current_user)):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Greetings / chitchat: reply conversationally without searching the documents.
    if rag.is_greeting(payload.question):
        return SearchResponse(**rag.greeting_response())

    # File-retrieval requests ("get me the firewall guide"): return the file for download
    # instead of running Q&A (which would confusingly say "I can't find an answer").
    if rag.is_file_request(payload.question):
        doc = _find_document_by_name(db, payload.question, user)
        if doc:
            return SearchResponse(**rag.file_response(doc))
        return SearchResponse(**rag.file_not_found_response())

    owner_ids = None if user.role == Role.admin else [user.id]

    date_from_ts = int(payload.date_from.timestamp()) if payload.date_from else None
    date_to_ts = int(payload.date_to.timestamp()) if payload.date_to else None

    metadata_filter = rag.build_metadata_filter(
        doc_type=payload.doc_type, tags=payload.tags,
        owner_ids=owner_ids, date_from_ts=date_from_ts, date_to_ts=date_to_ts,
    )

    try:
        matches = rag.retrieve(payload.question, top_k=payload.top_k, metadata_filter=metadata_filter)
        result = rag.answer(payload.question, matches)
    except RagInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RagConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RagAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return SearchResponse(**result)