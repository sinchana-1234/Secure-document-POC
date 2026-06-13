"""
AI search = source-grounded Q&A. Hybrid retrieval comes together here:
  SEMANTIC (the embedded question) + METADATA FILTER (type/tags/dept/date) +
  RBAC FILTER (a non-admin's search is silently restricted to owner_id == self, so
  RAG can never surface a chunk from a document the user can't see).
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User, Role
from app.schemas import SearchRequest, SearchResponse
from app.services import rag

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    owner_ids = None
    if user.role != Role.admin:
        owner_ids = [user.id]

    date_from_ts = int(payload.date_from.timestamp()) if payload.date_from else None
    date_to_ts = int(payload.date_to.timestamp()) if payload.date_to else None

    metadata_filter = rag.build_metadata_filter(
        doc_type=payload.doc_type, tags=payload.tags, 
        owner_ids=owner_ids, date_from_ts=date_from_ts, date_to_ts=date_to_ts,
    )
    matches = rag.retrieve(payload.question, top_k=payload.top_k, metadata_filter=metadata_filter)
    result = rag.answer(payload.question, matches)
    return SearchResponse(**result)