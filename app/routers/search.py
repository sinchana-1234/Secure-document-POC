"""
AI search = source-grounded Q&A. SEMANTIC + METADATA FILTER + RBAC FILTER (a non-admin's
search is silently restricted to owner_id == self). The router enforces RBAC, translates
filters, calls rag, and maps rag's typed errors to the right HTTP status.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User, Role
from app.schemas import SearchRequest, SearchResponse
from app.services import rag
from app.services.rag import RagInputError, RagConfigError, RagAPIError

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db),
           user: User = Depends(get_current_user)):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

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