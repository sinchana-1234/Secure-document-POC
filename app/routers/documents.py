"""
Document routes: upload, list (repository + filters), get, delete.
RBAC scope is built INTO the query: admin sees all, a user sees only their own rows,
so a user can never even read another user's document by id ("fetch then check" is
one forgotten check away from a leak).
"""
import json
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models import User, Role, Document, DocStatus
from app.schemas import DocumentOut, UploadResponse
from app.services import vectorstore
from app.services.pipeline import ingest, DuplicateError

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _scoped_query(db: Session, user: User):
    q = db.query(Document)
    if user.role != Role.admin:
        q = q.filter(Document.owner_id == user.id)
    return q


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")

    parsed_tags = []
    if tags:
        try:
            parsed_tags = json.loads(tags)
            if not isinstance(parsed_tags, list):
                parsed_tags = [str(parsed_tags)]
        except json.JSONDecodeError:
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        doc = ingest(db, file_bytes=contents, original_filename=file.filename, owner=user,
                     title=title, tags=parsed_tags)
    except DuplicateError as e:
        return UploadResponse(status="duplicate", message=str(e), duplicate_of_id=e.existing_id,
                              duplicate_kind=e.kind, similarity=e.score)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))   # missing keys -> clear 503, not hidden 500

    return UploadResponse(status=doc.status.value, document=DocumentOut.model_validate(doc))


@router.get("", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    q: Optional[str] = Query(None, description="keyword in title/filename/text"),
    doc_type: Optional[str] = None,
    tag: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
):
    query = _scoped_query(db, user)
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Document.title.ilike(like)) |
            (Document.original_filename.ilike(like)) |
            (Document.extracted_text.ilike(like))
        )
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    
    if status:
        query = query.filter(Document.status == status)
    if tag:
        query = query.filter(Document.tags.contains([tag]))
    if date_from:
        query = query.filter(Document.upload_date >= date_from)
    if date_to:
        query = query.filter(Document.upload_date <= date_to)

    return query.order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = _scoped_query(db, user).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = _scoped_query(db, user).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        if doc.status == DocStatus.indexed:
            vectorstore.delete_document(doc.id)
    except RuntimeError:
        pass
    if doc.stored_path and os.path.exists(doc.stored_path):
        os.remove(doc.stored_path)
    db.delete(doc)
    db.commit()
    return {"status": "deleted", "id": doc_id}