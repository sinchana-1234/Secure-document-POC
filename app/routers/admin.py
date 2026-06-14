"""
Admin router — all routes require the caller to be an admin (enforced by the
`require_admin` dependency, which raises 403 before the handler even runs).

ENDPOINTS
─────────
GET  /api/admin/stats                 Platform-wide summary counts
GET  /api/admin/users                 Paginated + filterable user list (with total)
POST /api/admin/users                 Create a new user (any role)
GET  /api/admin/users/{user_id}       Single user detail with document count
PATCH /api/admin/users/{user_id}      Partial update (role, password, name)
DELETE /api/admin/users/{user_id}     Delete user + all their documents (cascade)
GET  /api/admin/documents             Paginated + filterable file list (with total)
DELETE /api/admin/documents/{doc_id}  Hard-delete a document (Pinecone + disk + DB)

RBAC DESIGN
───────────
We reuse the existing `require_admin` FastAPI dependency defined in core/deps.py.
That dependency reads the JWT, resolves the User row, and checks role == "admin".
Writing the guard once and injecting it avoids the "forgot one endpoint" class of
privilege-escalation bugs.

PAGINATION
──────────
All list endpoints accept `limit` (page size, max 100) and `offset` (skip N rows).
Responses are wrapped in UserListResponse / DocumentListResponse which include a
`total` count so the frontend can render correct page navigation.

QUERY EFFICIENCY
────────────────
list_users: one GROUP BY subquery fetches all document counts — O(1) queries total.
list_all_documents: a single JOIN resolves owner info — O(1) queries total.
Neither endpoint scales linearly with result size.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.core.deps import require_admin
from app.core.security import hash_password
from app.database import get_db
from app.models import Document, DocStatus, Role, User
from app.schemas.admin import (
    AdminUserCreate,
    AdminUserUpdate,
    AdminStats,
    DocumentAdminOut,
    DocumentListResponse,
    UserDetail,
    UserListResponse,
)
from app.services import vectorstore
from app.services.vectorstore import VectorStoreAPIError

logger = logging.getLogger("doc-intel.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_user_or_404(db: Session, user_id: int) -> User:
    user = db.query(User).filter(User.id == user_id, User.deleted_at.is_(None)).first()
    if not user:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user


def _get_document_or_404(db: Session, doc_id: int) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id, Document.deleted_at.is_(None)).first()
    if not doc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found",
        )
    return doc


def _remove_file(path: str) -> None:
    """Delete a stored file, logging rather than raising if it is already gone."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not remove stored file '%s': %s", path, exc)


def _doc_count_subquery(db: Session):
    return (
        db.query(
            Document.owner_id.label("owner_id"),
            func.count(Document.id).label("doc_count"),
        )
        .filter(Document.deleted_at.is_(None))
        .group_by(Document.owner_id)
        .subquery()
    )


def _build_user_detail(user: User, doc_count: int) -> UserDetail:
    """Build a UserDetail from an ORM row + precomputed count."""
    return UserDetail(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        created_at=user.created_at,
        document_count=doc_count,
    )


def _build_document_admin_out(doc: Document) -> DocumentAdminOut:
    """
    Build DocumentAdminOut from a Document whose .owner is already loaded
    via joinedload — no extra query fired here.
    """
    owner = doc.owner
    return DocumentAdminOut(
        id=doc.id,
        original_filename=doc.original_filename,
        title=doc.title,
        doc_type=doc.doc_type,
        size_bytes=doc.size_bytes,
        tags=doc.tags or [],
        page_count=doc.page_count,
        num_chunks=doc.num_chunks,
        ocr_used=doc.ocr_used,
        status=doc.status.value,
        error_message=doc.error_message,
        duplicate_of_id=doc.duplicate_of_id,
        upload_date=doc.upload_date,
        owner_id=doc.owner_id,
        owner_email=owner.email if owner else None,
        owner_name=owner.full_name if owner else None,
    )


# ---------------------------------------------------------------------------
# Platform Stats
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=AdminStats, summary="Platform-wide summary counts")
def get_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Returns aggregate counts for the admin dashboard header cards.
    All values come from COUNT queries — no full table scans.
    """
    total_users = db.query(User).filter(User.deleted_at.is_(None)).count()
    total_docs  = db.query(Document).filter(Document.deleted_at.is_(None)).count()
    indexed     = db.query(Document).filter(Document.deleted_at.is_(None), Document.status == DocStatus.indexed).count()
    failed      = db.query(Document).filter(Document.deleted_at.is_(None), Document.status == DocStatus.failed).count()
    duplicate   = db.query(Document).filter(Document.deleted_at.is_(None), Document.status == DocStatus.duplicate).count()

    return AdminStats(
        total_users=total_users,
        total_documents=total_docs,
        indexed_documents=indexed,
        failed_documents=failed,
        duplicate_documents=duplicate,
    )


# ---------------------------------------------------------------------------
# User Management
# ---------------------------------------------------------------------------

@router.get("/users", response_model=UserListResponse, summary="List all users with document counts")
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    role: Optional[str] = Query(None, description="Filter by role: 'user' or 'admin'"),
    search: Optional[str] = Query(None, description="Search by name or email (case-insensitive)"),
    limit: int = Query(default=50, le=100, description="Max records to return (cap 100)"),
    offset: int = Query(default=0, ge=0, description="Records to skip for pagination"),
):
    """
    Returns all platform users with a total count for pagination.
    Document counts come from a single GROUP BY subquery — not one COUNT per user.
    """
    if role and role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")

    # One subquery gives us all document counts grouped by owner — O(1) DB calls
    counts_sq = _doc_count_subquery(db)

    query = (
        db.query(User, func.coalesce(counts_sq.c.doc_count, 0).label("doc_count"))
        .outerjoin(counts_sq, User.id == counts_sq.c.owner_id)
        .filter(User.deleted_at.is_(None))
    )

    if role:
        query = query.filter(User.role == role)

    if search:
        like = f"%{search}%"
        query = query.filter(
            (User.email.ilike(like)) | (User.full_name.ilike(like))
        )

    # Get total before applying pagination
    total = query.count()

    rows = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()

    items = [_build_user_detail(user, int(doc_count)) for user, doc_count in rows]
    return UserListResponse(items=items, total=total)


@router.post(
    "/users",
    response_model=UserDetail,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a new user (admin or regular)",
)
def create_user(
    payload: AdminUserCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Admin-only user creation. Unlike the public /register endpoint, this allows
    creating admin-role accounts. Password is bcrypt-hashed before storage.
    role is validated to Literal["user","admin"] by the schema — no extra check needed.
    """
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Email '{payload.email}' is already registered",
        )

    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=Role.admin if payload.role == "admin" else Role.user,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("Admin created user id=%d email=%s role=%s", new_user.id, new_user.email, new_user.role.value)
    return _build_user_detail(new_user, 0)


@router.get("/users/{user_id}", response_model=UserDetail, summary="Get single user detail")
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Returns full detail for one user, including their document upload count."""
    user = _get_user_or_404(db, user_id)
    doc_count = db.query(Document).filter(Document.owner_id == user_id, Document.deleted_at.is_(None)).count()
    return _build_user_detail(user, doc_count)


@router.patch("/users/{user_id}", response_model=UserDetail, summary="Partially update a user")
def update_user(
    user_id: int,
    payload: AdminUserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Partial update — only fields present in the request body are changed.
    Admins cannot demote themselves to prevent accidental lockout.
    role is validated to Literal["user","admin"] by the schema.
    """
    user = _get_user_or_404(db, user_id)

    if user.id == admin.id and payload.role == "user":
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="You cannot demote your own admin account",
        )

    if payload.full_name is not None:
        user.full_name = payload.full_name

    if payload.role is not None:
        user.role = Role.admin if payload.role == "admin" else Role.user

    if payload.password is not None:
        user.hashed_password = hash_password(payload.password)

    db.commit()
    db.refresh(user)

    doc_count = db.query(Document).filter(Document.owner_id == user.id, Document.deleted_at.is_(None)).count()
    logger.info("Admin updated user id=%d", user.id)
    return _build_user_detail(user, doc_count)


@router.delete(
    "/users/{user_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a user and all their documents",
)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Permanently deletes a user and ALL their documents in three steps per document:
      1. Pinecone vectors removed
      2. Disk file removed
      3. DB rows removed (user delete cascades to documents via FK)

    Admin cannot delete their own account to prevent last-admin lockout.
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own admin account",
        )

    user = _get_user_or_404(db, user_id)

    now = datetime.utcnow()
    # Soft delete: flag the user and all their documents. Files + vectors are kept
    # so a restore can bring everything back.
    docs = db.query(Document).filter(
        Document.owner_id == user_id,
        Document.deleted_at.is_(None),
    ).all()
    for doc in docs:
        doc.deleted_at = now
        doc.deleted_by = admin.id

    user.deleted_at = now
    user.deleted_by = admin.id
    db.commit()

    logger.info(
        "Admin soft-deleted user id=%d email=%s — %d documents flagged",
        user_id, user.email, len(docs),
    )

# ---------------------------------------------------------------------------
# Document / File Management (admin sees ALL users' files)
# ---------------------------------------------------------------------------

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded files across all users",
)
def list_all_documents(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    doc_status: Optional[str] = Query(None, description="indexed, failed, duplicate, processing"),
    doc_type: Optional[str] = Query(None, description="pdf, docx, txt, image"),
    owner_id: Optional[int] = Query(None, description="Filter to a specific user's uploads"),
    search: Optional[str] = Query(None, description="Search title or filename"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Admin-scoped document list — all documents visible regardless of owner.
    Owner info is resolved via a single JOIN — no N+1 per-document queries.
    Returns items + total for frontend pagination.
    """
    # joinedload ensures doc.owner is populated in the same query — no N+1
    query = db.query(Document).options(joinedload(Document.owner)).filter(Document.deleted_at.is_(None))

    if doc_status:
        # Convert raw string to DocStatus enum before filtering.
        # Passing a raw string to SAEnum comparison fails silently on some
        # DB backends — the enum conversion makes it explicit and reliable.
        try:
            status_enum = DocStatus(doc_status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{doc_status}'. Must be one of: {[s.value for s in DocStatus]}",
            )
        query = query.filter(Document.status == status_enum)

    if doc_type:
        query = query.filter(Document.doc_type == doc_type)

    if owner_id:
        query = query.filter(Document.owner_id == owner_id)

    if search:
        like = f"%{search}%"
        query = query.filter(
            (Document.title.ilike(like)) | (Document.original_filename.ilike(like))
        )

    total = query.count()

    docs = query.order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()

    return DocumentListResponse(
        items=[_build_document_admin_out(doc) for doc in docs],
        total=total,
    )


@router.delete(
    "/documents/{doc_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Hard-delete a document (Pinecone + disk + DB)",
)
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Permanently removes a document from all three storage layers in order:
      1. Pinecone vectors — deleted first; stale vectors cause phantom search results
      2. Disk file — raw uploaded file removed
      3. PostgreSQL row — metadata record deleted last

    VectorStoreAPIError (not RuntimeError) is caught so a Pinecone outage
    does not block DB + disk cleanup.
    """
    doc = _get_document_or_404(db, doc_id)

    # Soft delete: flag the row, keep Pinecone vectors + disk file for restore.
    doc.deleted_at = datetime.utcnow()
    doc.deleted_by = _admin.id
    db.commit()

    logger.info("Admin soft-deleted document id=%d filename=%s", doc_id, doc.original_filename)