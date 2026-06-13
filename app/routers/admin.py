"""
Admin router — all routes require the caller to be an admin (enforced by the
`require_admin` dependency, which raises 403 before the handler even runs).

ENDPOINTS
─────────
GET  /api/admin/stats                 Platform-wide summary counts
GET  /api/admin/users                 Paginated + filterable user list
POST /api/admin/users                 Create a new user (any role)
GET  /api/admin/users/{user_id}       Single user detail with document count
PATCH /api/admin/users/{user_id}      Partial update (role, dept, password, name)
DELETE /api/admin/users/{user_id}     Soft-safe delete (blocks self-delete)
GET  /api/admin/documents             Paginated + filterable file list (all users)
DELETE /api/admin/documents/{doc_id}  Hard-delete a document (disk + DB)

RBAC DESIGN
───────────
We reuse the existing `require_admin` FastAPI dependency defined in core/deps.py.
That dependency reads the JWT, resolves the User row, and checks role == "admin".
Writing the guard once and injecting it avoids the "forgot one endpoint" class of
privilege-escalation bugs.

PAGINATION
──────────
All list endpoints accept `limit` (page size, max 100) and `offset` (skip N rows).
This keeps response payloads bounded regardless of dataset size.
"""

import os
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.core.security import hash_password
from app.database import get_db
from app.models import User, Role, Document, DocStatus
from app.schemas.admin import (
    AdminUserCreate,
    AdminUserUpdate,
    UserDetail,
    DocumentAdminOut,
    AdminStats,
)
from app.services import vectorstore

logger = logging.getLogger("doc-intel.admin")

# All routes live under /api/admin — prefix enforced here, not in main.py
router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_user_or_404(db: Session, user_id: int) -> User:
    """
    Centralised lookup so every endpoint raises the same 404 shape.
    Avoids repeating the query + raise pattern in every handler.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found")
    return user


def _get_document_or_404(db: Session, doc_id: int) -> Document:
    """Same pattern for documents."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {doc_id} not found")
    return doc


def _build_user_detail(db: Session, user: User) -> UserDetail:
    """
    Attach the document count to a User ORM object and return the Pydantic schema.
    COUNT is a single aggregated query — no Python-level iteration over rows.
    """
    doc_count = db.query(Document).filter(Document.owner_id == user.id).count()
    return UserDetail(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        department=user.department,
        created_at=user.created_at,
        document_count=doc_count,
    )


def _build_document_admin_out(doc: Document, owner: Optional[User]) -> DocumentAdminOut:
    """
    Enrich a Document ORM object with its owner's email and name.
    Owner is already loaded by the join in the list query, so no extra hit.
    """
    return DocumentAdminOut(
        id=doc.id,
        original_filename=doc.original_filename,
        title=doc.title,
        doc_type=doc.doc_type,
        size_bytes=doc.size_bytes,
        department=doc.department,
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
    total_users = db.query(User).count()
    total_docs  = db.query(Document).count()

    # Simple individual counts — clear and cheap enough for an admin panel
    indexed   = db.query(Document).filter(Document.status == DocStatus.indexed).count()
    failed    = db.query(Document).filter(Document.status == DocStatus.failed).count()
    duplicate = db.query(Document).filter(Document.status == DocStatus.duplicate).count()

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

@router.get("/users", response_model=List[UserDetail], summary="List all users with document counts")
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    role: Optional[str] = Query(None, description="Filter by role: 'user' or 'admin'"),
    department: Optional[str] = Query(None, description="Filter by department"),
    search: Optional[str] = Query(None, description="Search by name or email (case-insensitive)"),
    limit: int = Query(default=50, le=100, description="Max records to return (cap 100)"),
    offset: int = Query(default=0, ge=0, description="Records to skip for pagination"),
):
    """
    Returns all platform users. Supports free-text search on email + full_name,
    and filtering by role / department. Each record includes a document_count.
    """
    query = db.query(User)

    if role:
        if role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
        query = query.filter(User.role == role)

    if department:
        query = query.filter(User.department.ilike(f"%{department}%"))

    if search:
        # ILIKE = case-insensitive LIKE in PostgreSQL
        like = f"%{search}%"
        query = query.filter(
            (User.email.ilike(like)) | (User.full_name.ilike(like))
        )

    # Stable ordering (newest first) so pagination is deterministic
    users = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()

    return [_build_user_detail(db, u) for u in users]


@router.post("/users", response_model=UserDetail, status_code=status.HTTP_201_CREATED,
             summary="Create a new user (admin or regular)")
def create_user(
    payload: AdminUserCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Admin-only user creation. Unlike the public /register endpoint, this allows
    creating admin-role accounts. Password is bcrypt-hashed before storage.
    """
    # Prevent duplicate emails at the application layer
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{payload.email}' is already registered",
        )

    role = Role.admin if payload.role == "admin" else Role.user

    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=role,
        department=payload.department,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("Admin created user id=%d email=%s role=%s", new_user.id, new_user.email, role.value)
    return _build_user_detail(db, new_user)


@router.get("/users/{user_id}", response_model=UserDetail, summary="Get single user detail")
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Returns full detail for one user, including their document upload count."""
    user = _get_user_or_404(db, user_id)
    return _build_user_detail(db, user)


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
    """
    user = _get_user_or_404(db, user_id)

    # Guard: admin cannot accidentally demote their own account
    if user.id == admin.id and payload.role == "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot demote your own admin account",
        )

    if payload.full_name is not None:
        user.full_name = payload.full_name

    if payload.department is not None:
        user.department = payload.department

    if payload.role is not None:
        if payload.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
        user.role = Role.admin if payload.role == "admin" else Role.user

    if payload.password is not None:
        # Re-hash the new password — never store plain text
        user.hashed_password = hash_password(payload.password)

    db.commit()
    db.refresh(user)

    logger.info("Admin updated user id=%d", user.id)
    return _build_user_detail(db, user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a user account")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Permanently deletes a user. Admin cannot delete their own account
    to prevent last-admin lockout.
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own admin account",
        )

    user = _get_user_or_404(db, user_id)
    db.delete(user)
    db.commit()

    logger.info("Admin deleted user id=%d email=%s", user_id, user.email)


# ---------------------------------------------------------------------------
# Document / File Management (admin sees ALL users' files)
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=List[DocumentAdminOut],
            summary="List all uploaded files across all users")
def list_all_documents(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    status: Optional[str] = Query(None, description="indexed, failed, duplicate, processing"),
    doc_type: Optional[str] = Query(None, description="pdf, docx, txt, image"),
    department: Optional[str] = Query(None),
    owner_id: Optional[int] = Query(None, description="Filter to a specific user's uploads"),
    search: Optional[str] = Query(None, description="Search title or filename"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Admin-scoped document list — no owner filter applied, ALL documents visible.
    Supports filtering by status, type, department, owner, and free-text search.
    """
    query = db.query(Document)

    if status:
        query = query.filter(Document.status == status)
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    if department:
        query = query.filter(Document.department.ilike(f"%{department}%"))
    if owner_id:
        query = query.filter(Document.owner_id == owner_id)
    if search:
        like = f"%{search}%"
        query = query.filter(
            (Document.title.ilike(like)) | (Document.original_filename.ilike(like))
        )

    # Newest first — most recent uploads are most interesting to reviewers
    docs = query.order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()

    results = []
    for doc in docs:
        owner = db.query(User).filter(User.id == doc.owner_id).first()
        results.append(_build_document_admin_out(doc, owner))

    return results


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Hard-delete a document (Pinecone + disk + DB)")
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
    """
    doc = _get_document_or_404(db, doc_id)

    # 1. Remove vector chunks from Pinecone
    if doc.status == DocStatus.indexed:
        try:
            vectorstore.delete_document(doc.id)
            logger.info("Deleted Pinecone vectors for document id=%d", doc.id)
        except RuntimeError as exc:
            # Log but don't abort — Pinecone being down shouldn't block DB cleanup
            logger.warning("Could not delete Pinecone vectors for doc %d: %s", doc.id, exc)

    # 2. Remove the raw file from disk
    if doc.stored_path and os.path.exists(doc.stored_path):
        os.remove(doc.stored_path)
        logger.info("Deleted file from disk: %s", doc.stored_path)

    # 3. Remove the metadata row from PostgreSQL
    db.delete(doc)
    db.commit()

    logger.info("Admin hard-deleted document id=%d filename=%s", doc_id, doc.original_filename)