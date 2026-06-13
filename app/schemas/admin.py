"""
Admin-specific Pydantic schemas.

WHY separate from auth.py schemas:
  - AdminUserCreate lets an admin set any role (including admin itself).
  - UserDetail extends UserOut with created_at + document stats so the
    admin list view has everything it needs in one call.
  - DocumentAdminOut extends DocumentOut with the owner's email so the
    admin file view doesn't need a second round-trip to resolve who uploaded it.

All schemas use `from_attributes = True` (Pydantic v2) so SQLAlchemy ORM
objects can be returned directly from route handlers without manual conversion.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------

class AdminUserCreate(BaseModel):
    """
    Payload for admin creating a new user.
    Unlike public registration, admin can explicitly assign any role without restriction.
    """
    email: EmailStr
    password: str = Field(min_length=6, description="Minimum 6 characters")
    full_name: Optional[str] = None
    role: str = Field(default="user", description="'user' or 'admin'")


class AdminUserUpdate(BaseModel):
    """
    Partial update payload — all fields optional so admin can patch just
    what's needed (e.g. promote a user to admin without touching anything else).
    """
    full_name: Optional[str] = None
    role: Optional[str] = None          # 'user' | 'admin'
    password: Optional[str] = Field(default=None, min_length=6)


class UserDetail(BaseModel):
    """
    Full user record returned by admin list/detail endpoints.
    Includes created_at and document count — extra fields useful for the
    admin dashboard that don't belong in the public-facing UserOut.
    """
    id: int
    email: EmailStr
    full_name: Optional[str]
    role: str
    created_at: Optional[datetime]
    document_count: int = 0             # computed in the route, not a DB column

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

class DocumentAdminOut(BaseModel):
    """
    Document record enriched with owner info for the admin files view.
    Owner email is joined in the query so we don't hit N+1 queries.
    """
    id: int
    original_filename: str
    title: Optional[str]
    doc_type: str
    size_bytes: int
    tags: Optional[List[str]] = []
    page_count: Optional[int]
    num_chunks: Optional[int]
    ocr_used: Optional[str]
    status: str
    error_message: Optional[str]
    duplicate_of_id: Optional[int]
    upload_date: datetime

    # Owner info — flattened here so the frontend doesn't need a join
    owner_id: int
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Summary / stats schemas
# ---------------------------------------------------------------------------

class AdminStats(BaseModel):
    """
    High-level platform stats shown on the admin dashboard header.
    All numbers come from simple COUNT queries — cheap to compute.
    """
    total_users: int
    total_documents: int
    indexed_documents: int
    failed_documents: int
    duplicate_documents: int