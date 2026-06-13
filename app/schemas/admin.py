"""
Admin-specific Pydantic schemas.

WHY separate from auth.py schemas:
  - AdminUserCreate lets an admin set any role (including admin itself).
  - UserDetail extends UserOut with created_at + document stats so the
    admin list view has everything it needs in one call.
  - DocumentAdminOut extends DocumentOut with the owner's email so the
    admin file view doesn't need a second round-trip to resolve who uploaded it.
  - UserListResponse / DocumentListResponse wrap list endpoints with a total
    count so the frontend can render correct pagination (Page 1 of N).

All schemas use model_config = ConfigDict(from_attributes=True) (Pydantic v2)
so SQLAlchemy ORM objects can be returned directly from route handlers.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------

class AdminUserCreate(BaseModel):
    """
    Payload for admin creating a new user.
    Unlike public registration, admin can explicitly assign any role.
    role is constrained to Literal["user", "admin"] — any other value is
    rejected by Pydantic before the route handler runs.
    """
    email: EmailStr
    password: str = Field(min_length=6, description="Minimum 6 characters")
    full_name: Optional[str] = None
    role: Literal["user", "admin"] = "user"


class AdminUserUpdate(BaseModel):
    """
    Partial update payload — all fields optional so admin can patch just
    what's needed (e.g. promote a user to admin without touching anything else).
    role is constrained to Literal so invalid values are caught at validation time.
    password uses a field_validator instead of Field(min_length=6) because
    Pydantic v2 applies min_length constraints even when the field is None
    in some edge cases — the validator only runs when a value is actually provided.
    """
    full_name: Optional[str] = None
    role: Optional[Literal["user", "admin"]] = None
    password: Optional[str] = None

    @field_validator("password", mode="before")
    @classmethod
    def password_min_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class UserDetail(BaseModel):
    """
    Full user record returned by admin list/detail endpoints.
    Includes created_at and document count — extra fields useful for the
    admin dashboard that don't belong in the public-facing UserOut.
    document_count is computed in the route (not a DB column).
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    created_at: Optional[datetime] = None
    document_count: int = 0


class UserListResponse(BaseModel):
    """
    Paginated wrapper for the user list endpoint.
    total lets the frontend compute page count without a second request.
    """
    items: List[UserDetail]
    total: int


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

class DocumentAdminOut(BaseModel):
    """
    Document record enriched with owner info for the admin files view.
    Owner email and name are resolved via JOIN in the route query — no N+1.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    title: Optional[str] = None
    doc_type: str
    size_bytes: int
    tags: Optional[List[str]] = []
    page_count: Optional[int] = None
    num_chunks: Optional[int] = None
    ocr_used: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    duplicate_of_id: Optional[int] = None
    upload_date: datetime

    # Owner info — flattened so the frontend doesn't need a second request
    owner_id: int
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None


class DocumentListResponse(BaseModel):
    """
    Paginated wrapper for the document list endpoint.
    total lets the frontend compute page count without a second request.
    """
    items: List[DocumentAdminOut]
    total: int


# ---------------------------------------------------------------------------
# Summary / stats schema
# ---------------------------------------------------------------------------

class AdminStats(BaseModel):
    """
    High-level platform stats shown on the admin dashboard header cards.
    All numbers come from COUNT queries — cheap to compute, never a full scan.
    """
    total_users: int
    total_documents: int
    indexed_documents: int
    failed_documents: int
    duplicate_documents: int