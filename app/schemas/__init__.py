"""Re-export every schema so callers can `from app.schemas import X`."""
from app.schemas.auth import UserCreate, UserOut, Token
from app.schemas.document import DocumentOut, UploadResponse
from app.schemas.search import SearchRequest, SourceRef, SearchResponse
from app.schemas.admin import (
    AdminUserCreate,
    AdminUserUpdate,
    UserDetail,
    DocumentAdminOut,
    AdminStats,
)

__all__ = [
    "UserCreate", "UserOut", "Token",
    "DocumentOut", "UploadResponse",
    "SearchRequest", "SourceRef", "SearchResponse",
    "AdminUserCreate", "AdminUserUpdate", "UserDetail", "DocumentAdminOut", "AdminStats",
]