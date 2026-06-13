"""Re-export every schema so callers can `from app.schemas import X`."""
from app.schemas.auth import UserCreate, UserOut, Token
from app.schemas.document import DocumentOut, UploadResponse
from app.schemas.search import SearchRequest, SourceRef, SearchResponse

__all__ = [
    "UserCreate", "UserOut", "Token",
    "DocumentOut", "UploadResponse",
    "SearchRequest", "SourceRef", "SearchResponse",
]