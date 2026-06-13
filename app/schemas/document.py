"""Document-related request/response contracts."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    original_filename: str
    title: Optional[str]
    doc_type: str
    size_bytes: int
    department: Optional[str]
    tags: Optional[List[str]] = []
    page_count: Optional[int]
    num_chunks: Optional[int]
    ocr_used: Optional[str]
    status: str
    error_message: Optional[str]
    duplicate_of_id: Optional[int]
    owner_id: int
    upload_date: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    status: str
    document: Optional[DocumentOut] = None
    message: Optional[str] = None
    duplicate_of_id: Optional[int] = None
    duplicate_kind: Optional[str] = None
    similarity: Optional[float] = None