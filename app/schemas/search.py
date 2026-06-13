"""Search / RAG request/response contracts."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class SearchRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    doc_type: Optional[str] = None
    tags: Optional[List[str]] = None
    department: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class SourceRef(BaseModel):
    ref: int
    document_id: Optional[int]
    title: Optional[str]
    doc_type: Optional[str]
    chunk_index: Optional[int]
    score: float
    snippet: str


class SearchResponse(BaseModel):
    answer: str
    sources: List[SourceRef]