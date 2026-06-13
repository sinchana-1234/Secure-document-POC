import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database import Base

# JSONB on Postgres (indexable + containment queries); plain JSON elsewhere (portable).
TAGS_TYPE = JSON().with_variant(JSONB(), "postgresql")


class DocStatus(str, enum.Enum):
    pending = "pending"; processing = "processing"; indexed = "indexed"
    duplicate = "duplicate"; failed = "failed"


class Document(Base):
    __tablename__ = "documents"
    id               = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(512), nullable=False)
    stored_path       = Column(String(1024), nullable=False)
    file_hash         = Column(String(64), nullable=False, index=True)   # exact dedup
    mime_type         = Column(String(120), nullable=True)
    doc_type          = Column(String(40), nullable=False)
    size_bytes        = Column(Integer, nullable=False)
    owner_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title             = Column(String(512), nullable=True)
    tags              = Column(TAGS_TYPE, nullable=True, default=list)
    page_count        = Column(Integer, nullable=True)
    num_chunks        = Column(Integer, nullable=True, default=0)
    ocr_used          = Column(String(5), nullable=True)
    extracted_text    = Column(Text, nullable=True)
    status            = Column(SAEnum(DocStatus), nullable=False, default=DocStatus.pending)
    error_message     = Column(Text, nullable=True)
    duplicate_of_id   = Column(Integer, ForeignKey("documents.id"), nullable=True)
    upload_date       = Column(DateTime, default=datetime.utcnow, index=True)
    owner = relationship("User")