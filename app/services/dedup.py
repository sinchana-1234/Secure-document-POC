"""
Duplicate detection, cheap-first:
  LAYER 1 EXACT (free): SHA-256 file hash match -> stop before any extraction/embedding.
  LAYER 2 NEAR (semantic): same report re-exported with a new timestamp has different
          bytes but ~identical meaning; embed it and check Pinecone's nearest neighbour.
99% of real duplicates are exact re-uploads, so the free check almost always wins.
"""
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document
from app.services import vectorstore


def find_exact_duplicate(db: Session, file_hash: str) -> Optional[Document]:
    return db.query(Document).filter(Document.file_hash == file_hash).first()


def find_near_duplicate(sample_vector) -> Tuple[Optional[int], float]:
    nearest = vectorstore.find_nearest(sample_vector)
    if not nearest:
        return None, 0.0
    score = float(nearest.get("score", 0.0))
    if score >= settings.NEAR_DUP_THRESHOLD:
        doc_id = nearest.get("metadata", {}).get("document_id")
        return (int(doc_id) if doc_id is not None else None), score
    return None, score