"""
The ingestion pipeline = the conductor. Routes stay thin; the real sequence lives
here so it's testable and readable top-to-bottom:
  1. exact-dup check (hash, free)  2. save to disk  3. extract (+OCR)
  4. chunk -> embed  5. near-dup check (semantic)  6. upsert to Pinecone  7. mark indexed
"""
import os
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, DocStatus
from app.utils.hashing import sha256_of_bytes
from app.utils.chunking import chunk_text
from app.services import extraction, embeddings, vectorstore, dedup


class DuplicateError(Exception):
    def __init__(self, message: str, existing_id: int, kind: str, score: float = None):
        super().__init__(message)
        self.existing_id = existing_id
        self.kind = kind          # "exact" | "near"
        self.score = score


def _save_file(file_bytes: bytes, original_filename: str) -> str:
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    ext = os.path.splitext(original_filename)[1].lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"   # UUID name => no path traversal, no collisions
    stored_path = os.path.join(settings.STORAGE_DIR, stored_name)
    with open(stored_path, "wb") as f:
        f.write(file_bytes)
    return stored_path


def ingest(db: Session, *, file_bytes: bytes, original_filename: str, owner,
           title: str = None, tags: list = None) -> Document:
    tags = tags or []

    # 1. EXACT duplicate (before spending any money/CPU)
    file_hash = sha256_of_bytes(file_bytes)
    existing = dedup.find_exact_duplicate(db, file_hash)
    if existing:
        raise DuplicateError(f"Identical file already exists (document id {existing.id}).",
                             existing_id=existing.id, kind="exact")

    # 2. Save to disk
    stored_path = _save_file(file_bytes, original_filename)
    doc_type = extraction.detect_doc_type(original_filename)
    if doc_type == "unknown":
        os.remove(stored_path)
        raise ValueError(f"Unsupported file type: {original_filename}")

    doc = Document(
        original_filename=original_filename, stored_path=stored_path, file_hash=file_hash,
        doc_type=doc_type, size_bytes=len(file_bytes), owner_id=owner.id,
        title=title or original_filename, tags=tags, status=DocStatus.processing,
        upload_date=datetime.utcnow(),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        # 3. Extract
        result = extraction.extract(stored_path, original_filename)
        doc.doc_type = result.doc_type
        doc.page_count = result.page_count
        doc.ocr_used = "yes" if result.ocr_used else "no"
        doc.mime_type = result.mime_type
        doc.extracted_text = result.text[:200_000]

        chunks = chunk_text(result.text)
        if not chunks:
            doc.status = DocStatus.failed
            doc.error_message = "No extractable text found (empty or unreadable document)."
            db.commit()
            return doc

        # 4. Embed
        vectors = embeddings.embed_texts(chunks)

        # 5. NEAR-duplicate (first chunk's vector as a representative sample)
        near_id, score = dedup.find_near_duplicate(vectors[0])
        if near_id and near_id != doc.id:
            doc.status = DocStatus.duplicate
            doc.duplicate_of_id = near_id
            doc.error_message = f"Near-duplicate of document {near_id} (similarity {score:.3f})."
            db.commit()
            raise DuplicateError(f"Near-duplicate of document {near_id} (similarity {score:.3f}).",
                                 existing_id=near_id, kind="near", score=score)

        # 6. Upsert to Pinecone with metadata for hybrid filtering
        base_md = {
            "document_id": doc.id, "title": doc.title, "doc_type": doc.doc_type,
            "owner_id": doc.owner_id, "tags": tags,
            "upload_ts": int(doc.upload_date.timestamp()),
        }
        n = vectorstore.upsert_chunks(doc.id, chunks, vectors, base_md)

        # 7. Done
        doc.num_chunks = n
        doc.status = DocStatus.indexed
        db.commit()
        db.refresh(doc)
        return doc

    except DuplicateError:
        raise
    except Exception as e:
        doc.status = DocStatus.failed
        doc.error_message = str(e)[:1000]
        db.commit()
        raise