"""
pipeline.py — The ingestion conductor.

Routes stay thin; the real sequence lives here so it's testable top-to-bottom:
  1. Exact-dup check  (hash, free)
  2. Save to disk
  3. Extract text (+OCR if needed)
  4. Chunk → embed
  5. Near-dup check  (semantic, all chunk vectors, owner-scoped)
  6. Upsert to Pinecone
  7. Mark indexed

FILE LIFECYCLE
──────────────
The physical file is written in step 2.  Any failure after that point must
clean it up before raising — we never leave orphaned files on disk.
  - Unsupported type  → removed immediately after detection (step 2)
  - Near-duplicate    → removed before raising DuplicateError (step 5)
  - Any other error   → removed in the outer except block
Exact duplicates are caught before the file is written, so no cleanup needed.
"""
import logging
import os
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, DocStatus
from app.services import dedup, embeddings, extraction, vectorstore
from app.utils.chunking import chunk_text
from app.utils.hashing import sha256_of_bytes

logger = logging.getLogger("doc-poc.pipeline")

import logging
logger = logging.getLogger("doc-poc.pipeline")

class DuplicateError(Exception):
    def __init__(self, message: str, existing_id: int, kind: str, score: float = None):
        super().__init__(message)
        self.existing_id = existing_id
        self.kind = kind        # "exact" | "near"
        self.score = score


def _save_file(file_bytes: bytes, original_filename: str) -> str:
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    ext = os.path.splitext(original_filename)[1].lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"  # UUID name → no path traversal, no collisions
    stored_path = os.path.join(settings.STORAGE_DIR, stored_name)
    with open(stored_path, "wb") as f:
        f.write(file_bytes)
    return stored_path


def _remove_file(path: str) -> None:
    """Delete a stored file, logging rather than raising if it is already gone."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not remove stored file '%s': %s", path, exc)


def ingest(
    db: Session,
    *,
    file_bytes: bytes,
    original_filename: str,
    owner,
    title: str = None,
    tags: list = None,
) -> Document:
    tags = tags or []

    # 1. EXACT duplicate — before spending any money/CPU or touching disk
    file_hash = sha256_of_bytes(file_bytes)
    existing = dedup.find_exact_duplicate(db, file_hash)
    if existing:
        raise DuplicateError(
            f"Identical file already exists (document id {existing.id}).",
            existing_id=existing.id,
            kind="exact",
        )

    # 2. Save to disk
    stored_path = _save_file(file_bytes, original_filename)
    doc_type = extraction.detect_doc_type(original_filename)
    if doc_type == "unknown":
        _remove_file(stored_path)
        raise ValueError(f"Unsupported file type: {original_filename}")

    doc = Document(
        original_filename=original_filename,
        stored_path=stored_path,
        file_hash=file_hash,
        doc_type=doc_type,
        size_bytes=len(file_bytes),
        owner_id=owner.id,
        title=title or original_filename,
        tags=tags,
        status=DocStatus.processing,
        upload_date=datetime.utcnow(),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        # 3. Extract text (+OCR if needed)
        result = extraction.extract(stored_path, original_filename)
        doc.doc_type = result.doc_type
        doc.page_count = result.page_count
        doc.ocr_used = "yes" if result.ocr_used else "no"
        doc.mime_type = result.mime_type
        doc.extracted_text = result.text[:200_000]

        # ── AI Firewall checkpoint (input): scan the document for prompt injection ──
        from app.security.firewall import firewall
        from app.config import settings as _settings
        detection = firewall.scan_document(result.text)
        if detection.is_suspicious:
            logger.warning(
                "Firewall flagged document id=%s severity=%s categories=%s",
                doc.id, detection.severity, ",".join(detection.matched_categories),
            )
            if _settings.FIREWALL_MODE == "enforce":
                doc.status = DocStatus.failed
                doc.error_message = "Blocked by security firewall (possible injected instructions)."
                db.commit()
                raise DuplicateError(
                    "This document was blocked by the security firewall.",
                    existing_id=doc.id, kind="firewall",
                )

        chunks = chunk_text(result.text)
        if not chunks:
            doc.status = DocStatus.failed
            doc.error_message = "No extractable text found (empty or unreadable document)."
            db.commit()
            # File kept: the document record references it; admin may want to inspect it.
            return doc

        # 4. Embed all chunks
        vectors = embeddings.embed_texts(chunks)
        # 5. NEAR-duplicate check — scoped to this owner, self-excluded so a
        #    re-processed document cannot match its own previously upserted vectors.
        near_id, score = dedup.find_near_duplicate(
            vectors,
            owner_id=owner.id,
            exclude_doc_id=doc.id,
        )
        # A near-dup match against a soft-deleted document isn't a real duplicate —
        # that doc is hidden from the user, so blocking their re-upload would confuse.
        if near_id is not None:
            still_active = db.query(Document).filter(
                Document.id == near_id,
                Document.deleted_at.is_(None),
            ).first()
            if still_active is None:
                near_id = None
        if near_id is not None:
            doc.status = DocStatus.duplicate
            doc.duplicate_of_id = near_id
            doc.error_message = (
                f"Near-duplicate of document {near_id} (similarity {score:.3f})."
            )
            db.commit()
            # Remove the physical file — the document is a duplicate and will
            # not be indexed, so there is no reason to keep the bytes on disk.
            _remove_file(stored_path)
            raise DuplicateError(
                f"Near-duplicate of document {near_id} (similarity {score:.3f}).",
                existing_id=near_id,
                kind="near",
                score=score,
            )

        # 6. Upsert to Pinecone with metadata for hybrid filtering
        base_md = {
            "document_id": doc.id,
            "title": doc.title,
            "doc_type": doc.doc_type,
            "owner_id": doc.owner_id,
            "tags": tags,
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
        # Already handled above (status set, file removed) — just re-raise.
        raise

    except Exception as exc:
        # Mark the DB record as failed so operators can see it, then clean up
        # the physical file — a failed document has no valid indexed state so
        # keeping the bytes serves no purpose and wastes disk space.
        doc.status = DocStatus.failed
        doc.error_message = str(exc)[:1000]
        db.commit()
        _remove_file(stored_path)
        raise