"""
embeddings.py — turn text into vectors via the OpenAI embeddings API.

This is the bridge from "words" to "meaning": each piece of text becomes a fixed-length
list of numbers (1536 for text-embedding-3-small) where similar meaning => nearby vectors.
Everything downstream (Pinecone storage, semantic search, near-duplicate detection)
depends on this, so it's built defensively.
"""
from __future__ import annotations

import logging
from typing import List

from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
)

from app.config import settings

logger = logging.getLogger("doc-poc.embeddings")


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------
class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingConfigError(EmbeddingError):
    """Missing/invalid OpenAI key — a configuration problem the operator must fix."""


class EmbeddingAPIError(EmbeddingError):
    """Upstream OpenAI failure (rate limit, quota, network, server error)."""


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_MAX_INPUTS_PER_REQUEST = 100      # how many texts we send in one API call
_MAX_INPUT_CHARS = 24_000          # ~8k tokens safety cap (model limit is 8191 tokens)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Build the OpenAI client once, with sane timeouts + automatic retries."""
    global _client
    if not settings.OPENAI_API_KEY:
        raise EmbeddingConfigError("OPENAI_API_KEY is not set. Add it to backend/.env.")
    if _client is None:
        # max_retries: the SDK retries transient errors (timeouts, 5xx, rate spikes)
        # with exponential backoff before giving up.
        _client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3, timeout=30.0)
    return _client


def _sanitize(texts: List[str]) -> List[str]:
    """
    Make every input safe to send while keeping a 1:1 correspondence with the originals
    (the caller zips chunks with the returned vectors, so counts must match exactly):
      - empty/whitespace -> single space (the API rejects truly empty strings)
      - over-long       -> truncated to the safety cap
    """
    cleaned: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            logger.warning("Empty text passed to embeddings; substituting a placeholder to keep alignment.")
            t = " "
        if len(t) > _MAX_INPUT_CHARS:
            logger.warning("Truncating an over-long input (%d chars) to %d.", len(t), _MAX_INPUT_CHARS)
            t = t[:_MAX_INPUT_CHARS]
        cleaned.append(t)
    return cleaned


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts. Returns one vector per input, in the SAME order.
    Raises EmbeddingConfigError (bad key) or EmbeddingAPIError (upstream failure).
    """
    if not texts:
        return []

    client = _get_client()
    prepared = _sanitize(texts)
    vectors: List[List[float]] = []

    for start in range(0, len(prepared), _MAX_INPUTS_PER_REQUEST):
        batch = prepared[start:start + _MAX_INPUTS_PER_REQUEST]
        try:
            resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
        except AuthenticationError as e:
            raise EmbeddingConfigError("OpenAI rejected the API key (check OPENAI_API_KEY).") from e
        except RateLimitError as e:
            raise EmbeddingAPIError("OpenAI rate limit or quota exceeded — try again later.") from e
        except OpenAIError as e:
            raise EmbeddingAPIError(f"OpenAI embeddings request failed: {e}") from e

        # The API returns items with an `index`; sort by it so order is guaranteed
        # regardless of how the server batched the response.
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(item.embedding for item in ordered)

    # Catch a model/index mismatch here, not later in Pinecone.
    if vectors and len(vectors[0]) != settings.EMBEDDING_DIM:
        raise EmbeddingError(
            f"Embedding dimension {len(vectors[0])} does not match configured "
            f"EMBEDDING_DIM={settings.EMBEDDING_DIM}. Check EMBEDDING_MODEL vs EMBEDDING_DIM."
        )

    logger.info("Embedded %d text(s) into %d-dim vectors.", len(vectors),
                len(vectors[0]) if vectors else 0)
    return vectors


def embed_query(text: str) -> List[float]:
    """Embed a single search query. Raises if the query is empty."""
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed an empty query.")
    return embed_texts([text])[0]