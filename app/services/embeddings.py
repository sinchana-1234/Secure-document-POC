"""
Embeddings turn text into a 1536-number vector capturing meaning. A lazy singleton
client means app startup never crashes just because OPENAI_API_KEY isn't set yet —
we fail loudly only when embeddings are actually needed. Inputs are batched.
"""
from typing import List
from openai import OpenAI
from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to backend/.env to enable embeddings/RAG.")
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    if not texts:
        return []
    client = _get_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]