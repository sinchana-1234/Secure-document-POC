from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_NAME: str = "Secure Document Intelligence POC"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:4200", "http://127.0.0.1:4200"]

    # Auth / JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # File storage
    STORAGE_DIR: str = "storage"
    MAX_UPLOAD_MB: int = 50

    # OpenAI
    OPENAI_API_KEY: str                        # REQUIRED — from .env
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536                  # must match model above
    CHAT_MODEL: str = "gpt-4o-mini"

    # PostgreSQL
    DATABASE_URL: str                          # REQUIRED — from .env

    # Pinecone
    PINECONE_API_KEY: str                      # REQUIRED — from .env
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str
    PINECONE_DIMENSION: int                    # must match EMBEDDING_DIM
    PINECONE_METRIC: str

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # RAG retrieval
    RETRIEVAL_TOP_K: int = 6

    # Near-duplicate detection threshold (cosine similarity, 0–1)
    # 0.96 catches re-exports and minor edits; lower it to catch looser copies.
    NEAR_DUP_THRESHOLD: float = 0.96

    # OCR
    OCR_LANG: str = "eng"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()