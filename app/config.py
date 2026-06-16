from functools import lru_cache

from pydantic import field_validator, model_validator
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
    VISION_MODEL: str = "gpt-4o"
    # OpenAI
    OPENAI_API_KEY: str                         # REQUIRED — from .env
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536                   # must match model above
    CHAT_MODEL: str = "gpt-4o-mini"

    # PostgreSQL
    DATABASE_URL: str                           # REQUIRED — from .env
    # Pinecone
    PINECONE_API_KEY: str                       # REQUIRED — from .env
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str
    PINECONE_DIMENSION: int                     # must equal EMBEDDING_DIM — validated below
    PINECONE_METRIC: str

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # RAG retrieval
    RETRIEVAL_TOP_K: int = 6

    # Near-duplicate detection threshold (cosine similarity, 0–1)
    # 0.96 catches re-exports and minor edits; lower it to catch looser copies.
    NEAR_DUP_THRESHOLD: float = 0.96
    FIREWALL_MODE: str = "monitor"

    # OCR
    OCR_LANG: str = "eng"

    FIREWALL_MODE: str = "monitor"

    # ------------------------------------------------------------------
    # Field-level validators
    # These run per-field at parse time so misconfiguration is caught at
    # startup with a clear message rather than silently misbehaving later.
    # ------------------------------------------------------------------

    @field_validator("MAX_UPLOAD_MB")
    @classmethod
    def max_upload_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"MAX_UPLOAD_MB must be > 0, got {v}")
        return v

    @field_validator("RETRIEVAL_TOP_K")
    @classmethod
    def retrieval_top_k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"RETRIEVAL_TOP_K must be > 0, got {v}")
        return v

    @field_validator("NEAR_DUP_THRESHOLD")
    @classmethod
    def threshold_must_be_valid_similarity(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"NEAR_DUP_THRESHOLD must be in (0, 1], got {v}. "
                "A value of 0 would flag every document as a duplicate; "
                "a value > 1 is outside the cosine similarity range."
            )
        return v

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int) -> int:
        # CHUNK_SIZE may not be parsed yet when this runs (field order is not
        # guaranteed), so we only reject clearly invalid values here.
        # The cross-field check is done in the model validator below.
        if v < 0:
            raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {v}")
        return v

    # ------------------------------------------------------------------
    # Cross-field (model-level) validator
    # Runs after ALL fields are parsed, so both sides are available.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "Settings":
        # PINECONE_DIMENSION must match EMBEDDING_DIM exactly.
        # Divergence means the index was created with the wrong shape and
        # every upsert will fail with an opaque Pinecone dimension error.
        if self.PINECONE_DIMENSION != self.EMBEDDING_DIM:
            raise ValueError(
                f"PINECONE_DIMENSION ({self.PINECONE_DIMENSION}) must equal "
                f"EMBEDDING_DIM ({self.EMBEDDING_DIM}). "
                "Either recreate the Pinecone index or update the config to match."
            )

        # CHUNK_OVERLAP must be strictly less than CHUNK_SIZE.
        # If overlap >= chunk_size, the step size in chunk_text() collapses to 1,
        # producing thousands of single-character chunks and exploding embedding costs.
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE}). "
                "Overlap >= chunk size collapses the step to 1 character per chunk."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()