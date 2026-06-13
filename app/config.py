"""
Central configuration.

WHY a single settings object: every service (Pinecone, OpenAI, DB, JWT) needs
secrets. Reading os.getenv() scattered across 20 files is how you end up with
typos and missing-key bugs. We read ONCE here, validate, and import everywhere.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Secure Document Intelligence Platform"
    ENV: str = "dev"
    STORAGE_DIR: str = "storage"
    MAX_UPLOAD_MB: int = 50

    JWT_SECRET: str = "CHANGE_ME_super_secret_dev_key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8

    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/doc_intel"

    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"   # 1536 dims
    EMBEDDING_DIM: int = 1536
    CHAT_MODEL: str = "gpt-4o-mini"

    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "doc-intel"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 6
    NEAR_DUP_THRESHOLD: float = 0.96

    OCR_LANG: str = "eng"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()