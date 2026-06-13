from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App (safe defaults; not DB config)
    APP_NAME: str = "Secure Document Intelligence POC"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:4200", "http://127.0.0.1:4200"]
    # ----- Auth / JWT -----
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    # ----- OpenAI -----
    OPENAI_API_KEY: str                       # REQUIRED — from .env (it's a credential)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536                 # must match the model above

    # PostgreSQL (REQUIRED — only from .env, no default)
    DATABASE_URL: str

    # Pinecone (REQUIRED — only from .env, no default)
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str
    PINECONE_DIMENSION: int      # must match embedding model
    PINECONE_METRIC: str


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()