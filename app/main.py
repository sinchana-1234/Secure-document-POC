"""
Application entrypoint. Wires routers, CORS, logging, and DB bootstrap.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import User, Document  # noqa: F401
from app.routers import auth, documents, search, admin   # <-- added admin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("doc-intel")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENV)
    if settings.ENV == "prod" and settings.FIREWALL_MODE == "monitor":
        logger.warning(
            "SECURITY: Firewall is in MONITOR mode — attacks are logged but NOT blocked. "
            "Set FIREWALL_MODE=enforce in .env before going live."
        )
    Base.metadata.create_all(bind=engine)
    from app.services import vectorstore
    try:
        vectorstore.ensure_index()
    except Exception as e:
        logger.warning("Could not ensure Pinecone index at startup: %s", e)
    logger.info("Database tables ensured.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok", "app": settings.APP_NAME}


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(admin.router)  