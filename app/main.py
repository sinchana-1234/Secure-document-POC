"""
Application entrypoint. Wires routers, CORS, logging, and DB bootstrap.

WHY a lifespan handler instead of @app.on_event("startup"): on_event is deprecated.
The lifespan context manager is the supported way to run startup/shutdown code.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import User, Document  # noqa: F401  (imported so create_all sees the tables)
from app.routers import auth, documents, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("doc-intel")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENV)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
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