"""
Layer: INTERFACES
Purpose: FastAPI application entry point with middleware.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from interfaces.api.routes import router
from interfaces.api.industrial_routes import industrial_router
from interfaces.api.vision_routes import vision_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("PrivaCore Nexus starting — provider=%s", settings.cloud_provider.value)
    yield
    logger.info("PrivaCore Nexus shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="VaultMind",
        version=settings.app_version,
        description="GDPR-compliant industrial AI with privacy-first architecture",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    app.include_router(industrial_router, prefix="/api/v1/industrial")
    app.include_router(vision_router, prefix="/api/v1/industrial")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("interfaces.api.main:app", host="127.0.0.1", port=8002, reload=False)
