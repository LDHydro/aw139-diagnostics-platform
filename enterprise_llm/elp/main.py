"""
Application entry point.

Run with::

    uvicorn elp.main:app --host 0.0.0.0 --port 8080

or through the systemd unit / container image in ``deploy/``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import ROUTERS
from .config import get_settings
from .db import close_db, healthcheck, init_db
from .llm.embeddings import get_embedding_client, get_rerank_client
from .llm.router import close_router, get_router

log = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # These are chatty at INFO and drown out anything useful.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("starting %s v%s (%s)", settings.app_name, __version__, settings.environment)

    if settings.auth.mode == "disabled" and settings.is_production:
        raise RuntimeError(
            "ELP_AUTH__MODE=disabled is not permitted when "
            "ELP_ENVIRONMENT=production. Configure OIDC or LDAP first."
        )

    try:
        await init_db()
        log.info("database schema ready")
    except Exception as exc:
        # Starting without a database is pointless, but failing loudly with a
        # clear message beats a stack trace on every request.
        log.error("database initialisation failed: %s", exc)
        raise

    # Probe the model servers so a misconfigured endpoint shows up at start-up
    # rather than on the first user question.
    health = await get_router().health()
    for name, value in health.items():
        if value.get("status") != "ok":
            log.warning(
                "%s model endpoint is not answering (%s): %s",
                name, value.get("endpoint"), value.get("detail"),
            )
        else:
            log.info("%s model endpoint ready: %s", name, value.get("endpoint"))

    yield

    log.info("shutting down")
    await get_embedding_client().aclose()
    await get_rerank_client().aclose()
    await close_router()
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Self-hosted LLM platform: question answering over department "
            "governing documents with citations, federation with other "
            "internal AI systems, predictive maintenance scheduling, LaTeX "
            "generation and application-development assistance. "
            "Access is governed by Active Directory group membership."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", settings.auth.api_key_header],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every request so a log line, an audit row and a response agree."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "unhandled error on %s %s (request_id=%s)",
                request.method, request.url.path, request_id,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "internal error",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        elapsed_ms = (time.monotonic() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        return response

    for router in ROUTERS:
        app.include_router(router)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        """Liveness probe.  Deliberately unauthenticated and cheap."""
        database = await healthcheck()
        return {
            "status": "ok" if database.get("status") == "ok" else "degraded",
            "version": __version__,
            "environment": settings.environment,
            "database": database.get("status"),
            "auth_mode": settings.auth.mode,
        }

    @app.get("/", tags=["health"])
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "openai_compatible_base_url": "/v1",
        }

    return app


app = create_app()
