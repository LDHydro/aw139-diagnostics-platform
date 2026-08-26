"""Async database engine, session management and schema bootstrap."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """Declarative base for every ELP table."""


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.database_url,
            echo=s.db_echo,
            pool_size=s.db_pool_size,
            max_overflow=s.db_max_overflow,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a transactional session."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ----------------------------------------------------------------------
# Schema bootstrap
# ----------------------------------------------------------------------
# Indexes that SQLAlchemy cannot express portably: the pgvector HNSW index
# and the generated tsvector column used for keyword search.

_BOOTSTRAP_SQL: list[str] = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
]

_POST_CREATE_SQL: list[str] = [
    # Full-text search vector maintained by Postgres itself.
    """
    ALTER TABLE doc_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('simple', coalesce(heading, '')), 'A') ||
            setweight(to_tsvector('simple', coalesce(text, '')), 'B')
        ) STORED
    """,
    "CREATE INDEX IF NOT EXISTS ix_doc_chunks_tsv ON doc_chunks USING GIN (tsv)",
    # Approximate nearest-neighbour index for cosine distance.
    """
    CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding
        ON doc_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """,
    "CREATE INDEX IF NOT EXISTS ix_documents_groups ON documents USING GIN (allowed_groups)",
]


async def init_db(create_all: bool = True) -> None:
    """Create extensions, tables and the indexes that need raw SQL."""
    engine = get_engine()
    async with engine.begin() as conn:
        for stmt in _BOOTSTRAP_SQL:
            await conn.execute(text(stmt))

    if create_all:
        from . import models  # noqa: F401  (registers mappers)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async with engine.begin() as conn:
        for stmt in _POST_CREATE_SQL:
            try:
                await conn.execute(text(stmt))
            except Exception as exc:  # pragma: no cover - index already present
                log.warning("bootstrap statement skipped: %s", exc)


async def healthcheck() -> dict[str, Any]:
    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
            ext = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version = ext.scalar_one_or_none()
        return {"status": "ok", "pgvector": version}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def close_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
