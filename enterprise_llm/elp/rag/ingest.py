"""
Ingestion pipeline: file -> parsed blocks -> chunks -> embeddings -> Postgres.

Designed for a small corpus of large documents (10-20 governing manuals),
which is why it re-indexes a whole document atomically rather than trying to
diff at the chunk level: a governing document changes by revision, not by
paragraph, and a half-updated manual is worse than a stale one.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..llm.embeddings import EmbeddingClient, get_embedding_client
from ..models import DocChunk, Document, DocumentStatus
from .chunker import Chunk, chunk_blocks
from .parsers import ParseError, parse

log = logging.getLogger(__name__)


@dataclass
class IngestRequest:
    path: Path
    doc_key: str
    title: str = ""
    department: str = ""
    doc_type: str = "governing"
    revision: str = ""
    effective_date: date | None = None
    review_due_date: date | None = None
    # AD groups permitted to retrieve from this document; empty = all
    # authenticated callers.
    allowed_groups: list[str] = field(default_factory=list)
    classification: str = "internal"
    source_uri: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class IngestResult:
    document_id: str
    doc_key: str
    revision: str
    chunk_count: int
    page_count: int
    skipped: bool = False
    reason: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Ingestor:
    def __init__(self, embedder: EmbeddingClient | None = None) -> None:
        self.settings = get_settings()
        self.embedder = embedder or get_embedding_client()

    async def ingest(
        self,
        session: AsyncSession,
        request: IngestRequest,
        *,
        force: bool = False,
    ) -> IngestResult:
        path = request.path
        if not path.exists():
            raise ParseError(f"file not found: {path}")

        checksum = sha256_file(path)

        existing = (
            await session.execute(
                select(Document).where(
                    Document.doc_key == request.doc_key,
                    Document.revision == request.revision,
                )
            )
        ).scalar_one_or_none()

        if (
            existing is not None
            and existing.content_sha256 == checksum
            and existing.status == DocumentStatus.READY.value
            and not force
        ):
            return IngestResult(
                document_id=existing.id,
                doc_key=existing.doc_key,
                revision=existing.revision,
                chunk_count=existing.chunk_count,
                page_count=existing.page_count,
                skipped=True,
                reason="content unchanged since last index",
            )

        log.info("parsing %s (%s rev %s)", path.name, request.doc_key, request.revision or "-")
        parsed = parse(path)
        chunks = chunk_blocks(parsed.blocks, self.settings.rag)
        if not chunks:
            raise ParseError(f"no extractable text found in {path.name}")

        document = existing
        if document is None:
            document = Document(
                doc_key=request.doc_key,
                title=request.title or parsed.meta.get("title") or path.stem,
                revision=request.revision,
            )
            session.add(document)
        else:
            # Re-indexing an existing revision: drop the old passages first.
            await session.execute(
                delete(DocChunk).where(DocChunk.document_id == document.id)
            )

        document.title = request.title or document.title or path.stem
        document.department = request.department
        document.doc_type = request.doc_type
        document.effective_date = request.effective_date
        document.review_due_date = request.review_due_date
        document.source_uri = request.source_uri or str(path)
        document.content_sha256 = checksum
        document.page_count = parsed.page_count
        document.allowed_groups = list(request.allowed_groups)
        document.classification = request.classification
        document.status = DocumentStatus.INDEXING.value
        document.status_detail = ""
        document.meta = {**parsed.meta, **request.meta}
        await session.flush()

        try:
            await self._embed_and_store(session, document, chunks)
        except Exception as exc:
            document.status = DocumentStatus.FAILED.value
            document.status_detail = str(exc)[:500]
            raise

        document.chunk_count = len(chunks)
        document.status = DocumentStatus.READY.value

        # Any earlier revision of the same document is now historical.
        await self._supersede_older_revisions(session, document)
        await session.flush()

        log.info(
            "indexed %s rev %s: %d chunks across %d pages",
            document.doc_key, document.revision or "-", len(chunks), parsed.page_count,
        )
        return IngestResult(
            document_id=document.id,
            doc_key=document.doc_key,
            revision=document.revision,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
        )

    async def _embed_and_store(
        self, session: AsyncSession, document: Document, chunks: list[Chunk]
    ) -> None:
        vectors = await self.embedder.embed([c.embed_text() for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            session.add(
                DocChunk(
                    document_id=document.id,
                    ordinal=chunk.ordinal,
                    section_number=chunk.section_number,
                    section_path=chunk.section_path,
                    heading=chunk.heading,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=vector,
                    meta=chunk.meta,
                )
            )
        await session.flush()

    async def _supersede_older_revisions(
        self, session: AsyncSession, current: Document
    ) -> None:
        others = (
            await session.execute(
                select(Document).where(
                    Document.doc_key == current.doc_key,
                    Document.id != current.id,
                    Document.status != DocumentStatus.SUPERSEDED.value,
                )
            )
        ).scalars().all()
        for other in others:
            # Only supersede genuinely older revisions; an operator may keep
            # two live revisions deliberately (e.g. a fleet mid-transition).
            if _revision_sort_key(other.revision) < _revision_sort_key(current.revision):
                other.status = DocumentStatus.SUPERSEDED.value
                other.superseded_by = current.id
                log.info(
                    "marked %s rev %s superseded by rev %s",
                    other.doc_key, other.revision or "-", current.revision or "-",
                )


def _revision_sort_key(revision: str) -> tuple:
    """
    Order revisions like "A" < "B" < "1" < "2" < "Rev 10".

    Digits sort numerically after letters, which matches how most
    maintenance organisations issue revisions (initial letters, then
    numbered amendments).
    """
    if not revision:
        return (0, 0, "")
    cleaned = revision.strip().lower().removeprefix("rev").strip()
    if cleaned.isdigit():
        return (2, int(cleaned), "")
    if len(cleaned) == 1 and cleaned.isalpha():
        return (1, ord(cleaned), "")
    return (1, 0, cleaned)


async def delete_document(session: AsyncSession, doc_key: str, revision: str = "") -> int:
    """Remove a document (and its passages).  Returns rows deleted."""
    query = select(Document).where(Document.doc_key == doc_key)
    if revision:
        query = query.where(Document.revision == revision)
    documents = (await session.execute(query)).scalars().all()
    for document in documents:
        await session.delete(document)
    return len(documents)
