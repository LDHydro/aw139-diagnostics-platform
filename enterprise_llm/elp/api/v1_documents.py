"""Managing the department governing documents."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..db import get_session
from ..llm.client import InferenceError
from ..models import DocChunk, Document, DocumentStatus
from ..rag.ingest import Ingestor, IngestRequest, delete_document
from ..rag.parsers import SUPPORTED_SUFFIXES, ParseError
from .schemas import DocumentSummary, IngestMetadata, IngestResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

# Refuse anything larger than this outright rather than filling the disk.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _visible_filter(principal: Principal):
    """Documents this caller's AD groups permit seeing in listings."""
    from sqlalchemy import Text, bindparam, text
    from sqlalchemy.dialects.postgresql import ARRAY

    return text(
        "(cardinality(documents.allowed_groups) = 0 "
        "OR documents.allowed_groups && :visible_groups)"
    ).bindparams(
        bindparam("visible_groups", value=list(principal.groups), type_=ARRAY(Text))
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    principal: Principal = Depends(require_scope(Scope.DOCS_READ)),
    session: AsyncSession = Depends(get_session),
    department: str = "",
    status: str = "",
    include_superseded: bool = False,
) -> list[DocumentSummary]:
    query = select(Document).where(_visible_filter(principal))
    if department:
        query = query.where(Document.department == department)
    if status:
        query = query.where(Document.status == status)
    elif not include_superseded:
        query = query.where(Document.status != DocumentStatus.SUPERSEDED.value)

    rows = (
        await session.execute(query.order_by(Document.doc_key, Document.revision))
    ).scalars().all()
    return [DocumentSummary.model_validate(r, from_attributes=True) for r in rows]


@router.get("/stats")
async def corpus_stats(
    principal: Principal = Depends(require_scope(Scope.DOCS_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Corpus size and health, for an operations dashboard."""
    totals = (
        await session.execute(
            select(
                func.count(Document.id),
                func.coalesce(func.sum(Document.chunk_count), 0),
                func.coalesce(func.sum(Document.page_count), 0),
            ).where(_visible_filter(principal))
        )
    ).one()

    by_status = (
        await session.execute(
            select(Document.status, func.count(Document.id))
            .where(_visible_filter(principal))
            .group_by(Document.status)
        )
    ).all()

    unembedded = (
        await session.execute(
            select(func.count(DocChunk.id)).where(DocChunk.embedding.is_(None))
        )
    ).scalar_one()

    return {
        "documents": totals[0],
        "chunks": int(totals[1]),
        "pages": int(totals[2]),
        "by_status": {status: count for status, count in by_status},
        "chunks_missing_embeddings": unembedded,
    }


@router.get("/{doc_key}", response_model=list[DocumentSummary])
async def get_document(
    doc_key: str,
    principal: Principal = Depends(require_scope(Scope.DOCS_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentSummary]:
    """All revisions of one document."""
    rows = (
        await session.execute(
            select(Document)
            .where(Document.doc_key == doc_key, _visible_filter(principal))
            .order_by(Document.revision)
        )
    ).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no document '{doc_key}' that your groups permit reading",
        )
    return [DocumentSummary.model_validate(r, from_attributes=True) for r in rows]


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form(
        ...,
        description="JSON matching the IngestMetadata schema",
    ),
    principal: Principal = Depends(require_scope(Scope.DOCS_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    """Upload and index a governing document."""
    try:
        meta = IngestMetadata.model_validate_json(metadata)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid metadata: {exc}") from exc

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported file type '{suffix}'. Supported: "
                + ", ".join(sorted(SUPPORTED_SUFFIXES))
            ),
        )

    staging = Path(tempfile.mkdtemp(prefix="elp-upload-"))
    target = staging / filename
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                            "upload limit"
                        ),
                    )
                handle.write(chunk)

        ingest_request = IngestRequest(
            path=target,
            doc_key=meta.doc_key,
            title=meta.title,
            department=meta.department,
            doc_type=meta.doc_type,
            revision=meta.revision,
            effective_date=meta.effective_date,
            review_due_date=meta.review_due_date,
            allowed_groups=meta.allowed_groups,
            classification=meta.classification,
            source_uri=f"upload://{filename}",
            meta={"uploaded_by": principal.subject, "original_filename": filename},
        )
        try:
            result = await Ingestor().ingest(session, ingest_request, force=meta.force)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"the embedding model is not reachable, so the document "
                f"could not be indexed: {exc}",
            ) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    await audit.record(
        session,
        principal,
        "documents.upload",
        resource=f"{meta.doc_key} rev {meta.revision}",
        request=request,
        detail={
            "chunks": result.chunk_count,
            "pages": result.page_count,
            "skipped": result.skipped,
            "allowed_groups": meta.allowed_groups,
            "bytes": written,
        },
    )

    return IngestResponse(
        document_id=result.document_id,
        doc_key=result.doc_key,
        revision=result.revision,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        skipped=result.skipped,
        reason=result.reason,
    )


@router.patch("/{document_id}/access")
async def update_access(
    document_id: str,
    allowed_groups: list[str],
    request: Request,
    principal: Principal = Depends(require_scope(Scope.DOCS_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Change which Active Directory groups may read a document."""
    document = (
        await session.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    previous = list(document.allowed_groups)
    document.allowed_groups = allowed_groups

    await audit.record(
        session,
        principal,
        "documents.access_changed",
        resource=f"{document.doc_key} rev {document.revision}",
        request=request,
        detail={"from": previous, "to": allowed_groups},
    )
    return {"document_id": document_id, "allowed_groups": allowed_groups}


@router.delete("/{doc_key}")
async def remove_document(
    doc_key: str,
    request: Request,
    revision: str = "",
    principal: Principal = Depends(require_scope(Scope.DOCS_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    deleted = await delete_document(session, doc_key, revision)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no document '{doc_key}'")

    await audit.record(
        session,
        principal,
        "documents.delete",
        resource=f"{doc_key} rev {revision or 'all'}",
        request=request,
        detail={"deleted": deleted},
    )
    return {"doc_key": doc_key, "revisions_deleted": deleted}
