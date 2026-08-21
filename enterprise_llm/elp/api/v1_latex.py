"""LaTeX authoring and PDF generation."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..db import get_session
from ..latex.render import LatexError, get_renderer
from ..llm.client import InferenceError
from ..rag.retrieve import RetrievalFilter, get_retriever
from .schemas import LatexRequest, LatexResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/latex", tags=["latex"])


@router.post("", response_model=LatexResponse)
async def render_latex(
    payload: LatexRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.LATEX)),
    session: AsyncSession = Depends(get_session),
) -> LatexResponse:
    """
    Produce a LaTeX document and, by default, compile it to PDF.

    Supply ``source`` to compile LaTeX you already have, or ``brief`` to have
    the local model write it.  With ``ground_in_documents`` the brief is
    answered from the governing documents and the result carries citations.
    """
    if not payload.brief and not payload.source:
        raise HTTPException(
            status_code=400, detail="supply either 'brief' or 'source'"
        )

    started = time.monotonic()
    renderer = get_renderer()
    references: list[dict] = []
    sources_block = ""

    if payload.ground_in_documents and payload.brief:
        passages = await get_retriever().retrieve(
            session,
            payload.brief,
            principal,
            filters=RetrievalFilter(departments=payload.departments),
        )
        blocks = []
        for index, passage in enumerate(passages, start=1):
            marker = f"D{index}"
            blocks.append(f"[{marker}] {passage.citation()}\n{passage.text}")
            references.append(passage.to_reference(marker))
        sources_block = "\n\n---\n\n".join(blocks)

    try:
        if payload.source:
            source = payload.source
            result = await renderer.compile(source) if payload.compile else None
        elif payload.compile:
            source, result = await renderer.generate_and_compile(
                payload.brief,
                template=payload.template,
                sources=sources_block,
                document_class=payload.document_class,
            )
        else:
            source = await renderer.generate(
                payload.brief,
                template=payload.template,
                sources=sources_block,
                document_class=payload.document_class,
            )
            result = None
    except LatexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(
            status_code=503, detail=f"the local model is not reachable: {exc}"
        ) from exc

    await audit.record(
        session,
        principal,
        "latex.render",
        resource=(payload.brief or "supplied source")[:300],
        outcome="ok" if (result is None or result.success) else "compile_failed",
        latency_ms=(time.monotonic() - started) * 1000,
        request=request,
        detail={
            "compiled": bool(result and result.success),
            "grounded": payload.ground_in_documents,
            "references": audit.references_digest(references),
        },
    )

    return LatexResponse(
        source=source,
        compiled=bool(result and result.success),
        artifact_id=result.artifact_id if result else "",
        download_url=(
            f"/v1/latex/{result.artifact_id}/pdf" if result and result.success else ""
        ),
        page_count=result.page_count if result else 0,
        size_bytes=result.size_bytes if result else 0,
        errors=result.errors if result else [],
        references=references,
    )


@router.get("/{artifact_id}/pdf")
async def download_pdf(
    artifact_id: str,
    principal: Principal = Depends(require_scope(Scope.LATEX)),
) -> FileResponse:
    path = get_renderer().artifact_path(artifact_id)
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found or expired")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=f"{artifact_id}.pdf",
    )


@router.get("/templates")
async def list_templates(
    principal: Principal = Depends(require_scope(Scope.LATEX)),
) -> list[dict]:
    """Built-in preambles a brief can be rendered against."""
    from pathlib import Path

    from ..config import get_settings

    directory = Path(get_settings().latex.template_dir)
    if not directory.is_dir():
        return []
    return [
        {
            "name": path.stem,
            "filename": path.name,
            "content": path.read_text(encoding="utf-8"),
        }
        for path in sorted(directory.glob("*.tex"))
    ]
