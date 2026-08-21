"""
Plain-language question answering with references.

This is the endpoint every in-house application calls.  It retrieves from
the governing documents the caller is cleared to read, optionally consults
other internal AI systems, and returns an answer whose every claim carries a
resolvable citation.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..config import get_settings
from ..db import get_session
from ..federation.orchestrator import get_orchestrator
from ..llm.client import ChatMessage, InferenceError
from ..models import Conversation, Message
from ..rag.answer import get_synthesizer
from ..rag.retrieve import RetrievalFilter, get_retriever
from .schemas import AskRequest, AskResponse, SearchHit, SearchRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ask"])


async def _load_history(
    session: AsyncSession, conversation_id: str, principal: Principal
) -> tuple[Conversation | None, list[ChatMessage]]:
    conversation = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
    ).scalar_one_or_none()
    if conversation is None:
        return None, []
    if conversation.actor != principal.subject and not principal.is_admin:
        raise HTTPException(status_code=403, detail="this conversation belongs to another user")

    rows = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(6)
        )
    ).scalars().all()
    history = [ChatMessage(role=r.role, content=r.content) for r in reversed(rows)]
    return conversation, history


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.ASK)),
    session: AsyncSession = Depends(get_session),
) -> AskResponse:
    started = time.monotonic()
    settings = get_settings()

    conversation, history = (
        await _load_history(session, payload.conversation_id, principal)
        if payload.conversation_id
        else (None, [])
    )

    filters = RetrievalFilter(
        departments=payload.departments,
        doc_keys=payload.doc_keys,
        include_superseded=payload.include_superseded,
    )

    try:
        passages = await get_retriever().retrieve(
            session, payload.question, principal, filters=filters, top_k=payload.top_k
        )
    except InferenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"the local embedding model is not reachable: {exc}",
        ) from exc

    # Consult other internal AI systems when permitted and requested.
    peer_answers: list = []
    peer_warnings: list[str] = []
    wants_peers = (
        payload.consult_peers
        if payload.consult_peers is not None
        else settings.federation.auto_consult
    )
    if wants_peers and principal.has(Scope.FEDERATION):
        peer_answers, peer_warnings = await get_orchestrator().ask_peers(
            payload.question,
            principal,
            session,
            requested=payload.peers or None,
            context=(
                "You are being consulted by another internal system on behalf of "
                "an aviation maintenance organisation. Answer concisely and state "
                "plainly when you do not know."
            ),
        )
    elif wants_peers and payload.peers:
        peer_warnings.append(
            "you do not hold the federation:query permission, so no other AI "
            "systems were consulted"
        )

    try:
        result = await get_synthesizer().answer(
            payload.question, passages, peer_answers=peer_answers, history=history
        )
    except InferenceError as exc:
        raise HTTPException(
            status_code=503, detail=f"the local model is not reachable: {exc}"
        ) from exc

    result.warnings.extend(peer_warnings)
    latency_ms = (time.monotonic() - started) * 1000

    # Persist the turn so a conversation can be continued and reviewed.
    conversation_id = None
    if conversation is None and payload.conversation_id is None and payload.app:
        conversation = Conversation(
            actor=principal.subject,
            title=payload.question[:120],
            app=payload.app,
        )
        session.add(conversation)
        await session.flush()
    if conversation is not None:
        conversation_id = conversation.id
        session.add(
            Message(conversation_id=conversation.id, role="user", content=payload.question)
        )
        session.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=result.answer,
                references=result.references,
                meta={"confidence": result.confidence, "model": result.model},
            )
        )

    await audit.record(
        session,
        principal,
        "ask",
        resource=payload.question[:500],
        outcome="ok" if result.grounded else "ungrounded",
        latency_ms=latency_ms,
        request=request,
        detail={
            "confidence": round(result.confidence, 3),
            "passages": result.passages_considered,
            "peers": result.peers_consulted,
            "references": audit.references_digest(result.references),
            "app": payload.app,
        },
    )

    return AskResponse(
        answer=result.answer,
        references=result.references,
        confidence=result.confidence,
        grounded=result.grounded,
        passages_considered=result.passages_considered,
        peers_consulted=result.peers_consulted,
        conversation_id=conversation_id,
        model=result.model,
        latency_ms=latency_ms,
        warnings=result.warnings,
    )


@router.post("/search", response_model=list[SearchHit])
async def search(
    payload: SearchRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.DOCS_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[SearchHit]:
    """Retrieval without generation - for apps that render their own results."""
    filters = RetrievalFilter(
        departments=payload.departments, doc_keys=payload.doc_keys
    )
    passages = await get_retriever().retrieve(
        session, payload.query, principal, filters=filters, top_k=payload.top_k
    )

    await audit.record(
        session,
        principal,
        "search",
        resource=payload.query[:500],
        request=request,
        detail={"hits": len(passages)},
    )

    return [
        SearchHit(
            citation=p.citation(),
            document_key=p.doc_key,
            document_title=p.doc_title,
            revision=p.revision,
            section=p.section_number,
            section_path=p.section_path,
            page_start=p.page_start,
            page_end=p.page_end,
            score=round(p.score, 4),
            text=p.text if payload.include_text else "",
        )
        for p in passages
    ]
