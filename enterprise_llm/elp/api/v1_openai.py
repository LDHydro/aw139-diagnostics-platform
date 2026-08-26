"""
OpenAI-compatible endpoints.

Any in-house application, IDE assistant or script that already speaks the
OpenAI API can point its base URL here and its API key at a platform service
key.  Nothing else changes, and every call is authenticated, scoped and
audited on the way through.

Two virtual models are published alongside the raw one:

* ``elp-grounded``   - retrieves from the governing documents first and
  returns an answer with citations appended, so existing chat UIs get
  references without knowing anything about RAG.
* ``elp-code``       - routes to the code profile (or a dedicated code
  endpoint when one is configured).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..config import get_settings
from ..db import get_session
from ..llm.client import ChatMessage, InferenceError
from ..llm.embeddings import get_embedding_client
from ..llm.router import TaskKind, get_router
from ..rag.answer import get_synthesizer
from ..rag.retrieve import get_retriever

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

GROUNDED_MODEL = "elp-grounded"
CODE_MODEL = "elp-code"


class ChatCompletionMessage(BaseModel):
    role: str
    content: str | None = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatCompletionMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    top_p: float | None = None
    stream: bool = False
    user: str | None = None


class EmbeddingRequest(BaseModel):
    model: str = ""
    input: str | list[str]


def _last_user_message(messages: list[ChatCompletionMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


def _task_for(model: str) -> TaskKind:
    if model == CODE_MODEL:
        return TaskKind.CODE
    if model == GROUNDED_MODEL:
        return TaskKind.GROUNDED_ANSWER
    return TaskKind.CHAT


@router.get("/models")
async def list_models(
    principal: Principal = Depends(require_scope(Scope.CHAT)),
) -> dict[str, Any]:
    settings = get_settings().inference
    router_info = get_router().describe()
    created = int(time.time())

    models = [
        {
            "id": settings.chat_model,
            "object": "model",
            "created": created,
            "owned_by": "local",
            "description": "Local generalist model served by vLLM.",
        },
        {
            "id": GROUNDED_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "elp",
            "description": (
                "Answers from the department governing documents you are "
                "cleared to read, with citations."
            ),
        },
        {
            "id": CODE_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "elp",
            "description": "Code-tuned sampling profile for application development.",
        },
    ]
    if router_info["dedicated_code_model"]:
        models.append(
            {
                "id": router_info["code_model"],
                "object": "model",
                "created": created,
                "owned_by": "local",
                "description": "Dedicated code model endpoint.",
            }
        )
    return {"object": "list", "data": models}


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.CHAT)),
    session: AsyncSession = Depends(get_session),
):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    started = time.monotonic()
    model = payload.model or get_settings().inference.chat_model
    task = _task_for(model)

    if task is TaskKind.CODE and not principal.has(Scope.DEV):
        raise HTTPException(
            status_code=403,
            detail=f"the '{CODE_MODEL}' model requires the '{Scope.DEV}' permission",
        )

    # Grounded mode short-circuits into the RAG pipeline.
    if model == GROUNDED_MODEL:
        return await _grounded_completion(
            payload, request, principal, session, started
        )

    messages = [
        ChatMessage(role=m.role, content=m.content or "") for m in payload.messages
    ]
    client, profile = get_router().resolve(task)

    if payload.stream:
        return StreamingResponse(
            _stream_completion(client, messages, payload, profile, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        completion = await client.chat(
            messages,
            temperature=(
                profile.temperature if payload.temperature is None else payload.temperature
            ),
            max_tokens=payload.max_tokens or profile.max_tokens,
            top_p=payload.top_p or profile.top_p,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    await audit.record(
        session,
        principal,
        "chat.completions",
        resource=model,
        latency_ms=(time.monotonic() - started) * 1000,
        request=request,
        detail={
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        },
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": completion.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
        },
    }


async def _stream_completion(client, messages, payload, profile, model):
    """Relay the model's deltas as OpenAI-shaped server-sent events."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None = None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    yield frame({"role": "assistant", "content": ""})
    try:
        async for piece in client.stream(
            messages,
            temperature=(
                profile.temperature if payload.temperature is None else payload.temperature
            ),
            max_tokens=payload.max_tokens or profile.max_tokens,
            top_p=payload.top_p or profile.top_p,
        ):
            yield frame({"content": piece})
    except InferenceError as exc:
        # The response has already started, so the error has to travel in-band.
        yield frame({"content": f"\n\n[error: {exc}]"}, finish="stop")
        yield "data: [DONE]\n\n"
        return

    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"


async def _grounded_completion(
    payload: ChatCompletionRequest,
    request: Request,
    principal: Principal,
    session: AsyncSession,
    started: float,
) -> dict[str, Any]:
    """Serve ``elp-grounded`` through retrieval, returning citations inline."""
    if not principal.has(Scope.ASK):
        raise HTTPException(
            status_code=403,
            detail=f"the '{GROUNDED_MODEL}' model requires the '{Scope.ASK}' permission",
        )

    question = _last_user_message(payload.messages)
    if not question:
        raise HTTPException(status_code=400, detail="no user message to answer")

    history = [
        ChatMessage(role=m.role, content=m.content or "")
        for m in payload.messages[:-1]
        if m.role in {"user", "assistant"}
    ]

    passages = await get_retriever().retrieve(session, question, principal)
    result = await get_synthesizer().answer(question, passages, history=history)

    # Chat UIs have nowhere to render structured references, so append them.
    content = result.answer
    if result.references:
        lines = "\n".join(
            f"[{r['marker']}] {r['citation']}" for r in result.references
        )
        content = f"{content}\n\n---\nSources:\n{lines}"

    await audit.record(
        session,
        principal,
        "chat.completions.grounded",
        resource=question[:500],
        outcome="ok" if result.grounded else "ungrounded",
        latency_ms=(time.monotonic() - started) * 1000,
        request=request,
        detail={
            "confidence": round(result.confidence, 3),
            "references": audit.references_digest(result.references),
        },
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": GROUNDED_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        # Non-standard but harmless: clients that know to look get structure.
        "elp": {
            "references": result.references,
            "confidence": round(result.confidence, 3),
            "grounded": result.grounded,
            "warnings": result.warnings,
        },
    }


@router.post("/embeddings")
async def embeddings(
    payload: EmbeddingRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.CHAT)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(texts) > 256:
        raise HTTPException(
            status_code=400, detail="at most 256 inputs per embeddings request"
        )

    client = get_embedding_client()
    try:
        vectors = await client.embed(texts)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    await audit.record(
        session, principal, "embeddings", request=request, detail={"count": len(texts)}
    )

    return {
        "object": "list",
        "model": client.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts), "total_tokens": 0},
    }
