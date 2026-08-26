"""
Application development assistance.

Reads files from explicitly configured workspace roots only.  Every path is
resolved and checked against those roots before it is opened, so a crafted
``files`` entry cannot walk out of the workspace and read, say, the platform's
own environment file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..config import get_settings
from ..db import get_session
from ..llm.client import ChatMessage, InferenceError
from ..llm.router import TaskKind, get_router
from .schemas import DevRequest, DevResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/dev", tags=["development"])

_MODE_INSTRUCTIONS = {
    "explain": (
        "Explain how the supplied code works. Be concrete about control flow, "
        "data flow and edge cases. Do not restate the code line by line."
    ),
    "generate": (
        "Write the requested code. Match the conventions, naming and structure "
        "of the supplied files. Return complete, runnable code, not fragments, "
        "and say briefly what you changed and why."
    ),
    "review": (
        "Review the supplied code for correctness bugs, security problems and "
        "unnecessary complexity. Report concrete findings with file and line "
        "references. Do not invent problems; if the code is sound, say so."
    ),
    "patch": (
        "Return a unified diff (git apply compatible) implementing the request. "
        "Include correct file paths and sufficient context lines. Return only "
        "the diff."
    ),
    "test": (
        "Write tests for the supplied code using the test framework already "
        "present in the project. Cover the edge cases, not just the happy path."
    ),
}


def _resolve_roots() -> list[Path]:
    roots = []
    for raw in get_settings().dev.workspace_roots:
        try:
            roots.append(Path(raw).expanduser().resolve(strict=True))
        except (OSError, RuntimeError):
            log.warning("configured workspace root does not exist: %s", raw)
    return roots


def _safe_path(candidate: str, roots: list[Path], workspace: str = "") -> Path:
    """Resolve a requested path, refusing anything outside the allowed roots."""
    if not roots:
        raise HTTPException(
            status_code=503,
            detail=(
                "no workspace roots are configured; set "
                "ELP_DEV__WORKSPACE_ROOTS before using file context"
            ),
        )

    path = Path(candidate).expanduser()
    if not path.is_absolute():
        base = Path(workspace) if workspace else roots[0]
        path = base / path

    try:
        # strict=True resolves symlinks, so a symlink pointing out of the
        # workspace is caught by the containment check below.
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=f"file not found: {candidate}") from exc

    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved

    raise HTTPException(
        status_code=403,
        detail=f"'{candidate}' is outside the configured workspace roots",
    )


@router.get("/workspaces")
async def workspaces(
    principal: Principal = Depends(require_scope(Scope.DEV)),
) -> dict:
    settings = get_settings().dev
    return {
        "enabled": settings.enabled,
        "roots": [str(r) for r in _resolve_roots()],
        "max_file_bytes": settings.max_file_bytes,
        "max_files_per_request": settings.max_files_per_request,
        "excluded_dirs": settings.excluded_dirs,
    }


@router.post("", response_model=DevResponse)
async def assist(
    payload: DevRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.DEV)),
    session: AsyncSession = Depends(get_session),
) -> DevResponse:
    settings = get_settings().dev
    if not settings.enabled:
        raise HTTPException(
            status_code=503, detail="development assistance is disabled on this deployment"
        )
    if len(payload.files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=(
                f"at most {settings.max_files_per_request} files per request; "
                f"{len(payload.files)} were supplied"
            ),
        )

    started = time.monotonic()
    roots = _resolve_roots()
    context_blocks: list[str] = []
    included: list[str] = []
    truncated: list[str] = []

    for candidate in payload.files:
        path = _safe_path(candidate, roots, payload.workspace)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"not a file: {candidate}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(
                status_code=422, detail=f"could not read {candidate}: {exc}"
            ) from exc

        if len(content.encode("utf-8")) > settings.max_file_bytes:
            content = content[: settings.max_file_bytes]
            truncated.append(candidate)

        language = path.suffix.lstrip(".") or payload.language
        context_blocks.append(
            f"--- {path} ---\n```{language}\n{content}\n```"
        )
        included.append(str(path))

    instruction = _MODE_INSTRUCTIONS.get(payload.mode, _MODE_INSTRUCTIONS["generate"])
    system = (
        "You are a senior software engineer working inside a company's own "
        "codebase, on a self-hosted model. Nothing you are shown leaves the "
        f"building.\n\n{instruction}"
    )
    user_parts = [f"REQUEST\n=======\n{payload.instruction}"]
    if context_blocks:
        user_parts.append("FILES\n=====\n" + "\n\n".join(context_blocks))
    if payload.language:
        user_parts.append(f"Target language: {payload.language}")

    client, profile = get_router().resolve(TaskKind.CODE)
    try:
        completion = await client.chat(
            [ChatMessage("system", system), ChatMessage("user", "\n\n".join(user_parts))],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            top_p=profile.top_p,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = (time.monotonic() - started) * 1000
    await audit.record(
        session,
        principal,
        "dev.assist",
        resource=payload.mode,
        latency_ms=latency_ms,
        request=request,
        detail={"files": included, "truncated": truncated, "mode": payload.mode},
    )

    return DevResponse(
        output=completion.text,
        files_included=included,
        truncated=truncated,
        model=completion.model,
        latency_ms=latency_ms,
    )
