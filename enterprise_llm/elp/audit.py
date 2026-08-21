"""
Audit logging.

In a maintenance organisation the question "who was told what, by which
revision of which document, and when" is an airworthiness question, not an
IT one.  Every answered question records the sources that produced it, so a
past answer can be reconstructed even after the document is revised.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from .auth.principal import Principal
from .models import AuditEvent

log = logging.getLogger(__name__)


def client_ip(request: Request | None) -> str:
    if request is None:
        return ""
    # Behind the reverse proxy the real client is in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


async def record(
    session: AsyncSession,
    principal: Principal,
    action: str,
    *,
    resource: str = "",
    outcome: str = "ok",
    detail: dict[str, Any] | None = None,
    request: Request | None = None,
    latency_ms: float = 0.0,
) -> None:
    """Write one audit row.  Never raises - auditing must not break a request."""
    try:
        event = AuditEvent(
            request_id=getattr(getattr(request, "state", None), "request_id", "") or "",
            actor=principal.subject,
            actor_type=principal.kind,
            actor_groups=list(principal.groups),
            action=action,
            resource=resource[:2000],
            outcome=outcome,
            client_ip=client_ip(request),
            latency_ms=latency_ms,
            detail=detail or {},
        )
        session.add(event)
        await session.flush()
    except Exception as exc:  # noqa: BLE001
        log.error("failed to write audit event for %s/%s: %s", principal.subject, action, exc)


def references_digest(references: list[dict]) -> list[dict]:
    """Compact form of an answer's references, for the audit detail column."""
    return [
        {
            "marker": r.get("marker"),
            "type": r.get("type"),
            "citation": r.get("citation"),
            "document_key": r.get("document_key"),
            "revision": r.get("revision"),
            "system": r.get("system"),
        }
        for r in references
    ]
