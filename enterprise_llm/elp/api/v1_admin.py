"""Administration: identity, service accounts, peer systems, audit and health."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.apikeys import create_api_key, revoke_api_key
from ..auth.deps import current_principal, require_admin, require_scope
from ..auth.principal import ROLE_SCOPES, Principal, Scope
from ..config import get_settings
from ..db import get_session, healthcheck
from ..federation.orchestrator import get_orchestrator
from ..federation.registry import get_registry
from ..llm.embeddings import get_embedding_client, get_rerank_client
from ..llm.router import get_router
from ..models import ApiKey, AuditEvent, PeerSystem
from .schemas import ApiKeyRequest, ApiKeyResponse, PeerRequest, WhoAmIResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["administration"])


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------

@router.get("/whoami", response_model=WhoAmIResponse)
async def whoami(
    principal: Principal = Depends(current_principal),
) -> WhoAmIResponse:
    """
    What the platform believes about the caller.

    The first thing to check when someone reports "it says I don't have
    access": it shows exactly which AD groups arrived on the token and what
    they mapped to.
    """
    return WhoAmIResponse(
        subject=principal.subject,
        display_name=principal.display_name,
        email=principal.email,
        kind=principal.kind,
        groups=principal.groups,
        roles=principal.roles,
        scopes=sorted(principal.scopes),
        issuer=principal.issuer,
    )


@router.get("/roles")
async def roles(principal: Principal = Depends(require_admin)) -> dict:
    """The role catalogue and the AD group mapping currently in force."""
    auth = get_settings().auth
    return {
        "mode": auth.mode,
        "roles": {name: list(scopes) for name, scopes in ROLE_SCOPES.items()},
        "group_role_map": auth.group_role_map,
        "allowed_groups": auth.allowed_groups,
        "default_role": auth.default_role,
        "groups_claim": auth.groups_claim,
    }


# ----------------------------------------------------------------------
# Service accounts
# ----------------------------------------------------------------------

@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> list[ApiKeyResponse]:
    rows = (await session.execute(select(ApiKey).order_by(ApiKey.name))).scalars().all()
    return [
        ApiKeyResponse(
            id=r.id,
            name=r.name,
            key_prefix=r.key_prefix,
            scopes=list(r.scopes),
            groups=list(r.groups),
            active=r.active,
            expires_at=r.expires_at,
        )
        for r in rows
    ]


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=201)
async def issue_api_key(
    payload: ApiKeyRequest,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiKeyResponse:
    """
    Issue a credential for an in-house application.

    The plaintext key is returned exactly once; only its hash is stored.
    """
    unknown = set(payload.scopes) - set(Scope.ALL)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown scope(s): {', '.join(sorted(unknown))}. Valid scopes: "
                + ", ".join(Scope.ALL)
            ),
        )
    existing = (
        await session.execute(select(ApiKey).where(ApiKey.name == payload.name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"a key named '{payload.name}' already exists"
        )

    record, plaintext = await create_api_key(
        session,
        name=payload.name,
        scopes=payload.scopes,
        groups=payload.groups,
        owner_email=payload.owner_email,
        description=payload.description,
        expires_at=payload.expires_at,
        rate_limit_per_minute=payload.rate_limit_per_minute,
    )

    await audit.record(
        session,
        principal,
        "admin.api_key_created",
        resource=payload.name,
        request=request,
        detail={"scopes": payload.scopes, "groups": payload.groups},
    )

    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        scopes=list(record.scopes),
        groups=list(record.groups),
        active=record.active,
        expires_at=record.expires_at,
        api_key=plaintext,
    )


@router.delete("/api-keys/{name}")
async def delete_api_key(
    name: str,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if not await revoke_api_key(session, name):
        raise HTTPException(status_code=404, detail=f"no key named '{name}'")
    await audit.record(
        session, principal, "admin.api_key_revoked", resource=name, request=request
    )
    return {"name": name, "active": False}


# ----------------------------------------------------------------------
# Peer AI systems
# ----------------------------------------------------------------------

@router.get("/peers")
async def list_peers(
    principal: Principal = Depends(require_scope(Scope.FEDERATION)),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Internal AI systems this caller may consult."""
    peers = await get_registry().visible_to(principal, session)
    return [
        {
            "name": p.name,
            "display_name": p.display_name or p.name,
            "description": p.description,
            "protocol": p.protocol,
            "model": p.model,
            "capabilities": p.capabilities,
            "enabled": p.enabled,
            "citable": p.citable,
        }
        for p in peers
    ]


@router.post("/peers", status_code=201)
async def register_peer(
    payload: PeerRequest,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Register another internal AI system.

    Credentials are never stored here - ``auth_env_var`` names the
    environment variable the gateway reads at call time.
    """
    existing = (
        await session.execute(select(PeerSystem).where(PeerSystem.name == payload.name))
    ).scalar_one_or_none()
    peer = existing or PeerSystem(name=payload.name, base_url=payload.base_url)
    if existing is None:
        session.add(peer)

    for field_name, value in payload.model_dump().items():
        setattr(peer, field_name, value)
    await session.flush()

    await audit.record(
        session,
        principal,
        "admin.peer_registered" if existing is None else "admin.peer_updated",
        resource=payload.name,
        request=request,
        detail={"protocol": payload.protocol, "base_url": payload.base_url},
    )
    return {"name": peer.name, "created": existing is None}


@router.delete("/peers/{name}")
async def delete_peer(
    name: str,
    request: Request,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict:
    peer = (
        await session.execute(select(PeerSystem).where(PeerSystem.name == name))
    ).scalar_one_or_none()
    if peer is None:
        raise HTTPException(status_code=404, detail=f"no peer named '{name}'")
    await session.delete(peer)
    await audit.record(
        session, principal, "admin.peer_deleted", resource=name, request=request
    )
    return {"name": name, "deleted": True}


# ----------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------

@router.get("/audit")
async def list_audit(
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    actor: str = "",
    action: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    query = select(AuditEvent).order_by(desc(AuditEvent.ts))
    if actor:
        query = query.where(AuditEvent.actor == actor)
    if action:
        query = query.where(AuditEvent.action == action)

    rows = (
        await session.execute(query.limit(min(limit, 1000)).offset(offset))
    ).scalars().all()
    return [
        {
            "id": r.id,
            "ts": r.ts.isoformat(),
            "actor": r.actor,
            "actor_type": r.actor_type,
            "action": r.action,
            "resource": r.resource,
            "outcome": r.outcome,
            "client_ip": r.client_ip,
            "latency_ms": r.latency_ms,
            "detail": r.detail,
        }
        for r in rows
    ]


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

@router.get("/health/deep")
async def deep_health(
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Probe every dependency.  Use this after a deployment or a GPU reset."""
    database = await healthcheck()
    models = await get_router().health()
    embeddings = await get_embedding_client().health()
    reranker = await get_rerank_client().health()
    peers = await get_orchestrator().health(session)

    components = {
        "database": database,
        "models": models,
        "embeddings": embeddings,
        "reranker": reranker,
        "peers": peers,
    }
    degraded = [
        name
        for name, value in components.items()
        if isinstance(value, dict) and value.get("status") == "error"
    ]
    if any(p.get("status") == "error" for p in peers):
        degraded.append("peers")

    return {
        "status": "degraded" if degraded else "ok",
        "degraded": sorted(set(degraded)),
        "components": components,
        "routing": get_router().describe(),
    }
