"""
Sign-in endpoints.

Only reachable in LDAP mode.  In OIDC mode the identity provider owns the
login flow entirely and these routes report that, rather than offering a
second, weaker way in.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.oidc import AuthError
from ..auth.principal import Principal
from ..auth.sessions import issue_session_token
from ..config import get_settings
from ..db import get_session

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=256)
    password: str = Field(..., min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    subject: str
    display_name: str
    roles: list[str]
    groups: list[str]


@router.get("/config")
async def auth_config() -> dict:
    """What a client needs to know to sign a user in."""
    auth = get_settings().auth
    if auth.mode == "oidc":
        return {
            "mode": "oidc",
            "issuer": auth.oidc_issuer,
            "client_id": auth.oidc_client_id,
            "audience": auth.oidc_audience or auth.oidc_client_id,
            "authorization_flow": "authorization_code_with_pkce",
            "note": (
                "Obtain a token from the identity provider and present it as "
                "'Authorization: Bearer <token>'."
            ),
        }
    if auth.mode == "ldap":
        return {"mode": "ldap", "login_endpoint": "/v1/auth/login"}
    return {"mode": "disabled", "note": "authentication is disabled (development only)"}


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> LoginResponse:
    settings = get_settings()
    if settings.auth.mode != "ldap":
        raise HTTPException(
            status_code=404,
            detail=(
                "password login is not enabled; this deployment authenticates "
                "through the corporate identity provider"
            ),
        )

    from ..auth.ldap_auth import authenticate

    try:
        # ldap3 is synchronous; run it off the event loop so a slow domain
        # controller cannot stall every other request on the server.
        import anyio

        principal = await anyio.to_thread.run_sync(
            authenticate, payload.username, payload.password, settings.auth
        )
    except AuthError as exc:
        await audit.record(
            session,
            Principal(subject=payload.username, kind="user"),
            "auth.login_failed",
            outcome="denied",
            request=request,
            detail={"reason": str(exc)},
        )
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    ttl = 8 * 3600
    token = issue_session_token(principal, ttl)

    await audit.record(
        session,
        principal,
        "auth.login",
        request=request,
        detail={"roles": principal.roles, "group_count": len(principal.groups)},
    )

    return LoginResponse(
        access_token=token,
        expires_in=ttl,
        subject=principal.subject,
        display_name=principal.display_name,
        roles=principal.roles,
        groups=principal.groups,
    )
