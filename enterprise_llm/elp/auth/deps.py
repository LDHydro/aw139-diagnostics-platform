"""FastAPI dependencies that turn a request into an authorised Principal."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ..config import get_settings
from ..db import get_sessionmaker
from .apikeys import resolve_api_key
from .oidc import AuthError, get_validator
from .principal import DEV_SUPERUSER, Principal, Scope

log = logging.getLogger(__name__)


async def current_principal(request: Request) -> Principal:
    """
    Resolve the caller from either an SSO bearer token or a service API key.

    Order matters: a bearer token identifies a person and always wins over
    an API key, so a user session is never silently downgraded to whatever
    the calling app's service account can see.

    Deliberately does not take a database session as a dependency: validating
    an SSO token needs only the identity provider's public keys, so a
    database outage must not turn every request into an authentication
    failure. The API-key path opens its own short-lived session, since that
    is the only branch that needs one.
    """
    settings = get_settings()
    auth = settings.auth

    if auth.mode == "disabled":
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="authentication is disabled but environment is production",
            )
        request.state.principal = DEV_SUPERUSER
        return DEV_SUPERUSER

    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        try:
            if auth.mode == "oidc":
                principal = await get_validator().principal_from_token(token)
            else:
                # In LDAP mode we issue our own session tokens; the login
                # route stores them and they are presented the same way.
                from .sessions import resolve_session_token

                principal = await resolve_session_token(token)
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        request.state.principal = principal
        return principal

    api_key = request.headers.get(auth.api_key_header, "")
    if api_key:
        try:
            async with get_sessionmaker()() as session:
                principal = await resolve_api_key(session, api_key)
                await session.commit()
        except HTTPException:
            raise
        except Exception as exc:
            log.error("could not verify API key: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API key verification is unavailable (database unreachable)",
            ) from exc
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or expired API key",
            )
        request.state.principal = principal
        return principal

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "authentication required: present an SSO bearer token or an "
            f"{auth.api_key_header} header"
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scope(*scopes: str) -> Callable[..., Coroutine[Any, Any, Principal]]:
    """Dependency factory enforcing that the caller holds *any* of ``scopes``."""

    async def _dependency(
        principal: Principal = Depends(current_principal),
    ) -> Principal:
        if not principal.has_any(*scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"'{principal.subject}' lacks the required permission "
                    f"({' or '.join(scopes)}); roles held: {', '.join(principal.roles) or 'none'}"
                ),
            )
        return principal

    return _dependency


require_admin = require_scope(Scope.ADMIN)
