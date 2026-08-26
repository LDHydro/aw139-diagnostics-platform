"""
Short-lived signed session tokens.

Only used in LDAP mode, where the platform performs the credential check
itself and therefore has to issue something the browser can present on
subsequent requests.  In OIDC mode the IdP's own access token is used and
none of this is reachable.
"""

from __future__ import annotations

import time

import jwt

from ..config import get_settings
from .oidc import AuthError
from .principal import Principal, scopes_for_roles

_ALGORITHM = "HS256"
_DEFAULT_TTL_SECONDS = 8 * 3600


def _secret() -> str:
    secret = get_settings().auth.oidc_client_secret
    if not secret:
        raise AuthError(
            "ELP_AUTH__OIDC_CLIENT_SECRET must be set to sign session tokens "
            "in LDAP mode"
        )
    return secret


def issue_session_token(principal: Principal, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> str:
    now = int(time.time())
    payload = {
        "sub": principal.subject,
        "name": principal.display_name,
        "email": principal.email,
        "groups": principal.groups,
        "roles": principal.roles,
        "iat": now,
        "exp": now + ttl_seconds,
        "iss": "elp-ldap",
    }
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


async def resolve_session_token(token: str) -> Principal:
    try:
        claims = jwt.decode(
            token,
            _secret(),
            algorithms=[_ALGORITHM],
            issuer="elp-ldap",
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("session has expired, please sign in again") from exc
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid session token: {exc}") from exc

    roles = list(claims.get("roles") or ["reader"])
    return Principal(
        subject=str(claims["sub"]),
        display_name=str(claims.get("name", "")),
        email=str(claims.get("email", "")),
        kind="user",
        groups=list(claims.get("groups") or []),
        roles=roles,
        scopes=scopes_for_roles(roles),
        credential_id=str(claims["sub"]),
        issuer="elp-ldap",
        token_expires_at=claims.get("exp"),
    )
