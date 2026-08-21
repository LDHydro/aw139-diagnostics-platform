"""
OIDC bearer-token validation for Microsoft Entra ID, ADFS 2019+ and Keycloak.

The platform never handles user passwords.  The in-house application (or the
browser) obtains a token from the corporate IdP and presents it here; we
validate the signature against the IdP's published JWKS and read Active
Directory group membership from the token.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

from ..config import AuthSettings, get_settings
from .principal import Principal, scopes_for_roles

log = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when a credential cannot be validated."""


@dataclass
class _JwksCache:
    keys: dict[str, Any]
    fetched_at: float


class OidcValidator:
    """Caches IdP discovery + JWKS and validates bearer tokens."""

    # Never re-fetch JWKS more often than this, even on unknown key IDs;
    # otherwise a stream of bogus tokens becomes a DoS against the IdP.
    _MIN_REFRESH_INTERVAL = 60.0

    def __init__(self, settings: AuthSettings | None = None) -> None:
        self.settings = settings or get_settings().auth
        self._jwks: _JwksCache | None = None
        self._jwks_url: str = self.settings.oidc_jwks_url
        self._issuer: str = self.settings.oidc_issuer
        self._lock = asyncio.Lock()
        self._last_refresh_attempt = 0.0

    # ------------------------------------------------------------------
    # Discovery / key material
    # ------------------------------------------------------------------

    async def _discover(self, client: httpx.AsyncClient) -> None:
        if self._jwks_url:
            return
        if not self._issuer:
            raise AuthError("ELP_AUTH__OIDC_ISSUER is not configured")
        url = self._issuer.rstrip("/") + "/.well-known/openid-configuration"
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        doc = resp.json()
        self._jwks_url = doc["jwks_uri"]
        # Trust the issuer the IdP advertises, which may differ in trailing
        # slash or tenant placeholder from what was configured.
        self._issuer = doc.get("issuer", self._issuer)

    async def _load_jwks(self, force: bool = False) -> dict[str, Any]:
        ttl = self.settings.jwks_cache_seconds
        now = time.monotonic()
        if (
            not force
            and self._jwks is not None
            and (now - self._jwks.fetched_at) < ttl
        ):
            return self._jwks.keys

        async with self._lock:
            # Another coroutine may have refreshed while we waited.
            if (
                not force
                and self._jwks is not None
                and (time.monotonic() - self._jwks.fetched_at) < ttl
            ):
                return self._jwks.keys
            if (
                force
                and self._jwks is not None
                and (time.monotonic() - self._last_refresh_attempt) < self._MIN_REFRESH_INTERVAL
            ):
                return self._jwks.keys
            self._last_refresh_attempt = time.monotonic()

            async with httpx.AsyncClient() as client:
                await self._discover(client)
                resp = await client.get(self._jwks_url, timeout=10.0)
                resp.raise_for_status()
                payload = resp.json()

            keys: dict[str, Any] = {}
            for jwk in payload.get("keys", []):
                kid = jwk.get("kid")
                if not kid or jwk.get("kty") != "RSA":
                    continue
                try:
                    keys[kid] = RSAAlgorithm.from_jwk(jwk)
                except Exception as exc:  # pragma: no cover - malformed key
                    log.warning("skipping unusable JWK %s: %s", kid, exc)
            if not keys:
                raise AuthError("IdP published no usable RSA signing keys")
            self._jwks = _JwksCache(keys=keys, fetched_at=time.monotonic())
            log.info("loaded %d signing keys from %s", len(keys), self._jwks_url)
            return keys

    async def _key_for(self, kid: str | None) -> Any:
        if not kid:
            raise AuthError("token header has no key id")
        keys = await self._load_jwks()
        if kid not in keys:
            # Key rotation: refresh once and try again.
            keys = await self._load_jwks(force=True)
        if kid not in keys:
            raise AuthError(f"unknown signing key id: {kid}")
        return keys[kid]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def validate(self, token: str) -> dict[str, Any]:
        """Verify a bearer token and return its claims."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthError(f"malformed token: {exc}") from exc

        key = await self._key_for(header.get("kid"))
        audience = self.settings.oidc_audience or self.settings.oidc_client_id

        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256", "RS384", "RS512"],
                audience=audience or None,
                issuer=self._issuer or None,
                leeway=self.settings.leeway_seconds,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": bool(self._issuer),
                    "verify_aud": bool(audience),
                    "require": ["exp", "iat"],
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError("token audience does not match this platform") from exc
        except jwt.PyJWTError as exc:
            raise AuthError(f"token rejected: {exc}") from exc

        return claims

    async def principal_from_token(self, token: str) -> Principal:
        claims = await self.validate(token)
        return build_principal(claims, self.settings)


# ----------------------------------------------------------------------
# Claim -> Principal mapping
# ----------------------------------------------------------------------

def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def extract_groups(claims: dict[str, Any], settings: AuthSettings) -> list[str]:
    """
    Pull AD group membership out of the token.

    Entra ID emits ``groups`` as an array of group *object IDs* unless the
    app registration is configured to emit sAMAccountName.  When a user
    belongs to more than ~200 groups Entra omits the claim entirely and
    instead sets ``_claim_names``/``_claim_sources`` pointing at Graph -
    the "groups overage" case.  We do not call Graph inline (it would put a
    network round-trip on every request); instead we fall back to app roles,
    which are never truncated.
    """
    groups = _as_list(claims.get(settings.groups_claim))
    if groups:
        return groups

    overage = isinstance(claims.get("_claim_names"), dict) and (
        settings.groups_claim in claims["_claim_names"]
    )
    if overage and settings.allow_group_overage_fallback:
        roles = _as_list(claims.get(settings.roles_claim))
        if roles:
            log.info(
                "groups overage for sub=%s; using %d app role(s) instead",
                claims.get("sub", "?"), len(roles),
            )
            return roles
        log.warning(
            "groups overage for sub=%s and no app roles present - "
            "assign app roles in the Entra app registration",
            claims.get("sub", "?"),
        )

    # ADFS and Keycloak commonly put group names on the roles claim.
    return _as_list(claims.get(settings.roles_claim))


def map_roles(groups: list[str], settings: AuthSettings) -> list[str]:
    """Translate AD groups into platform roles using the configured map."""
    lookup = {k.lower(): v for k, v in settings.group_role_map.items()}
    roles: list[str] = []
    for group in groups:
        role = lookup.get(group.lower())
        if role and role not in roles:
            roles.append(role)
    if not roles:
        roles = [settings.default_role]
    return roles


def build_principal(claims: dict[str, Any], settings: AuthSettings) -> Principal:
    groups = extract_groups(claims, settings)

    if settings.allowed_groups:
        permitted = {g.lower() for g in settings.allowed_groups}
        if not any(g.lower() in permitted for g in groups):
            raise AuthError(
                "your Active Directory groups do not grant access to this platform"
            )

    roles = map_roles(groups, settings)
    subject = (
        claims.get("preferred_username")
        or claims.get("upn")
        or claims.get("email")
        or claims.get("sub", "unknown")
    )
    return Principal(
        subject=str(subject),
        display_name=str(claims.get("name", "")),
        email=str(claims.get("email") or claims.get("upn") or ""),
        kind="user",
        groups=groups,
        roles=roles,
        scopes=scopes_for_roles(roles),
        credential_id=str(claims.get("sub", "")),
        issuer=str(claims.get("iss", "")),
        token_expires_at=claims.get("exp"),
    )


_validator: OidcValidator | None = None


def get_validator() -> OidcValidator:
    global _validator
    if _validator is None:
        _validator = OidcValidator()
    return _validator
