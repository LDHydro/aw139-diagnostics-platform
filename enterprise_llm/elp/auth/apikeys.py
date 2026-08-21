"""
Service-account API keys for in-house applications.

An application (the AW139 diagnostics UI, a scheduling tool, a CI job)
authenticates with a long-lived key instead of a user token.  The key still
carries AD groups, so document-level ACLs apply exactly as they do for a
person: an app cannot read a document its owning group cannot read.

Only a SHA-256 hash is stored.  The plaintext is returned once, at creation.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import ApiKey
from .principal import Principal, scopes_for_roles

_KEY_BYTES = 32


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Return ``(plaintext, hash, prefix)`` for a fresh credential."""
    prefix = get_settings().auth.api_key_prefix
    raw = prefix + secrets.token_urlsafe(_KEY_BYTES)
    return raw, _hash(raw), raw[: len(prefix) + 8]


async def create_api_key(
    session: AsyncSession,
    *,
    name: str,
    scopes: list[str],
    groups: list[str],
    owner_email: str = "",
    description: str = "",
    expires_at: datetime | None = None,
    rate_limit_per_minute: int = 120,
) -> tuple[ApiKey, str]:
    raw, digest, prefix = generate_key()
    record = ApiKey(
        name=name,
        description=description,
        key_hash=digest,
        key_prefix=prefix,
        scopes=scopes,
        groups=groups,
        owner_email=owner_email,
        expires_at=expires_at,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    session.add(record)
    await session.flush()
    return record, raw


async def resolve_api_key(session: AsyncSession, raw: str) -> Principal | None:
    """Validate a presented key and return the caller it identifies."""
    if not raw:
        return None
    digest = _hash(raw.strip())
    result = await session.execute(select(ApiKey).where(ApiKey.key_hash == digest))
    record = result.scalar_one_or_none()
    if record is None or not record.active:
        return None

    now = datetime.now(UTC)
    if record.expires_at is not None:
        expires = record.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires < now:
            return None

    record.last_used_at = now

    # A key may be granted scopes directly, or inherit them from roles.
    scopes = frozenset(record.scopes) if record.scopes else scopes_for_roles(["service"])
    return Principal(
        subject=f"service:{record.name}",
        display_name=record.name,
        email=record.owner_email,
        kind="service",
        groups=list(record.groups),
        roles=["service"],
        scopes=scopes,
        credential_id=record.id,
        issuer="elp-api-key",
    )


async def revoke_api_key(session: AsyncSession, name: str) -> bool:
    result = await session.execute(select(ApiKey).where(ApiKey.name == name))
    record = result.scalar_one_or_none()
    if record is None:
        return False
    record.active = False
    return True
