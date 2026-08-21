"""
Direct Active Directory bind, for sites without an OIDC identity provider.

This is the fallback path.  Prefer OIDC (Entra ID or ADFS): it keeps
passwords out of this platform entirely and supports MFA.  LDAP bind is
offered because plenty of maintenance networks still run a plain on-prem
domain controller with no federation in front of it.
"""

from __future__ import annotations

import logging
import re

from ..config import AuthSettings, get_settings
from .oidc import AuthError
from .principal import Principal, scopes_for_roles

log = logging.getLogger(__name__)

# "CN=AW139-Engineering,OU=Groups,DC=corp,DC=example,DC=com" -> "AW139-Engineering"
_CN_RE = re.compile(r"^CN=([^,]+)", re.IGNORECASE)


def _group_name(dn: str) -> str:
    match = _CN_RE.match(dn.strip())
    return match.group(1) if match else dn.strip()


def authenticate(username: str, password: str, settings: AuthSettings | None = None) -> Principal:
    """Bind as the user and read their group membership."""
    settings = settings or get_settings().auth
    if not password:
        raise AuthError("password required")
    if not settings.ldap_url or not settings.ldap_user_base_dn:
        raise AuthError("LDAP is not configured")

    try:
        from ldap3 import ALL, SUBTREE, Connection, Server
        from ldap3.core.exceptions import LDAPException
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AuthError(
            "ldap3 is not installed; install it or switch to ELP_AUTH__MODE=oidc"
        ) from exc

    server = Server(settings.ldap_url, use_ssl=settings.ldap_use_ssl, get_info=ALL)

    # Step 1: find the user's DN using the service account.
    search_filter = settings.ldap_user_filter.replace("{username}", _escape(username))
    try:
        with Connection(
            server,
            user=settings.ldap_bind_dn or None,
            password=settings.ldap_bind_password or None,
            auto_bind=True,
        ) as conn:
            conn.search(
                search_base=settings.ldap_user_base_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=[
                    settings.ldap_group_attribute,
                    "displayName",
                    "mail",
                    "sAMAccountName",
                ],
            )
            if not conn.entries:
                raise AuthError("user not found in directory")
            entry = conn.entries[0]
            user_dn = entry.entry_dn
            groups = [
                _group_name(dn)
                for dn in entry[settings.ldap_group_attribute].values
            ]
            display_name = str(entry["displayName"].value or "") if "displayName" in entry else ""
            email = str(entry["mail"].value or "") if "mail" in entry else ""

        # Step 2: bind as the user to verify the password.
        with Connection(server, user=user_dn, password=password, auto_bind=True):
            pass
    except LDAPException as exc:
        log.info("LDAP authentication failed for %s: %s", username, exc)
        raise AuthError("invalid username or password") from exc

    if settings.allowed_groups:
        permitted = {g.lower() for g in settings.allowed_groups}
        if not any(g.lower() in permitted for g in groups):
            raise AuthError(
                "your Active Directory groups do not grant access to this platform"
            )

    from .oidc import map_roles

    roles = map_roles(groups, settings)
    return Principal(
        subject=username,
        display_name=display_name,
        email=email,
        kind="user",
        groups=groups,
        roles=roles,
        scopes=scopes_for_roles(roles),
        credential_id=user_dn,
        issuer="ldap",
    )


def _escape(value: str) -> str:
    """Escape LDAP filter metacharacters (RFC 4515)."""
    out = []
    for ch in value:
        if ch in "\\*()\0":
            out.append(f"\\{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)
