"""Authentication and authorisation for the Enterprise LLM Platform."""

from .deps import current_principal, require_admin, require_scope
from .oidc import AuthError
from .principal import Principal, Scope, scopes_for_roles

__all__ = [
    "AuthError",
    "Principal",
    "Scope",
    "current_principal",
    "require_admin",
    "require_scope",
    "scopes_for_roles",
]
