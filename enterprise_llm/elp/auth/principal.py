"""Caller identity, roles and scopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


class Scope:
    """Fine-grained permissions granted to a caller."""

    ASK = "ask"                       # plain-language Q&A over the documents
    CHAT = "chat"                     # raw / OpenAI-compatible completions
    DOCS_READ = "docs:read"
    DOCS_WRITE = "docs:write"         # upload, re-index, retire documents
    MAINT_READ = "maint:read"
    MAINT_WRITE = "maint:write"       # record utilisation, plan, cancel
    MAINT_APPROVE = "maint:approve"   # approve deferrals past a due point
    LATEX = "latex"
    DEV = "dev"                       # code assistance
    FEDERATION = "federation:query"   # consult other internal AI systems
    REPORTS = "reports"               # request and run reports
    REPORTS_APPROVE = "reports:approve"  # approve a query for unattended runs
    ADMIN = "admin"

    ALL = (
        ASK, CHAT, DOCS_READ, DOCS_WRITE, MAINT_READ, MAINT_WRITE,
        MAINT_APPROVE, LATEX, DEV, FEDERATION, REPORTS, REPORTS_APPROVE, ADMIN,
    )


# Roles are what you map AD groups onto; scopes are what the code checks.
ROLE_SCOPES: dict[str, tuple[str, ...]] = {
    "admin": tuple(Scope.ALL),
    "engineer": (
        Scope.ASK, Scope.CHAT, Scope.DOCS_READ, Scope.DOCS_WRITE,
        Scope.MAINT_READ, Scope.MAINT_WRITE, Scope.LATEX, Scope.DEV,
        Scope.FEDERATION, Scope.REPORTS, Scope.REPORTS_APPROVE,
    ),
    "planner": (
        Scope.ASK, Scope.CHAT, Scope.DOCS_READ, Scope.MAINT_READ,
        Scope.MAINT_WRITE, Scope.LATEX, Scope.FEDERATION, Scope.REPORTS,
    ),
    "maintenance_manager": (
        Scope.ASK, Scope.CHAT, Scope.DOCS_READ, Scope.MAINT_READ,
        Scope.MAINT_WRITE, Scope.MAINT_APPROVE, Scope.LATEX, Scope.FEDERATION,
        Scope.REPORTS, Scope.REPORTS_APPROVE,
    ),
    "developer": (
        Scope.ASK, Scope.CHAT, Scope.DOCS_READ, Scope.DEV, Scope.LATEX,
        Scope.FEDERATION,
    ),
    "reader": (Scope.ASK, Scope.CHAT, Scope.DOCS_READ, Scope.MAINT_READ, Scope.REPORTS),
    "service": (Scope.ASK, Scope.CHAT, Scope.DOCS_READ),
}


@dataclass(slots=True)
class Principal:
    """An authenticated caller: a person via SSO, or an in-house application."""

    subject: str
    display_name: str = ""
    email: str = ""
    kind: Literal["user", "service"] = "user"
    # AD group names or object IDs carried on the token.
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    scopes: frozenset[str] = frozenset()
    # Identifier of the credential used, for audit.
    credential_id: str = ""
    issuer: str = ""
    token_expires_at: int | None = None

    def has(self, scope: str) -> bool:
        return Scope.ADMIN in self.scopes or scope in self.scopes

    def has_any(self, *scopes: str) -> bool:
        return any(self.has(s) for s in scopes)

    @property
    def is_admin(self) -> bool:
        return Scope.ADMIN in self.scopes

    def audit_dict(self) -> dict:
        return {
            "subject": self.subject,
            "kind": self.kind,
            "roles": self.roles,
            "groups": self.groups,
            "credential_id": self.credential_id,
        }


def scopes_for_roles(roles: list[str]) -> frozenset[str]:
    granted: set[str] = set()
    for role in roles:
        granted.update(ROLE_SCOPES.get(role, ()))
    return frozenset(granted)


ANONYMOUS = Principal(
    subject="anonymous",
    display_name="Anonymous",
    kind="user",
    roles=["reader"],
    scopes=scopes_for_roles(["reader"]),
)

# Used only when ELP_AUTH__MODE=disabled (isolated development).
DEV_SUPERUSER = Principal(
    subject="dev@localhost",
    display_name="Local Development",
    kind="user",
    groups=["*"],
    roles=["admin"],
    scopes=frozenset(Scope.ALL),
    credential_id="auth-disabled",
)
