"""Operational reporting against NAMIS, driven by plain-language requests."""

from .authoring import QueryDraft, draft_query, narrate
from .cron import CronError, describe, is_due, next_run, parse
from .datasource import (
    DataSourceError,
    NamisSource,
    QueryResult,
    TableInfo,
    close_source,
    get_source,
    render_schema_catalogue,
)
from .render import SUPPORTED_FORMATS, Artifact, write_artifacts
from .runner import execute, run_due
from .service import (
    ReportError,
    approve_definition,
    can_access,
    create_definition,
    due_definitions,
    fingerprint,
    get_definition,
    get_run,
    is_approved,
    list_definitions,
    list_runs,
    prune_runs,
    require_access,
    set_schedule,
    update_definition,
)
from .sqlguard import GuardResult, UnsafeQuery, validate

__all__ = [
    "SUPPORTED_FORMATS",
    "Artifact",
    "CronError",
    "DataSourceError",
    "GuardResult",
    "NamisSource",
    "QueryDraft",
    "QueryResult",
    "ReportError",
    "TableInfo",
    "UnsafeQuery",
    "approve_definition",
    "can_access",
    "close_source",
    "create_definition",
    "describe",
    "draft_query",
    "due_definitions",
    "execute",
    "fingerprint",
    "get_definition",
    "get_run",
    "get_source",
    "is_approved",
    "is_due",
    "list_definitions",
    "list_runs",
    "narrate",
    "next_run",
    "parse",
    "prune_runs",
    "render_schema_catalogue",
    "require_access",
    "run_due",
    "set_schedule",
    "update_definition",
    "validate",
    "write_artifacts",
]
