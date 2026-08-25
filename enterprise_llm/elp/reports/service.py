"""
Report definitions: saving, approving and scheduling.

The approval model is the point of this module. A report that runs
unattended is a standing instruction to a production database, so:

* The query is generated **once**, at draft time, and stored.
* A person with ``reports:approve`` reviews and approves it.
* The approval is bound to a fingerprint of the query text. Edit the query
  and the fingerprint stops matching, the report drops back to draft, and
  the schedule stops firing until someone approves it again.

That last rule is what stops "just a small tweak" from putting an unreviewed
query on a nightly timer.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.principal import Principal, Scope
from ..config import ReportSettings, get_settings
from ..models import ReportDefinition, ReportRun, ReportStatus
from .cron import CronError, next_run, parse
from .sqlguard import UnsafeQuery, validate

log = logging.getLogger(__name__)


class ReportError(RuntimeError):
    """A report operation could not be completed."""


def fingerprint(query: str) -> str:
    """Stable hash of a query, ignoring incidental whitespace."""
    normalised = " ".join(query.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def can_access(principal: Principal, definition: ReportDefinition) -> bool:
    """Whether this caller may run the report and read its results."""
    if principal.is_admin:
        return True
    if not definition.allowed_groups:
        return True
    permitted = {g.lower() for g in definition.allowed_groups}
    return bool({g.lower() for g in principal.groups} & permitted)


def require_access(principal: Principal, definition: ReportDefinition) -> None:
    if not can_access(principal, definition):
        raise ReportError(
            f"'{definition.name}' is restricted to "
            f"{', '.join(definition.allowed_groups)}, which your Active "
            "Directory groups do not cover"
        )


def is_approved(definition: ReportDefinition) -> bool:
    """
    Whether the stored query is the one that was approved.

    Compares the fingerprint rather than trusting the status column, so an
    edit that bypassed the normal path is still caught.
    """
    if definition.status != ReportStatus.APPROVED.value:
        return False
    return bool(
        definition.approved_query_hash
        and definition.approved_query_hash == fingerprint(definition.query)
    )


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------

async def get_definition(session: AsyncSession, identifier: str) -> ReportDefinition:
    row = (
        await session.execute(
            select(ReportDefinition).where(ReportDefinition.id == identifier)
        )
    ).scalar_one_or_none()
    if row is None:
        row = (
            await session.execute(
                select(ReportDefinition).where(ReportDefinition.name == identifier)
            )
        ).scalar_one_or_none()
    if row is None:
        raise ReportError(f"no report named '{identifier}'")
    return row


async def list_definitions(
    session: AsyncSession, principal: Principal, *, include_disabled: bool = False
) -> list[ReportDefinition]:
    query = select(ReportDefinition)
    if not include_disabled:
        query = query.where(ReportDefinition.status != ReportStatus.DISABLED.value)
    rows = (
        await session.execute(query.order_by(ReportDefinition.name))
    ).scalars().all()
    return [row for row in rows if can_access(principal, row)]


async def create_definition(
    session: AsyncSession,
    *,
    name: str,
    request_text: str,
    query: str,
    principal: Principal,
    description: str = "",
    source: str = "namis",
    query_language: str = "sql",
    parameters: dict | None = None,
    output_formats: list[str] | None = None,
    narrative: bool = True,
    allowed_groups: list[str] | None = None,
    settings: ReportSettings | None = None,
) -> ReportDefinition:
    settings = settings or get_settings().reports

    existing = (
        await session.execute(
            select(ReportDefinition).where(ReportDefinition.name == name)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ReportError(f"a report named '{name}' already exists")

    if query_language == "sql":
        try:
            guard = validate(query, settings)
            query = guard.sql
        except UnsafeQuery as exc:
            raise ReportError(f"the query was rejected: {exc}") from exc

    definition = ReportDefinition(
        name=name,
        description=description,
        request_text=request_text,
        source=source,
        query_language=query_language,
        query=query,
        parameters=parameters or {},
        output_formats=output_formats or ["markdown", "csv"],
        narrative=narrative,
        allowed_groups=allowed_groups or [],
        owner=principal.subject,
        created_by=principal.subject,
        status=ReportStatus.DRAFT.value,
    )
    session.add(definition)
    await session.flush()
    log.info("report '%s' created by %s", name, principal.subject)
    return definition


async def update_definition(
    session: AsyncSession,
    definition: ReportDefinition,
    principal: Principal,
    *,
    settings: ReportSettings | None = None,
    **fields,
) -> tuple[ReportDefinition, list[str]]:
    """
    Update a definition.

    Changing the query revokes approval, which also stops any schedule. That
    is deliberate: an edited query has not been reviewed, and a schedule is
    exactly where an unreviewed query would do the most damage.
    """
    settings = settings or get_settings().reports
    notes: list[str] = []

    new_query = fields.pop("query", None)
    if new_query is not None and new_query.strip() != definition.query.strip():
        if definition.query_language == "sql":
            try:
                guard = validate(new_query, settings)
                new_query = guard.sql
            except UnsafeQuery as exc:
                raise ReportError(f"the query was rejected: {exc}") from exc
        definition.query = new_query
        if definition.status == ReportStatus.APPROVED.value:
            definition.status = ReportStatus.DRAFT.value
            definition.approved_query_hash = ""
            definition.approved_by = ""
            definition.approved_at = None
            notes.append(
                "the query changed, so approval was revoked and the report "
                "returned to draft"
            )
            if definition.schedule_enabled:
                definition.schedule_enabled = False
                definition.next_run_at = None
                notes.append(
                    "the schedule was disabled; re-approve the report to resume it"
                )

    for key, value in fields.items():
        if value is not None and hasattr(definition, key):
            setattr(definition, key, value)

    await session.flush()
    return definition, notes


async def approve_definition(
    session: AsyncSession,
    definition: ReportDefinition,
    principal: Principal,
    *,
    settings: ReportSettings | None = None,
) -> ReportDefinition:
    """Approve the stored query for unattended execution."""
    settings = settings or get_settings().reports

    if not principal.has(Scope.REPORTS_APPROVE):
        raise ReportError(
            f"approving a report for unattended execution requires the "
            f"'{Scope.REPORTS_APPROVE}' permission"
        )
    if not definition.query.strip():
        raise ReportError("this report has no query to approve")

    # Re-validate at approval time: the allowlist may have tightened since
    # the query was drafted.
    if definition.query_language == "sql":
        try:
            validate(definition.query, settings)
        except UnsafeQuery as exc:
            raise ReportError(
                f"the stored query no longer passes validation and cannot be "
                f"approved: {exc}"
            ) from exc

    definition.status = ReportStatus.APPROVED.value
    definition.approved_query_hash = fingerprint(definition.query)
    definition.approved_by = principal.subject
    definition.approved_at = datetime.now(UTC)
    await session.flush()

    log.info("report '%s' approved by %s", definition.name, principal.subject)
    return definition


async def set_schedule(
    session: AsyncSession,
    definition: ReportDefinition,
    *,
    cron: str,
    timezone_name: str = "UTC",
    enabled: bool = True,
    settings: ReportSettings | None = None,
) -> ReportDefinition:
    settings = settings or get_settings().reports

    if enabled:
        if settings.require_approval_for_schedule and not is_approved(definition):
            raise ReportError(
                f"'{definition.name}' must be approved before it can run on a "
                "schedule. An unreviewed query running unattended against "
                "NAMIS is exactly what approval exists to prevent."
            )
        try:
            parse(cron)
        except CronError as exc:
            raise ReportError(f"the schedule is not valid: {exc}") from exc

    definition.schedule_cron = cron
    definition.schedule_timezone = timezone_name
    definition.schedule_enabled = enabled
    definition.next_run_at = (
        next_run(cron, after=datetime.now(UTC), timezone_name=timezone_name)
        if enabled
        else None
    )
    await session.flush()

    log.info(
        "report '%s' schedule set to '%s' (%s), enabled=%s",
        definition.name, cron, timezone_name, enabled,
    )
    return definition


async def due_definitions(
    session: AsyncSession, *, now: datetime | None = None
) -> list[ReportDefinition]:
    """Scheduled reports whose next firing has arrived."""
    from .cron import is_due

    now = now or datetime.now(UTC)
    rows = (
        await session.execute(
            select(ReportDefinition).where(
                ReportDefinition.schedule_enabled.is_(True),
                ReportDefinition.status == ReportStatus.APPROVED.value,
            )
        )
    ).scalars().all()

    due: list[ReportDefinition] = []
    for row in rows:
        if not row.schedule_cron:
            continue
        if not is_approved(row):
            log.warning(
                "report '%s' is scheduled but its query no longer matches its "
                "approval; skipping",
                row.name,
            )
            continue
        try:
            if is_due(
                row.schedule_cron,
                last_run=row.last_run_at,
                now=now,
                timezone_name=row.schedule_timezone,
            ):
                due.append(row)
        except CronError as exc:
            log.error("report '%s' has an invalid schedule: %s", row.name, exc)
    return due


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------

async def list_runs(
    session: AsyncSession, definition_id: str, *, limit: int = 25
) -> list[ReportRun]:
    return list(
        (
            await session.execute(
                select(ReportRun)
                .where(ReportRun.definition_id == definition_id)
                .order_by(ReportRun.started_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )


async def get_run(session: AsyncSession, run_id: str) -> ReportRun:
    row = (
        await session.execute(select(ReportRun).where(ReportRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise ReportError(f"no report run with id '{run_id}'")
    return row


async def prune_runs(
    session: AsyncSession, *, retention_days: int | None = None
) -> int:
    """Delete run history past the retention window. Artifacts go with it."""
    import shutil
    from datetime import timedelta
    from pathlib import Path

    settings = get_settings().reports
    retention_days = retention_days or settings.run_retention_days
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    rows = (
        await session.execute(select(ReportRun).where(ReportRun.started_at < cutoff))
    ).scalars().all()

    for row in rows:
        directory = Path(settings.artifact_dir) / row.id
        shutil.rmtree(directory, ignore_errors=True)
        await session.delete(row)

    if rows:
        log.info("pruned %d report run(s) older than %d days", len(rows), retention_days)
    return len(rows)
