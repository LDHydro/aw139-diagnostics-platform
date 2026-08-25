"""
Executing a report end to end.

Query NAMIS, narrate the result, render the artifacts, record the run.

Every failure mode produces a recorded run rather than an exception that
vanishes into a log: a scheduled report that silently stopped working is the
worst outcome here, because nobody notices until someone asks why the
numbers stopped arriving.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import ReportSettings, get_settings
from ..llm.client import InferenceError
from ..models import ReportDefinition, ReportRun, RunStatus
from .authoring import narrate
from .cron import next_run
from .datasource import DataSourceError, QueryResult, get_source
from .render import write_artifacts
from .service import ReportError, is_approved
from .sqlguard import UnsafeQuery, validate

log = logging.getLogger(__name__)


async def execute(
    session: AsyncSession,
    definition: ReportDefinition,
    *,
    actor: str,
    trigger: str = "manual",
    parameters: dict | None = None,
    settings: ReportSettings | None = None,
) -> ReportRun:
    """
    Run a report and record the outcome.

    Returns the :class:`ReportRun` in every case, including failure. Callers
    should check ``run.status`` rather than relying on exceptions.
    """
    settings = settings or get_settings().reports
    started = time.monotonic()

    run = ReportRun(
        definition_id=definition.id,
        status=RunStatus.RUNNING.value,
        trigger=trigger,
        actor=actor,
        parameters_used={**(definition.parameters or {}), **(parameters or {})},
    )
    session.add(run)
    await session.flush()

    def _fail(status: str, message: str, warnings: list[str] | None = None) -> ReportRun:
        run.status = status
        run.error = message
        run.warnings = warnings or []
        run.finished_at = datetime.now(UTC)
        run.duration_ms = (time.monotonic() - started) * 1000
        log.error(
            "report '%s' run %s: %s", definition.name, status, message
        )
        return run

    # A scheduled run must be executing the approved query, not whatever is
    # currently stored. Checked here as well as at schedule time, because
    # the two are separated by however long the schedule has been running.
    if (
        trigger == "scheduled"
        and settings.require_approval_for_schedule
        and not is_approved(definition)
    ):
        return _fail(
            RunStatus.BLOCKED.value,
            "the stored query does not match its approval, so the scheduled "
            "run was refused. Re-approve the report to resume it.",
        )

    if definition.query_language == "sql":
        try:
            guard = validate(definition.query, settings)
        except UnsafeQuery as exc:
            return _fail(
                RunStatus.BLOCKED.value,
                f"the stored query no longer passes validation: {exc}",
            )
        query_to_run = guard.sql
        warnings = list(guard.notes)
    else:
        query_to_run = definition.query
        warnings = []

    run.query_executed = query_to_run

    # --- Query -------------------------------------------------------
    try:
        source = get_source(settings)
        result: QueryResult = await source.run(query_to_run, run.parameters_used)
    except DataSourceError as exc:
        return _fail(RunStatus.FAILED.value, str(exc), warnings)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            RunStatus.FAILED.value, f"{type(exc).__name__}: {exc}", warnings
        )

    warnings.extend(result.warnings)
    run.row_count = result.row_count
    run.truncated = result.truncated

    # --- Narration ----------------------------------------------------
    narrative_text = ""
    if definition.narrative:
        try:
            narrative_text = await narrate(
                definition.request_text, result, settings=settings
            )
        except InferenceError as exc:
            # The data is the report; the prose is a convenience. Losing the
            # model must not lose the numbers.
            warnings.append(
                f"the summary could not be generated (the local model is not "
                f"reachable): {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"the summary could not be generated: {exc}")
    run.narrative = narrative_text

    # --- Artifacts -----------------------------------------------------
    try:
        artifacts, render_warnings = await write_artifacts(
            result,
            run_id=run.id,
            title=definition.name,
            formats=list(definition.output_formats or ["markdown"]),
            request_text=definition.request_text,
            narrative=narrative_text,
            settings=settings,
        )
        warnings.extend(render_warnings)
        run.artifacts = [a.to_dict() for a in artifacts]
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"artifacts could not be written: {exc}")
        run.artifacts = []

    run.status = RunStatus.SUCCEEDED.value
    run.warnings = warnings
    run.finished_at = datetime.now(UTC)
    run.duration_ms = (time.monotonic() - started) * 1000

    definition.last_run_at = run.finished_at
    if definition.schedule_enabled and definition.schedule_cron:
        definition.next_run_at = next_run(
            definition.schedule_cron,
            after=run.finished_at,
            timezone_name=definition.schedule_timezone,
        )

    await session.flush()
    log.info(
        "report '%s' produced %d row(s) in %.0f ms (%s)",
        definition.name, run.row_count, run.duration_ms, trigger,
    )
    return run


async def run_due(
    session: AsyncSession, *, now: datetime | None = None
) -> list[ReportRun]:
    """
    Execute every scheduled report that is due.

    Reports run one after another rather than concurrently: they hit a
    production system, and a burst of simultaneous queries at 03:00 is how a
    reporting job becomes an outage.
    """
    from .service import due_definitions

    now = now or datetime.now(UTC)
    definitions = await due_definitions(session, now=now)
    if not definitions:
        return []

    log.info("%d scheduled report(s) due", len(definitions))
    runs: list[ReportRun] = []
    for definition in definitions:
        try:
            run = await execute(
                session, definition, actor="scheduler", trigger="scheduled"
            )
            runs.append(run)
        except ReportError as exc:
            log.error("report '%s' could not run: %s", definition.name, exc)
        # Commit after each report so one failure does not discard the
        # successful runs that preceded it.
        await session.commit()
    return runs
