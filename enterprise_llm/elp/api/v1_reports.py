"""
Operational reporting.

Operations asks for a report in plain language; the platform authors a
read-only query against NAMIS, runs it, and produces the deliverable. Saved
reports can run on a schedule.

The workflow is deliberately two-step for anything scheduled:

    draft  ->  review  ->  save  ->  approve  ->  schedule

Ad-hoc runs (``/ask`` and ``/run``) skip approval, because a person is
sitting there watching the result and the query is still read-only, guarded
and row-limited. Unattended runs do not skip it, because nobody is watching.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..config import get_settings
from ..db import get_session
from ..llm.client import InferenceError
from ..models import ReportDefinition, ReportRun
from ..reports import (
    DataSourceError,
    ReportError,
    approve_definition,
    create_definition,
    describe,
    draft_query,
    execute,
    get_definition,
    get_run,
    get_source,
    is_approved,
    list_definitions,
    list_runs,
    require_access,
    set_schedule,
    update_definition,
)
from .schemas import (
    ReportAskRequest,
    ReportCreateRequest,
    ReportDraftRequest,
    ReportDraftResponse,
    ReportImportResult,
    ReportRunSummary,
    ReportScheduleRequest,
    ReportSummary,
    ReportUpdateRequest,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/reports", tags=["reports"])


def _report_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _summary(definition: ReportDefinition) -> ReportSummary:
    schedule_description = ""
    if definition.schedule_cron:
        try:
            schedule_description = describe(definition.schedule_cron)
        except Exception:  # noqa: BLE001 - a bad cron must not break listing
            schedule_description = "invalid schedule"

    return ReportSummary(
        id=definition.id,
        name=definition.name,
        description=definition.description,
        request_text=definition.request_text,
        status=definition.status,
        query_language=definition.query_language,
        output_formats=list(definition.output_formats or []),
        narrative=definition.narrative,
        allowed_groups=list(definition.allowed_groups or []),
        owner=definition.owner,
        approved_by=definition.approved_by,
        approved_at=definition.approved_at,
        approval_current=is_approved(definition),
        schedule_cron=definition.schedule_cron,
        schedule_timezone=definition.schedule_timezone,
        schedule_enabled=definition.schedule_enabled,
        schedule_description=schedule_description,
        last_run_at=definition.last_run_at,
        next_run_at=definition.next_run_at,
    )


def _run_summary(run: ReportRun) -> ReportRunSummary:
    return ReportRunSummary(
        id=run.id,
        definition_id=run.definition_id,
        status=run.status,
        trigger=run.trigger,
        actor=run.actor,
        started_at=run.started_at,
        finished_at=run.finished_at,
        row_count=run.row_count,
        truncated=run.truncated,
        duration_ms=run.duration_ms,
        narrative=run.narrative,
        artifacts=list(run.artifacts or []),
        error=run.error,
        warnings=list(run.warnings or []),
    )


async def _plan_report(
    request_text: str, mode: str
) -> ReportDraftResponse:
    """
    Plan a report, preferring a definition over free-form SQL.

    Structured first because the model is an untrusted suggester and the
    compiler is the gatekeeper: a hallucinated column becomes a rejection
    rather than a query. Free-form SQL is the fallback for requests the
    definition model cannot express - window functions, unions, anything
    needing SQL the compiler does not emit.
    """
    from ..reports.authoring import draft_structured
    from ..reports.catalog import get_catalog

    catalog = get_catalog()
    fell_back = False
    structured_rejection = ""

    if mode in {"auto", "structured"} and catalog is not None and len(catalog):
        planned = await draft_structured(request_text, catalog)
        if planned.valid and planned.report is not None:
            return ReportDraftResponse(
                request_text=request_text,
                query_language="structured",
                query=planned.sql,
                definition=planned.report.to_dict(),
                explanation=planned.explanation,
                assumptions=planned.assumptions,
                tables=planned.tables,
                tables_offered=planned.tables_offered,
                columns=planned.columns,
                valid=True,
                repaired=planned.repaired,
                warnings=planned.warnings,
            )
        structured_rejection = planned.rejection
        if mode == "structured":
            return ReportDraftResponse(
                request_text=request_text,
                query_language="structured",
                query="",
                definition=planned.report.to_dict() if planned.report else None,
                explanation=planned.explanation,
                assumptions=planned.assumptions,
                tables_offered=planned.tables_offered,
                valid=False,
                rejection=planned.rejection,
                repaired=planned.repaired,
            )
        fell_back = True
        log.info(
            "structured planning did not compile (%s); falling back to SQL",
            structured_rejection,
        )
    elif mode == "structured":
        return ReportDraftResponse(
            request_text=request_text,
            query_language="structured",
            query="",
            valid=False,
            rejection=(
                "structured planning needs the NAMIS catalogue; set "
                "ELP_REPORTS__CATALOG_PATH"
            ),
        )

    tables = await _schema_tables()
    result = await draft_query(request_text, tables)
    warnings = list(result.warnings)
    if fell_back and structured_rejection:
        warnings.append(
            "a structured definition could not be produced "
            f"({structured_rejection}); this is free-form SQL and needs a "
            "closer read before approval"
        )

    return ReportDraftResponse(
        request_text=request_text,
        query_language="sql",
        query=result.query,
        explanation=result.explanation,
        assumptions=result.assumptions,
        tables=result.tables,
        tables_offered=result.tables_offered,
        valid=result.valid,
        rejection=result.rejection,
        repaired=result.repaired,
        fell_back_to_sql=fell_back,
        warnings=warnings,
    )


async def _schema_tables():
    try:
        return await get_source().describe_schema()
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ----------------------------------------------------------------------
# NAMIS schema
# ----------------------------------------------------------------------

@router.get("/schema")
async def namis_schema(
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
) -> dict:
    """What the reporting account can see in NAMIS."""
    tables = await _schema_tables()
    return {
        "table_count": len(tables),
        "tables": [
            {
                "name": table.qualified,
                "comment": table.comment,
                "columns": [
                    {"name": c.name, "type": c.type, "nullable": c.nullable}
                    for c in table.columns
                ],
            }
            for table in tables
        ],
    }


# ----------------------------------------------------------------------
# Drafting and ad-hoc
# ----------------------------------------------------------------------

@router.post("/draft", response_model=ReportDraftResponse)
async def draft(
    payload: ReportDraftRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportDraftResponse:
    """
    Turn a plain-language request into a query, without running or saving it.

    Review the query and the assumptions before saving. The assumptions are
    where the model tells you what it had to guess, and they are usually the
    difference between the report you wanted and the one you asked for.
    """
    try:
        response = await _plan_report(payload.request_text, payload.mode)
    except InferenceError as exc:
        raise HTTPException(
            status_code=503, detail=f"the local model is not reachable: {exc}"
        ) from exc
    except ValueError as exc:
        raise _report_error(exc) from exc

    await audit.record(
        session,
        principal,
        "reports.draft",
        resource=payload.request_text[:300],
        outcome="ok" if response.valid else "rejected",
        request=request,
        detail={
            "query_language": response.query_language,
            "tables": response.tables,
            "valid": response.valid,
            "rejection": response.rejection,
            "fell_back_to_sql": response.fell_back_to_sql,
        },
    )
    return response


@router.post("/ask")
async def ask(
    payload: ReportAskRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Ad-hoc report: draft, run and return in one call, without saving.

    Runs an unapproved query, which is acceptable here and not for a
    schedule: the caller is authenticated, the query is read-only and
    row-limited, and a person is looking at the result.
    """
    import json as _json

    try:
        drafted = await _plan_report(payload.request_text, payload.mode)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not drafted.valid:
        await audit.record(
            session, principal, "reports.ask", resource=payload.request_text[:300],
            outcome="rejected", request=request, detail={"rejection": drafted.rejection},
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"a report could not be produced for that request: "
                f"{drafted.rejection}"
            ),
        )

    # A transient definition so the runner path is identical to a saved one.
    definition = ReportDefinition(
        name=f"Ad-hoc: {payload.request_text[:80]}",
        request_text=payload.request_text,
        query_language=drafted.query_language,
        query=(
            _json.dumps(drafted.definition)
            if drafted.query_language == "structured"
            else drafted.query
        ),
        output_formats=payload.output_formats,
        narrative=payload.narrative,
        owner=principal.subject,
        created_by=principal.subject,
    )
    session.add(definition)
    await session.flush()

    run = await execute(session, definition, actor=principal.subject, trigger="manual")

    await audit.record(
        session,
        principal,
        "reports.ask",
        resource=payload.request_text[:300],
        outcome=run.status,
        request=request,
        detail={
            "rows": run.row_count,
            "tables": drafted.tables,
            "query_language": drafted.query_language,
        },
    )

    return {
        "request_text": payload.request_text,
        "query_language": drafted.query_language,
        "query": drafted.query,
        "definition": drafted.definition,
        "explanation": drafted.explanation,
        "assumptions": drafted.assumptions,
        "warnings": drafted.warnings,
        "run": _run_summary(run).model_dump(mode="json"),
        "definition_id": definition.id,
        "note": (
            "This was an ad-hoc run and the query was not approved for "
            "unattended execution. Save and approve it to put it on a schedule."
        ),
    }


@router.post("/import", response_model=list[ReportImportResult])
async def import_saved_reports(
    request: Request,
    files: list[UploadFile] = File(..., description="Saved-report .json files"),
    save: bool = False,
    allowed_groups: str = "",
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> list[ReportImportResult]:
    """
    Import reports saved by the NAMIS report generator.

    Copy `%LOCALAPPDATA%\\NamisReports\\saved-reports\\*.json` and upload them.
    Each becomes a structured definition, which is compiled against the
    catalogue immediately — so a report referring to a table or column the
    catalogue does not know is reported now rather than the first time it
    runs on a schedule.

    Runs read-only unless `save=true`. Imported reports are created as
    drafts and still need approval before they can be scheduled.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from ..reports.catalog import get_catalog
    from ..reports.import_saved import import_directory

    staging = _Path(tempfile.mkdtemp(prefix="elp-saved-reports-"))
    try:
        for upload in files:
            name = _Path(upload.filename or "report.json").name
            if not name.lower().endswith(".json"):
                continue
            (staging / name).write_bytes(await upload.read())

        catalog = get_catalog()
        imported = import_directory(
            staging, catalog, max_rows=get_settings().reports.max_rows
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    groups = [g.strip() for g in allowed_groups.split(",") if g.strip()]
    results: list[ReportImportResult] = []

    for entry in imported:
        payload = ReportImportResult(**entry.to_dict())
        if save:
            if not entry.compiles:
                payload.save_error = (
                    "not saved: the definition does not compile against the "
                    "catalogue"
                )
            else:
                try:
                    await create_definition(
                        session,
                        name=entry.name,
                        request_text=entry.description
                        or f"Imported from {entry.source_file}",
                        query=_json.dumps(entry.report.to_dict()),
                        principal=principal,
                        description=entry.description,
                        query_language="structured",
                        output_formats=["xlsx", "csv", "markdown"],
                        allowed_groups=groups,
                    )
                    payload.saved = True
                except ReportError as exc:
                    payload.save_error = str(exc)
        results.append(payload)

    await audit.record(
        session,
        principal,
        "reports.import",
        resource=f"{len(files)} file(s)",
        request=request,
        detail={
            "imported": len(results),
            "compiled": sum(1 for r in results if r.compiles),
            "saved": sum(1 for r in results if r.saved),
        },
    )
    return results


# ----------------------------------------------------------------------
# Definitions
# ----------------------------------------------------------------------

@router.get("", response_model=list[ReportSummary])
async def list_reports(
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
    include_disabled: bool = False,
) -> list[ReportSummary]:
    rows = await list_definitions(session, principal, include_disabled=include_disabled)
    return [_summary(row) for row in rows]


@router.post("", response_model=ReportSummary, status_code=201)
async def create_report(
    payload: ReportCreateRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportSummary:
    """Save a report. It is created as a draft and cannot be scheduled yet."""
    try:
        definition = await create_definition(
            session,
            name=payload.name,
            request_text=payload.request_text,
            query=payload.query,
            principal=principal,
            description=payload.description,
            source=payload.source,
            query_language=payload.query_language,
            parameters=payload.parameters,
            output_formats=payload.output_formats,
            narrative=payload.narrative,
            allowed_groups=payload.allowed_groups,
        )
    except ReportError as exc:
        raise _report_error(exc) from exc

    await audit.record(
        session, principal, "reports.create", resource=payload.name,
        request=request, detail={"allowed_groups": payload.allowed_groups},
    )
    return _summary(definition)


@router.get("/{identifier}", response_model=ReportSummary)
async def get_report(
    identifier: str,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportSummary:
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _summary(definition)


@router.patch("/{identifier}", response_model=ReportSummary)
async def patch_report(
    identifier: str,
    payload: ReportUpdateRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportSummary:
    """
    Update a report.

    Changing the query revokes its approval and disables any schedule.
    """
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
        definition, notes = await update_definition(
            session, definition, principal, **payload.model_dump(exclude_unset=True)
        )
    except ReportError as exc:
        raise _report_error(exc) from exc

    await audit.record(
        session, principal, "reports.update", resource=definition.name,
        request=request, detail={"notes": notes},
    )
    summary = _summary(definition)
    if notes:
        log.info("report '%s' updated: %s", definition.name, "; ".join(notes))
    return summary


@router.post("/{identifier}/approve", response_model=ReportSummary)
async def approve_report(
    identifier: str,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS_APPROVE)),
    session: AsyncSession = Depends(get_session),
) -> ReportSummary:
    """
    Approve the stored query for unattended execution.

    Read the query before approving it. Approval binds to the exact text; a
    later edit revokes it automatically.
    """
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
        definition = await approve_definition(session, definition, principal)
    except ReportError as exc:
        raise _report_error(exc) from exc

    await audit.record(
        session,
        principal,
        "reports.approve",
        resource=definition.name,
        request=request,
        detail={
            "query": definition.query[:2000],
            "query_hash": definition.approved_query_hash,
        },
    )
    return _summary(definition)


@router.put("/{identifier}/schedule", response_model=ReportSummary)
async def schedule_report(
    identifier: str,
    payload: ReportScheduleRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS_APPROVE)),
    session: AsyncSession = Depends(get_session),
) -> ReportSummary:
    """Put an approved report on a schedule."""
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
        definition = await set_schedule(
            session,
            definition,
            cron=payload.cron,
            timezone_name=payload.timezone,
            enabled=payload.enabled,
        )
    except ReportError as exc:
        raise _report_error(exc) from exc

    await audit.record(
        session,
        principal,
        "reports.schedule",
        resource=definition.name,
        request=request,
        detail={
            "cron": payload.cron,
            "timezone": payload.timezone,
            "enabled": payload.enabled,
            "next_run_at": definition.next_run_at.isoformat()
            if definition.next_run_at
            else None,
        },
    )
    return _summary(definition)


@router.delete("/{identifier}")
async def delete_report(
    identifier: str,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    name = definition.name
    await session.delete(definition)
    await audit.record(
        session, principal, "reports.delete", resource=name, request=request
    )
    return {"name": name, "deleted": True}


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------

@router.post("/{identifier}/run", response_model=ReportRunSummary)
async def run_report(
    identifier: str,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportRunSummary:
    """Run a saved report now."""
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run = await execute(session, definition, actor=principal.subject, trigger="manual")

    await audit.record(
        session,
        principal,
        "reports.run",
        resource=definition.name,
        outcome=run.status,
        latency_ms=run.duration_ms,
        request=request,
        detail={"rows": run.row_count, "run_id": run.id, "error": run.error},
    )
    return _run_summary(run)


@router.get("/{identifier}/runs", response_model=list[ReportRunSummary])
async def report_runs(
    identifier: str,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
    limit: int = 25,
) -> list[ReportRunSummary]:
    try:
        definition = await get_definition(session, identifier)
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    runs = await list_runs(session, definition.id, limit=min(limit, 200))
    return [_run_summary(r) for r in runs]


@router.get("/runs/{run_id}", response_model=ReportRunSummary)
async def single_run(
    run_id: str,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> ReportRunSummary:
    try:
        run = await get_run(session, run_id)
        definition = await get_definition(session, run.definition_id)
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _run_summary(run)


@router.get("/runs/{run_id}/artifact/{output_format}")
async def download_artifact(
    run_id: str,
    output_format: str,
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Download one rendered artifact from a run."""
    try:
        run = await get_run(session, run_id)
        definition = await get_definition(session, run.definition_id)
        # The ACL is on the definition, so it must be checked here too:
        # otherwise a run id would be enough to read a restricted report.
        require_access(principal, definition)
    except ReportError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    entry = next(
        (a for a in (run.artifacts or []) if a.get("format") == output_format), None
    )
    if entry is None:
        available = ", ".join(a.get("format", "?") for a in (run.artifacts or []))
        raise HTTPException(
            status_code=404,
            detail=(
                f"this run has no '{output_format}' artifact"
                + (f"; available: {available}" if available else "")
            ),
        )

    path = Path(entry["path"])
    if not path.is_file():
        raise HTTPException(
            status_code=410,
            detail="the artifact file is no longer on disk; re-run the report",
        )

    media_types = {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "html": "text/html",
        "json": "application/json",
        "markdown": "text/markdown",
    }
    return FileResponse(
        str(path),
        media_type=media_types.get(output_format, "application/octet-stream"),
        filename=path.name,
    )


@router.get("/status/scheduled")
async def scheduled_status(
    principal: Principal = Depends(require_scope(Scope.REPORTS)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Every scheduled report and when it next fires."""
    rows = await list_definitions(session, principal)
    scheduled = [r for r in rows if r.schedule_enabled]
    now = datetime.now(UTC)

    return {
        "count": len(scheduled),
        "reports": [
            {
                "name": r.name,
                "cron": r.schedule_cron,
                "timezone": r.schedule_timezone,
                "description": describe(r.schedule_cron) if r.schedule_cron else "",
                "approval_current": is_approved(r),
                "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
                "next_run_at": r.next_run_at.isoformat() if r.next_run_at else None,
                "overdue": bool(r.next_run_at and r.next_run_at < now),
            }
            for r in scheduled
        ],
    }
