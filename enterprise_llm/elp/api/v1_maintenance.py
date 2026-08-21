"""
Predictive maintenance scheduling API.

Forecast when each task card falls due, plan the visits, and handle what
happens when reality intervenes: a slot is cancelled, a task is deferred, an
aircraft flies more than expected.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..db import get_session
from ..llm.client import ChatMessage, InferenceError
from ..llm.router import TaskKind, get_router
from ..maintenance import events as event_ops
from ..maintenance import service as maint
from ..maintenance.schedule_io import import_schedule
from ..models import Aircraft, MaintenanceEvent
from .schemas import (
    CancelEventRequest,
    CompleteEventRequest,
    DeferRequest,
    ForecastResponse,
    PlanRequest,
    RescheduleRequest,
    UtilizationEntry,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/maintenance", tags=["maintenance"])


def _maintenance_error(exc: maint.MaintenanceError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


# ----------------------------------------------------------------------
# Fleet and forecasting
# ----------------------------------------------------------------------

@router.get("/fleet")
async def fleet(
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    include_retired: bool = False,
) -> list[dict]:
    query = select(Aircraft)
    if not include_retired:
        query = query.where(Aircraft.in_service.is_(True))
    rows = (await session.execute(query.order_by(Aircraft.tail_number))).scalars().all()
    return [
        {
            "id": r.id,
            "tail_number": r.tail_number,
            "model": r.model,
            "serial_number": r.serial_number,
            "configuration": r.configuration,
            "base_station": r.base_station,
            "in_service": r.in_service,
            "flight_hours": r.current_flight_hours,
            "cycles": r.current_cycles,
            "landings": r.current_landings,
            "counters_as_of": r.counters_as_of.isoformat() if r.counters_as_of else None,
        }
        for r in rows
    ]


@router.get("/aircraft/{tail_number}/forecast", response_model=ForecastResponse)
async def forecast(
    tail_number: str,
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    horizon_days: int | None = None,
    status: str = "",
) -> ForecastResponse:
    """When every applicable task card falls due on this aircraft."""
    try:
        state, rate, forecasts, _specs = await maint.forecast_for_aircraft(
            session,
            tail_number,
            horizon_days=horizon_days,
            status_filter=[s.strip() for s in status.split(",") if s.strip()] or None,
        )
    except maint.MaintenanceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    counts: dict[str, int] = {}
    for item in forecasts:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1

    return ForecastResponse(
        tail_number=state.tail_number,
        model=state.model,
        serial_number=state.serial_number,
        as_of=state.as_of,
        utilization={
            "daily_flight_hours": rate.daily_flight_hours,
            "daily_cycles": rate.daily_cycles,
            "daily_landings": rate.daily_landings,
            "samples": rate.samples,
            "confidence": rate.confidence,
            "source": rate.source,
            "description": rate.describe(),
        },
        forecasts=[f.to_dict() for f in forecasts],
        summary={
            "task_count": len(forecasts),
            "by_status": counts,
            "next_due": forecasts[0].to_dict() if forecasts else None,
        },
    )


@router.post("/plan")
async def plan(
    payload: PlanRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Build a maintenance plan, optionally committing it to the schedule."""
    if payload.commit and not principal.has(Scope.MAINT_WRITE):
        raise HTTPException(
            status_code=403,
            detail=f"committing a plan requires the '{Scope.MAINT_WRITE}' permission",
        )

    started = time.monotonic()
    try:
        result = await maint.plan_for_fleet(
            session,
            horizon_days=payload.horizon_days,
            tail_numbers=payload.tail_numbers or None,
        )
    except maint.MaintenanceError as exc:
        raise _maintenance_error(exc) from exc

    response = result.to_dict()

    if payload.commit:
        created = await event_ops.persist_plan(
            session, result.events, created_by=principal.subject
        )
        response["committed_event_ids"] = [e.id for e in created]

    if payload.explain:
        response["explanation"] = await _explain_plan(result)

    await audit.record(
        session,
        principal,
        "maintenance.plan",
        resource=",".join(payload.tail_numbers) or "fleet",
        latency_ms=(time.monotonic() - started) * 1000,
        request=request,
        detail={
            "events": len(result.events),
            "committed": payload.commit,
            "horizon_days": payload.horizon_days,
        },
    )
    return response


async def _explain_plan(result) -> str:
    """Narrate a plan in plain language for a scheduling meeting."""
    lines = []
    for event in result.events[:25]:
        codes = ", ".join(t.task_code for t in event.tasks)
        lines.append(
            f"- {event.tail_number}: {event.start} to {event.end} "
            f"({event.downtime_days}d, {event.total_man_hours:.0f} man-hours) - {codes}"
        )
    warnings = "\n".join(f"- {w}" for w in result.warnings[:15]) or "- none"

    prompt = (
        "Summarise this draft maintenance plan for a planning meeting. State "
        "what drives the schedule, where the pressure points are, and what "
        "decisions the planners need to make. Be concise and concrete. Do not "
        "invent tasks, dates or limits that are not listed.\n\n"
        f"PLANNED VISITS\n{chr(10).join(lines) or '- none'}\n\n"
        f"WARNINGS\n{warnings}"
    )
    client, profile = get_router().resolve(TaskKind.EXPLAIN_SCHEDULE)
    try:
        completion = await client.chat(
            [ChatMessage("user", prompt)],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
        return completion.text
    except InferenceError as exc:
        return f"(explanation unavailable: the local model is not reachable - {exc})"


# ----------------------------------------------------------------------
# Recording reality
# ----------------------------------------------------------------------

@router.post("/utilization")
async def post_utilization(
    entries: list[UtilizationEntry],
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Log daily flying.  Re-posting a day corrects it rather than adding to it."""
    if not entries:
        raise HTTPException(status_code=400, detail="no entries supplied")
    if len(entries) > 2000:
        raise HTTPException(status_code=400, detail="at most 2000 entries per request")

    recorded = 0
    problems: list[str] = []
    for entry in entries:
        try:
            await maint.record_utilization(
                session,
                entry.tail_number,
                entry.day,
                flight_hours=entry.flight_hours,
                cycles=entry.cycles,
                landings=entry.landings,
                source=entry.source,
            )
            recorded += 1
        except maint.MaintenanceError as exc:
            problems.append(f"{entry.tail_number} {entry.day}: {exc}")

    await audit.record(
        session,
        principal,
        "maintenance.utilization",
        request=request,
        detail={"recorded": recorded, "rejected": len(problems)},
    )
    return {"recorded": recorded, "problems": problems}


@router.post("/schedule/import")
async def import_maintenance_schedule(
    request: Request,
    file: UploadFile = File(..., description="CSV, TSV, JSON or XLSX task-card export"),
    default_models: str = Form(default="", description="Comma-separated aircraft models"),
    source_document_key: str = Form(default=""),
    replace_existing: bool = Form(default=False),
    dry_run: bool = Form(default=False),
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Load the standard maintenance schedule (task cards and intervals)."""
    filename = Path(file.filename or "schedule.csv").name
    staging = Path(tempfile.mkdtemp(prefix="elp-schedule-"))
    target = staging / filename
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)

        try:
            result = await import_schedule(
                session,
                target,
                default_models=[
                    m.strip() for m in default_models.split(",") if m.strip()
                ]
                or None,
                source_document_key=source_document_key,
                replace_existing=replace_existing,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    await audit.record(
        session,
        principal,
        "maintenance.schedule_import",
        resource=filename,
        request=request,
        detail=result.to_dict(),
    )
    return result.to_dict()


# ----------------------------------------------------------------------
# Events: schedule, cancel, reschedule, complete
# ----------------------------------------------------------------------

@router.get("/events")
async def list_events(
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    tail_number: str = "",
    status: str = "",
    since: date | None = None,
    until: date | None = None,
) -> list[dict]:
    query = select(MaintenanceEvent)
    if tail_number:
        aircraft = await maint.get_aircraft(session, tail_number)
        query = query.where(MaintenanceEvent.aircraft_id == aircraft.id)
    if status:
        query = query.where(MaintenanceEvent.status == status)
    if since:
        query = query.where(MaintenanceEvent.scheduled_end >= since)
    if until:
        query = query.where(MaintenanceEvent.scheduled_start <= until)

    rows = (
        await session.execute(query.order_by(MaintenanceEvent.scheduled_start))
    ).scalars().all()
    return [
        {
            "id": r.id,
            "aircraft_id": r.aircraft_id,
            "title": r.title,
            "status": r.status,
            "scheduled_start": r.scheduled_start.isoformat(),
            "scheduled_end": r.scheduled_end.isoformat(),
            "station": r.station,
            "estimated_man_hours": r.estimated_man_hours,
            "task_codes": [i.task.task_code for i in r.items if i.task],
            "cancellation_reason": r.cancellation_reason,
            "replaces_event_id": r.replaces_event_id,
            "rationale": (r.meta or {}).get("rationale", []),
            "warnings": (r.meta or {}).get("warnings", []),
        }
        for r in rows
    ]


@router.post("/events/{event_id}/cancel")
async def cancel(
    event_id: str,
    payload: CancelEventRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Cancel a visit.

    The tasks that were on it are re-forecast and replanned in the same
    transaction, so work never silently disappears from the schedule.
    """
    try:
        outcome = await event_ops.cancel_event(
            session,
            event_id,
            reason=payload.reason,
            actor=principal.subject,
            reschedule=payload.reschedule,
        )
    except maint.MaintenanceError as exc:
        raise _maintenance_error(exc) from exc

    await audit.record(
        session,
        principal,
        "maintenance.event_cancelled",
        resource=event_id,
        outcome="ok" if not outcome.at_risk else "at_risk",
        request=request,
        detail=outcome.to_dict(),
    )
    return outcome.to_dict()


@router.post("/events/{event_id}/reschedule")
async def reschedule(
    event_id: str,
    payload: RescheduleRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        event, warnings = await event_ops.reschedule_event(
            session,
            event_id,
            payload.new_start,
            actor=principal.subject,
            reason=payload.reason,
        )
    except maint.MaintenanceError as exc:
        raise _maintenance_error(exc) from exc

    await audit.record(
        session,
        principal,
        "maintenance.event_rescheduled",
        resource=event_id,
        request=request,
        detail={"new_start": payload.new_start.isoformat(), "warnings": warnings},
    )
    return {
        "event_id": event.id,
        "scheduled_start": event.scheduled_start.isoformat(),
        "scheduled_end": event.scheduled_end.isoformat(),
        "warnings": warnings,
    }


@router.post("/events/{event_id}/confirm")
async def confirm(
    event_id: str,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        event = await event_ops.confirm_event(
            session, event_id, actor=principal.subject
        )
    except maint.MaintenanceError as exc:
        raise _maintenance_error(exc) from exc

    await audit.record(
        session, principal, "maintenance.event_confirmed", resource=event_id, request=request
    )
    return {"event_id": event.id, "status": event.status}


@router.post("/events/{event_id}/complete")
async def complete(
    event_id: str,
    payload: CompleteEventRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        result = await event_ops.complete_event(
            session,
            event_id,
            completed_on=payload.completed_on,
            actor=principal.subject,
            work_order=payload.work_order,
            completed_task_codes=payload.completed_task_codes,
        )
    except maint.MaintenanceError as exc:
        raise _maintenance_error(exc) from exc

    await audit.record(
        session,
        principal,
        "maintenance.event_completed",
        resource=event_id,
        request=request,
        detail=result,
    )
    return result


@router.post("/defer")
async def defer(
    payload: DeferRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_APPROVE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Approve running a task past its due point.

    Requires the maint:approve permission specifically - deferring a limit is
    an airworthiness decision, not routine planning.
    """
    try:
        deferral = await event_ops.defer_task(
            session,
            payload.tail_number,
            payload.task_code,
            until=payload.until,
            reason=payload.reason,
            approved_by=principal.subject,
            extension_flight_hours=payload.extension_flight_hours,
            authority_reference=payload.authority_reference,
        )
    except maint.MaintenanceError as exc:
        await audit.record(
            session,
            principal,
            "maintenance.defer_refused",
            resource=f"{payload.tail_number}/{payload.task_code}",
            outcome="refused",
            request=request,
            detail={"reason": str(exc)},
        )
        raise _maintenance_error(exc) from exc

    await audit.record(
        session,
        principal,
        "maintenance.deferred",
        resource=f"{payload.tail_number}/{payload.task_code}",
        request=request,
        detail={
            "until": payload.until.isoformat(),
            "reason": payload.reason,
            "authority_reference": payload.authority_reference,
        },
    )
    return {
        "deferral_id": deferral.id,
        "task_code": payload.task_code,
        "tail_number": payload.tail_number,
        "expires_on": deferral.expires_on.isoformat(),
        "approved_by": deferral.approved_by,
    }
