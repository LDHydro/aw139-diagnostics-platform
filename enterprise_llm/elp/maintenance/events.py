"""
The lifecycle of a maintenance event: confirm, cancel, defer, reschedule, close.

Cancellation is the interesting case.  A cancelled slot does not make the
work go away - it makes it *later*, and something has to decide whether
later is still legal.  ``cancel_event`` re-forecasts every task that was on
the cancelled visit, rebuilds a replacement plan, and reports explicitly on
any task that can no longer be completed before its hard limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import (
    Deferral,
    EventStatus,
    EventTask,
    MaintenanceEvent,
    MaintenanceTask,
)
from .forecast import DueStatus
from .planner import MaintenancePlanner
from .service import (
    MaintenanceError,
    forecast_for_aircraft,
    get_aircraft,
    record_compliance,
)

log = logging.getLogger(__name__)


@dataclass
class CancellationOutcome:
    cancelled_event_id: str
    replacement_event_ids: list[str] = field(default_factory=list)
    # Tasks that cannot be completed before their hard limit any more.
    at_risk: list[dict] = field(default_factory=list)
    rescheduled: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cancelled_event_id": self.cancelled_event_id,
            "replacement_event_ids": self.replacement_event_ids,
            "rescheduled": self.rescheduled,
            "at_risk": self.at_risk,
            "warnings": self.warnings,
        }


async def get_event(session: AsyncSession, event_id: str) -> MaintenanceEvent:
    row = (
        await session.execute(
            select(MaintenanceEvent).where(MaintenanceEvent.id == event_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise MaintenanceError(f"no maintenance event with id '{event_id}'")
    return row


# ----------------------------------------------------------------------
# Creating and confirming
# ----------------------------------------------------------------------

async def persist_plan(
    session: AsyncSession,
    plan_events: list,
    *,
    created_by: str = "",
    status: str = EventStatus.PLANNED.value,
    replaces_event_id: str | None = None,
) -> list[MaintenanceEvent]:
    """Write planner output into the schedule."""
    created: list[MaintenanceEvent] = []
    for planned in plan_events:
        event = MaintenanceEvent(
            aircraft_id=planned.aircraft_id,
            title=planned.title,
            status=status,
            scheduled_start=planned.start,
            scheduled_end=planned.end,
            station=planned.station,
            estimated_man_hours=planned.total_man_hours,
            estimated_downtime_hours=planned.downtime_days
            * get_settings().maintenance.shift_hours_per_day,
            created_by=created_by,
            replaces_event_id=replaces_event_id,
            meta={"rationale": planned.rationale, "warnings": planned.warnings},
        )
        session.add(event)
        await session.flush()

        for task in planned.tasks:
            session.add(
                EventTask(
                    event_id=event.id,
                    task_id=task.task_id,
                    due_on=task.due_on,
                    hard_limit_on=task.hard_limit_on,
                    driving_basis=task.driving_basis,
                    remaining_margin=task.interval_waste,
                )
            )
        created.append(event)

    await session.flush()
    return created


async def confirm_event(
    session: AsyncSession, event_id: str, *, actor: str = ""
) -> MaintenanceEvent:
    event = await get_event(session, event_id)
    if event.status in (EventStatus.COMPLETED.value, EventStatus.CANCELLED.value):
        raise MaintenanceError(
            f"event is {event.status} and cannot be confirmed"
        )
    event.status = EventStatus.CONFIRMED.value
    if actor:
        event.notes = f"{event.notes}\nConfirmed by {actor}".strip()
    await session.flush()
    return event


async def reschedule_event(
    session: AsyncSession,
    event_id: str,
    new_start: date,
    *,
    actor: str = "",
    reason: str = "",
) -> tuple[MaintenanceEvent, list[str]]:
    """Move a visit, checking that nothing on it breaches its hard limit."""
    event = await get_event(session, event_id)
    if event.status in (EventStatus.COMPLETED.value, EventStatus.CANCELLED.value):
        raise MaintenanceError(f"event is {event.status} and cannot be moved")

    duration = (event.scheduled_end - event.scheduled_start).days
    new_end = new_start + timedelta(days=duration)

    warnings: list[str] = []
    for item in event.items:
        if item.hard_limit_on and new_end > item.hard_limit_on:
            warnings.append(
                f"{item.task.task_code if item.task else item.task_id} has a hard "
                f"limit of {item.hard_limit_on}; the rescheduled visit ends {new_end}"
            )

    event.scheduled_start = new_start
    event.scheduled_end = new_end
    note = f"Rescheduled to {new_start} by {actor or 'system'}"
    if reason:
        note += f": {reason}"
    event.notes = f"{event.notes}\n{note}".strip()
    await session.flush()
    return event, warnings


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------

async def cancel_event(
    session: AsyncSession,
    event_id: str,
    *,
    reason: str,
    actor: str = "",
    reschedule: bool = True,
    today: date | None = None,
) -> CancellationOutcome:
    """
    Cancel a visit and work out what happens to the tasks that were on it.

    This is where a schedule usually goes wrong in practice: the slot is
    dropped, the work quietly disappears from the plan, and a limit is
    exceeded weeks later.  The replacement plan is generated in the same
    transaction as the cancellation so that cannot happen.
    """
    today = today or date.today()
    event = await get_event(session, event_id)
    if event.status == EventStatus.COMPLETED.value:
        raise MaintenanceError("a completed event cannot be cancelled")
    if event.status == EventStatus.CANCELLED.value:
        raise MaintenanceError("event is already cancelled")

    affected_task_ids = [item.task_id for item in event.items]

    event.status = EventStatus.CANCELLED.value
    event.cancellation_reason = reason
    event.cancelled_at = datetime.now(UTC)
    event.cancelled_by = actor
    for item in event.items:
        item.status = "removed"
    await session.flush()

    outcome = CancellationOutcome(cancelled_event_id=event.id)
    if not affected_task_ids:
        outcome.warnings.append("the cancelled event had no tasks assigned")
        return outcome

    if not reschedule:
        outcome.warnings.append(
            f"{len(affected_task_ids)} task(s) released from the schedule without "
            "a replacement visit; they will reappear on the next forecast"
        )
        return outcome

    # Re-forecast from current state, then plan only the affected tasks.
    aircraft = await get_aircraft(session, event.aircraft_id)
    _state, _rate, forecasts, specs = await forecast_for_aircraft(
        session, aircraft.id, today=today
    )
    affected = {tid for tid in affected_task_ids}
    relevant = [f for f in forecasts if f.task_id in affected]

    for forecast in relevant:
        if forecast.status in (DueStatus.GROUNDED, DueStatus.OVERDUE) or (
            forecast.hard_limit_on and forecast.hard_limit_on < today
        ):
            outcome.at_risk.append(
                {
                    "task_code": forecast.task_code,
                    "title": forecast.task_title,
                    "status": forecast.status.value,
                    "due_on": forecast.due_on.isoformat() if forecast.due_on else None,
                    "hard_limit_on": (
                        forecast.hard_limit_on.isoformat()
                        if forecast.hard_limit_on
                        else None
                    ),
                    "detail": (
                        "already past its limit; the aircraft cannot be released "
                        "until this is cleared or a deferral is approved"
                    ),
                }
            )

    planner = MaintenancePlanner()
    plan = planner.plan_aircraft(
        relevant, specs, station=event.station, today=today
    )
    replacements = await persist_plan(
        session,
        plan.events,
        created_by=actor,
        status=EventStatus.PLANNED.value,
        replaces_event_id=event.id,
    )

    outcome.replacement_event_ids = [e.id for e in replacements]
    outcome.warnings.extend(plan.warnings)
    for new_event, planned in zip(replacements, plan.events, strict=True):
        outcome.rescheduled.append(
            {
                "event_id": new_event.id,
                "start": planned.start.isoformat(),
                "end": planned.end.isoformat(),
                "task_codes": [t.task_code for t in planned.tasks],
                "warnings": planned.warnings,
            }
        )
        outcome.warnings.extend(planned.warnings)

    log.info(
        "cancelled event %s (%s): %d task(s) moved onto %d replacement visit(s)",
        event.id, reason, len(affected_task_ids), len(replacements),
    )
    return outcome


# ----------------------------------------------------------------------
# Deferral
# ----------------------------------------------------------------------

async def defer_task(
    session: AsyncSession,
    identifier: str,
    task_code: str,
    *,
    until: date,
    reason: str,
    approved_by: str,
    extension_flight_hours: float = 0.0,
    authority_reference: str = "",
    event_id: str | None = None,
    today: date | None = None,
) -> Deferral:
    """
    Approve running a task past its nominal due point.

    Refused outright when the task card forbids deferral (airworthiness
    limitations, most ADs) or when the requested extension exceeds the
    limit the card itself allows.
    """
    today = today or date.today()
    aircraft = await get_aircraft(session, identifier)
    task = (
        await session.execute(
            select(MaintenanceTask).where(MaintenanceTask.task_code == task_code)
        )
    ).scalar_one_or_none()
    if task is None:
        raise MaintenanceError(f"no task card with code '{task_code}'")

    if not task.can_be_deferred or task.is_airworthiness_limitation:
        raise MaintenanceError(
            f"{task.task_code} is marked as non-deferrable"
            + (" (airworthiness limitation)" if task.is_airworthiness_limitation else "")
            + " in the maintenance programme; it must be accomplished before the limit"
        )
    if until <= today:
        raise MaintenanceError("a deferral must expire in the future")
    if task.max_deferral_days:
        max_until = today + timedelta(days=task.max_deferral_days)
        if until > max_until:
            raise MaintenanceError(
                f"{task.task_code} allows a maximum deferral of "
                f"{task.max_deferral_days} day(s), i.e. no later than {max_until}"
            )
    if not approved_by:
        raise MaintenanceError("a deferral requires a named approver")

    deferral = Deferral(
        aircraft_id=aircraft.id,
        task_id=task.id,
        event_id=event_id,
        reason=reason,
        approved_by=approved_by,
        expires_on=until,
        extension_flight_hours=extension_flight_hours,
        authority_reference=authority_reference,
    )
    session.add(deferral)
    await session.flush()

    log.info(
        "deferred %s on %s until %s (approved by %s)",
        task.task_code, aircraft.tail_number, until, approved_by,
    )
    return deferral


async def release_deferral(
    session: AsyncSession, deferral_id: str, *, actor: str = ""
) -> Deferral:
    row = (
        await session.execute(select(Deferral).where(Deferral.id == deferral_id))
    ).scalar_one_or_none()
    if row is None:
        raise MaintenanceError(f"no deferral with id '{deferral_id}'")
    row.released = True
    if actor:
        row.reason = f"{row.reason}\nReleased by {actor}".strip()
    await session.flush()
    return row


# ----------------------------------------------------------------------
# Completion
# ----------------------------------------------------------------------

async def complete_event(
    session: AsyncSession,
    event_id: str,
    *,
    completed_on: date | None = None,
    actor: str = "",
    work_order: str = "",
    completed_task_codes: list[str] | None = None,
) -> dict:
    """
    Close a visit out, signing off its tasks.

    Tasks not in ``completed_task_codes`` stay open and are pushed back into
    the forecast rather than being silently marked complete.
    """
    completed_on = completed_on or date.today()
    event = await get_event(session, event_id)
    if event.status == EventStatus.CANCELLED.value:
        raise MaintenanceError("a cancelled event cannot be completed")

    aircraft = await get_aircraft(session, event.aircraft_id)
    signed_off: list[str] = []
    still_open: list[str] = []

    for item in event.items:
        task = item.task
        if task is None:
            continue
        wanted = (
            completed_task_codes is None or task.task_code in completed_task_codes
        )
        if not wanted:
            item.status = "deferred"
            still_open.append(task.task_code)
            continue

        await record_compliance(
            session,
            aircraft.id,
            task.task_code,
            completed_on,
            work_order=work_order or event.id[:8],
            performed_by=actor,
            notes=f"Completed on event {event.id}",
        )
        item.status = "completed"
        signed_off.append(task.task_code)

    event.status = EventStatus.COMPLETED.value
    event.notes = (
        f"{event.notes}\nCompleted {completed_on} by {actor or 'unknown'}"
    ).strip()
    await session.flush()

    return {
        "event_id": event.id,
        "completed_on": completed_on.isoformat(),
        "signed_off": signed_off,
        "still_open": still_open,
        "warnings": (
            [
                f"{len(still_open)} task(s) were not signed off and remain due: "
                + ", ".join(still_open)
            ]
            if still_open
            else []
        ),
    }
