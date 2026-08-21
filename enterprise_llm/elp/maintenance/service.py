"""
Database-facing maintenance service.

Loads fleet state out of Postgres, converts it into the plain dataclasses
the forecasting and planning engines work on, and writes results back.
Keeping the engines free of ORM types is what makes them testable.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import MaintenanceSettings, get_settings
from ..models import (
    Aircraft,
    ComplianceRecord,
    Deferral,
    MaintenanceTask,
    UtilizationRecord,
)
from .forecast import (
    AircraftState,
    ComplianceState,
    DeferralState,
    DueForecast,
    TaskSpec,
    UtilizationDay,
    UtilizationRate,
    estimate_utilization,
    forecast_aircraft,
)
from .planner import MaintenancePlanner, PlanResult, plan_fleet

log = logging.getLogger(__name__)


class MaintenanceError(RuntimeError):
    """A maintenance operation could not be completed."""


# ----------------------------------------------------------------------
# ORM -> dataclass conversion
# ----------------------------------------------------------------------

def to_aircraft_state(row: Aircraft) -> AircraftState:
    in_service_since = None
    raw = (row.meta or {}).get("in_service_since")
    if raw:
        try:
            in_service_since = date.fromisoformat(str(raw))
        except ValueError:
            log.warning(
                "aircraft %s has an unparseable in_service_since: %r",
                row.tail_number, raw,
            )
    return AircraftState(
        id=row.id,
        tail_number=row.tail_number,
        model=row.model,
        serial_number=row.serial_number,
        configuration=row.configuration,
        flight_hours=row.current_flight_hours,
        cycles=row.current_cycles,
        landings=row.current_landings,
        as_of=row.counters_as_of.date() if row.counters_as_of else date.today(),
        in_service_since=in_service_since,
        base_station=row.base_station,
    )


def to_task_spec(row: MaintenanceTask) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        task_code=row.task_code,
        title=row.title,
        ata_chapter=row.ata_chapter,
        task_type=row.task_type,
        interval_flight_hours=row.interval_flight_hours,
        interval_cycles=row.interval_cycles,
        interval_landings=row.interval_landings,
        interval_calendar_days=row.interval_calendar_days,
        tolerance_flight_hours=row.tolerance_flight_hours,
        tolerance_calendar_days=row.tolerance_calendar_days,
        estimated_man_hours=row.estimated_man_hours,
        estimated_downtime_hours=row.estimated_downtime_hours,
        technicians_required=row.technicians_required,
        requires_hangar=row.requires_hangar,
        is_airworthiness_limitation=row.is_airworthiness_limitation,
        can_be_deferred=row.can_be_deferred,
        max_deferral_days=row.max_deferral_days,
        source_document_key=row.source_document_key,
        source_reference=row.source_reference,
    )


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

async def get_aircraft(session: AsyncSession, identifier: str) -> Aircraft:
    """Look an aircraft up by tail number or id."""
    row = (
        await session.execute(
            select(Aircraft).where(Aircraft.tail_number == identifier)
        )
    ).scalar_one_or_none()
    if row is None:
        row = (
            await session.execute(select(Aircraft).where(Aircraft.id == identifier))
        ).scalar_one_or_none()
    if row is None:
        raise MaintenanceError(f"no aircraft found for '{identifier}'")
    return row


async def load_utilization(
    session: AsyncSession,
    aircraft_id: str,
    settings: MaintenanceSettings | None = None,
    *,
    today: date | None = None,
) -> list[UtilizationDay]:
    settings = settings or get_settings().maintenance
    today = today or date.today()
    since = today - timedelta(days=settings.utilization_window_days)
    rows = (
        await session.execute(
            select(UtilizationRecord)
            .where(
                UtilizationRecord.aircraft_id == aircraft_id,
                UtilizationRecord.day >= since,
            )
            .order_by(UtilizationRecord.day)
        )
    ).scalars().all()
    return [
        UtilizationDay(
            day=r.day,
            flight_hours=r.flight_hours,
            cycles=r.cycles,
            landings=r.landings,
        )
        for r in rows
    ]


async def load_applicable_tasks(
    session: AsyncSession, aircraft: Aircraft
) -> list[MaintenanceTask]:
    rows = (
        await session.execute(
            select(MaintenanceTask).where(MaintenanceTask.active.is_(True))
        )
    ).scalars().all()
    return [t for t in rows if t.applies_to(aircraft)]


async def load_latest_compliance(
    session: AsyncSession, aircraft_id: str
) -> dict[str, ComplianceState]:
    """Most recent accomplishment per task."""
    rows = (
        await session.execute(
            select(ComplianceRecord)
            .where(ComplianceRecord.aircraft_id == aircraft_id)
            .order_by(ComplianceRecord.completed_on.desc())
        )
    ).scalars().all()

    latest: dict[str, ComplianceState] = {}
    for row in rows:
        if row.task_id in latest:
            continue  # rows are newest-first, so the first hit wins
        latest[row.task_id] = ComplianceState(
            completed_on=row.completed_on,
            at_flight_hours=row.at_flight_hours,
            at_cycles=row.at_cycles,
            at_landings=row.at_landings,
            work_order=row.work_order,
        )
    return latest


async def load_active_deferrals(
    session: AsyncSession, aircraft_id: str, *, today: date | None = None
) -> dict[str, DeferralState]:
    today = today or date.today()
    rows = (
        await session.execute(
            select(Deferral).where(
                Deferral.aircraft_id == aircraft_id,
                Deferral.released.is_(False),
                Deferral.expires_on >= today,
            )
        )
    ).scalars().all()
    return {
        row.task_id: DeferralState(
            expires_on=row.expires_on,
            extension_flight_hours=row.extension_flight_hours,
            approved_by=row.approved_by,
            reason=row.reason,
        )
        for row in rows
    }


# ----------------------------------------------------------------------
# Forecast / plan
# ----------------------------------------------------------------------

async def forecast_for_aircraft(
    session: AsyncSession,
    identifier: str,
    *,
    today: date | None = None,
    horizon_days: int | None = None,
    status_filter: list[str] | None = None,
) -> tuple[AircraftState, UtilizationRate, list[DueForecast], dict[str, TaskSpec]]:
    settings = get_settings().maintenance
    today = today or date.today()

    aircraft_row = await get_aircraft(session, identifier)
    state = to_aircraft_state(aircraft_row)

    utilization = await load_utilization(session, aircraft_row.id, settings, today=today)
    rate = estimate_utilization(utilization, settings, today=today)

    task_rows = await load_applicable_tasks(session, aircraft_row)
    specs = {row.id: to_task_spec(row) for row in task_rows}

    compliance = await load_latest_compliance(session, aircraft_row.id)
    deferrals = await load_active_deferrals(session, aircraft_row.id, today=today)

    forecasts = forecast_aircraft(
        state,
        list(specs.values()),
        rate,
        compliance,
        deferrals,
        settings,
        today=today,
        horizon_days=horizon_days,
    )

    if status_filter:
        wanted = {s.lower() for s in status_filter}
        forecasts = [f for f in forecasts if f.status.value in wanted]

    return state, rate, forecasts, specs


async def plan_for_aircraft(
    session: AsyncSession,
    identifier: str,
    *,
    today: date | None = None,
    horizon_days: int | None = None,
) -> tuple[AircraftState, UtilizationRate, PlanResult]:
    state, rate, forecasts, specs = await forecast_for_aircraft(
        session, identifier, today=today, horizon_days=horizon_days
    )
    planner = MaintenancePlanner()
    plan = planner.plan_aircraft(
        forecasts,
        specs,
        station=state.base_station,
        today=today,
        horizon_days=horizon_days,
    )
    return state, rate, plan


async def plan_for_fleet(
    session: AsyncSession,
    *,
    today: date | None = None,
    horizon_days: int | None = None,
    tail_numbers: list[str] | None = None,
) -> PlanResult:
    query = select(Aircraft).where(Aircraft.in_service.is_(True))
    if tail_numbers:
        query = query.where(Aircraft.tail_number.in_(tail_numbers))
    aircraft_rows = (await session.execute(query)).scalars().all()

    per_aircraft: dict[str, tuple[list[DueForecast], dict[str, TaskSpec], str]] = {}
    for row in aircraft_rows:
        _state, _rate, forecasts, specs = await forecast_for_aircraft(
            session, row.id, today=today, horizon_days=horizon_days
        )
        per_aircraft[row.id] = (forecasts, specs, row.base_station)

    return plan_fleet(per_aircraft, today=today, horizon_days=horizon_days)


# ----------------------------------------------------------------------
# Writes
# ----------------------------------------------------------------------

async def record_utilization(
    session: AsyncSession,
    identifier: str,
    day: date,
    *,
    flight_hours: float = 0.0,
    cycles: int = 0,
    landings: int = 0,
    source: str = "manual",
    update_counters: bool = True,
) -> UtilizationRecord:
    """
    Log a day's flying and roll the aircraft's counters forward.

    Re-posting the same day replaces the earlier figure rather than adding
    to it, and the aircraft totals are adjusted by the difference so a
    correction does not double-count.
    """
    aircraft = await get_aircraft(session, identifier)

    existing = (
        await session.execute(
            select(UtilizationRecord).where(
                UtilizationRecord.aircraft_id == aircraft.id,
                UtilizationRecord.day == day,
            )
        )
    ).scalar_one_or_none()

    if existing is None:
        record = UtilizationRecord(
            aircraft_id=aircraft.id,
            day=day,
            flight_hours=flight_hours,
            cycles=cycles,
            landings=landings,
            source=source,
        )
        session.add(record)
        delta_fh, delta_cy, delta_ld = flight_hours, cycles, landings
    else:
        delta_fh = flight_hours - existing.flight_hours
        delta_cy = cycles - existing.cycles
        delta_ld = landings - existing.landings
        existing.flight_hours = flight_hours
        existing.cycles = cycles
        existing.landings = landings
        existing.source = source
        record = existing

    if update_counters:
        aircraft.current_flight_hours = round(
            aircraft.current_flight_hours + delta_fh, 2
        )
        aircraft.current_cycles += delta_cy
        aircraft.current_landings += delta_ld
        from datetime import datetime

        aircraft.counters_as_of = datetime.now(UTC)

    await session.flush()
    return record


async def record_compliance(
    session: AsyncSession,
    identifier: str,
    task_code: str,
    completed_on: date,
    *,
    work_order: str = "",
    performed_by: str = "",
    notes: str = "",
    at_flight_hours: float | None = None,
    at_cycles: int | None = None,
    at_landings: int | None = None,
) -> ComplianceRecord:
    """Sign a task off, resetting its interval from this point."""
    aircraft = await get_aircraft(session, identifier)
    task = (
        await session.execute(
            select(MaintenanceTask).where(MaintenanceTask.task_code == task_code)
        )
    ).scalar_one_or_none()
    if task is None:
        raise MaintenanceError(f"no task card with code '{task_code}'")

    record = ComplianceRecord(
        aircraft_id=aircraft.id,
        task_id=task.id,
        completed_on=completed_on,
        at_flight_hours=(
            aircraft.current_flight_hours if at_flight_hours is None else at_flight_hours
        ),
        at_cycles=aircraft.current_cycles if at_cycles is None else at_cycles,
        at_landings=aircraft.current_landings if at_landings is None else at_landings,
        work_order=work_order,
        performed_by=performed_by,
        notes=notes,
    )
    session.add(record)

    # Signing a task off releases any deferral that was holding it open.
    deferrals = (
        await session.execute(
            select(Deferral).where(
                Deferral.aircraft_id == aircraft.id,
                Deferral.task_id == task.id,
                Deferral.released.is_(False),
            )
        )
    ).scalars().all()
    for deferral in deferrals:
        deferral.released = True

    await session.flush()
    return record
