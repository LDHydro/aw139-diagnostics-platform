"""
Utilisation forecasting and due-date projection.

The scheduling problem in a maintenance organisation is not "when is this
task due" - the manual already says that - it is "on what *calendar date*
will this aircraft reach that point, given how it is actually being flown".
That is what this module answers.

A task card may carry several intervals at once (600 flight hours *or* 12
months *or* 1000 landings).  Each is projected onto the calendar
independently and the earliest one drives the schedule.  Everything here is
a pure function over plain dataclasses so it can be tested without a
database or a model server.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from ..config import MaintenanceSettings, get_settings

# A projection further out than this is reported as "not in the horizon"
# rather than as a precise but meaningless date.
_FAR_FUTURE_DAYS = 3650


class Basis(str, Enum):
    FLIGHT_HOURS = "flight_hours"
    CYCLES = "cycles"
    LANDINGS = "landings"
    CALENDAR = "calendar_days"


class DueStatus(str, Enum):
    OK = "ok"
    DUE_SOON = "due_soon"
    DUE = "due"
    OVERDUE = "overdue"
    # Past the tolerance: the aircraft cannot legally fly until it is done.
    GROUNDED = "grounded"


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------

@dataclass
class AircraftState:
    id: str
    tail_number: str
    model: str = ""
    serial_number: str = ""
    configuration: str = ""
    flight_hours: float = 0.0
    cycles: int = 0
    landings: int = 0
    as_of: date = field(default_factory=date.today)
    in_service_since: date | None = None
    base_station: str = ""


@dataclass
class UtilizationDay:
    day: date
    flight_hours: float = 0.0
    cycles: int = 0
    landings: int = 0


@dataclass
class TaskSpec:
    id: str
    task_code: str
    title: str = ""
    ata_chapter: str = ""
    task_type: str = "inspection"
    interval_flight_hours: float | None = None
    interval_cycles: int | None = None
    interval_landings: int | None = None
    interval_calendar_days: int | None = None
    tolerance_flight_hours: float = 0.0
    tolerance_calendar_days: int = 0
    estimated_man_hours: float = 1.0
    estimated_downtime_hours: float = 1.0
    technicians_required: int = 1
    requires_hangar: bool = False
    is_airworthiness_limitation: bool = False
    can_be_deferred: bool = True
    max_deferral_days: int = 0
    source_document_key: str = ""
    source_reference: str = ""


@dataclass
class ComplianceState:
    """When the task was last accomplished, and at what counter readings."""

    completed_on: date
    at_flight_hours: float = 0.0
    at_cycles: int = 0
    at_landings: int = 0
    work_order: str = ""


@dataclass
class DeferralState:
    expires_on: date
    extension_flight_hours: float = 0.0
    approved_by: str = ""
    reason: str = ""


# ----------------------------------------------------------------------
# Utilisation rate
# ----------------------------------------------------------------------

@dataclass
class UtilizationRate:
    daily_flight_hours: float
    daily_cycles: float
    daily_landings: float
    samples: int = 0
    stdev_flight_hours: float = 0.0
    confidence: float = 0.0
    source: str = "history"

    def describe(self) -> str:
        return (
            f"{self.daily_flight_hours:.2f} FH/day, {self.daily_cycles:.2f} cycles/day "
            f"({self.source}, {self.samples} day(s) of history, "
            f"confidence {self.confidence:.0%})"
        )


def estimate_utilization(
    records: list[UtilizationDay],
    settings: MaintenanceSettings | None = None,
    *,
    today: date | None = None,
) -> UtilizationRate:
    """
    Exponentially-weighted daily utilisation over the configured window.

    Two details matter for correctness:

    * **Non-flying days count.** Most logbook feeds only emit rows for days
      the aircraft flew.  Averaging those rows alone overstates utilisation
      badly (a helicopter flown twice a week would look like a daily
      workhorse), so gaps inside the observed span are filled with zeros.
    * **Recent weeks weigh more**, because a change in operational tempo -
      a contract starting, a season ending - should move the forecast
      quickly rather than being diluted by three months of history.
    """
    settings = settings or get_settings().maintenance
    today = today or date.today()
    window_start = today - timedelta(days=settings.utilization_window_days)

    in_window = sorted(
        (r for r in records if window_start <= r.day <= today), key=lambda r: r.day
    )
    if not in_window:
        return UtilizationRate(
            daily_flight_hours=settings.default_daily_flight_hours,
            daily_cycles=settings.default_daily_cycles,
            daily_landings=settings.default_daily_cycles,
            samples=0,
            confidence=0.15,
            source="default",
        )

    by_day = {r.day: r for r in in_window}
    span_start, span_end = in_window[0].day, in_window[-1].day
    dense: list[UtilizationDay] = []
    cursor = span_start
    while cursor <= span_end:
        dense.append(by_day.get(cursor, UtilizationDay(day=cursor)))
        cursor += timedelta(days=1)

    # Smooth over whole weeks, not individual days.  Flying patterns are
    # strongly weekly (weekdays busy, weekends idle), and a day-level EWMA
    # lands wherever the last few days happened to fall - the same fleet
    # would forecast differently depending on whether you asked on a Friday
    # or a Sunday.  Bucketing into 7-day periods removes that phase
    # sensitivity while still weighting recent weeks more heavily.
    buckets: list[tuple[float, float, float]] = []
    for start in range(len(dense), 0, -7):
        window = dense[max(0, start - 7) : start]
        if not window:
            continue
        days = len(window)
        buckets.append(
            (
                sum(r.flight_hours for r in window) / days,
                sum(r.cycles for r in window) / days,
                sum(r.landings for r in window) / days,
            )
        )
    buckets.reverse()  # oldest first, so the EWMA ends on the newest week

    alpha = settings.utilization_alpha
    ewma_fh, ewma_cy, ewma_ld = buckets[0]
    for bucket_fh, bucket_cy, bucket_ld in buckets[1:]:
        ewma_fh = alpha * bucket_fh + (1 - alpha) * ewma_fh
        ewma_cy = alpha * bucket_cy + (1 - alpha) * ewma_cy
        ewma_ld = alpha * bucket_ld + (1 - alpha) * ewma_ld

    # Variability is measured between weeks, which is what actually makes a
    # projection uncertain; day-to-day on/off flying is normal and expected.
    weekly = [b[0] for b in buckets]
    stdev = statistics.pstdev(weekly) if len(weekly) > 1 else 0.0
    mean = statistics.fmean(weekly) if weekly else 0.0

    # Confidence rises with sample count and falls as utilisation becomes
    # erratic relative to its own mean.
    sample_factor = min(1.0, len(dense) / max(1, settings.utilization_window_days * 0.5))
    variability = (stdev / mean) if mean > 0 else 1.0
    stability_factor = 1.0 / (1.0 + variability)
    confidence = max(0.1, min(0.95, 0.4 * sample_factor + 0.6 * stability_factor))

    return UtilizationRate(
        daily_flight_hours=round(ewma_fh, 4),
        daily_cycles=round(ewma_cy, 4),
        daily_landings=round(ewma_ld, 4),
        samples=len(dense),
        stdev_flight_hours=round(stdev, 4),
        confidence=round(confidence, 3),
        source="history",
    )


# ----------------------------------------------------------------------
# Due forecast
# ----------------------------------------------------------------------

@dataclass
class BasisProjection:
    basis: Basis
    interval: float
    remaining: float
    due_on: date | None
    # Fraction of the interval still available, clamped at 0.
    remaining_fraction: float
    unit: str

    @property
    def consumed_fraction(self) -> float:
        return max(0.0, 1.0 - self.remaining_fraction)


@dataclass
class DueForecast:
    aircraft_id: str
    tail_number: str
    task_id: str
    task_code: str
    task_title: str
    driving_basis: Basis
    due_on: date | None
    hard_limit_on: date | None
    remaining_fraction: float
    status: DueStatus
    confidence: float
    projections: list[BasisProjection] = field(default_factory=list)
    last_completed_on: date | None = None
    deferred_until: date | None = None
    is_airworthiness_limitation: bool = False
    source_document_key: str = ""
    source_reference: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def days_until_due(self) -> int | None:
        if self.due_on is None:
            return None
        return (self.due_on - date.today()).days

    def to_dict(self) -> dict:
        return {
            "aircraft_id": self.aircraft_id,
            "tail_number": self.tail_number,
            "task_id": self.task_id,
            "task_code": self.task_code,
            "task_title": self.task_title,
            "driving_basis": self.driving_basis.value,
            "due_on": self.due_on.isoformat() if self.due_on else None,
            "hard_limit_on": (
                self.hard_limit_on.isoformat() if self.hard_limit_on else None
            ),
            "days_until_due": self.days_until_due,
            "remaining_fraction": round(self.remaining_fraction, 4),
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "last_completed_on": (
                self.last_completed_on.isoformat() if self.last_completed_on else None
            ),
            "deferred_until": (
                self.deferred_until.isoformat() if self.deferred_until else None
            ),
            "is_airworthiness_limitation": self.is_airworthiness_limitation,
            "source": {
                "document_key": self.source_document_key,
                "reference": self.source_reference,
            },
            "projections": [
                {
                    "basis": p.basis.value,
                    "interval": p.interval,
                    "remaining": round(p.remaining, 2),
                    "unit": p.unit,
                    "due_on": p.due_on.isoformat() if p.due_on else None,
                    "remaining_fraction": round(p.remaining_fraction, 4),
                }
                for p in self.projections
            ],
            "notes": self.notes,
        }


def _project_days(remaining: float, daily_rate: float) -> int | None:
    """Calendar days until ``remaining`` units are consumed at ``daily_rate``."""
    if remaining <= 0:
        return 0
    if daily_rate <= 0:
        return None  # not consumed at all on current usage
    return math.ceil(remaining / daily_rate)


def forecast_task(
    aircraft: AircraftState,
    task: TaskSpec,
    rate: UtilizationRate,
    compliance: ComplianceState | None = None,
    deferral: DeferralState | None = None,
    settings: MaintenanceSettings | None = None,
    *,
    today: date | None = None,
) -> DueForecast:
    """Project every interval on a task card onto the calendar."""
    settings = settings or get_settings().maintenance
    today = today or date.today()
    notes: list[str] = []

    # Baseline: last accomplishment, or entry into service for a task that
    # has never been done.
    if compliance is not None:
        base_date = compliance.completed_on
        base_fh = compliance.at_flight_hours
        base_cycles = compliance.at_cycles
        base_landings = compliance.at_landings
    else:
        base_date = aircraft.in_service_since or today
        base_fh = 0.0
        base_cycles = 0
        base_landings = 0
        notes.append(
            "no compliance record found; interval computed from entry into service"
        )

    projections: list[BasisProjection] = []

    if task.interval_flight_hours:
        due_fh = base_fh + task.interval_flight_hours
        remaining = due_fh - aircraft.flight_hours
        days = _project_days(remaining, rate.daily_flight_hours)
        projections.append(
            BasisProjection(
                basis=Basis.FLIGHT_HOURS,
                interval=task.interval_flight_hours,
                remaining=remaining,
                due_on=today + timedelta(days=days) if days is not None else None,
                remaining_fraction=max(0.0, remaining / task.interval_flight_hours),
                unit="FH",
            )
        )

    if task.interval_cycles:
        due_cycles = base_cycles + task.interval_cycles
        remaining = due_cycles - aircraft.cycles
        days = _project_days(remaining, rate.daily_cycles)
        projections.append(
            BasisProjection(
                basis=Basis.CYCLES,
                interval=float(task.interval_cycles),
                remaining=remaining,
                due_on=today + timedelta(days=days) if days is not None else None,
                remaining_fraction=max(0.0, remaining / task.interval_cycles),
                unit="cycles",
            )
        )

    if task.interval_landings:
        due_landings = base_landings + task.interval_landings
        remaining = due_landings - aircraft.landings
        days = _project_days(remaining, rate.daily_landings)
        projections.append(
            BasisProjection(
                basis=Basis.LANDINGS,
                interval=float(task.interval_landings),
                remaining=remaining,
                due_on=today + timedelta(days=days) if days is not None else None,
                remaining_fraction=max(0.0, remaining / task.interval_landings),
                unit="landings",
            )
        )

    if task.interval_calendar_days:
        due_date = base_date + timedelta(days=task.interval_calendar_days)
        remaining_days = (due_date - today).days
        projections.append(
            BasisProjection(
                basis=Basis.CALENDAR,
                interval=float(task.interval_calendar_days),
                remaining=float(remaining_days),
                due_on=due_date,
                remaining_fraction=max(
                    0.0, remaining_days / task.interval_calendar_days
                ),
                unit="days",
            )
        )

    if not projections:
        # A task card with no interval is non-recurring (one-off SB, AD with
        # a single compliance point).  Report it as satisfied once complied.
        return DueForecast(
            aircraft_id=aircraft.id,
            tail_number=aircraft.tail_number,
            task_id=task.id,
            task_code=task.task_code,
            task_title=task.title,
            driving_basis=Basis.CALENDAR,
            due_on=None,
            hard_limit_on=None,
            remaining_fraction=1.0,
            status=DueStatus.OK if compliance else DueStatus.DUE,
            confidence=rate.confidence,
            projections=[],
            last_completed_on=compliance.completed_on if compliance else None,
            is_airworthiness_limitation=task.is_airworthiness_limitation,
            source_document_key=task.source_document_key,
            source_reference=task.source_reference,
            notes=notes + ["non-recurring task (no repeat interval defined)"],
        )

    # The earliest projected date governs.  A basis that is never consumed
    # (due_on is None) cannot drive the schedule.
    datable = [p for p in projections if p.due_on is not None]
    if datable:
        driver = min(datable, key=lambda p: p.due_on)  # type: ignore[arg-type,return-value]
    else:
        driver = min(projections, key=lambda p: p.remaining_fraction)
        notes.append(
            "current utilisation does not consume this interval; "
            "no calendar due date can be projected"
        )

    due_on = driver.due_on
    remaining_fraction = min(p.remaining_fraction for p in projections)

    # Tolerance extends the hard limit past the nominal due point.
    hard_limit_on = due_on
    if due_on is not None:
        extra_days = task.tolerance_calendar_days
        if task.tolerance_flight_hours and rate.daily_flight_hours > 0:
            extra_days = max(
                extra_days,
                math.floor(task.tolerance_flight_hours / rate.daily_flight_hours),
            )
        hard_limit_on = due_on + timedelta(days=extra_days)

    deferred_until = None
    if deferral is not None:
        deferred_until = deferral.expires_on
        if hard_limit_on is None or deferral.expires_on > hard_limit_on:
            hard_limit_on = deferral.expires_on
        notes.append(
            f"approved deferral in force until {deferral.expires_on.isoformat()}"
            + (f" ({deferral.reason})" if deferral.reason else "")
        )

    status = _status_for(
        due_on, hard_limit_on, remaining_fraction, today, settings
    )

    # A projection is only as good as the utilisation estimate behind it,
    # except for calendar tasks, which need no forecast at all.
    confidence = 1.0 if driver.basis is Basis.CALENDAR else rate.confidence

    return DueForecast(
        aircraft_id=aircraft.id,
        tail_number=aircraft.tail_number,
        task_id=task.id,
        task_code=task.task_code,
        task_title=task.title,
        driving_basis=driver.basis,
        due_on=due_on,
        hard_limit_on=hard_limit_on,
        remaining_fraction=remaining_fraction,
        status=status,
        confidence=confidence,
        projections=projections,
        last_completed_on=compliance.completed_on if compliance else None,
        deferred_until=deferred_until,
        is_airworthiness_limitation=task.is_airworthiness_limitation,
        source_document_key=task.source_document_key,
        source_reference=task.source_reference,
        notes=notes,
    )


def _status_for(
    due_on: date | None,
    hard_limit_on: date | None,
    remaining_fraction: float,
    today: date,
    settings: MaintenanceSettings,
) -> DueStatus:
    if hard_limit_on is not None and hard_limit_on < today:
        return DueStatus.GROUNDED
    if due_on is not None and due_on < today:
        return DueStatus.OVERDUE
    if remaining_fraction <= 0:
        return DueStatus.DUE
    if remaining_fraction <= settings.warning_threshold_pct:
        return DueStatus.DUE_SOON
    if due_on is not None and (due_on - today).days <= settings.bundling_window_days:
        return DueStatus.DUE_SOON
    return DueStatus.OK


def forecast_aircraft(
    aircraft: AircraftState,
    tasks: list[TaskSpec],
    rate: UtilizationRate,
    compliance: dict[str, ComplianceState] | None = None,
    deferrals: dict[str, DeferralState] | None = None,
    settings: MaintenanceSettings | None = None,
    *,
    today: date | None = None,
    horizon_days: int | None = None,
) -> list[DueForecast]:
    """Forecast every applicable task, soonest first."""
    settings = settings or get_settings().maintenance
    today = today or date.today()
    horizon_days = horizon_days or settings.forecast_horizon_days
    horizon = today + timedelta(days=horizon_days)
    compliance = compliance or {}
    deferrals = deferrals or {}

    results: list[DueForecast] = []
    for task in tasks:
        forecast = forecast_task(
            aircraft,
            task,
            rate,
            compliance.get(task.id),
            deferrals.get(task.id),
            settings,
            today=today,
        )
        if forecast.due_on is not None and forecast.due_on > horizon:
            continue
        results.append(forecast)

    far = today + timedelta(days=_FAR_FUTURE_DAYS)
    results.sort(key=lambda f: (f.due_on or far, -int(f.is_airworthiness_limitation)))
    return results
