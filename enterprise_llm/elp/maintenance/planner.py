"""
Turning a list of due dates into an actual maintenance plan.

Forecasting says *when* each task becomes due.  Planning decides *when the
aircraft comes in*, which is a different problem: tasks should be bundled so
one visit clears several cards, but not bundled so aggressively that useful
interval is thrown away, and not so many aircraft should be down at once
that the operation cannot fly.

The planner is greedy rather than optimal, deliberately: a planner whose
output a human cannot predict or explain does not get used.  Every decision
it makes is recorded on the event as a rationale line.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..config import MaintenanceSettings, get_settings
from .forecast import DueForecast, DueStatus, TaskSpec

log = logging.getLogger(__name__)


@dataclass
class PlannedTask:
    task_id: str
    task_code: str
    title: str
    due_on: date | None
    hard_limit_on: date | None
    driving_basis: str
    man_hours: float
    downtime_hours: float
    technicians: int
    requires_hangar: bool
    is_airworthiness_limitation: bool
    # Fraction of the interval discarded by doing this task early.
    interval_waste: float = 0.0
    reason: str = ""


@dataclass
class PlannedEvent:
    aircraft_id: str
    tail_number: str
    start: date
    end: date
    tasks: list[PlannedTask] = field(default_factory=list)
    station: str = ""
    total_man_hours: float = 0.0
    downtime_days: int = 1
    requires_hangar: bool = False
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        if not self.tasks:
            return "Maintenance event"
        driver = self.tasks[0]
        extra = len(self.tasks) - 1
        suffix = f" (+{extra} task{'s' if extra != 1 else ''})" if extra else ""
        return f"{driver.task_code} {driver.title}"[:120] + suffix

    def to_dict(self) -> dict:
        return {
            "aircraft_id": self.aircraft_id,
            "tail_number": self.tail_number,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "station": self.station,
            "downtime_days": self.downtime_days,
            "total_man_hours": round(self.total_man_hours, 1),
            "requires_hangar": self.requires_hangar,
            "task_count": len(self.tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "task_code": t.task_code,
                    "title": t.title,
                    "due_on": t.due_on.isoformat() if t.due_on else None,
                    "hard_limit_on": (
                        t.hard_limit_on.isoformat() if t.hard_limit_on else None
                    ),
                    "driving_basis": t.driving_basis,
                    "man_hours": t.man_hours,
                    "interval_waste": round(t.interval_waste, 3),
                    "is_airworthiness_limitation": t.is_airworthiness_limitation,
                    "reason": t.reason,
                }
                for t in self.tasks
            ],
            "rationale": self.rationale,
            "warnings": self.warnings,
        }


@dataclass
class PlanResult:
    events: list[PlannedEvent] = field(default_factory=list)
    unplanned: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "unplanned": self.unplanned,
            "warnings": self.warnings,
            "summary": {
                "event_count": len(self.events),
                "task_count": sum(len(e.tasks) for e in self.events),
                "total_man_hours": round(
                    sum(e.total_man_hours for e in self.events), 1
                ),
                "total_downtime_days": sum(e.downtime_days for e in self.events),
            },
        }


class MaintenancePlanner:
    def __init__(self, settings: MaintenanceSettings | None = None) -> None:
        self.settings = settings or get_settings().maintenance

    # ------------------------------------------------------------------

    def plan_aircraft(
        self,
        forecasts: list[DueForecast],
        tasks: dict[str, TaskSpec],
        *,
        station: str = "",
        today: date | None = None,
        horizon_days: int | None = None,
        blackout_dates: set[date] | None = None,
    ) -> PlanResult:
        """Bundle one aircraft's due tasks into visits."""
        today = today or date.today()
        horizon_days = horizon_days or self.settings.planning_horizon_days
        horizon = today + timedelta(days=horizon_days)
        blackout_dates = blackout_dates or set()

        result = PlanResult()
        candidates = [
            f
            for f in forecasts
            if f.due_on is not None and f.due_on <= horizon
        ]
        # Tasks with no projectable due date still need to be visible.
        for forecast in forecasts:
            if forecast.due_on is None:
                result.unplanned.append(
                    {
                        "task_code": forecast.task_code,
                        "title": forecast.task_title,
                        "reason": (
                            "no calendar due date can be projected at the current "
                            "utilisation rate"
                        ),
                        "status": forecast.status.value,
                    }
                )

        candidates.sort(key=lambda f: (f.due_on, not f.is_airworthiness_limitation))
        remaining = list(candidates)

        while remaining:
            anchor = remaining.pop(0)
            spec = tasks.get(anchor.task_id)
            if spec is None:
                result.warnings.append(
                    f"task {anchor.task_code} has no specification; skipped"
                )
                continue

            event = self._open_event(anchor, spec, station, today, blackout_dates)
            bundled = self._bundle(event, anchor, remaining, tasks)
            for forecast in bundled:
                remaining.remove(forecast)

            self._finalise(event, blackout_dates)
            result.events.append(event)

        result.events.sort(key=lambda e: e.start)
        self._flag_conflicts(result)
        return result

    # ------------------------------------------------------------------

    def _open_event(
        self,
        anchor: DueForecast,
        spec: TaskSpec,
        station: str,
        today: date,
        blackout_dates: set[date],
    ) -> PlannedEvent:
        # Never schedule in the past; an overdue task is scheduled today.
        start = max(anchor.due_on or today, today)
        while start in blackout_dates:
            start += timedelta(days=1)

        event = PlannedEvent(
            aircraft_id=anchor.aircraft_id,
            tail_number=anchor.tail_number,
            start=start,
            end=start,
            station=station,
        )
        event.tasks.append(
            PlannedTask(
                task_id=anchor.task_id,
                task_code=anchor.task_code,
                title=anchor.task_title,
                due_on=anchor.due_on,
                hard_limit_on=anchor.hard_limit_on,
                driving_basis=anchor.driving_basis.value,
                man_hours=spec.estimated_man_hours,
                downtime_hours=spec.estimated_downtime_hours,
                technicians=spec.technicians_required,
                requires_hangar=spec.requires_hangar,
                is_airworthiness_limitation=spec.is_airworthiness_limitation,
                reason=f"drives the visit; due {anchor.due_on} on {anchor.driving_basis.value}",
            )
        )
        event.rationale.append(
            f"Visit anchored on {anchor.task_code}, due {anchor.due_on} "
            f"({anchor.driving_basis.value})."
        )
        if anchor.status in (DueStatus.OVERDUE, DueStatus.GROUNDED):
            event.warnings.append(
                f"{anchor.task_code} is already {anchor.status.value}; "
                "the aircraft may not be released to service until it is cleared"
            )
        return event

    def _bundle(
        self,
        event: PlannedEvent,
        anchor: DueForecast,
        remaining: list[DueForecast],
        tasks: dict[str, TaskSpec],
    ) -> list[DueForecast]:
        """Pull in nearby tasks that can be done in the same visit."""
        window_end = event.start + timedelta(days=self.settings.bundling_window_days)
        capacity = (
            self.settings.shift_hours_per_day * self.settings.technicians_per_shift
        )
        # A visit may run several days, so allow up to the bundling window
        # worth of labour before refusing more work.
        max_man_hours = capacity * max(1, self.settings.bundling_window_days / 2)

        bundled: list[DueForecast] = []
        for forecast in remaining:
            if forecast.due_on is None or forecast.due_on > window_end:
                continue
            spec = tasks.get(forecast.task_id)
            if spec is None:
                continue

            waste = self._interval_waste(forecast, event.start)
            if waste > self.settings.max_interval_waste_pct:
                # Doing it now would discard too much life; it will get its
                # own visit closer to its due date.
                continue

            projected = event.total_man_hours + spec.estimated_man_hours
            if projected > max_man_hours and len(event.tasks) > 1:
                event.rationale.append(
                    f"Stopped bundling at {len(event.tasks)} tasks: adding "
                    f"{forecast.task_code} would exceed the labour available "
                    f"in this visit."
                )
                break

            event.tasks.append(
                PlannedTask(
                    task_id=forecast.task_id,
                    task_code=forecast.task_code,
                    title=forecast.task_title,
                    due_on=forecast.due_on,
                    hard_limit_on=forecast.hard_limit_on,
                    driving_basis=forecast.driving_basis.value,
                    man_hours=spec.estimated_man_hours,
                    downtime_hours=spec.estimated_downtime_hours,
                    technicians=spec.technicians_required,
                    requires_hangar=spec.requires_hangar,
                    is_airworthiness_limitation=spec.is_airworthiness_limitation,
                    interval_waste=waste,
                    reason=(
                        f"due {forecast.due_on}, within "
                        f"{self.settings.bundling_window_days} days of the anchor; "
                        f"{waste:.0%} of interval given up"
                    ),
                )
            )
            bundled.append(forecast)
            event.total_man_hours = projected

        if bundled:
            event.rationale.append(
                f"Bundled {len(bundled)} additional task(s) due within "
                f"{self.settings.bundling_window_days} days to avoid a second visit."
            )
        return bundled

    def _interval_waste(self, forecast: DueForecast, planned_start: date) -> float:
        """
        Fraction of the task's interval discarded by doing it on ``planned_start``.

        Doing a 600-hour inspection 60 hours early throws away 10% of the
        interval.  Over a fleet and a year that is real money, so it is
        surfaced rather than hidden inside the bundling decision.
        """
        if forecast.due_on is None:
            return 0.0
        days_early = (forecast.due_on - planned_start).days
        if days_early <= 0:
            return 0.0

        driver = next(
            (p for p in forecast.projections if p.basis is forecast.driving_basis),
            None,
        )
        if driver is None or driver.interval <= 0:
            return 0.0

        # Convert days early into a fraction of the interval by using the
        # task's own consumption rate over its full interval.
        total_days = None
        if driver.due_on and forecast.last_completed_on:
            total_days = (driver.due_on - forecast.last_completed_on).days
        if not total_days or total_days <= 0:
            # Fall back to the remaining fraction as an upper bound.
            return min(1.0, days_early / max(1, self.settings.forecast_horizon_days))
        return min(1.0, days_early / total_days)

    def _finalise(self, event: PlannedEvent, blackout_dates: set[date]) -> None:
        capacity = (
            self.settings.shift_hours_per_day * self.settings.technicians_per_shift
        )
        event.total_man_hours = sum(t.man_hours for t in event.tasks)
        event.requires_hangar = any(t.requires_hangar for t in event.tasks)

        # Downtime is the greater of the labour requirement and the longest
        # single task's elapsed time (a cure or a leak check does not go
        # faster with more technicians).
        labour_days = math.ceil(event.total_man_hours / capacity) if capacity else 1
        elapsed_days = math.ceil(
            max((t.downtime_hours for t in event.tasks), default=1.0)
            / self.settings.shift_hours_per_day
        )
        event.downtime_days = max(1, labour_days, elapsed_days)

        end = event.start
        remaining_days = event.downtime_days - 1
        while remaining_days > 0:
            end += timedelta(days=1)
            if end not in blackout_dates:
                remaining_days -= 1
        event.end = end

        # A bundled task must not be pushed past its own hard limit.
        for task in event.tasks:
            if task.hard_limit_on and event.end > task.hard_limit_on:
                event.warnings.append(
                    f"{task.task_code} has a hard limit of {task.hard_limit_on} but "
                    f"this visit ends {event.end}; shorten the visit or split it out"
                )

        event.rationale.append(
            f"{event.total_man_hours:.1f} man-hours over {event.downtime_days} day(s) "
            f"at {self.settings.technicians_per_shift} technician(s) x "
            f"{self.settings.shift_hours_per_day:.0f}h."
        )

    def _flag_conflicts(self, result: PlanResult) -> None:
        """Warn when too many aircraft would be down on the same day."""
        limit = self.settings.max_concurrent_aircraft_down
        if limit <= 0:
            return

        down_by_day: dict[date, set[str]] = {}
        for event in result.events:
            cursor = event.start
            while cursor <= event.end:
                down_by_day.setdefault(cursor, set()).add(event.tail_number)
                cursor += timedelta(days=1)

        breaches = sorted(
            (day, tails) for day, tails in down_by_day.items() if len(tails) > limit
        )
        for day, tails in breaches[:10]:
            result.warnings.append(
                f"{day.isoformat()}: {len(tails)} aircraft scheduled down "
                f"({', '.join(sorted(tails))}) against a limit of {limit}"
            )
        if len(breaches) > 10:
            result.warnings.append(
                f"...and {len(breaches) - 10} further day(s) over the concurrency limit"
            )


def plan_fleet(
    per_aircraft: dict[str, tuple[list[DueForecast], dict[str, TaskSpec], str]],
    settings: MaintenanceSettings | None = None,
    *,
    today: date | None = None,
    horizon_days: int | None = None,
) -> PlanResult:
    """
    Plan several aircraft together.

    Each aircraft is planned independently, then the combined plan is
    checked for fleet-level conflicts - the point at which "everything is
    individually fine" turns into "nothing can fly on Tuesday".
    """
    planner = MaintenancePlanner(settings)
    combined = PlanResult()
    for _aircraft_id, (forecasts, tasks, station) in per_aircraft.items():
        result = planner.plan_aircraft(
            forecasts, tasks, station=station, today=today, horizon_days=horizon_days
        )
        combined.events.extend(result.events)
        combined.unplanned.extend(result.unplanned)
        combined.warnings.extend(w for w in result.warnings if "aircraft scheduled down" not in w)

    combined.events.sort(key=lambda e: (e.start, e.tail_number))
    planner._flag_conflicts(combined)
    return combined
