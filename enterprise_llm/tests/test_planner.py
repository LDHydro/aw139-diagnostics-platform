"""Bundling due tasks into maintenance visits."""

from __future__ import annotations

from datetime import timedelta

from elp.maintenance.forecast import (
    ComplianceState,
    TaskSpec,
    UtilizationDay,
    estimate_utilization,
    forecast_aircraft,
)
from elp.maintenance.planner import MaintenancePlanner, plan_fleet
from tests.test_forecast import TODAY, aircraft, weekday_history


def _plan(tasks, compliance, settings, horizon_days=180):
    rate = estimate_utilization(weekday_history(), settings, today=TODAY)
    forecasts = forecast_aircraft(
        aircraft(), tasks, rate, compliance, settings=settings, today=TODAY,
        horizon_days=horizon_days,
    )
    specs = {t.id: t for t in tasks}
    planner = MaintenancePlanner(settings)
    return planner.plan_aircraft(
        forecasts, specs, station="SBJD", today=TODAY, horizon_days=horizon_days
    )


def test_nearby_tasks_are_bundled_into_one_visit(maintenance_settings):
    tasks = [
        TaskSpec(id="a", task_code="A", title="Task A", interval_calendar_days=100,
                 estimated_man_hours=4),
        TaskSpec(id="b", task_code="B", title="Task B", interval_calendar_days=105,
                 estimated_man_hours=4),
    ]
    compliance = {
        "a": ComplianceState(completed_on=TODAY - timedelta(days=70)),
        "b": ComplianceState(completed_on=TODAY - timedelta(days=70)),
    }
    result = _plan(tasks, compliance, maintenance_settings)

    assert len(result.events) == 1, "tasks days apart should share a visit"
    assert {t.task_code for t in result.events[0].tasks} == {"A", "B"}


def test_distant_tasks_get_separate_visits(maintenance_settings):
    tasks = [
        TaskSpec(id="a", task_code="A", title="Task A", interval_calendar_days=100),
        TaskSpec(id="b", task_code="B", title="Task B", interval_calendar_days=160),
    ]
    compliance = {
        "a": ComplianceState(completed_on=TODAY - timedelta(days=70)),
        "b": ComplianceState(completed_on=TODAY - timedelta(days=70)),
    }
    result = _plan(tasks, compliance, maintenance_settings)
    assert len(result.events) == 2


def test_a_visit_is_never_scheduled_in_the_past(maintenance_settings):
    """An overdue task is scheduled today, not on the date it was due."""
    tasks = [TaskSpec(id="a", task_code="LATE", title="Overdue", interval_calendar_days=30)]
    compliance = {"a": ComplianceState(completed_on=TODAY - timedelta(days=90))}

    result = _plan(tasks, compliance, maintenance_settings)

    assert result.events[0].start >= TODAY
    assert any("overdue" in w or "grounded" in w for w in result.events[0].warnings)


def test_downtime_reflects_labour_and_elapsed_time(maintenance_settings):
    """
    More technicians shorten labour-bound work but not a cure or a leak check.
    """
    labour_bound = TaskSpec(
        id="a", task_code="BIG", title="Heavy check",
        interval_calendar_days=100, estimated_man_hours=96, estimated_downtime_hours=8,
    )
    elapsed_bound = TaskSpec(
        id="b", task_code="CURE", title="Sealant cure",
        interval_calendar_days=100, estimated_man_hours=2, estimated_downtime_hours=48,
    )
    compliance_a = {"a": ComplianceState(completed_on=TODAY - timedelta(days=70))}
    compliance_b = {"b": ComplianceState(completed_on=TODAY - timedelta(days=70))}

    # 96 man-hours / (4 techs * 8h) = 3 days.
    assert _plan([labour_bound], compliance_a, maintenance_settings).events[0].downtime_days == 3
    # 48 elapsed hours / 8h shifts = 6 days regardless of manpower.
    assert _plan([elapsed_bound], compliance_b, maintenance_settings).events[0].downtime_days == 6


def test_bundling_is_refused_when_it_wastes_too_much_interval(maintenance_settings):
    """
    Pulling a task weeks early throws away interval that costs real money.

    A task due in 13 days whose interval is only 20 days would lose 65% of
    its life if bundled today, so it must get its own visit.
    """
    maintenance_settings.max_interval_waste_pct = 0.25
    tasks = [
        TaskSpec(id="a", task_code="NOW", title="Due now", interval_calendar_days=200),
        TaskSpec(id="b", task_code="SHORT", title="Short interval", interval_calendar_days=20),
    ]
    compliance = {
        "a": ComplianceState(completed_on=TODAY - timedelta(days=200)),
        "b": ComplianceState(completed_on=TODAY - timedelta(days=7)),
    }
    result = _plan(tasks, compliance, maintenance_settings)

    assert len(result.events) == 2, "a short-interval task must not be pulled forward"


def test_fleet_concurrency_breach_is_reported(maintenance_settings):
    """Individually fine, collectively grounding: three aircraft down at once."""
    maintenance_settings.max_concurrent_aircraft_down = 2
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)

    per_aircraft = {}
    for index, tail in enumerate(["PP-AAA", "PP-BBB", "PP-CCC"]):
        task = TaskSpec(
            id=f"t{index}", task_code=f"T{index}", title="Check",
            interval_calendar_days=100, estimated_man_hours=32,
        )
        state = aircraft(id=f"ac{index}", tail_number=tail)
        forecasts = forecast_aircraft(
            state, [task], rate,
            {task.id: ComplianceState(completed_on=TODAY - timedelta(days=70))},
            settings=maintenance_settings, today=TODAY,
        )
        per_aircraft[state.id] = (forecasts, {task.id: task}, "SBJD")

    result = plan_fleet(per_aircraft, maintenance_settings, today=TODAY, horizon_days=180)

    assert len(result.events) == 3
    assert any("aircraft scheduled down" in w for w in result.warnings)


def test_every_event_carries_a_rationale(maintenance_settings):
    """A planner nobody can interrogate is a planner nobody uses."""
    tasks = [TaskSpec(id="a", task_code="A", title="Task A", interval_calendar_days=100)]
    compliance = {"a": ComplianceState(completed_on=TODAY - timedelta(days=70))}

    event = _plan(tasks, compliance, maintenance_settings).events[0]

    assert event.rationale
    assert any("anchored on A" in line for line in event.rationale)
    assert event.tasks[0].reason


def test_tasks_with_no_projectable_date_are_surfaced_not_dropped(maintenance_settings):
    """An hours-based task on an idle aircraft must still be visible."""
    idle = estimate_utilization(
        [UtilizationDay(day=TODAY - timedelta(days=i)) for i in range(90)],
        maintenance_settings, today=TODAY,
    )
    task = TaskSpec(id="a", task_code="FH", title="Hours only", interval_flight_hours=100)
    forecasts = forecast_aircraft(
        aircraft(), [task], idle,
        {"a": ComplianceState(completed_on=TODAY, at_flight_hours=4820.0)},
        settings=maintenance_settings, today=TODAY,
    )
    result = MaintenancePlanner(maintenance_settings).plan_aircraft(
        forecasts, {"a": task}, today=TODAY
    )

    assert not result.events
    assert result.unplanned and result.unplanned[0]["task_code"] == "FH"
