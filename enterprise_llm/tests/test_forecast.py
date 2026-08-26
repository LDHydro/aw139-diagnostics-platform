"""Utilisation forecasting and due-date projection."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from elp.maintenance.forecast import (
    AircraftState,
    Basis,
    ComplianceState,
    DeferralState,
    DueStatus,
    TaskSpec,
    UtilizationDay,
    estimate_utilization,
    forecast_aircraft,
    forecast_task,
)

TODAY = date(2026, 8, 21)


def weekday_history(days: int = 90, hours: float = 2.5) -> list[UtilizationDay]:
    """Flies Monday to Friday, idle at weekends - a common pattern."""
    return [
        UtilizationDay(day=day, flight_hours=hours, cycles=4, landings=6)
        for day in (TODAY - timedelta(days=days - i) for i in range(days))
        if day.weekday() < 5
    ]


def aircraft(**overrides) -> AircraftState:
    defaults = dict(
        id="ac1",
        tail_number="PP-ABC",
        model="AW139",
        flight_hours=4820.0,
        cycles=7300,
        landings=11200,
        as_of=TODAY,
        in_service_since=date(2019, 3, 1),
    )
    defaults.update(overrides)
    return AircraftState(**defaults)


# ----------------------------------------------------------------------
# Utilisation
# ----------------------------------------------------------------------

def test_non_flying_days_are_counted(maintenance_settings):
    """
    Logbook feeds usually emit rows only for days flown.

    Averaging just those rows would say 2.5 FH/day for an aircraft that
    actually averages 2.5 * 5/7 = 1.79.  Over a 600-hour interval that is a
    forecast error of roughly two months.
    """
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    assert rate.source == "history"
    assert 1.6 < rate.daily_flight_hours < 2.0


def test_weekly_pattern_does_not_shift_the_estimate_by_day_of_week(maintenance_settings):
    """Asking on a Sunday must not give a different answer than a Wednesday."""
    estimates = []
    for offset in range(7):
        as_of = TODAY - timedelta(days=offset)
        history = [
            UtilizationDay(day=day, flight_hours=2.5, cycles=4, landings=6)
            for day in (as_of - timedelta(days=90 - i) for i in range(90))
            if day.weekday() < 5
        ]
        estimates.append(
            estimate_utilization(history, maintenance_settings, today=as_of).daily_flight_hours
        )

    spread = max(estimates) - min(estimates)
    assert spread < 0.15, f"day-of-week sensitivity too high: {estimates}"


def test_no_history_falls_back_to_the_default_rate(maintenance_settings):
    rate = estimate_utilization([], maintenance_settings, today=TODAY)
    assert rate.source == "default"
    assert rate.daily_flight_hours == maintenance_settings.default_daily_flight_hours
    # A guess must not be reported as a confident forecast.
    assert rate.confidence < 0.3


def test_erratic_flying_lowers_confidence(maintenance_settings):
    steady = [
        UtilizationDay(day=TODAY - timedelta(days=i), flight_hours=2.0) for i in range(90)
    ]
    erratic = [
        UtilizationDay(day=TODAY - timedelta(days=i), flight_hours=12.0 if i % 14 == 0 else 0.0)
        for i in range(90)
    ]
    assert (
        estimate_utilization(steady, maintenance_settings, today=TODAY).confidence
        > estimate_utilization(erratic, maintenance_settings, today=TODAY).confidence
    )


# ----------------------------------------------------------------------
# Due projection
# ----------------------------------------------------------------------

def test_earliest_interval_drives_the_due_date(maintenance_settings):
    """600 FH or 12 months, whichever comes first."""
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(
        id="t1",
        task_code="600H",
        title="600-hour inspection",
        interval_flight_hours=600,
        interval_calendar_days=365,
    )
    # Last done 290 days ago at 4400 FH: 180 FH remain (~100 days at this
    # rate), but only 75 calendar days remain. Calendar must win.
    compliance = ComplianceState(
        completed_on=TODAY - timedelta(days=290), at_flight_hours=4400.0
    )

    result = forecast_task(aircraft(), task, rate, compliance, settings=maintenance_settings, today=TODAY)

    assert result.driving_basis is Basis.CALENDAR
    assert result.due_on == compliance.completed_on + timedelta(days=365)
    assert len(result.projections) == 2


def test_flight_hours_drive_when_they_run_out_first(maintenance_settings):
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(
        id="t2", task_code="100H", title="100-hour", interval_flight_hours=100
    )
    compliance = ComplianceState(
        completed_on=TODAY - timedelta(days=30), at_flight_hours=4780.0
    )

    result = forecast_task(aircraft(), task, rate, compliance, settings=maintenance_settings, today=TODAY)

    assert result.driving_basis is Basis.FLIGHT_HOURS
    # 60 FH remain at ~1.8 FH/day -> roughly a month out.
    assert 25 <= (result.due_on - TODAY).days <= 40


def test_tolerance_extends_the_hard_limit(maintenance_settings):
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(
        id="t3",
        task_code="ANN",
        title="Annual",
        interval_calendar_days=365,
        tolerance_calendar_days=15,
    )
    compliance = ComplianceState(completed_on=TODAY - timedelta(days=300))

    result = forecast_task(aircraft(), task, rate, compliance, settings=maintenance_settings, today=TODAY)

    assert result.hard_limit_on == result.due_on + timedelta(days=15)


def test_overdue_and_grounded_are_distinguished(maintenance_settings):
    """Past due is a planning problem; past the hard limit grounds the aircraft."""
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(
        id="t4",
        task_code="ANN",
        title="Annual",
        interval_calendar_days=365,
        tolerance_calendar_days=15,
    )

    overdue = forecast_task(
        aircraft(), task, rate,
        ComplianceState(completed_on=TODAY - timedelta(days=370)),
        settings=maintenance_settings, today=TODAY,
    )
    grounded = forecast_task(
        aircraft(), task, rate,
        ComplianceState(completed_on=TODAY - timedelta(days=400)),
        settings=maintenance_settings, today=TODAY,
    )

    assert overdue.status is DueStatus.OVERDUE
    assert grounded.status is DueStatus.GROUNDED


def test_a_task_never_done_is_measured_from_entry_into_service(maintenance_settings):
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(id="t5", task_code="NEW", title="Never done", interval_calendar_days=365)

    result = forecast_task(aircraft(), task, rate, None, settings=maintenance_settings, today=TODAY)

    assert result.last_completed_on is None
    assert any("entry into service" in note for note in result.notes)


def test_calendar_tasks_are_certain_regardless_of_utilisation(maintenance_settings):
    """A 12-month check needs no flying-rate forecast, so confidence is full."""
    poor_rate = estimate_utilization([], maintenance_settings, today=TODAY)
    assert poor_rate.confidence < 0.3

    task = TaskSpec(id="t6", task_code="CAL", title="Calendar", interval_calendar_days=180)
    result = forecast_task(
        aircraft(), task, poor_rate,
        ComplianceState(completed_on=TODAY - timedelta(days=30)),
        settings=maintenance_settings, today=TODAY,
    )
    assert result.confidence == pytest.approx(1.0)


def test_a_grounded_aircraft_gets_no_flight_hour_due_date(maintenance_settings):
    """With zero flying, an hours-based task is never consumed."""
    idle = estimate_utilization(
        [UtilizationDay(day=TODAY - timedelta(days=i)) for i in range(90)],
        maintenance_settings, today=TODAY,
    )
    task = TaskSpec(id="t7", task_code="FH", title="Hours only", interval_flight_hours=100)

    result = forecast_task(
        aircraft(), task, idle,
        ComplianceState(completed_on=TODAY - timedelta(days=10), at_flight_hours=4800.0),
        settings=maintenance_settings, today=TODAY,
    )
    assert result.due_on is None
    assert any("does not consume" in note for note in result.notes)


def test_deferral_extends_the_hard_limit(maintenance_settings):
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    task = TaskSpec(id="t8", task_code="DEF", title="Deferrable", interval_calendar_days=365)
    compliance = ComplianceState(completed_on=TODAY - timedelta(days=360))
    deferral = DeferralState(expires_on=TODAY + timedelta(days=45), reason="parts on order")

    result = forecast_task(
        aircraft(), task, rate, compliance, deferral,
        settings=maintenance_settings, today=TODAY,
    )

    assert result.deferred_until == deferral.expires_on
    assert result.hard_limit_on == deferral.expires_on
    assert any("deferral in force" in note for note in result.notes)


def test_fleet_forecast_is_sorted_and_respects_the_horizon(maintenance_settings):
    rate = estimate_utilization(weekday_history(), maintenance_settings, today=TODAY)
    tasks = [
        TaskSpec(id="a", task_code="SOON", title="Soon", interval_calendar_days=30),
        TaskSpec(id="b", task_code="LATER", title="Later", interval_calendar_days=300),
        TaskSpec(id="c", task_code="FAR", title="Far", interval_calendar_days=2000),
    ]
    compliance = {t.id: ComplianceState(completed_on=TODAY) for t in tasks}

    results = forecast_aircraft(
        aircraft(), tasks, rate, compliance,
        settings=maintenance_settings, today=TODAY, horizon_days=365,
    )

    assert [r.task_code for r in results] == ["SOON", "LATER"]
    assert results[0].due_on < results[1].due_on
