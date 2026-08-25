"""Cron parsing and scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from elp.reports.cron import CronError, describe, is_due, next_run, parse

UTC = ZoneInfo("UTC")


def dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression",
    ["0 6 * * *", "*/15 * * * *", "0 6 * * 1-5", "0 0 1 1 *", "30 5 * * MON",
     "0 8 1,15 * *", "@daily", "@weekly", "0 6-18/2 * * *"],
)
def test_valid_expressions_parse(expression):
    assert parse(expression)


@pytest.mark.parametrize(
    "expression",
    ["", "0 6 * *", "0 6 * * * *", "99 * * * *", "0 25 * * *", "* * 32 * *",
     "* * * 13 *", "* * * * 9", "0 6 5-1 * *", "*/0 * * * *", "abc * * * *"],
)
def test_invalid_expressions_are_rejected(expression):
    with pytest.raises(CronError):
        parse(expression)


def test_aliases_expand():
    assert parse("@daily").hours == parse("0 0 * * *").hours
    assert parse("@weekly").days_of_week == frozenset({0})


def test_sunday_is_both_zero_and_seven():
    assert parse("0 0 * * 7").days_of_week == frozenset({0})
    assert parse("0 0 * * 0").days_of_week == parse("0 0 * * 7").days_of_week


def test_names_are_accepted_for_days_and_months():
    assert parse("0 6 * * mon").days_of_week == frozenset({1})
    assert parse("0 6 * jan *").months == frozenset({1})


# ----------------------------------------------------------------------
# The day-of-month / day-of-week OR rule
# ----------------------------------------------------------------------

def test_both_day_fields_restricted_means_or():
    """
    `0 6 1 * MON` fires on the 1st AND on every Monday.

    This is the rule people get wrong. Reading it as AND would make a report
    fire roughly once a year instead of weekly.
    """
    schedule = parse("0 6 1 * MON")
    assert schedule.matches(dt(2026, 9, 1, 6, 0))    # a Tuesday, but the 1st
    assert schedule.matches(dt(2026, 9, 7, 6, 0))    # a Monday, not the 1st
    assert not schedule.matches(dt(2026, 9, 8, 6, 0))  # neither


def test_only_day_of_month_restricted():
    schedule = parse("0 6 15 * *")
    assert schedule.matches(dt(2026, 9, 15, 6, 0))
    assert not schedule.matches(dt(2026, 9, 16, 6, 0))


def test_only_day_of_week_restricted():
    schedule = parse("0 6 * * 5")  # Friday
    assert schedule.matches(dt(2026, 8, 28, 6, 0))
    assert not schedule.matches(dt(2026, 8, 27, 6, 0))


# ----------------------------------------------------------------------
# Next occurrence
# ----------------------------------------------------------------------

def test_next_run_is_strictly_after_the_reference():
    base = dt(2026, 8, 25, 6, 0)
    assert next_run("0 6 * * *", after=base) == dt(2026, 8, 26, 6, 0)


def test_next_run_skips_to_the_next_weekday():
    # 2026-08-28 is a Friday; the next weekday run is Monday the 31st.
    assert next_run("0 6 * * 1-5", after=dt(2026, 8, 28, 7, 0)) == dt(2026, 8, 31, 6, 0)


def test_next_run_crosses_a_year_boundary():
    assert next_run("0 0 1 1 *", after=dt(2026, 6, 1, 0, 0)) == dt(2027, 1, 1, 0, 0)


def test_next_run_handles_a_step_within_the_hour():
    assert next_run("*/15 * * * *", after=dt(2026, 8, 25, 10, 31)) == dt(2026, 8, 25, 10, 45)


def test_next_run_is_returned_in_utc_for_a_local_schedule():
    """06:00 in São Paulo (UTC-3) is 09:00 UTC."""
    result = next_run(
        "0 6 * * *", after=dt(2026, 8, 25, 0, 0), timezone_name="America/Sao_Paulo"
    )
    assert result.hour == 9


def test_unknown_timezone_is_rejected():
    with pytest.raises(CronError, match="unknown timezone"):
        next_run("0 6 * * *", after=dt(2026, 8, 25, 0, 0), timezone_name="Mars/Olympus")


# ----------------------------------------------------------------------
# Due detection
# ----------------------------------------------------------------------

def test_a_missed_run_is_picked_up_late():
    """
    The runner may have been busy or the host rebooting.

    Driving from the last run rather than an exact minute match is what stops
    a report being silently skipped.
    """
    assert is_due(
        "0 6 * * *",
        last_run=dt(2026, 8, 24, 6, 0),
        now=dt(2026, 8, 25, 7, 5),
    )


def test_a_report_already_run_today_is_not_due_again():
    assert not is_due(
        "0 6 * * *",
        last_run=dt(2026, 8, 25, 6, 0),
        now=dt(2026, 8, 25, 7, 5),
    )


def test_a_never_run_report_fires_only_within_the_grace_window():
    # An occurrence 10 minutes ago is picked up...
    assert is_due("0 6 * * *", last_run=None, now=dt(2026, 8, 25, 6, 10))
    # ...but one from many hours ago has passed.
    assert not is_due(
        "0 6 * * *", last_run=None, now=dt(2026, 8, 25, 23, 0), grace_minutes=90
    )


def test_a_schedule_not_yet_reached_is_not_due():
    assert not is_due(
        "0 6 * * *",
        last_run=dt(2026, 8, 25, 6, 0),
        now=dt(2026, 8, 25, 6, 30) + timedelta(0),
    )


# ----------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------

def test_describe_is_human_readable():
    assert "06:00" in describe("0 6 * * *")
    assert "mon" in describe("0 6 * * 1")
    assert "or on day" in describe("0 6 1 * MON")
