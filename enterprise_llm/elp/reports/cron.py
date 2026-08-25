"""
A small, dependency-free cron matcher for scheduled reports.

Standard five fields: ``minute hour day-of-month month day-of-week``.
Supports ``*``, ``n``, ``a-b``, ``*/n``, ``a-b/n``, comma lists, and the
usual three-letter month and weekday names.

The one rule people get wrong, and the reason this is tested rather than
assumed: **when both day-of-month and day-of-week are restricted, cron ORs
them.** ``0 6 1 * MON`` fires on the first of the month *and* on every
Monday, not only on Mondays that fall on the first. When either field is
``*`` the other simply applies.

Schedules are evaluated in the report's own timezone, so "every weekday at
06:00" means 06:00 where the department is, across daylight-saving changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
}

_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}

_FIELD_BOUNDS = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
_FIELD_NAMES = ["minute", "hour", "day-of-month", "month", "day-of-week"]

# How far ahead next_run will look before giving up. A schedule with no
# occurrence inside four years is a mistake, not a schedule.
_MAX_LOOKAHEAD_DAYS = 366 * 4


class CronError(ValueError):
    """The cron expression could not be parsed."""


def _parse_value(token: str, index: int) -> int:
    token = token.strip().lower()
    if index == 3 and token in _MONTHS:
        return _MONTHS[token]
    if index == 4 and token in _WEEKDAYS:
        return _WEEKDAYS[token]
    if not re.fullmatch(r"\d+", token):
        raise CronError(f"'{token}' is not a valid {_FIELD_NAMES[index]} value")
    return int(token)


def _parse_field(field: str, index: int) -> set[int]:
    low, high = _FIELD_BOUNDS[index]
    values: set[int] = set()

    for part in field.split(","):
        part = part.strip()
        if not part:
            raise CronError(f"empty {_FIELD_NAMES[index]} entry in '{field}'")

        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not re.fullmatch(r"\d+", step_text) or int(step_text) == 0:
                raise CronError(f"'{step_text}' is not a valid step in '{field}'")
            step = int(step_text)
            part = part.strip() or "*"

        if part == "*":
            start, end = low, high
        elif "-" in part[1:]:  # [1:] so a negative-looking token still errors
            start_text, _, end_text = part.partition("-")
            start = _parse_value(start_text, index)
            end = _parse_value(end_text, index)
        else:
            start = end = _parse_value(part, index)

        if start > end:
            raise CronError(
                f"range {start}-{end} is inverted in the {_FIELD_NAMES[index]} field"
            )
        if start < low or end > high:
            raise CronError(
                f"{_FIELD_NAMES[index]} value out of range: {part} "
                f"(permitted {low}-{high})"
            )
        values.update(range(start, end + 1, step))

    if index == 4 and 7 in values:
        # Both 0 and 7 mean Sunday.
        values.discard(7)
        values.add(0)
    return values


@dataclass(frozen=True)
class CronSchedule:
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool
    expression: str

    def matches(self, moment: datetime) -> bool:
        if moment.minute not in self.minutes:
            return False
        if moment.hour not in self.hours:
            return False
        if moment.month not in self.months:
            return False

        # Python: Monday=0..Sunday=6. Cron: Sunday=0..Saturday=6.
        cron_dow = (moment.weekday() + 1) % 7
        dom_hit = moment.day in self.days_of_month
        dow_hit = cron_dow in self.days_of_week

        if self.dom_restricted and self.dow_restricted:
            # The OR rule. Both restricted means either may trigger.
            return dom_hit or dow_hit
        if self.dom_restricted:
            return dom_hit
        if self.dow_restricted:
            return dow_hit
        return True

    def _day_matches(self, moment: datetime) -> bool:
        if moment.month not in self.months:
            return False
        cron_dow = (moment.weekday() + 1) % 7
        dom_hit = moment.day in self.days_of_month
        dow_hit = cron_dow in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            return dom_hit or dow_hit
        if self.dom_restricted:
            return dom_hit
        if self.dow_restricted:
            return dow_hit
        return True

    def next_after(self, after: datetime) -> datetime | None:
        """
        First firing strictly after ``after``.

        Skips whole non-matching days rather than stepping minute by minute,
        so a yearly schedule resolves in milliseconds.
        """
        cursor = (after + timedelta(minutes=1)).replace(second=0, microsecond=0)
        limit = after + timedelta(days=_MAX_LOOKAHEAD_DAYS)

        while cursor <= limit:
            if not self._day_matches(cursor):
                # Jump to the start of the next day.
                cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            for hour in sorted(self.hours):
                if hour < cursor.hour:
                    continue
                for minute in sorted(self.minutes):
                    if hour == cursor.hour and minute < cursor.minute:
                        continue
                    return cursor.replace(hour=hour, minute=minute)
            cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0)
        return None


def parse(expression: str) -> CronSchedule:
    """Parse a five-field cron expression or an ``@`` alias."""
    if not expression or not expression.strip():
        raise CronError("the schedule is empty")

    text = expression.strip().lower()
    text = _ALIASES.get(text, text)

    fields = text.split()
    if len(fields) != 5:
        raise CronError(
            f"expected 5 fields (minute hour day-of-month month day-of-week), "
            f"got {len(fields)}: '{expression}'"
        )

    parsed = [_parse_field(field, index) for index, field in enumerate(fields)]
    return CronSchedule(
        minutes=frozenset(parsed[0]),
        hours=frozenset(parsed[1]),
        days_of_month=frozenset(parsed[2]),
        months=frozenset(parsed[3]),
        days_of_week=frozenset(parsed[4]),
        dom_restricted=fields[2] != "*",
        dow_restricted=fields[4] != "*",
        expression=expression.strip(),
    )


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CronError(f"unknown timezone '{name}'") from exc


def is_due(
    expression: str,
    *,
    last_run: datetime | None,
    now: datetime,
    timezone_name: str = "UTC",
    grace_minutes: int = 90,
) -> bool:
    """
    Whether a schedule should fire at ``now``.

    Driven by the last firing rather than by an exact minute match, so a
    report is not silently skipped because the runner was busy, the host was
    rebooting, or the timer drifted. ``grace_minutes`` bounds how late a
    missed run may still be picked up - beyond that the occasion has passed
    and firing would be more confusing than helpful.
    """
    schedule = parse(expression)
    zone = resolve_timezone(timezone_name)
    local_now = now.astimezone(zone)

    if last_run is None:
        # Never run: fire if an occurrence fell within the grace window.
        window_start = local_now - timedelta(minutes=grace_minutes)
        occurrence = schedule.next_after(window_start)
        return occurrence is not None and occurrence <= local_now

    local_last = last_run.astimezone(zone)
    occurrence = schedule.next_after(local_last)
    if occurrence is None:
        return False
    return occurrence <= local_now


def next_run(
    expression: str, *, after: datetime, timezone_name: str = "UTC"
) -> datetime | None:
    """Next firing after ``after``, returned in UTC."""
    schedule = parse(expression)
    zone = resolve_timezone(timezone_name)
    local = schedule.next_after(after.astimezone(zone))
    return local.astimezone(ZoneInfo("UTC")) if local else None


def describe(expression: str) -> str:
    """A short human-readable rendering, for confirming a schedule."""
    schedule = parse(expression)
    parts = []
    if len(schedule.minutes) == 1 and len(schedule.hours) == 1:
        parts.append(
            f"at {next(iter(schedule.hours)):02d}:{next(iter(schedule.minutes)):02d}"
        )
    elif len(schedule.minutes) == 1:
        parts.append(f"at {next(iter(schedule.minutes))} minutes past the hour")
    else:
        parts.append(f"{len(schedule.minutes)} time(s) per hour")

    if schedule.dow_restricted:
        names = {v: k for k, v in _WEEKDAYS.items()}
        days = ", ".join(names[d] for d in sorted(schedule.days_of_week))
        parts.append(f"on {days}")
    if schedule.dom_restricted:
        days = ", ".join(str(d) for d in sorted(schedule.days_of_month))
        joiner = "or on day" if schedule.dow_restricted else "on day"
        parts.append(f"{joiner} {days} of the month")
    if len(schedule.months) < 12:
        names = {v: k for k, v in _MONTHS.items()}
        parts.append("in " + ", ".join(names[m] for m in sorted(schedule.months)))

    return " ".join(parts)
