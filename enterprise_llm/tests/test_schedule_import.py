"""Importing the customer-supplied standard maintenance schedule."""

from __future__ import annotations

import csv

import pytest

from elp.maintenance.schedule_io import (
    _build_column_map,
    parse_row,
    read_rows,
    validate,
)


def _map_and_parse(headers: list[str], row: dict) -> dict:
    return parse_row(row, _build_column_map(headers))


def test_column_aliases_are_matched_across_layouts():
    """Operators name their columns differently; all of these must land."""
    for headers, expected in [
        (["Task Code", "Task Description", "Interval FH"], {"task_code", "title", "interval_flight_hours"}),
        (["TASKNO", "Title", "FH"], {"task_code", "title", "interval_flight_hours"}),
        (["Card No.", "Description", "Flight Hours"], {"task_code", "title", "interval_flight_hours"}),
        (["Reference", "Name", "Hrs"], {"task_code", "title", "interval_flight_hours"}),
    ]:
        assert set(_build_column_map(headers).values()) >= expected, headers


def test_months_are_converted_to_days():
    parsed = _map_and_parse(
        ["Task", "Description", "Months"],
        {"Task": "ANN", "Description": "Annual inspection", "Months": "12"},
    )
    assert parsed["interval_calendar_days"] == 360


def test_explicit_days_win_over_months():
    parsed = _map_and_parse(
        ["Task", "Description", "Months", "Calendar Days"],
        {"Task": "X", "Description": "Y", "Months": "12", "Calendar Days": "365"},
    )
    assert parsed["interval_calendar_days"] == 365


def test_messy_numeric_cells_are_parsed():
    parsed = _map_and_parse(
        ["Task", "Description", "Interval FH", "Man Hours"],
        {"Task": "A", "Description": "B", "Interval FH": "1,200 hrs", "Man Hours": "≈ 12.5"},
    )
    assert parsed["interval_flight_hours"] == 1200.0
    assert parsed["estimated_man_hours"] == 12.5


def test_boolean_cells_accept_operator_spellings():
    for value, expected in [("Y", True), ("yes", True), ("X", True), ("N", False),
                            ("no", False), ("", False), ("-", False)]:
        parsed = _map_and_parse(
            ["Task", "Description", "Hangar"],
            {"Task": "A", "Description": "B", "Hangar": value},
        )
        assert parsed["requires_hangar"] is expected, value


def test_deferrable_defaults_to_true_when_the_column_is_blank():
    """Absence of a 'can defer' column must not silently forbid deferral."""
    parsed = _map_and_parse(["Task", "Description"], {"Task": "A", "Description": "B"})
    assert parsed["can_be_deferred"] is True


def test_applicability_lists_are_split_and_all_means_empty():
    parsed = _map_and_parse(
        ["Task", "Description", "Model", "Configuration"],
        {"Task": "A", "Description": "B", "Model": "AW139; AW169", "Configuration": "ALL"},
    )
    assert parsed["applicable_models"] == ["AW139", "AW169"]
    assert parsed["applicable_configurations"] == []


def test_rows_without_any_interval_are_rejected():
    """A recurring task with no interval is a limit nobody would be tracking."""
    parsed = _map_and_parse(
        ["Task", "Description"], {"Task": "A", "Description": "Inspect something"}
    )
    problem = validate(parsed)
    assert problem is not None and "interval" in problem


def test_one_time_directives_are_allowed_without_an_interval():
    parsed = _map_and_parse(
        ["Task", "Description", "Type"],
        {"Task": "AD-2024-11", "Description": "One-time inspection", "Type": "AD"},
    )
    assert validate(parsed) is None


def test_rows_missing_a_code_or_title_are_rejected():
    assert validate(_map_and_parse(["Task", "Description"], {"Task": "", "Description": "x"}))
    assert validate(_map_and_parse(["Task", "Description"], {"Task": "A", "Description": ""}))


def test_csv_round_trip(tmp_path):
    path = tmp_path / "schedule.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Task Code", "Task Description", "ATA", "Interval FH", "Months", "Man Hours"])
        writer.writerow(["AW139-100H", "100-hour inspection", "05", "100", "", "12"])
        writer.writerow(["AW139-ANN", "Annual inspection", "05", "", "12", "40"])

    rows, headers = read_rows(path)
    assert len(rows) == 2

    column_map = _build_column_map(headers)
    first = parse_row(rows[0], column_map)
    second = parse_row(rows[1], column_map)

    assert first["interval_flight_hours"] == 100.0
    assert first["estimated_man_hours"] == 12.0
    assert second["interval_calendar_days"] == 360
    assert validate(first) is None and validate(second) is None


def test_json_schedule_is_supported(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(
        '{"tasks": [{"task_code": "X1", "title": "Check", "interval_calendar_days": 90}]}',
        encoding="utf-8",
    )
    rows, headers = read_rows(path)
    assert len(rows) == 1
    parsed = parse_row(rows[0], _build_column_map(headers))
    assert parsed["task_code"] == "X1"
    assert validate(parsed) is None


def test_unsupported_format_is_refused(tmp_path):
    path = tmp_path / "schedule.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schedule format"):
        read_rows(path)
