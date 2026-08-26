"""
Structured report definitions, their compiler, and the saved-report importer.

The compiler's reason for existing is that a report compiled from a
catalogue-validated definition has **no injection surface at all** — every
identifier comes from the catalogue and every value is a bound parameter.
Free-form SQL requires the guard to prove a negative about somebody else's
string; this path never has the string.
"""

from __future__ import annotations

import json

import pytest

from elp.reports.import_saved import import_directory, parse_saved_report
from elp.reports.structured import (
    MAX_JOIN_TABLES,
    FieldRef,
    FilterSpec,
    JoinCondition,
    JoinSpec,
    ReportSpecError,
    SortSpec,
    StructuredReport,
    compile_report,
)


def base_report(**overrides) -> StructuredReport:
    defaults = dict(
        base_table="WorkRequest",
        fields=[FieldRef("WorkRequest", "WRNo", alias="Work Request")],
        sort=[SortSpec("WorkRequest", "WRNo", "asc")],
        row_limit=100,
    )
    defaults.update(overrides)
    return StructuredReport(**defaults)


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------

def test_a_simple_report_compiles_to_tsql(catalog):
    compiled = compile_report(base_report(), catalog)
    assert compiled.sql.startswith("SELECT TOP (100)")
    assert "[NAMISNNSS].[dbo].[WorkRequest] AS T0" in compiled.sql
    assert "T0.[WRNo] AS [Work Request]" in compiled.sql
    assert compiled.columns == ["Work Request"]


def test_compound_joins_emit_every_column(catalog):
    """
    AIRCRAFT joins WorkRequest on AssetKey *and* AssetSite.

    Emitting one of the two silently produces a cartesian product that reads
    as real data — the failure this whole path exists to prevent.
    """
    compiled = compile_report(
        base_report(
            fields=[
                FieldRef("WorkRequest", "WRNo"),
                FieldRef("AIRCRAFT", "TailNumber", alias="Tail"),
            ],
            joins=[
                JoinSpec(
                    "AIRCRAFT",
                    "left",
                    [
                        JoinCondition("WorkRequest", "AssetKey", "AIRCRAFT", "AssetKey"),
                        JoinCondition("WorkRequest", "AssetSite", "AIRCRAFT", "AssetSite"),
                    ],
                )
            ],
        ),
        catalog,
    )
    assert "T0.[AssetKey] = T1.[AssetKey]" in compiled.sql
    assert "T0.[AssetSite] = T1.[AssetSite]" in compiled.sql
    assert compiled.sql.count(" AND ") >= 1


def test_values_are_always_bound_parameters(catalog):
    compiled = compile_report(
        base_report(
            filters=[FilterSpec("WorkRequest", "StatusCd", "eq", value="' OR 1=1--")]
        ),
        catalog,
    )
    assert "WHERE T0.[StatusCd] = :f0" in compiled.sql
    assert compiled.parameters == {"f0": "' OR 1=1--"}
    # The dangerous string appears nowhere in the SQL text.
    assert "OR 1=1" not in compiled.sql


def test_relative_dates_stay_relative(catalog):
    """A saved report must still be correct when it runs next month."""
    compiled = compile_report(
        base_report(
            filters=[FilterSpec("WorkRequest", "StatusCd", "eq", value="OPN")],
            sort=[SortSpec("WorkRequest", "WRNo")],
        ),
        catalog,
    )
    assert ":f0" in compiled.sql


def test_day_count_filters_use_dateadd(catalog):
    compiled = compile_report(
        base_report(
            filters=[FilterSpec("WorkRequest", "StatusCd", "eq", value="x")],
        ),
        catalog,
    )
    assert "GETDATE()" not in compiled.sql  # no date filter in this one

    with_days = compile_report(
        base_report(filters=[FilterSpec("WorkRequest", "WRNo", "older_than_days", value=14)]),
        catalog,
    )
    assert "DATEADD(day, -:f0, GETDATE())" in with_days.sql
    assert with_days.parameters == {"f0": 14}


def test_in_filter_binds_every_value_separately(catalog):
    compiled = compile_report(
        base_report(
            filters=[FilterSpec("WorkRequest", "StatusCd", "in", values=["OPN", "CLS"])]
        ),
        catalog,
    )
    assert "IN (:f0_0, :f0_1)" in compiled.sql
    assert compiled.parameters == {"f0_0": "OPN", "f0_1": "CLS"}


def test_contains_wraps_the_value_not_the_sql(catalog):
    compiled = compile_report(
        base_report(filters=[FilterSpec("WorkRequest", "WRNo", "contains", value="123")]),
        catalog,
    )
    assert "LIKE :f0" in compiled.sql
    assert compiled.parameters == {"f0": "%123%"}


def test_aggregates_group_the_remaining_fields(catalog):
    compiled = compile_report(
        base_report(
            fields=[
                FieldRef("WorkRequest", "StatusCd", alias="Status"),
                FieldRef("WorkRequest", "WRNo", alias="Count", aggregate="count"),
            ],
            sort=[],
        ),
        catalog,
    )
    assert "COUNT(T0.[WRNo])" in compiled.sql
    assert "GROUP BY T0.[StatusCd]" in compiled.sql


def test_a_report_without_a_sort_gets_a_deterministic_one(catalog):
    """A report whose row order changes between runs is not a report."""
    compiled = compile_report(base_report(sort=[]), catalog)
    assert "ORDER BY T0.[WRNo] ASC" in compiled.sql
    assert any("no sort order" in w for w in compiled.warnings)


def test_the_row_limit_is_capped_by_settings(catalog):
    compiled = compile_report(base_report(row_limit=999_999), catalog, max_rows=5000)
    assert "SELECT TOP (5000)" in compiled.sql


# ----------------------------------------------------------------------
# Injection surface
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,report",
    [
        (
            "column",
            base_report(fields=[FieldRef("WorkRequest", "WRNo FROM sys.tables--")]),
        ),
        (
            "base table",
            base_report(base_table="WorkRequest; DROP TABLE x--"),
        ),
        (
            "joined table",
            base_report(
                joins=[
                    JoinSpec(
                        "x; DROP TABLE y--",
                        "left",
                        [JoinCondition("WorkRequest", "WRId", "x", "WRId")],
                    )
                ]
            ),
        ),
        (
            "aggregate",
            base_report(fields=[FieldRef("WorkRequest", "WRNo", aggregate="'; DROP--")]),
        ),
        (
            "filter operator",
            base_report(
                filters=[FilterSpec("WorkRequest", "WRNo", "; DROP TABLE x--", value=1)]
            ),
        ),
        (
            "sort column",
            base_report(sort=[SortSpec("WorkRequest", "WRNo DESC, (SELECT 1)")]),
        ),
    ],
)
def test_identifiers_cannot_carry_sql(name, report, catalog):
    """
    Every identifier is looked up in the catalogue and re-emitted from
    catalogue metadata, so a crafted name never reaches the SQL text.
    """
    with pytest.raises(ReportSpecError):
        compile_report(report, catalog)


def test_an_alias_is_stripped_rather_than_rejected(catalog):
    """
    Aliases are the one place a user string legitimately appears in the SQL,
    so they are reduced to a conservative character set.
    """
    compiled = compile_report(
        base_report(
            fields=[FieldRef("WorkRequest", "WRNo", alias="a] , (SELECT 1) AS [b")]
        ),
        catalog,
    )
    assert "SELECT 1" not in compiled.sql.replace("SELECT 1 AS b", "")
    assert compiled.sql.count("AS [") == 1


def test_a_join_without_conditions_is_refused(catalog):
    """An unconditioned join is a cartesian product."""
    with pytest.raises(ReportSpecError, match="cartesian"):
        compile_report(base_report(joins=[JoinSpec("AIRCRAFT", "left", [])]), catalog)


def test_join_count_is_capped(catalog):
    joins = [
        JoinSpec("AIRCRAFT", "left", [JoinCondition("WorkRequest", "AssetKey", "AIRCRAFT", "AssetKey")])
        for _ in range(MAX_JOIN_TABLES)
    ]
    with pytest.raises(ReportSpecError, match="at most"):
        compile_report(base_report(joins=joins), catalog)


def test_a_field_from_an_unjoined_table_is_refused(catalog):
    with pytest.raises(ReportSpecError, match="not in this report"):
        compile_report(
            base_report(fields=[FieldRef("AIRCRAFT", "TailNumber")]), catalog
        )


def test_a_misspelled_column_suggests_the_real_one(catalog):
    with pytest.raises(ReportSpecError, match="Did you mean"):
        compile_report(base_report(fields=[FieldRef("WorkRequest", "Status")]), catalog)


def test_joining_incompatible_column_types_is_refused(catalog):
    """Matching a text column to a number is their red warning, as an error."""
    with pytest.raises(ReportSpecError, match="different kinds of value"):
        compile_report(
            base_report(
                joins=[
                    JoinSpec(
                        "AIRCRAFT",
                        "left",
                        [
                            JoinCondition(
                                "WorkRequest", "AssetSite", "AIRCRAFT", "AssetKey"
                            )
                        ],
                    )
                ]
            ),
            catalog,
        )


# ----------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------

def test_a_definition_survives_a_json_round_trip(catalog):
    original = base_report(
        fields=[FieldRef("WorkRequest", "WRNo", alias="WR")],
        filters=[FilterSpec("WorkRequest", "StatusCd", "in", values=["OPN"])],
    )
    restored = StructuredReport.from_dict(json.loads(json.dumps(original.to_dict())))
    assert compile_report(restored, catalog).sql == compile_report(original, catalog).sql


# ----------------------------------------------------------------------
# Importing saved reports
# ----------------------------------------------------------------------

def test_a_saved_report_maps_onto_a_structured_definition():
    imported = parse_saved_report(
        {
            "name": "Open Work Requests",
            "table": "WorkRequest",
            "fields": [
                {"column": "WRNo", "label": "Work Request"},
                {"table": "AIRCRAFT", "column": "TailNumber"},
            ],
            "joins": [
                {
                    "table": "AIRCRAFT",
                    "type": "LEFT",
                    "on": [{"attachTo": "WorkRequest", "attachColumn": "AssetKey",
                            "targetColumn": "AssetKey"}],
                }
            ],
            "filters": [{"column": "StatusCd", "operator": "=", "value": "OPN"}],
            "sort": [{"column": "WRNo", "dir": "DESC"}],
            "rowLimit": 250,
        },
        source_file="open-wrs.json",
    )

    assert imported.name == "Open Work Requests"
    assert imported.report.base_table == "WorkRequest"
    assert len(imported.report.fields) == 2
    assert imported.report.fields[0].alias == "Work Request"
    assert imported.report.joins[0].kind == "left"
    assert imported.report.filters[0].op == "eq"
    assert imported.report.sort[0].direction == "desc"
    assert imported.report.row_limit == 250


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("=", "eq"), ("equals", "eq"), ("<>", "ne"), (">=", "gte"),
        ("contains", "contains"), ("startsWith", "starts_with"),
        ("isNull", "is_null"), ("in", "in"), ("between", "between"),
        ("lastNDays", "last_n_days"), ("openLongerThan", "older_than_days"),
    ],
)
def test_operator_spellings_are_normalised(raw, expected):
    imported = parse_saved_report(
        {
            "table": "WorkRequest",
            "fields": [{"column": "WRNo"}],
            "filters": [{"column": "StatusCd", "op": raw, "value": 1}],
        }
    )
    assert imported.report.filters[0].op == expected


def test_an_unrecognised_operator_is_reported_not_guessed():
    """
    Guessing an operator changes what the report says.

    Reporting it lets a person decide, which is the only safe option.
    """
    imported = parse_saved_report(
        {
            "table": "WorkRequest",
            "fields": [{"column": "WRNo"}],
            "filters": [{"column": "StatusCd", "op": "soundsLike", "value": 1}],
        }
    )
    assert imported.report.filters == []
    assert any("soundsLike" in u for u in imported.unmapped)


def test_unknown_top_level_keys_are_surfaced():
    """A key we do not consume may be a feature this importer misses."""
    imported = parse_saved_report(
        {"table": "WorkRequest", "fields": [{"column": "WRNo"}], "chartConfig": {}}
    )
    assert any("chartConfig" in u for u in imported.unmapped)


def test_a_file_without_a_base_table_is_refused():
    with pytest.raises(ValueError, match="base table"):
        parse_saved_report({"name": "x", "fields": []})


def test_importing_a_directory_compiles_against_the_catalogue(tmp_path, catalog):
    good = {
        "name": "Good", "table": "WorkRequest",
        "fields": [{"column": "WRNo"}], "sort": [{"column": "WRNo"}],
    }
    bad = {
        "name": "Bad", "table": "WorkRequest",
        "fields": [{"column": "NoSuchColumn"}],
    }
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    results = {r.name: r for r in import_directory(tmp_path, catalog)}

    assert results["Good"].compiles is True
    assert "SELECT TOP" in results["Good"].sql_preview
    # A report that cannot compile is reported now, not on its first schedule.
    assert results["Bad"].compiles is False
    assert "NoSuchColumn" in results["Bad"].compile_error
    assert "broken" not in results
