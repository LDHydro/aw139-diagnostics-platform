"""
Report approval, access control and rendering.

The approval model is the safety property worth testing hardest: a report
that runs unattended is a standing instruction to a production database, and
the whole point of binding approval to a fingerprint is that an edit cannot
quietly inherit an earlier review.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elp.auth.principal import Principal, Scope, scopes_for_roles
from elp.latex.render import validate_source
from elp.models import ReportDefinition, ReportStatus
from elp.reports.datasource import QueryResult
from elp.reports.render import to_csv, to_html, to_latex, to_markdown
from elp.reports.service import can_access, fingerprint, is_approved


def definition(**overrides) -> ReportDefinition:
    defaults = dict(
        name="Open work orders",
        request_text="all open work orders by tail",
        query="SELECT tail_number FROM work_orders WHERE closed_at IS NULL LIMIT 100",
        status=ReportStatus.DRAFT.value,
    )
    defaults.update(overrides)
    return ReportDefinition(**defaults)


def principal(role: str = "planner", groups: list[str] | None = None) -> Principal:
    return Principal(
        subject="user@corp",
        roles=[role],
        groups=groups or [],
        scopes=scopes_for_roles([role]),
    )


# ----------------------------------------------------------------------
# Fingerprinting
# ----------------------------------------------------------------------

def test_fingerprint_ignores_incidental_whitespace():
    """Reformatting a query is not a change of meaning."""
    a = fingerprint("SELECT a FROM t WHERE x = 1")
    b = fingerprint("SELECT   a\n  FROM t\nWHERE x = 1")
    assert a == b


def test_fingerprint_changes_when_the_query_changes():
    assert fingerprint("SELECT a FROM t") != fingerprint("SELECT b FROM t")
    # Even a limit change matters: it changes what the report says.
    assert fingerprint("SELECT a FROM t LIMIT 10") != fingerprint(
        "SELECT a FROM t LIMIT 1000"
    )


# ----------------------------------------------------------------------
# Approval
# ----------------------------------------------------------------------

def test_a_draft_is_not_approved():
    assert not is_approved(definition())


def test_an_approved_report_with_a_matching_hash_is_approved():
    report = definition(status=ReportStatus.APPROVED.value)
    report.approved_query_hash = fingerprint(report.query)
    report.approved_by = "manager@corp"
    report.approved_at = datetime.now(UTC)
    assert is_approved(report)


def test_editing_the_query_invalidates_an_existing_approval():
    """
    The core safety property.

    Approval binds to the exact query text. Editing it after approval must
    not inherit that review, because a schedule is precisely where an
    unreviewed query would do the most damage.
    """
    report = definition(status=ReportStatus.APPROVED.value)
    report.approved_query_hash = fingerprint(report.query)
    assert is_approved(report)

    report.query = "SELECT * FROM work_orders"
    assert not is_approved(report)


def test_a_status_of_approved_without_a_hash_is_not_trusted():
    """Guards against a row edited directly in the database."""
    report = definition(status=ReportStatus.APPROVED.value, approved_query_hash="")
    assert not is_approved(report)


def test_a_forged_status_with_a_stale_hash_is_not_trusted():
    report = definition(status=ReportStatus.APPROVED.value)
    report.approved_query_hash = fingerprint("SELECT something_else FROM t")
    assert not is_approved(report)


def test_approving_requires_the_approve_scope():
    assert not principal("planner").has(Scope.REPORTS_APPROVE)
    assert principal("planner").has(Scope.REPORTS)
    assert principal("maintenance_manager").has(Scope.REPORTS_APPROVE)
    assert principal("engineer").has(Scope.REPORTS_APPROVE)


def test_operations_can_request_reports_without_being_able_to_schedule_them():
    """A reader may ask for and run reports, but not put one on a timer."""
    reader = principal("reader")
    assert reader.has(Scope.REPORTS)
    assert not reader.has(Scope.REPORTS_APPROVE)


# ----------------------------------------------------------------------
# Access control
# ----------------------------------------------------------------------

def test_a_report_with_no_groups_is_open_to_any_report_user():
    assert can_access(principal("reader"), definition(allowed_groups=[]))


def test_a_restricted_report_needs_a_matching_group():
    report = definition(allowed_groups=["Finance"])
    assert not can_access(principal("reader", groups=["AW139-Line"]), report)
    assert can_access(principal("reader", groups=["Finance"]), report)


def test_group_matching_is_case_insensitive():
    report = definition(allowed_groups=["AW139-Engineering"])
    assert can_access(principal("reader", groups=["aw139-engineering"]), report)


def test_admins_reach_every_report():
    report = definition(allowed_groups=["Finance"])
    assert can_access(principal("admin"), report)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

@pytest.fixture
def result() -> QueryResult:
    return QueryResult(
        columns=["tail_number", "wo_number", "status"],
        rows=[
            ["PP-ABC", "WO-1001", "OPEN"],
            ["PP-DEF", "WO-1002", "AOG | urgent"],
            ["PP-GHI", "WO_1003", None],
        ],
        row_count=3,
        duration_ms=42.0,
        redacted_columns=["technician_name"],
        warnings=["no row limit was specified, so LIMIT 5000 was added"],
    )


def test_markdown_escapes_pipes_inside_values(result):
    """An unescaped pipe silently breaks the table into the wrong columns."""
    rendered = to_markdown(result, title="Work Orders")
    assert r"AOG \| urgent" in rendered


def test_markdown_reports_redaction_and_warnings(result):
    rendered = to_markdown(result, title="Work Orders")
    assert "technician_name" in rendered
    assert "LIMIT 5000" in rendered


def test_csv_round_trips_nulls_as_empty(result):
    lines = to_csv(result).strip().splitlines()
    assert lines[0] == "tail_number,wo_number,status"
    assert lines[3].endswith(",")


def test_html_escapes_values_and_titles(result):
    rendered = to_html(result, title="<script>alert(1)</script>")
    assert "<script>alert" not in rendered
    assert "&lt;script&gt;" in rendered


def test_latex_escapes_special_characters(result):
    rendered = to_latex(result, title="Work Orders & Defects")
    assert r"WO\_1003" in rendered
    assert r"\&" in rendered


def test_generated_latex_passes_the_compile_sandbox(result):
    """
    Rendering reuses the platform's LaTeX service, so its output has to
    satisfy the same sandbox as anything else compiled here.
    """
    validate_source(to_latex(result, title="Work Orders", narrative="Two open."))


def test_an_empty_result_renders_without_error():
    empty = QueryResult(columns=["a"], rows=[], row_count=0)
    assert "No rows returned" in to_markdown(empty, title="Nothing")
    assert "No rows returned" in to_html(empty, title="Nothing")
    validate_source(to_latex(empty, title="Nothing"))


# ----------------------------------------------------------------------
# Read-only guarantees
# ----------------------------------------------------------------------

def _datasource_ast():
    import ast
    from pathlib import Path

    import elp.reports.datasource as module

    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_namis_datasource_never_commits():
    """
    Architectural guard.

    Nothing in the NAMIS path may commit a transaction. This is asserted
    against the source rather than trusted to review, because a future
    change that adds a `commit()` here would be silent, plausible-looking,
    and would turn a read-only integration into a writable one.
    """
    import ast

    commits = [
        node
        for node in ast.walk(_datasource_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit"
    ]
    assert not commits, (
        "elp/reports/datasource.py contains a commit() call. The NAMIS "
        "connection is read-only and must never commit."
    )


def test_the_namis_datasource_rolls_back_after_reading():
    """A read-only query still opens a transaction; it must be closed."""
    import ast

    rollbacks = [
        node
        for node in ast.walk(_datasource_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rollback"
    ]
    assert rollbacks, "the NAMIS query path should roll back rather than commit"


def test_the_connection_is_read_only_at_the_server():
    """
    The strongest layer that application code controls.

    `default_transaction_read_only` applies to every transaction on the
    connection, so a code path that forgets the per-transaction SET still
    cannot write.
    """
    from elp.reports.datasource import SqlNamisSource

    source = SqlNamisSource()
    asyncpg_args = source._connect_args(
        "postgresql+asyncpg://u:p@host/db"
    )
    assert (
        asyncpg_args["server_settings"]["default_transaction_read_only"] == "on"
    )

    psycopg_args = source._connect_args("postgresql+psycopg://u:p@host/db")
    assert "default_transaction_read_only=on" in psycopg_args["options"]


def test_namis_traffic_is_identifiable_to_a_dba():
    """pg_stat_activity should show who is querying."""
    from elp.reports.datasource import SqlNamisSource

    args = SqlNamisSource()._connect_args("postgresql+asyncpg://u:p@host/db")
    assert args["server_settings"]["application_name"] == "elp-reports"


def test_no_write_statement_can_reach_the_database(open_settings=None):
    """
    End-to-end statement of the property, at the layer that filters.

    Every one of these is refused before a connection is even opened.
    """
    from elp.config import ReportSettings
    from elp.reports.sqlguard import UnsafeQuery, validate

    settings = ReportSettings(sql_dialect="postgresql")
    writes = [
        "INSERT INTO work_orders (id) VALUES (1)",
        "UPDATE work_orders SET status = 'CLOSED'",
        "DELETE FROM work_orders",
        "TRUNCATE work_orders",
        "DROP TABLE work_orders",
        "ALTER TABLE work_orders ADD COLUMN x int",
        "CREATE TABLE evil (x int)",
        "SELECT * INTO evil FROM work_orders",
        "SELECT 1; UPDATE work_orders SET status = 'X'",
        "GRANT ALL ON work_orders TO PUBLIC",
    ]
    for statement in writes:
        with pytest.raises(UnsafeQuery):
            validate(statement, settings)


# ----------------------------------------------------------------------
# Excel export
# ----------------------------------------------------------------------

def test_excel_preserves_real_types(tmp_path):
    """
    The point of Excel over CSV is that dates sort as dates and numbers sum.

    Stringifying everything would make the export no better than the CSV it
    sits beside.
    """
    from datetime import datetime
    from decimal import Decimal

    from elp.reports.render import to_xlsx

    openpyxl = pytest.importorskip("openpyxl")

    result = QueryResult(
        columns=["Work Request", "Opened", "Hours", "Status"],
        rows=[
            ["WO-1001", datetime(2026, 8, 1, 9, 30), Decimal("12.5"), "OPEN"],
            ["WO-1002", None, None, None],
        ],
        row_count=2,
    )
    path = tmp_path / "report.xlsx"
    to_xlsx(result, path, title="Open Work Orders")

    sheet = openpyxl.load_workbook(path).active
    values = list(sheet.iter_rows(min_row=5, max_row=5, values_only=True))[0]
    assert isinstance(values[1], datetime)
    assert isinstance(values[2], (int, float))
    assert values[0] == "WO-1001"


def test_excel_header_is_frozen_and_filterable(tmp_path):
    from elp.reports.render import to_xlsx

    openpyxl = pytest.importorskip("openpyxl")

    result = QueryResult(columns=["a", "b"], rows=[["x", "y"]], row_count=1)
    path = tmp_path / "r.xlsx"
    to_xlsx(result, path, title="R")

    sheet = openpyxl.load_workbook(path).active
    assert sheet.freeze_panes == "A5"
    assert sheet.auto_filter.ref is not None
    assert sheet.cell(row=4, column=1).font.bold


def test_excel_sheet_name_is_sanitised(tmp_path):
    """Excel refuses names over 31 characters or containing []:*?/\\"""
    from elp.reports.render import to_xlsx

    openpyxl = pytest.importorskip("openpyxl")

    result = QueryResult(columns=["a"], rows=[["x"]], row_count=1)
    path = tmp_path / "r.xlsx"
    to_xlsx(result, path, title="Work/Requests: [2026] *all* very long title indeed")

    sheet = openpyxl.load_workbook(path).active
    assert len(sheet.title) <= 31
    assert not set(sheet.title) & set("[]:*?/\\")


def test_pdf_caps_its_column_count(result):
    """
    A PDF wider than twelve columns is unreadable in landscape.

    The full set is still in the CSV and Excel artifacts.
    """
    from elp.reports.render import PDF_MAX_COLUMNS, to_latex

    wide = QueryResult(
        columns=[f"column_{i}" for i in range(20)],
        rows=[[i for i in range(20)]],
        row_count=1,
    )
    rendered = to_latex(wide, title="Wide report")
    assert rendered.count(r"\textbf{") == PDF_MAX_COLUMNS
    validate_source(rendered)
