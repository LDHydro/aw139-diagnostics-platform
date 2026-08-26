"""
NAMIS specifics: T-SQL dialect handling and the exported field catalogue.

NAMIS is Microsoft SQL Server across four databases on one instance. The
dialect differences tested here are the kind that fail loudly rather than
subtly — emitting `LIMIT` against T-SQL is a syntax error on every generated
query — plus the T-SQL-only escape routes the guard has to know about.
"""

from __future__ import annotations

import pytest

from elp.config import ReportSettings
from elp.reports.sqlguard import UnsafeQuery, enforce_limit, validate


@pytest.fixture
def mssql():
    return ReportSettings(sql_dialect="mssql", max_rows=5000)


# ----------------------------------------------------------------------
# T-SQL row limiting
# ----------------------------------------------------------------------

def test_tsql_uses_top_not_limit(mssql):
    """
    `LIMIT` is a syntax error in T-SQL.

    Getting this wrong does not degrade quietly — every generated query
    fails at the server.
    """
    result = validate("SELECT WRNo FROM WorkRequest ORDER BY WRNo", mssql)
    assert "TOP (5000)" in result.sql
    assert "LIMIT" not in result.sql.upper()


def test_top_is_injected_into_the_outermost_select_of_a_cte(mssql):
    """The CTE body is a subquery; the limit belongs on the main SELECT."""
    result = validate(
        "WITH x AS (SELECT WRNo FROM WorkRequest) SELECT * FROM x ORDER BY WRNo",
        mssql,
    )
    assert result.sql.index("TOP (5000)") > result.sql.index("WITH")
    assert "WITH x AS (SELECT WRNo" in result.sql  # inner SELECT untouched


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP (50) WRNo FROM WorkRequest ORDER BY WRNo",
        "SELECT TOP 50 WRNo FROM WorkRequest ORDER BY WRNo",
        "SELECT WRNo FROM WorkRequest ORDER BY WRNo OFFSET 0 ROWS FETCH NEXT 25 ROWS ONLY",
    ],
)
def test_an_existing_row_bound_is_respected(sql, mssql):
    result = validate(sql, mssql)
    assert result.limit_applied is None


def test_postgres_still_gets_limit():
    settings = ReportSettings(sql_dialect="postgresql")
    result = validate("SELECT * FROM work_orders", settings)
    assert result.sql.rstrip().endswith("LIMIT 5000")


def test_limit_syntax_is_chosen_per_dialect():
    sql = "SELECT a FROM t"
    assert "TOP (10)" in enforce_limit(sql, 10, "mssql")[0]
    assert "LIMIT 10" in enforce_limit(sql, 10, "postgresql")[0]


# ----------------------------------------------------------------------
# T-SQL identifiers
# ----------------------------------------------------------------------

def test_bracket_quoted_identifiers_are_understood(mssql):
    """
    `[dbo].[WorkRequest]` must resolve to a table name.

    Without bracket handling the dotted-name regex sees nothing and every
    bracket-quoted reference slips past the allowlist unchecked.
    """
    result = validate(
        "SELECT [WRNo] FROM [NAMISNNSS].[dbo].[WorkRequest] ORDER BY [WRNo]", mssql
    )
    assert "namisnnss.dbo.workrequest" in result.tables


def test_three_part_names_reach_the_other_databases(mssql):
    """NAMIS spans four databases on one instance; this is normal here."""
    result = validate(
        "SELECT * FROM [AMO_NASAWeb].[dbo].[FlightRecordHeaders] ORDER BY 1", mssql
    )
    assert "amo_nasaweb.dbo.flightrecordheaders" in result.tables


def test_four_part_names_are_refused_as_linked_servers(mssql):
    """A four-part name is a route out of this instance entirely."""
    with pytest.raises(UnsafeQuery, match="four-part name"):
        validate(
            "SELECT * FROM [OTHERSRV].[AMO_NASAWeb].[dbo].[FlightRecordHeaders]", mssql
        )


# ----------------------------------------------------------------------
# T-SQL escape routes
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,sql",
    [
        ("OPENROWSET", "SELECT * FROM OPENROWSET('SQLNCLI', 'x', 'SELECT 1')"),
        ("OPENQUERY", "SELECT * FROM OPENQUERY(LINKED, 'SELECT 1')"),
        ("OPENDATASOURCE", "SELECT * FROM OPENDATASOURCE('x','y').db.dbo.t"),
        ("xp_cmdshell", "SELECT * FROM t WHERE x = (SELECT xp_cmdshell('dir'))"),
        ("xp_dirtree", "SELECT * FROM t WHERE x = (SELECT xp_dirtree('C:'))"),
        ("sp_OACreate", "SELECT * FROM t WHERE x = (SELECT sp_OACreate('a'))"),
        ("sp_setapprole", "SELECT * FROM t WHERE x = (SELECT sp_setapprole('a','b'))"),
        ("WAITFOR", "SELECT * FROM t WHERE 1=1 WAITFOR DELAY '00:10:00'"),
        ("DBCC", "SELECT * FROM t WHERE 1=1 DBCC CHECKDB"),
        ("SHUTDOWN", "SELECT 1 SHUTDOWN"),
        ("BACKUP", "BACKUP DATABASE NAMISNNSS TO DISK='x'"),
        ("BULK", "SELECT * FROM t WHERE 1=1 BULK INSERT t FROM 'f'"),
    ],
)
def test_tsql_escape_routes_are_refused(name, sql, mssql):
    with pytest.raises(UnsafeQuery):
        validate(sql, mssql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM [master].[dbo].[sysdatabases]",
        "SELECT * FROM msdb.dbo.sysjobs",
        "SELECT * FROM sysobjects",
        "SELECT * FROM tempdb.dbo.x",
    ],
)
def test_system_databases_are_refused_in_any_position(sql, mssql):
    """
    In `master.dbo.sysdatabases` the dangerous part is the FIRST qualifier.

    Checking only the qualifier nearest the table would see `dbo` and pass.
    """
    with pytest.raises(UnsafeQuery):
        validate(sql, mssql)


# ----------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------

def test_catalogue_loads_tables_columns_and_joins(catalog):
    assert len(catalog) == 4
    assert catalog.column_count == 11
    assert len(catalog.relationships) == 2
    assert catalog.databases == ["AMO_NASAWeb", "NAMISNNSS"]


def test_catalogue_exposes_three_part_names(catalog):
    spec = catalog.get("WorkRequest")
    assert spec.qualified == "[NAMISNNSS].[dbo].[WorkRequest]"
    assert spec.plain == "namisnnss.dbo.workrequest"


def test_lookup_is_case_and_qualification_insensitive(catalog):
    assert catalog.get("workrequest") is catalog.get("WorkRequest")
    assert catalog.get("[NAMISNNSS].[dbo].[WorkRequest]") is catalog.get("WorkRequest")


def test_group_membership_backfills_a_missing_group(catalog):
    """The group index names tables the table entries do not self-declare."""
    assert catalog.get("WRStatusHist").group == "Work Requests"


def test_allowlist_covers_every_way_a_table_may_be_written(catalog):
    names = set(catalog.allowed_table_names())
    assert "workrequest" in names
    assert "dbo.workrequest" in names
    assert "namisnnss.dbo.workrequest" in names


def test_selection_finds_the_tables_a_request_is_about(catalog):
    chosen = [t.name for t in catalog.select_for_request("open work requests", limit=4)]
    assert "WorkRequest" in chosen


def test_selection_pulls_in_joinable_neighbours(catalog):
    """
    A request about work requests almost always needs the tables that join
    to them, and the model cannot ask for more once the prompt is built.
    """
    chosen = [t.name for t in catalog.select_for_request("work request status", limit=6)]
    assert "WorkRequest" in chosen
    assert "WRStatusHist" in chosen or "AIRCRAFT" in chosen


def test_selection_returns_nothing_for_an_empty_request(catalog):
    assert catalog.select_for_request("the and of") == []


def test_compound_joins_are_rendered_in_full(catalog):
    """
    AIRCRAFT joins WorkRequest on both AssetKey and AssetSite.

    Rendering only the first column would produce a cartesian product that
    looks like real data — the worst kind of wrong report.
    """
    rendered = catalog.render_for_prompt(
        [catalog.get("AIRCRAFT"), catalog.get("WorkRequest")]
    )
    assert "AssetKey = " in rendered
    assert "AssetSite = " in rendered
    assert "KNOWN JOINS" in rendered


def test_rendering_marks_primary_keys_and_types(catalog):
    rendered = catalog.render_for_prompt([catalog.get("WorkRequest")])
    assert "WRId uniqueidentifier PK" in rendered
    assert "WRNo char(7)" in rendered


def test_rendering_caps_wide_tables(catalog):
    spec = catalog.get("WorkRequest")
    rendered = catalog.render_for_prompt([spec], max_columns_per_table=2)
    assert "+3 more" in rendered


def test_a_missing_catalogue_is_reported_not_swallowed(tmp_path):
    from elp.reports.catalog import load_catalog

    with pytest.raises(FileNotFoundError, match="ELP_REPORTS__CATALOG_PATH"):
        load_catalog(tmp_path / "absent.json")
