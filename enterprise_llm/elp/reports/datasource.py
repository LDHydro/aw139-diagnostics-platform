"""
Connecting to NAMIS.

Two adapters: a direct read-only database connection, and a REST API for
sites where direct database access is not granted. Both are configured
entirely from environment variables - no credential is ever stored in a
report definition, a database row, or a log line.

The SQL adapter opens every transaction READ ONLY and sets a statement
timeout, so a query that slips past the validator still cannot write and
still cannot run all night. That is defence in depth, not a substitute for
provisioning NAMIS with a read-only account: **do that first.**
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..config import ReportSettings, get_settings
from .sqlguard import UnsafeQuery, redact_columns, validate

log = logging.getLogger(__name__)

_REDACTED = "[redacted]"


class DataSourceError(RuntimeError):
    """NAMIS could not be reached, or refused the request."""


# SQL Server error numbers, mapped to guidance that names the fix. Taken
# from what the existing NAMIS report generator learned in this environment:
# these are the failures that actually happen, and the raw driver text for
# each is unhelpful to the person who asked for a report.
_SQL_ERROR_GUIDANCE: list[tuple[str, str]] = [
    (
        "18456",
        "Login failed. The reporting account is not provisioned on this SQL "
        "Server - that needs a DBA, not a configuration change.",
    ),
    (
        "4060",
        "Login failed for the requested database. Check Initial Catalog and "
        "that the account has access to NAMISNNSS.",
    ),
    (
        "229",
        "Permission denied on an object. The account can log in but lacks "
        "SELECT on a table the report needs. Either grant it, or configure "
        "the application role (ELP_REPORTS__NAMIS_APP_ROLE).",
    ),
    (
        "230",
        "Permission denied on a column. The account lacks SELECT on one of "
        "the columns the report selects.",
    ),
    (
        "15151",
        "The application role or its password is wrong. The password may have "
        "been rotated - see security finding F-1.",
    ),
    (
        "15421",
        "The application role name or password is not valid. Check "
        "ELP_REPORTS__NAMIS_APP_ROLE and its password environment variable.",
    ),
    (
        "208",
        "Invalid object name. The table exists in the catalogue but not in "
        "this database - check the three-part name and that all four "
        "databases live on the instance you connected to.",
    ),
]


def _translate_sql_error(exc: Exception) -> str:
    """Turn a driver exception into something the requester can act on."""
    message = str(exc)
    for number, guidance in _SQL_ERROR_GUIDANCE:
        if number in message:
            return f"{guidance} (SQL error {number})"
    lowered = message.lower()
    if "certificate" in lowered:
        return (
            "the server's TLS certificate was not trusted. Add "
            "TrustServerCertificate=true to the connection string as an interim "
            "measure inside the VPN, or install a trusted certificate. " + message
        )
    if "timeout" in lowered or "timed out" in lowered:
        return f"the connection timed out - check VPN reachability. {message}"
    if "login timeout" in lowered or "server was not found" in lowered:
        return f"the SQL Server could not be reached. {message}"
    return message


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    duration_ms: float = 0.0
    truncated: bool = False
    redacted_columns: list[str] = field(default_factory=list)
    query_executed: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True


@dataclass
class TableInfo:
    name: str
    schema: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)
    comment: str = ""

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


class NamisSource:
    """Base adapter."""

    kind = "abstract"

    async def describe_schema(self) -> list[TableInfo]:
        raise NotImplementedError

    async def run(self, query: str, parameters: dict | None = None) -> QueryResult:
        raise NotImplementedError

    async def health(self) -> dict:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


# ----------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------

class SqlNamisSource(NamisSource):
    kind = "sql"

    def __init__(self, settings: ReportSettings | None = None) -> None:
        self.settings = settings or get_settings().reports
        self._engine: AsyncEngine | None = None
        self._schema_cache: list[TableInfo] | None = None

    def _dsn(self) -> str:
        dsn = self.settings.namis_dsn
        if not dsn:
            raise DataSourceError(
                "ELP_REPORTS__NAMIS_DSN is not set. Point it at a READ-ONLY "
                "NAMIS database account."
            )
        password = os.environ.get(self.settings.namis_auth_env_var, "")
        # Allow the password to be supplied separately from the DSN so it
        # never appears in configuration files or process listings.
        if password and "${PASSWORD}" in dsn:
            dsn = dsn.replace("${PASSWORD}", password)
        return dsn

    def _connect_args(self, dsn: str) -> dict:
        """
        Driver options that make the connection read-only at the server.

        This is stronger than the per-transaction ``SET TRANSACTION READ
        ONLY`` below: ``default_transaction_read_only`` applies to every
        transaction on the connection, so even a code path that forgot to
        set it cannot write. Both are belt and braces around the real
        control, which is that the NAMIS account has SELECT and nothing else.

        ``application_name`` is a courtesy to whoever runs NAMIS: this
        traffic is identifiable in pg_stat_activity.
        """
        if "asyncpg" in dsn:
            return {
                "server_settings": {
                    "default_transaction_read_only": "on",
                    "application_name": "elp-reports",
                }
            }
        if "psycopg" in dsn:
            return {
                "options": "-c default_transaction_read_only=on",
                "application_name": "elp-reports",
            }
        if "mssql" in dsn or "pyodbc" in dsn or "aioodbc" in dsn:
            # SQL Server has no session-level read-only switch. The controls
            # are the login's permissions (checked by verify_read_only) and
            # the statement validator. ApplicationIntent is a routing hint,
            # not a permission - it only takes effect against an availability
            # group replica - but it costs nothing and is correct intent.
            return {}
        return {}

    def engine(self) -> AsyncEngine:
        if self._engine is None:
            dsn = self._dsn()
            self._engine = create_async_engine(
                dsn,
                pool_size=3,
                max_overflow=2,
                pool_pre_ping=True,
                # Reports are infrequent; holding connections open against a
                # production system all day is rude.
                pool_recycle=1800,
                connect_args=self._connect_args(dsn),
            )
        return self._engine

    async def verify_read_only(self) -> dict:
        """
        Prove the account cannot write, rather than assuming it.

        Attempts a harmless write (a temporary table) and expects the server
        to refuse. If it succeeds, the connection is NOT read-only and the
        result says so loudly - that is a provisioning problem no amount of
        application-level validation makes safe.

        Run at commissioning and from ``/v1/health/deep``.
        """
        try:
            async with self.engine().connect() as connection:
                if connection.dialect.name == "mssql":
                    return await self._verify_read_only_mssql(connection)
                if connection.dialect.name != "postgresql":
                    return {
                        "read_only": None,
                        "detail": (
                            f"cannot verify automatically on "
                            f"{connection.dialect.name}; confirm the account has "
                            "SELECT only"
                        ),
                    }
                probe = "CREATE TEMP TABLE _elp_readonly_probe (x integer)"
                try:
                    await connection.execute(text(probe))
                except Exception as exc:
                    await connection.rollback()
                    message = str(exc).lower()
                    if "read-only" in message or "read only" in message or "permission" in message:
                        return {
                            "read_only": True,
                            "detail": "the server refused a write, as it should",
                        }
                    return {
                        "read_only": None,
                        "detail": f"the write probe failed for another reason: {exc}",
                    }

                # The write succeeded. That is the finding.
                await connection.rollback()
                log.error(
                    "SECURITY: the NAMIS connection accepted a write. The account "
                    "is not read-only. Fix this before running reports."
                )
                return {
                    "read_only": False,
                    "detail": (
                        "the NAMIS connection ACCEPTED a write. The account is not "
                        "read-only. Application-level validation is not a "
                        "substitute - grant SELECT only."
                    ),
                }
        except Exception as exc:
            return {"read_only": None, "detail": f"could not run the probe: {exc}"}

    async def describe_schema(self, refresh: bool = False) -> list[TableInfo]:
        """
        Introspect the tables the reporting account can see.

        This is what grounds query authoring: without it the model invents
        plausible column names, which is the single largest source of
        wrong-looking reports.
        """
        if self._schema_cache is not None and not refresh:
            return self._schema_cache

        # The exported catalogue is richer than anything introspection can
        # recover - it carries the join relationships - so prefer it and fall
        # back to live introspection only when it is absent.
        from .catalog import get_catalog

        catalog = get_catalog()
        if catalog is not None and len(catalog):
            tables = [
                TableInfo(
                    name=spec.table or spec.name,
                    schema=spec.schema,
                    comment=spec.group,
                    columns=[
                        ColumnInfo(name=c.name, type=c.sql, nullable=c.nullable)
                        for c in spec.columns
                    ],
                )
                for spec in catalog.tables.values()
            ]
            self._schema_cache = tables
            log.info("using the NAMIS catalogue: %d table(s)", len(tables))
            return tables

        allowed_schemas = [s.lower() for s in self.settings.allowed_schemas]
        allowed_tables = {t.lower() for t in self.settings.allowed_tables}

        def _introspect(connection) -> list[TableInfo]:
            from sqlalchemy import inspect

            inspector = inspect(connection)
            schemas = allowed_schemas or [inspector.default_schema_name or ""]
            tables: list[TableInfo] = []
            for schema in schemas:
                for name in inspector.get_table_names(schema=schema or None):
                    qualified = f"{schema}.{name}".lower() if schema else name.lower()
                    if allowed_tables and not (
                        qualified in allowed_tables or name.lower() in allowed_tables
                    ):
                        continue
                    columns = [
                        ColumnInfo(
                            name=column["name"],
                            type=str(column.get("type", "")),
                            nullable=bool(column.get("nullable", True)),
                        )
                        for column in inspector.get_columns(name, schema=schema or None)
                    ]
                    tables.append(TableInfo(name=name, schema=schema or "", columns=columns))
            return tables

        try:
            async with self.engine().connect() as connection:
                tables = await connection.run_sync(_introspect)
        except Exception as exc:
            raise DataSourceError(f"could not introspect NAMIS: {exc}") from exc

        self._schema_cache = tables
        log.info("introspected %d NAMIS table(s)", len(tables))
        return tables

    async def _verify_read_only_mssql(self, connection) -> dict:
        """
        Ask SQL Server what this login may actually do.

        A temp-table probe is useless here: creating ``#temp`` objects only
        needs rights in tempdb, which ``public`` holds by default, so a
        genuinely read-only login would still pass it. Interrogating the
        login's own permissions is both safe - it writes nothing - and
        precise about what is wrong.
        """
        probe = text(
            "SELECT "
            "  CAST(ISNULL(IS_MEMBER('db_owner'), 0) AS int)      AS is_owner, "
            "  CAST(ISNULL(IS_MEMBER('db_datawriter'), 0) AS int) AS is_writer, "
            "  (SELECT COUNT(*) FROM sys.fn_my_permissions(NULL, 'DATABASE') "
            "     WHERE permission_name IN "
            "     ('INSERT','UPDATE','DELETE','ALTER','CONTROL','CREATE TABLE')) "
            "                                                      AS write_perms"
        )
        try:
            row = (await connection.execute(probe)).one()
        except Exception as exc:
            return {
                "read_only": None,
                "detail": f"could not read this login's permissions: {exc}",
            }

        is_owner, is_writer, write_perms = int(row[0]), int(row[1]), int(row[2])
        if is_owner or is_writer or write_perms:
            reasons = []
            if is_owner:
                reasons.append("it is a member of db_owner")
            if is_writer:
                reasons.append("it is a member of db_datawriter")
            if write_perms:
                reasons.append(f"it holds {write_perms} database-level write permission(s)")
            log.error(
                "SECURITY: the NAMIS reporting login can write. %s", "; ".join(reasons)
            )
            return {
                "read_only": False,
                "detail": (
                    "the NAMIS login CAN WRITE: "
                    + "; ".join(reasons)
                    + ". Grant db_datareader (or explicit SELECT) and nothing else."
                ),
            }

        return {
            "read_only": True,
            "detail": (
                "the login holds no database-level write permission and is in "
                "neither db_owner nor db_datawriter"
            ),
        }

    async def _begin_session(self, connection) -> None:
        """Per-connection setup before the report query runs."""
        dialect = connection.dialect.name

        if dialect == "postgresql":
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            await connection.execute(
                text(
                    f"SET LOCAL statement_timeout = "
                    f"{int(self.settings.statement_timeout_ms)}"
                )
            )
            return

        if dialect in {"mysql", "mariadb"}:
            await connection.execute(text("SET SESSION TRANSACTION READ ONLY"))
            await connection.execute(
                text(
                    f"SET SESSION max_execution_time = "
                    f"{int(self.settings.statement_timeout_ms)}"
                )
            )
            return

        if dialect == "mssql":
            # Reporting against a live OLTP system takes shared locks that can
            # block the technicians using it. The isolation level is the lever;
            # see ELP_REPORTS__MSSQL_ISOLATION_LEVEL for the trade.
            level = self.settings.mssql_isolation_level
            await connection.execute(
                text(f"SET TRANSACTION ISOLATION LEVEL {level}")
            )

            # The application role, when the reporting login lacks direct
            # table rights. Leaving both settings empty is the better posture:
            # queries then run with the login's own permissions.
            role = self.settings.namis_app_role
            password = os.environ.get(self.settings.namis_app_role_env_var, "")
            if role and password:
                try:
                    await connection.execute(
                        text("EXEC sp_setapprole @rolename = :role, @password = :pwd"),
                        {"role": role, "pwd": password},
                    )
                except Exception as exc:
                    raise DataSourceError(
                        f"the NAMIS application role '{role}' could not be "
                        f"activated: {_translate_sql_error(exc)}"
                    ) from exc
            elif role and not password:
                raise DataSourceError(
                    f"ELP_REPORTS__NAMIS_APP_ROLE is set to '{role}' but "
                    f"${self.settings.namis_app_role_env_var} is empty"
                )

    async def run(self, query: str, parameters: dict | None = None) -> QueryResult:
        try:
            guard = validate(query, self.settings)
        except UnsafeQuery as exc:
            raise DataSourceError(str(exc)) from exc

        engine = self.engine()
        started = time.monotonic()

        try:
            async with engine.connect() as connection:
                await self._begin_session(connection)

                # A driver-side timeout is not guaranteed on every dialect, so
                # the wall clock is enforced here as well. A report that runs
                # for ten minutes against production is an incident.
                timeout_s = max(1.0, self.settings.statement_timeout_ms / 1000)
                result = await asyncio.wait_for(
                    connection.execute(text(guard.sql), parameters or {}),
                    timeout=timeout_s,
                )
                columns = list(result.keys())
                fetched = result.fetchmany(self.settings.max_rows + 1)
                # Never commit: a read-only report has nothing to persist.
                await connection.rollback()
        except TimeoutError as exc:
            raise DataSourceError(
                f"the query exceeded {self.settings.statement_timeout_ms} ms and "
                "was abandoned. Narrow the date range or add a more selective "
                "filter."
            ) from exc
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"NAMIS query failed: {_translate_sql_error(exc)}"
            ) from exc

        duration_ms = (time.monotonic() - started) * 1000
        truncated = len(fetched) > self.settings.max_rows
        rows = [list(row) for row in fetched[: self.settings.max_rows]]

        redacted = redact_columns(columns, self.settings)
        for row in rows:
            for index in redacted:
                if index < len(row):
                    row[index] = _REDACTED

        warnings = list(guard.notes)
        if truncated:
            warnings.append(
                f"the result was truncated at {self.settings.max_rows} rows; "
                "narrow the report or raise ELP_REPORTS__MAX_ROWS"
            )

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            duration_ms=duration_ms,
            truncated=truncated,
            redacted_columns=[columns[i] for i in sorted(redacted)],
            query_executed=guard.sql,
            warnings=warnings,
        )

    async def health(self) -> dict:
        try:
            async with self.engine().connect() as connection:
                await connection.execute(text("SELECT 1"))
            tables = await self.describe_schema()
            probe = await self.verify_read_only()
            # A writable connection is a failed health check, not a note.
            status = "error" if probe.get("read_only") is False else "ok"
            return {
                "status": status,
                "kind": "sql",
                "tables_visible": len(tables),
                "read_only": probe.get("read_only"),
                "read_only_detail": probe.get("detail", ""),
            }
        except Exception as exc:
            return {"status": "error", "kind": "sql", "detail": str(exc)}

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None


# ----------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------

class RestNamisSource(NamisSource):
    """
    NAMIS behind an HTTP API.

    The "query" is a JSON body posted to the configured endpoint, so the SQL
    guard does not apply. Whatever NAMIS exposes over REST is by definition
    what it is willing to serve, but the row and cell caps still apply.
    """

    kind = "rest"

    def __init__(self, settings: ReportSettings | None = None) -> None:
        self.settings = settings or get_settings().reports
        self._schema_cache: list[TableInfo] | None = None

    def _headers(self) -> dict[str, str]:
        if self.settings.namis_auth_type == "none":
            return {}
        secret = os.environ.get(self.settings.namis_auth_env_var, "")
        if not secret:
            raise DataSourceError(
                f"NAMIS expects its credential in ${self.settings.namis_auth_env_var}, "
                "which is not set"
            )
        if self.settings.namis_auth_type == "bearer":
            return {self.settings.namis_auth_header: f"Bearer {secret}"}
        return {self.settings.namis_auth_header: secret}

    async def describe_schema(self, refresh: bool = False) -> list[TableInfo]:
        if self._schema_cache is not None and not refresh:
            return self._schema_cache
        base = self.settings.namis_rest_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{base}/schema", headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise DataSourceError(f"could not read the NAMIS schema: {exc}") from exc

        tables = [
            TableInfo(
                name=entry.get("name", ""),
                schema=entry.get("schema", ""),
                comment=entry.get("description", ""),
                columns=[
                    ColumnInfo(
                        name=column.get("name", ""),
                        type=str(column.get("type", "")),
                        nullable=bool(column.get("nullable", True)),
                    )
                    for column in entry.get("columns", [])
                ],
            )
            for entry in payload.get("tables", [])
        ]
        self._schema_cache = tables
        return tables

    async def run(self, query: str, parameters: dict | None = None) -> QueryResult:
        import json

        base = self.settings.namis_rest_base_url.rstrip("/")
        try:
            body = json.loads(query) if query.strip().startswith("{") else {"query": query}
        except json.JSONDecodeError as exc:
            raise DataSourceError(f"the REST query is not valid JSON: {exc}") from exc
        if parameters:
            body.setdefault("parameters", {}).update(parameters)

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.statement_timeout_ms / 1000
            ) as client:
                response = await client.post(
                    f"{base}/query", headers=self._headers(), json=body
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise DataSourceError(f"NAMIS query failed: {exc}") from exc

        duration_ms = (time.monotonic() - started) * 1000
        rows_payload = payload.get("rows", payload.get("data", []))
        if rows_payload and isinstance(rows_payload[0], dict):
            columns = list(payload.get("columns") or rows_payload[0].keys())
            rows = [[row.get(column) for column in columns] for row in rows_payload]
        else:
            columns = list(payload.get("columns", []))
            rows = [list(row) for row in rows_payload]

        truncated = len(rows) > self.settings.max_rows
        rows = rows[: self.settings.max_rows]

        redacted = redact_columns(columns, self.settings)
        for row in rows:
            for index in redacted:
                if index < len(row):
                    row[index] = _REDACTED

        warnings = []
        if truncated:
            warnings.append(
                f"the result was truncated at {self.settings.max_rows} rows"
            )

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            duration_ms=duration_ms,
            truncated=truncated,
            redacted_columns=[columns[i] for i in sorted(redacted)],
            query_executed=query,
            warnings=warnings,
        )

    async def health(self) -> dict:
        try:
            tables = await self.describe_schema()
            return {"status": "ok", "kind": "rest", "tables_visible": len(tables)}
        except Exception as exc:
            return {"status": "error", "kind": "rest", "detail": str(exc)}


# ----------------------------------------------------------------------

_source: NamisSource | None = None


def get_source(settings: ReportSettings | None = None) -> NamisSource:
    global _source
    settings = settings or get_settings().reports
    if _source is None:
        if settings.namis_kind == "sql":
            _source = SqlNamisSource(settings)
        elif settings.namis_kind == "rest":
            _source = RestNamisSource(settings)
        else:
            raise DataSourceError(
                "NAMIS is not configured. Set ELP_REPORTS__NAMIS_KIND to 'sql' "
                "or 'rest' and supply the connection details."
            )
    return _source


async def close_source() -> None:
    global _source
    if _source is not None:
        await _source.aclose()
    _source = None


def render_schema_catalogue(tables: list[TableInfo], max_tables: int = 60) -> str:
    """Compact schema description used to ground query authoring."""
    lines: list[str] = []
    for table in tables[:max_tables]:
        columns = ", ".join(f"{c.name} {c.type}" for c in table.columns)
        line = f"{table.qualified}({columns})"
        if table.comment:
            line += f"  -- {table.comment}"
        lines.append(line)
    if len(tables) > max_tables:
        lines.append(f"... and {len(tables) - max_tables} further table(s)")
    return "\n".join(lines)
