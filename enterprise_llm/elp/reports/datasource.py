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

    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._dsn(),
                pool_size=3,
                max_overflow=2,
                pool_pre_ping=True,
                # Reports are infrequent; holding connections open against a
                # production system all day is rude.
                pool_recycle=1800,
            )
        return self._engine

    async def describe_schema(self, refresh: bool = False) -> list[TableInfo]:
        """
        Introspect the tables the reporting account can see.

        This is what grounds query authoring: without it the model invents
        plausible column names, which is the single largest source of
        wrong-looking reports.
        """
        if self._schema_cache is not None and not refresh:
            return self._schema_cache

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

    async def run(self, query: str, parameters: dict | None = None) -> QueryResult:
        try:
            guard = validate(query, self.settings)
        except UnsafeQuery as exc:
            raise DataSourceError(str(exc)) from exc

        engine = self.engine()
        started = time.monotonic()

        try:
            async with engine.connect() as connection:
                # Layer two: even a mis-provisioned account cannot write
                # inside a read-only transaction.
                dialect = connection.dialect.name
                if dialect == "postgresql":
                    await connection.execute(text("SET TRANSACTION READ ONLY"))
                    await connection.execute(
                        text(f"SET LOCAL statement_timeout = {int(self.settings.statement_timeout_ms)}")
                    )
                elif dialect in {"mysql", "mariadb"}:
                    await connection.execute(
                        text("SET SESSION TRANSACTION READ ONLY")
                    )
                    await connection.execute(
                        text(f"SET SESSION max_execution_time = {int(self.settings.statement_timeout_ms)}")
                    )

                result = await connection.execute(text(guard.sql), parameters or {})
                columns = list(result.keys())
                fetched = result.fetchmany(self.settings.max_rows + 1)
                # Never commit: a read-only report has nothing to persist.
                await connection.rollback()
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"NAMIS query failed: {exc}") from exc

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
            return {"status": "ok", "kind": "sql", "tables_visible": len(tables)}
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
