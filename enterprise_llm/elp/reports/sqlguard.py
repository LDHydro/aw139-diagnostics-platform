"""
Read-only validation of generated SQL.

A language model writes these queries and a scheduler runs them unattended
against a production system. That combination means the query text is
untrusted input, and it is validated the way untrusted input is validated:
allowlist what is permitted, reject everything else, and refuse to be clever
about ambiguous cases.

Defence is layered, because any single layer can be defeated:

1. **The database account is read-only.** This is the layer that actually
   holds; everything below is there to catch mistakes before they reach it.
   Nothing here substitutes for provisioning NAMIS access correctly.
2. **The transaction is opened READ ONLY**, so even a mis-provisioned
   account cannot write.
3. **The statement is validated** - single statement, SELECT or WITH only,
   no DDL, DML or dangerous functions, tables within the allowlist.
4. **A LIMIT is enforced**, so a missing WHERE clause is slow rather than
   catastrophic.

The validator works on a comment-stripped, string-literal-masked copy of the
statement, so keywords hidden inside comments or quoted text cannot smuggle
anything past it - and equally, a legitimate string containing the word
"delete" does not trip it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ReportSettings, get_settings


class UnsafeQuery(ValueError):
    """The statement is not a safe read-only query."""


# Statements that modify data, schema or server state.
_FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "merge", "upsert", "truncate",
    "drop", "alter", "create", "rename", "comment",
    "grant", "revoke", "vacuum", "analyze", "cluster", "reindex",
    "call", "do", "execute", "prepare", "deallocate",
    "copy", "import", "load", "outfile", "dumpfile",
    "listen", "notify", "lock", "set", "reset", "discard",
    "begin", "commit", "rollback", "savepoint", "start",
    "refresh", "attach", "detach", "pragma",
    # T-SQL. WAITFOR takes no parentheses, so the function scan misses it,
    # and WAITFOR DELAY is a free denial-of-service against the connection.
    "waitfor", "shutdown", "kill", "backup", "restore", "dbcc", "bulk",
}

# Functions and constructs that read files, run programs, reach the network
# or waste server time.
_FORBIDDEN_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "pg_logical_emit_message", "pg_file_write", "pg_file_unlink",
    "lo_import", "lo_export", "lo_put",
    "dblink", "dblink_exec", "dblink_connect",
    "query_to_xml", "xmlexec",
    "load_extension", "system", "shell", "xp_cmdshell",
    "benchmark", "sleep", "waitfor", "sp_executesql",
    "into_outfile", "load_file",
    # SQL Server: file, linked-server and OS reach.
    "openrowset", "opendatasource", "openquery", "openxml",
    "xp_dirtree", "xp_fileexist", "xp_regread", "xp_subdirs",
    "sp_oacreate", "sp_oamethod", "sp_oagetproperty", "sp_addlinkedserver",
    "sp_setapprole", "sp_configure", "fn_get_audit_file", "fn_trace_gettable",
}

# Catalog and system schemas: reading them leaks structure and credentials
# metadata, and no operational report needs them.
_FORBIDDEN_SCHEMAS = {
    "pg_catalog", "information_schema", "pg_toast", "pg_temp",
    "mysql", "sys", "performance_schema", "master", "msdb", "tempdb", "model",
}

# T-SQL quotes identifiers with brackets: [AMO_NASAWeb].[dbo].[WORKREQUEST]
_BRACKET_IDENT = re.compile(r"\[([^\]]*)\]")
# SQL Server system catalogue views. Not prefixed pg_, so named explicitly.
_SYSTEM_TABLES = {
    "sysdatabases", "sysobjects", "syscolumns", "sysusers", "syslogins",
    "sysprocesses", "sysservers", "sysaltfiles", "sysfiles", "sysindexes",
    "sysconfigures", "syscomments", "sysdepends", "sysmessages",
}

_SINGLE_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SINGLE_QUOTED = re.compile(r"'(?:''|\\.|[^'])*'", re.DOTALL)
_DOUBLE_QUOTED = re.compile(r'"(?:""|[^"])*"')
_DOLLAR_QUOTED = re.compile(r"\$([A-Za-z_]\w*)?\$.*?\$\1?\$", re.DOTALL)

# FROM/JOIN targets, capturing an optional schema qualifier.
_TABLE_REF = re.compile(
    r"\b(?:from|join|into|update)\s+"
    r"(?!\s*\()"                       # skip subqueries: FROM ( SELECT ...
    r"([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*){0,3})",
    re.IGNORECASE,
)
_FUNCTION_CALL = re.compile(r"\b([A-Za-z_][\w$]*)\s*\(")
_LIMIT_PRESENT = re.compile(
    r"\blimit\s+\d+"          # PostgreSQL / MySQL
    r"|\bfetch\s+first\b"     # ANSI
    r"|\bfetch\s+next\b"      # T-SQL OFFSET ... FETCH NEXT
    r"|\btop\s*\(?\s*\d+"    # T-SQL TOP n / TOP (n)
    r"|\btop\s+percent\b",
    re.IGNORECASE,
)
_SELECT_TOKEN = re.compile(r"\bselect\b", re.IGNORECASE)


@dataclass
class GuardResult:
    sql: str
    tables: list[str] = field(default_factory=list)
    limit_applied: int | None = None
    notes: list[str] = field(default_factory=list)


def _mask(sql: str) -> str:
    """
    Strip comments and blank out literal contents.

    Keywords inside a comment or a quoted string must not be scanned - both
    because an attacker would hide them there, and because a perfectly
    legitimate `WHERE status = 'deleted'` should not be rejected.
    Replacements preserve length so offsets stay meaningful.
    """
    masked = _BLOCK_COMMENT.sub(lambda m: " " * len(m.group()), sql)
    masked = _SINGLE_LINE_COMMENT.sub(lambda m: " " * len(m.group()), masked)
    masked = _DOLLAR_QUOTED.sub(lambda m: "'" + " " * (len(m.group()) - 2) + "'", masked)
    masked = _SINGLE_QUOTED.sub(lambda m: "'" + " " * (len(m.group()) - 2) + "'", masked)
    # Double quotes are identifiers in standard SQL, so keep their content
    # for table-name extraction but normalise the quoting.
    masked = _DOUBLE_QUOTED.sub(lambda m: m.group().replace('"', " "), masked)
    # T-SQL bracket quoting: [AMO_NASAWeb].[dbo].[WORKREQUEST] must reduce to
    # AMO_NASAWeb.dbo.WORKREQUEST, or the dotted-name regex below sees
    # nothing and every three-part reference slips past the allowlist.
    masked = _BRACKET_IDENT.sub(lambda m: m.group(1), masked)
    return masked


def _statements(masked: str) -> list[str]:
    """Split on semicolons that are not inside a masked literal."""
    parts = [p.strip() for p in masked.split(";")]
    return [p for p in parts if p]


def extract_tables(masked: str) -> list[str]:
    """Table references in a masked statement, lower-cased."""
    found: list[str] = []
    for match in _TABLE_REF.finditer(masked):
        name = match.group(1).lower()
        # A CTE name is not a real table; callers filter those separately.
        if name not in found:
            found.append(name)
    return found


def _cte_names(masked: str) -> set[str]:
    """Names bound by a WITH clause, which are not physical tables."""
    names: set[str] = set()
    for match in re.finditer(
        r"(?:\bwith\b|,)\s+([A-Za-z_][\w$]*)\s+as\s*\(", masked, re.IGNORECASE
    ):
        names.add(match.group(1).lower())
    return names


def check_read_only(sql: str) -> str:
    """Reject anything that is not a single read-only statement."""
    if not sql or not sql.strip():
        raise UnsafeQuery("the query is empty")

    masked = _mask(sql)
    statements = _statements(masked)
    if len(statements) > 1:
        raise UnsafeQuery(
            f"the query contains {len(statements)} statements; only a single "
            "statement is permitted"
        )
    if not statements:
        raise UnsafeQuery("the query contains no statement once comments are removed")

    body = statements[0]
    first = body.split(None, 1)[0].lower() if body.split() else ""
    if first not in {"select", "with"}:
        raise UnsafeQuery(
            f"the query starts with '{first.upper() or '?'}'; only SELECT and "
            "WITH are permitted"
        )

    lowered = body.lower()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            # SELECT ... INTO creates a table; a plain FETCH FIRST does not.
            raise UnsafeQuery(
                f"the query uses '{keyword.upper()}', which is not permitted in a "
                "read-only report query"
            )

    # SELECT ... INTO creates a table. It is not caught by the keyword scan
    # because "into" is legitimate in "INSERT INTO", and it is not caught by
    # the table allowlist when that list is empty - which is the default.
    if re.search(r"\binto\b", lowered) and not re.search(r"\binsert\s+into\b", lowered):
        raise UnsafeQuery(
            "the query uses SELECT ... INTO, which creates a table and is not "
            "permitted in a read-only report query"
        )

    for match in _FUNCTION_CALL.finditer(lowered):
        function = match.group(1)
        if function in _FORBIDDEN_FUNCTIONS:
            raise UnsafeQuery(
                f"the query calls '{function}()', which is not permitted"
            )

    return body


def check_tables(
    masked_body: str, settings: ReportSettings | None = None
) -> list[str]:
    """Confirm every table referenced is within the allowlist."""
    settings = settings or get_settings().reports
    ctes = _cte_names(masked_body)
    tables = [t for t in extract_tables(masked_body) if t not in ctes]

    allowed_tables = {t.lower() for t in settings.allowed_tables}
    allowed_schemas = {s.lower() for s in settings.allowed_schemas}

    # When no explicit allowlist is configured, the exported catalogue is the
    # allowlist. A table absent from it is one the NAMIS report generator
    # never knew about, which is not something operations should be reporting
    # on by accident.
    if not allowed_tables and not allowed_schemas:
        from .catalog import get_catalog

        catalog = get_catalog()
        if catalog is not None and len(catalog):
            allowed_tables = set(catalog.allowed_table_names())

    for table in tables:
        parts = table.split(".")
        if len(parts) > 3:
            # server.database.schema.table - a linked server, i.e. a route
            # out of this database entirely. Never permitted in a report.
            raise UnsafeQuery(
                f"the query references '{table}' as a four-part name, which "
                "reaches another server through a linked-server definition. "
                "Reports may only read the databases on this instance."
            )
        # Three-part names are normal here: NAMIS reaches AMO_NASAWeb,
        # OpsDBAMO and WebSupport_NASAWeb on the same instance.
        schema = parts[-2] if len(parts) >= 2 else ""
        bare = parts[-1]

        # Check every qualifier, not just the one nearest the table: in
        # master.dbo.sysdatabases the dangerous part is the database name,
        # which sits two positions from the end.
        for qualifier in parts[:-1]:
            if qualifier in _FORBIDDEN_SCHEMAS:
                raise UnsafeQuery(
                    f"the query reads from the system schema or database "
                    f"'{qualifier}', which is not permitted"
                )
        if (
            bare.startswith("pg_")
            or bare.startswith("sqlite_")
            or bare in _SYSTEM_TABLES
        ):
            raise UnsafeQuery(
                f"the query reads the system table '{table}', which is not permitted"
            )

        if allowed_tables or allowed_schemas:
            table_ok = table in allowed_tables or bare in allowed_tables
            schema_ok = bool(schema) and schema in allowed_schemas
            if not (table_ok or schema_ok):
                raise UnsafeQuery(
                    f"the query reads '{table}', which is not in the reporting "
                    "allowlist. Add it to ELP_REPORTS__ALLOWED_TABLES if "
                    "operations should be able to report on it."
                )
    return tables


def _main_select_index(masked: str) -> int | None:
    """
    Offset of the statement's outermost SELECT.

    For a plain query that is the first token. For ``WITH x AS (...) SELECT``
    the CTE bodies sit inside parentheses, so the first SELECT at paren depth
    zero is still the main one.
    """
    depth = 0
    for index, character in enumerate(masked):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and character in "sS":
            match = _SELECT_TOKEN.match(masked, index)
            if match:
                return index
    return None


def enforce_limit(
    sql: str, max_rows: int, dialect: str = "postgresql"
) -> tuple[str, int | None]:
    """
    Bound the row count when the query does not bound itself.

    A report query without a limit is the difference between a slow morning
    and an incident, and the model omits one more often than you would like.

    The syntax is not portable: PostgreSQL and MySQL take a trailing
    ``LIMIT n``, SQL Server takes ``TOP (n)`` immediately after SELECT.
    Emitting the wrong one does not degrade gracefully - it is a syntax
    error, and every generated query would fail.
    """
    masked = _mask(sql)
    if _LIMIT_PRESENT.search(masked):
        return sql, None

    trimmed = sql.rstrip().rstrip(";")

    if dialect in {"mssql", "sqlserver"}:
        index = _main_select_index(_mask(trimmed))
        if index is None:
            # Nothing recognisable to inject into. The fetch cap in the data
            # source still bounds the result, so this is a note, not a fault.
            return trimmed, None
        insert_at = index + len("select")
        return f"{trimmed[:insert_at]} TOP ({max_rows}){trimmed[insert_at:]}", max_rows

    return f"{trimmed}\nLIMIT {max_rows}", max_rows


def validate(sql: str, settings: ReportSettings | None = None) -> GuardResult:
    """
    Full validation. Returns the query that should actually be executed.

    Raises :class:`UnsafeQuery` with a plain-language reason, which is shown
    to the person who asked for the report - they are usually best placed to
    rephrase the request.
    """
    settings = settings or get_settings().reports

    body = check_read_only(sql)
    tables = check_tables(body, settings)

    notes: list[str] = []
    safe_sql, limit_applied = enforce_limit(
        sql.strip().rstrip(";"), settings.max_rows, settings.sql_dialect
    )
    if limit_applied:
        notes.append(
            f"no row limit was specified, so LIMIT {limit_applied} was added"
        )
    if not tables:
        notes.append(
            "no table references were detected; check the query does what you expect"
        )

    return GuardResult(
        sql=safe_sql, tables=tables, limit_applied=limit_applied, notes=notes
    )


def redact_columns(
    columns: list[str], settings: ReportSettings | None = None
) -> set[int]:
    """Indices of columns whose values must not leave the database."""
    settings = settings or get_settings().reports
    patterns = [p.lower() for p in settings.redacted_columns]
    return {
        index
        for index, name in enumerate(columns)
        if any(pattern in (name or "").lower() for pattern in patterns)
    }
