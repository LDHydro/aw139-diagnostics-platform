"""
Turning a plain-language report request into a query, and results back into prose.

Two model calls, with different risk profiles and different guardrails.

**Authoring** happens once, at draft time, and its output is reviewed by a
person before it can ever run unattended. The model is grounded in the real
introspected schema, so it cannot invent column names, and its output is put
through the SQL guard before anyone sees it. One repair round is allowed
using the validator's own message, because models produce nearly-valid SQL
far more often than invalid SQL.

**Narration** happens on every run, and is the riskier of the two: it faces
real data and its output goes to management. It is given only a sample of
rows and is told, firmly, that every figure it states must be one it can see.
Numbers in a report that nobody can reproduce are worse than no report.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..config import ReportSettings, get_settings
from ..llm.client import ChatMessage
from ..llm.router import TaskKind, get_router
from .datasource import QueryResult, TableInfo, render_schema_catalogue
from .sqlguard import UnsafeQuery, validate

log = logging.getLogger(__name__)

_AUTHOR_PROMPT = """\
You write read-only {dialect_name} queries for an aviation maintenance \
department's operational reports, against the NAMIS system.

SCHEMA — these are the only tables and columns that exist. Column types are \
shown; respect them.
{schema}

RULES
- Output a SINGLE SELECT (or WITH ... SELECT) statement. Nothing else.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, GRANT, EXEC, \
BACKUP, DBCC or any other statement that changes data, schema or server state.
- Use ONLY tables and columns from the schema above. If the request needs \
something that is not there, say so in the EXPLANATION instead of inventing \
a column name.
- Join using the KNOWN JOINS listed above, exactly as given. Where a join has \
several columns it is a compound key — use every column, or you will get a \
cartesian product that looks like real data.
{dialect_rules}
- Alias computed columns with clear, human-readable names: operations staff \
read this output, not engineers.
- Always include an explicit ORDER BY so results are stable between runs.

OUTPUT FORMAT — exactly these three sections, in this order:

SQL:
<the statement>

EXPLANATION:
<two or three sentences describing what the query returns, in plain language>

ASSUMPTIONS:
<one bullet per assumption you had to make; write "none" if there were none>
"""

# Dialect differences that produce a syntax error rather than a subtly wrong
# answer, so they are stated explicitly rather than left to the model.
_DIALECT_RULES = {
    "mssql": """\
- This is Microsoft SQL Server (T-SQL). Limit rows with SELECT TOP (n), \
NOT with LIMIT — LIMIT is a syntax error here.
- Quote identifiers with square brackets and write three-part names for \
tables outside the default database, e.g. [AMO_NASAWeb].[dbo].[FlightRecordHeaders]. \
Never write a four-part name: that reaches a linked server and is refused.
- Date arithmetic uses GETDATE() and DATEADD/DATEDIFF, e.g. \
WHERE CreatedDt >= DATEADD(day, -30, GETDATE()). There is no INTERVAL syntax.
- Use ISNULL() or COALESCE(), and CAST/CONVERT for type changes.""",
    "postgresql": """\
- This is PostgreSQL. Limit rows with LIMIT n.
- Date arithmetic uses CURRENT_DATE - INTERVAL '30 days'.""",
    "mysql": """\
- This is MySQL. Limit rows with LIMIT n.
- Date arithmetic uses DATE_SUB(NOW(), INTERVAL 30 DAY).""",
}

_DIALECT_NAMES = {
    "mssql": "Microsoft SQL Server (T-SQL)",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
}

_REPAIR_PROMPT = """\
The query you wrote was rejected by the safety validator.

REJECTION: {error}

Return a corrected query in the same three-section format. Do not argue with \
the validator - it is not going to change its mind. If the request genuinely \
cannot be satisfied with a single read-only SELECT over the permitted tables, \
say so in the EXPLANATION and return `SELECT 1 WHERE false` as the SQL."""

_NARRATE_PROMPT = """\
You summarise an operational report for an aviation maintenance department.

RULES - a report nobody can reproduce is worse than no report:
- State ONLY figures you can see in the data below. Never estimate, \
extrapolate, or round to a "nicer" number.
- You are shown a SAMPLE of {sample_size} row(s) out of {row_count}. Do not \
describe the whole dataset as though you have seen it. Totals and counts \
given to you separately are reliable; anything you would have to compute \
across unseen rows is not.
- Lead with what changed or what needs attention, not with a description of \
the table. The reader can see the table.
- If nothing in the data warrants attention, say so plainly in one sentence.
- No preamble, no "here is your report", no closing pleasantries.
- Three short paragraphs at most."""


@dataclass
class QueryDraft:
    query: str
    explanation: str = ""
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    repaired: bool = False
    valid: bool = False
    rejection: str = ""
    # Catalogue tables shown to the model for this request.
    tables_offered: list[str] = field(default_factory=list)


def _extract_section(text: str, name: str) -> str:
    """Pull one labelled section out of the model's response."""
    pattern = re.compile(
        rf"^{name}:\s*\n?(.*?)(?=^\s*(?:SQL|EXPLANATION|ASSUMPTIONS):|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


def parse_draft(response: str) -> QueryDraft:
    sql = _strip_fences(_extract_section(response, "SQL"))
    explanation = _extract_section(response, "EXPLANATION")
    assumptions_text = _extract_section(response, "ASSUMPTIONS")

    assumptions = [
        line.strip(" -*\t")
        for line in assumptions_text.splitlines()
        if line.strip(" -*\t") and line.strip(" -*\t").lower() != "none"
    ]

    if not sql:
        # The model ignored the format; fall back to the whole response,
        # which the validator will reject if it is not a query.
        sql = _strip_fences(response)

    return QueryDraft(query=sql, explanation=explanation, assumptions=assumptions)


async def draft_query(
    request_text: str,
    tables: list[TableInfo],
    *,
    settings: ReportSettings | None = None,
    repair_attempts: int = 1,
) -> QueryDraft:
    """Author a query for a plain-language request, grounded in the real schema."""
    settings = settings or get_settings().reports

    # Prefer the exported catalogue: it carries the join relationships, and
    # it lets the schema shown be narrowed to what this request needs. The
    # full NAMIS schema is 8,000-odd columns - it would not fit in the
    # context window, and if it did the relevant tables would be buried.
    from .catalog import get_catalog

    catalog = get_catalog()
    selected_names: list[str] = []
    if catalog is not None and len(catalog):
        chosen = catalog.select_for_request(
            request_text, limit=settings.catalog_tables_per_request
        )
        if chosen:
            schema = catalog.render_for_prompt(chosen)
            selected_names = [c.name for c in chosen]
            log.info(
                "grounding the query in %d catalogued table(s): %s",
                len(chosen), ", ".join(selected_names[:8]),
            )
        else:
            schema = render_schema_catalogue(tables)
    elif tables:
        schema = render_schema_catalogue(tables)
    else:
        raise ValueError(
            "no NAMIS tables are visible and no catalogue is loaded, so a query "
            "cannot be authored. Check the reporting account's permissions, or "
            "set ELP_REPORTS__CATALOG_PATH."
        )

    dialect = settings.sql_dialect
    client, profile = get_router().resolve(TaskKind.CODE)

    messages = [
        ChatMessage(
            "system",
            _AUTHOR_PROMPT.format(
                schema=schema,
                dialect_name=_DIALECT_NAMES.get(dialect, dialect),
                dialect_rules=_DIALECT_RULES.get(dialect, ""),
            ),
        ),
        ChatMessage("user", f"REPORT REQUEST\n{request_text}"),
    ]

    draft = QueryDraft(query="")
    for attempt in range(repair_attempts + 1):
        completion = await client.chat(
            messages, temperature=0.0, max_tokens=profile.max_tokens
        )
        draft = parse_draft(completion.text)
        draft.repaired = attempt > 0

        try:
            guard = validate(draft.query, settings)
        except UnsafeQuery as exc:
            draft.valid = False
            draft.rejection = str(exc)
            if attempt < repair_attempts:
                log.info("authored query rejected (%s); repairing", exc)
                messages.append(ChatMessage("assistant", completion.text))
                messages.append(
                    ChatMessage("user", _REPAIR_PROMPT.format(error=exc))
                )
                continue
            return draft

        draft.valid = True
        draft.rejection = ""
        draft.tables = guard.tables
        draft.tables_offered = selected_names
        draft.warnings = list(guard.notes)
        # Surface the query that will actually run, LIMIT included, so the
        # approver reviews the real thing rather than a prettier version.
        draft.query = guard.sql
        return draft

    return draft


async def narrate(
    request_text: str,
    result: QueryResult,
    *,
    settings: ReportSettings | None = None,
) -> str:
    """Write a plain-language summary of a result set."""
    settings = settings or get_settings().reports
    if not result.rows:
        return (
            "The report returned no rows. Either there is nothing to report for "
            "this period, or the filters are narrower than intended."
        )

    sample = result.rows[: settings.narration_row_sample]
    header = " | ".join(result.columns)
    lines = [header, "-" * min(len(header), 120)]
    for row in sample:
        cells = []
        for value in row:
            text = "" if value is None else str(value)
            if len(text) > settings.max_cell_chars:
                text = text[: settings.max_cell_chars] + "…"
            cells.append(text)
        lines.append(" | ".join(cells))

    body = "\n".join(lines)
    if result.redacted_columns:
        body += (
            f"\n\n(columns withheld for privacy: "
            f"{', '.join(result.redacted_columns)})"
        )

    prompt = (
        f"REPORT REQUEST\n{request_text}\n\n"
        f"TOTAL ROWS RETURNED: {result.row_count}"
        + (" (truncated)" if result.truncated else "")
        + f"\n\nDATA SAMPLE\n{body}"
    )

    client, profile = get_router().resolve(TaskKind.SUMMARIZE)
    completion = await client.chat(
        [
            ChatMessage(
                "system",
                _NARRATE_PROMPT.format(
                    sample_size=len(sample), row_count=result.row_count
                ),
            ),
            ChatMessage("user", prompt),
        ],
        temperature=0.1,
        max_tokens=profile.max_tokens,
    )
    return completion.text.strip()
