"""
Importing reports saved by the NAMIS report generator.

Its saved reports are per-user JSON files under
``%LOCALAPPDATA%\\NamisReports\\saved-reports\\`` - one per report, holding a
base table, fields, joins, filters, sort and a row limit. That is the same
shape as :mod:`elp.reports.structured`, so an imported report becomes a
structured definition and inherits everything the platform adds: access
control, approval, scheduling, narration and the wider set of export formats.

**The key mapping here is inferred, not confirmed.** It was written from the
generator's user guide rather than from a sample file, so it accepts several
spellings for each field and reports anything it could not place instead of
guessing. Run it with ``dry_run`` first against one real export and send back
the ``unmapped`` list - that turns an inference into a fact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import NamisCatalog
from .structured import (
    FieldRef,
    FilterSpec,
    JoinCondition,
    JoinSpec,
    ReportSpecError,
    SortSpec,
    StructuredReport,
    compile_report,
)

log = logging.getLogger(__name__)

# Each platform field, and the key spellings an export might use for it.
_KEYS: dict[str, tuple[str, ...]] = {
    "name": ("name", "reportName", "title"),
    "description": ("description", "notes", "comment"),
    "base_table": ("table", "baseTable", "primaryTable", "rootTable", "from"),
    "fields": ("fields", "columns", "selectedFields", "select"),
    "joins": ("joins", "joinedTables", "relations"),
    "filters": ("filters", "where", "criteria", "conditions"),
    "sort": ("sort", "sorts", "orderBy", "sortBy"),
    "row_limit": ("rowLimit", "limit", "maxRows", "top"),
    "resolve_lookups": ("resolveLookups", "namesForCodedIds", "friendlyNames"),
    "filter_logic": ("filterLogic", "logic", "combine"),
}

# Their filter operators, mapped onto ours.
_OPERATORS: dict[str, str] = {
    "=": "eq", "==": "eq", "eq": "eq", "equals": "eq", "is": "eq",
    "!=": "ne", "<>": "ne", "ne": "ne", "notequals": "ne", "isnot": "ne",
    ">": "gt", "gt": "gt", "greaterthan": "gt",
    ">=": "gte", "gte": "gte", "atleast": "gte",
    "<": "lt", "lt": "lt", "lessthan": "lt",
    "<=": "lte", "lte": "lte", "atmost": "lte",
    "contains": "contains", "like": "contains",
    "startswith": "starts_with", "beginswith": "starts_with",
    "endswith": "ends_with",
    "isnull": "is_null", "isempty": "is_null", "isblank": "is_null",
    "isnotnull": "not_null", "isnotempty": "not_null",
    "istrue": "is_true", "yes": "is_true",
    "isfalse": "is_false", "no": "is_false",
    "in": "in", "oneof": "in",
    "notin": "not_in",
    "between": "between", "range": "between",
    "lastndays": "last_n_days", "inlastdays": "last_n_days", "recent": "last_n_days",
    "nextndays": "next_n_days", "duewithindays": "next_n_days",
    "olderthandays": "older_than_days", "openlongerthan": "older_than_days",
}


def _pick(data: dict, name: str, default: Any = None) -> Any:
    for key in _KEYS[name]:
        if key in data:
            return data[key]
        # Case-insensitive fallback: exports are not consistent about casing.
        for actual in data:
            if actual.lower() == key.lower():
                return data[actual]
    return default


def _normalise_operator(raw: Any) -> str:
    text = str(raw or "eq").strip().lower().replace(" ", "").replace("_", "")
    return _OPERATORS.get(text, "")


def _table_of(entry: dict, fallback: str) -> str:
    for key in ("table", "tableName", "entity", "source"):
        if entry.get(key):
            return str(entry[key])
    return fallback


def _column_of(entry: dict) -> str:
    for key in ("column", "field", "name", "columnName"):
        if entry.get(key):
            return str(entry[key])
    return ""


@dataclass
class ImportedReport:
    name: str
    description: str
    report: StructuredReport
    unmapped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_file: str = ""
    compiles: bool = False
    compile_error: str = ""
    sql_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "source_file": self.source_file,
            "base_table": self.report.base_table,
            "field_count": len(self.report.fields),
            "join_count": len(self.report.joins),
            "filter_count": len(self.report.filters),
            "compiles": self.compiles,
            "compile_error": self.compile_error,
            "unmapped": self.unmapped,
            "warnings": self.warnings,
            "sql_preview": self.sql_preview,
        }


def parse_saved_report(data: dict, *, source_file: str = "") -> ImportedReport:
    """Map one saved-report document onto a structured definition."""
    unmapped: list[str] = []
    warnings: list[str] = []

    name = str(_pick(data, "name") or Path(source_file).stem or "Imported report")
    base_table = str(_pick(data, "base_table") or "")
    if not base_table:
        raise ValueError(
            "the file names no base table. Expected one of: "
            + ", ".join(_KEYS["base_table"])
        )

    fields: list[FieldRef] = []
    for entry in _pick(data, "fields", []) or []:
        if isinstance(entry, str):
            fields.append(FieldRef(table=base_table, column=entry))
            continue
        if not isinstance(entry, dict):
            continue
        column = _column_of(entry)
        if not column:
            unmapped.append(f"field without a column name: {entry!r:.80}")
            continue
        fields.append(
            FieldRef(
                table=_table_of(entry, base_table),
                column=column,
                alias=str(entry.get("alias") or entry.get("label") or ""),
                aggregate=str(entry.get("aggregate") or entry.get("agg") or ""),
            )
        )

    joins: list[JoinSpec] = []
    for entry in _pick(data, "joins", []) or []:
        if not isinstance(entry, dict):
            continue
        table = _table_of(entry, "")
        if not table:
            unmapped.append(f"join without a table: {entry!r:.80}")
            continue
        kind = str(entry.get("kind") or entry.get("type") or entry.get("joinType") or "left")
        kind = "inner" if "inner" in kind.lower() else "left"

        conditions: list[JoinCondition] = []
        raw_conditions = entry.get("on") or entry.get("conditions") or []
        if isinstance(raw_conditions, dict):
            raw_conditions = [raw_conditions]
        for condition in raw_conditions:
            if not isinstance(condition, dict):
                continue
            left_table = (
                condition.get("left_table") or condition.get("leftTable")
                or condition.get("attachTo") or base_table
            )
            right_table = (
                condition.get("right_table") or condition.get("rightTable") or table
            )
            left_column = (
                condition.get("left_column") or condition.get("leftColumn")
                or condition.get("attachColumn")
            )
            right_column = (
                condition.get("right_column") or condition.get("rightColumn")
                or condition.get("targetColumn")
            )
            if not left_column or not right_column:
                unmapped.append(f"join condition without columns: {condition!r:.60}")
                continue
            conditions.append(
                JoinCondition(
                    left_table=str(left_table),
                    left_column=str(left_column),
                    right_table=str(right_table),
                    right_column=str(right_column),
                )
            )
        if not conditions:
            unmapped.append(f"join to '{table}' had no usable conditions")
            continue
        joins.append(JoinSpec(table=table, kind=kind, on=conditions))

    filters: list[FilterSpec] = []
    for entry in _pick(data, "filters", []) or []:
        if not isinstance(entry, dict):
            continue
        column = _column_of(entry)
        operator = _normalise_operator(
            entry.get("op") or entry.get("operator") or entry.get("comparison")
        )
        if not column:
            unmapped.append(f"filter without a column: {entry!r:.70}")
            continue
        if not operator:
            unmapped.append(
                f"filter on {column}: unrecognised operator "
                f"{entry.get('op') or entry.get('operator')!r}"
            )
            continue
        values = entry.get("values")
        filters.append(
            FilterSpec(
                table=_table_of(entry, base_table),
                column=column,
                op=operator,
                value=entry.get("value") if entry.get("value") is not None else entry.get("val"),
                values=list(values) if isinstance(values, list) else [],
            )
        )

    sort: list[SortSpec] = []
    for entry in _pick(data, "sort", []) or []:
        if isinstance(entry, str):
            sort.append(SortSpec(table=base_table, column=entry))
            continue
        if not isinstance(entry, dict):
            continue
        column = _column_of(entry)
        if not column:
            continue
        direction = str(
            entry.get("direction") or entry.get("dir") or entry.get("order") or "asc"
        )
        sort.append(
            SortSpec(
                table=_table_of(entry, base_table),
                column=column,
                direction="desc" if direction.lower().startswith("d") else "asc",
            )
        )

    row_limit = _pick(data, "row_limit", 1000)
    try:
        row_limit = int(row_limit)
    except (TypeError, ValueError):
        warnings.append(f"row limit {row_limit!r} was not a number; using 1000")
        row_limit = 1000

    logic = str(_pick(data, "filter_logic", "and") or "and").lower()

    report = StructuredReport(
        base_table=base_table,
        fields=fields,
        joins=joins,
        filters=filters,
        filter_logic="or" if logic.startswith("or") else "and",
        sort=sort,
        row_limit=row_limit,
        resolve_lookups=bool(_pick(data, "resolve_lookups", True)),
    )

    # Anything at the top level we did not consume is worth reporting: it may
    # be a feature this importer does not yet know about.
    consumed = {k.lower() for group in _KEYS.values() for k in group}
    for key in data:
        if key.lower() not in consumed:
            unmapped.append(f"unrecognised top-level key: {key}")

    return ImportedReport(
        name=name,
        description=str(_pick(data, "description") or ""),
        report=report,
        unmapped=unmapped,
        warnings=warnings,
        source_file=source_file,
    )


def import_directory(
    directory: str | Path, catalog: NamisCatalog | None = None, *, max_rows: int = 5000
) -> list[ImportedReport]:
    """
    Read every saved report in a directory.

    When a catalogue is supplied each import is compiled immediately, so a
    report that references a table or column the catalogue does not know is
    reported now rather than the first time it runs on a schedule.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"no such directory: {directory}")

    results: list[ImportedReport] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.error("%s is not valid JSON: %s", path.name, exc)
            continue
        if not isinstance(data, dict):
            log.error("%s does not contain a report object", path.name)
            continue

        try:
            imported = parse_saved_report(data, source_file=path.name)
        except ValueError as exc:
            log.error("%s could not be imported: %s", path.name, exc)
            continue

        if catalog is not None:
            try:
                compiled = compile_report(imported.report, catalog, max_rows=max_rows)
                imported.compiles = True
                imported.sql_preview = compiled.sql
                imported.warnings.extend(compiled.warnings)
            except ReportSpecError as exc:
                imported.compiles = False
                imported.compile_error = str(exc)

        results.append(imported)

    log.info(
        "read %d saved report(s) from %s; %d compile against the catalogue",
        len(results), directory, sum(1 for r in results if r.compiles),
    )
    return results
