"""
Structured report definitions, compiled to parameterised T-SQL.

The NAMIS report generator does not ask anyone to write SQL. A report is a
base table, some fields, some joins, some filters and a sort order, and the
tool turns that into a query. This module does the same thing, and it exists
for a reason beyond parity.

**A compiled structured report has no injection surface at all.** Every
identifier is resolved through the catalogue and emitted from catalogue
metadata - a column name that is not in the catalogue never reaches the SQL
text, it raises. Every value is emitted as a bound parameter. There is no
concatenation of user input anywhere. Contrast that with free-form SQL, where
the guard has to *prove a negative* about a string somebody else wrote.

Both paths are supported: raw SQL where expressiveness is needed, structured
where safety and portability matter more. Imported reports and anything the
model can express structurally should use this one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import NamisCatalog, TableSpec

log = logging.getLogger(__name__)

# Their builder caps a report at eight joined tables; beyond that the query
# plans stop being predictable and the output stops being readable.
MAX_JOIN_TABLES = 8

# Aggregate functions a field may carry.
AGGREGATES = frozenset({"count", "count_distinct", "sum", "avg", "min", "max"})

JOIN_KINDS = {"inner": "INNER JOIN", "left": "LEFT JOIN"}

# Operator -> SQL template. `{c}` is the (catalogue-resolved) column
# expression, `{p}` the bound parameter name. Nothing else is interpolated.
_OPERATORS: dict[str, str] = {
    "eq": "{c} = :{p}",
    "ne": "{c} <> :{p}",
    "gt": "{c} > :{p}",
    "gte": "{c} >= :{p}",
    "lt": "{c} < :{p}",
    "lte": "{c} <= :{p}",
    "contains": "{c} LIKE :{p}",
    "starts_with": "{c} LIKE :{p}",
    "ends_with": "{c} LIKE :{p}",
    "is_null": "{c} IS NULL",
    "not_null": "{c} IS NOT NULL",
    "is_true": "{c} = 1",
    "is_false": "{c} = 0",
    "in": "{c} IN ({p})",
    "not_in": "{c} NOT IN ({p})",
    "between": "{c} BETWEEN :{p}_lo AND :{p}_hi",
    # Relative dates, so a saved report stays correct next month.
    "last_n_days": "{c} >= DATEADD(day, -:{p}, GETDATE())",
    "next_n_days": "{c} BETWEEN GETDATE() AND DATEADD(day, :{p}, GETDATE())",
    "older_than_days": "{c} < DATEADD(day, -:{p}, GETDATE())",
}

# Operators that take no value at all.
_NULLARY = {"is_null", "not_null", "is_true", "is_false"}
# Operators whose operand must be a number of days.
_DAY_COUNT = {"last_n_days", "next_n_days", "older_than_days"}

_TEXT_KINDS = {"text", "string"}
_NUMBER_KINDS = {"number", "numeric", "int", "decimal", "bool"}
_DATE_KINDS = {"date", "datetime", "time"}


class ReportSpecError(ValueError):
    """The structured definition is not valid against the catalogue."""


# ----------------------------------------------------------------------
# The definition
# ----------------------------------------------------------------------

@dataclass
class FieldRef:
    table: str
    column: str
    alias: str = ""
    aggregate: str = ""

    def output_name(self) -> str:
        if self.alias:
            return self.alias
        if self.aggregate:
            return f"{self.aggregate}_{self.column}"
        return self.column


@dataclass
class JoinCondition:
    left_table: str
    left_column: str
    right_table: str
    right_column: str


@dataclass
class JoinSpec:
    table: str
    kind: str = "left"
    on: list[JoinCondition] = field(default_factory=list)


@dataclass
class FilterSpec:
    table: str
    column: str
    op: str
    value: Any = None
    values: list[Any] = field(default_factory=list)


@dataclass
class SortSpec:
    table: str
    column: str
    direction: str = "asc"


@dataclass
class StructuredReport:
    base_table: str
    fields: list[FieldRef] = field(default_factory=list)
    joins: list[JoinSpec] = field(default_factory=list)
    filters: list[FilterSpec] = field(default_factory=list)
    # How the filters combine. Their builder ANDs them.
    filter_logic: str = "and"
    sort: list[SortSpec] = field(default_factory=list)
    group_by: list[FieldRef] = field(default_factory=list)
    row_limit: int = 1000
    # Resolve coded IDs to names where the catalogue knows the value list.
    resolve_lookups: bool = True

    def to_dict(self) -> dict:
        return {
            "base_table": self.base_table,
            "fields": [
                {
                    "table": f.table, "column": f.column,
                    "alias": f.alias, "aggregate": f.aggregate,
                }
                for f in self.fields
            ],
            "joins": [
                {
                    "table": j.table,
                    "kind": j.kind,
                    "on": [
                        {
                            "left_table": c.left_table, "left_column": c.left_column,
                            "right_table": c.right_table, "right_column": c.right_column,
                        }
                        for c in j.on
                    ],
                }
                for j in self.joins
            ],
            "filters": [
                {
                    "table": f.table, "column": f.column, "op": f.op,
                    "value": f.value, "values": f.values,
                }
                for f in self.filters
            ],
            "filter_logic": self.filter_logic,
            "sort": [
                {"table": s.table, "column": s.column, "direction": s.direction}
                for s in self.sort
            ],
            "group_by": [
                {"table": g.table, "column": g.column} for g in self.group_by
            ],
            "row_limit": self.row_limit,
            "resolve_lookups": self.resolve_lookups,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StructuredReport:
        return cls(
            base_table=str(data.get("base_table") or data.get("table") or ""),
            fields=[
                FieldRef(
                    table=str(f.get("table", "")),
                    column=str(f.get("column", "")),
                    alias=str(f.get("alias", "") or ""),
                    aggregate=str(f.get("aggregate", "") or ""),
                )
                for f in data.get("fields", [])
                if isinstance(f, dict)
            ],
            joins=[
                JoinSpec(
                    table=str(j.get("table", "")),
                    kind=str(j.get("kind", "left")).lower(),
                    on=[
                        JoinCondition(
                            left_table=str(c.get("left_table", "")),
                            left_column=str(c.get("left_column", "")),
                            right_table=str(c.get("right_table", "")),
                            right_column=str(c.get("right_column", "")),
                        )
                        for c in j.get("on", [])
                        if isinstance(c, dict)
                    ],
                )
                for j in data.get("joins", [])
                if isinstance(j, dict)
            ],
            filters=[
                FilterSpec(
                    table=str(f.get("table", "")),
                    column=str(f.get("column", "")),
                    op=str(f.get("op", "eq")).lower(),
                    value=f.get("value"),
                    values=list(f.get("values") or []),
                )
                for f in data.get("filters", [])
                if isinstance(f, dict)
            ],
            filter_logic=str(data.get("filter_logic", "and")).lower(),
            sort=[
                SortSpec(
                    table=str(s.get("table", "")),
                    column=str(s.get("column", "")),
                    direction=str(s.get("direction", "asc")).lower(),
                )
                for s in data.get("sort", [])
                if isinstance(s, dict)
            ],
            group_by=[
                FieldRef(table=str(g.get("table", "")), column=str(g.get("column", "")))
                for g in data.get("group_by", [])
                if isinstance(g, dict)
            ],
            row_limit=int(data.get("row_limit") or 1000),
            resolve_lookups=bool(data.get("resolve_lookups", True)),
        )


@dataclass
class CompiledQuery:
    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------

_SAFE_ALIAS = re.compile(r"[^A-Za-z0-9_ ]")


def _quote_alias(alias: str) -> str:
    """
    Emit an output alias safely.

    Aliases are the one place a user-supplied *string* legitimately appears
    in the SQL text, so it is stripped to a conservative character set and
    bracket-quoted. Anything else would be an injection point.
    """
    cleaned = _SAFE_ALIAS.sub("", alias).strip()
    return cleaned[:120] or "column"


def _kind_of(spec: TableSpec, column: str) -> str:
    found = spec.column(column)
    return (found.kind if found else "").lower()


class ReportCompiler:
    """Turns a validated structured definition into parameterised T-SQL."""

    def __init__(self, catalog: NamisCatalog, *, max_rows: int = 5000) -> None:
        self.catalog = catalog
        self.max_rows = max_rows

    # -- validation helpers -------------------------------------------

    def _table(self, name: str) -> TableSpec:
        spec = self.catalog.get(name)
        if spec is None:
            raise ReportSpecError(
                f"'{name}' is not in the NAMIS catalogue. Check the spelling, or "
                "regenerate the catalogue if the table is new."
            )
        return spec

    def _column(self, table: str, column: str) -> str:
        spec = self._table(table)
        found = spec.column(column)
        if found is None:
            close = [
                c.name for c in spec.columns if column.lower() in c.name.lower()
            ][:4]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ReportSpecError(
                f"'{table}' has no column '{column}'.{hint}"
            )
        # Return the catalogue's spelling, never the caller's.
        return found.name

    # -- compilation ---------------------------------------------------

    def compile(self, report: StructuredReport) -> CompiledQuery:
        if not report.base_table:
            raise ReportSpecError("the report has no base table")
        if not report.fields:
            raise ReportSpecError("the report selects no fields")

        base = self._table(report.base_table)
        aliases: dict[str, str] = {base.name: "T0"}
        warnings: list[str] = []

        # --- joins ----------------------------------------------------
        if len(report.joins) + 1 > MAX_JOIN_TABLES:
            raise ReportSpecError(
                f"a report may join at most {MAX_JOIN_TABLES} tables; this one "
                f"uses {len(report.joins) + 1}. Remove one, or split the report."
            )

        join_sql: list[str] = []
        for index, join in enumerate(report.joins, start=1):
            joined = self._table(join.table)
            if joined.name in aliases:
                raise ReportSpecError(
                    f"'{join.table}' is already in this report"
                )
            kind = JOIN_KINDS.get(join.kind)
            if kind is None:
                raise ReportSpecError(
                    f"unknown join type '{join.kind}'; use 'inner' or 'left'"
                )
            if not join.on:
                raise ReportSpecError(
                    f"the join to '{join.table}' has no join condition, which "
                    "would produce a cartesian product"
                )

            aliases[joined.name] = f"T{index}"
            conditions: list[str] = []
            for condition in join.on:
                left_spec = self._table(condition.left_table)
                right_spec = self._table(condition.right_table)
                if left_spec.name not in aliases or right_spec.name not in aliases:
                    raise ReportSpecError(
                        f"the join condition references '{condition.left_table}' or "
                        f"'{condition.right_table}' before it is part of the report"
                    )
                left_column = self._column(condition.left_table, condition.left_column)
                right_column = self._column(condition.right_table, condition.right_column)

                left_kind = _kind_of(left_spec, left_column)
                right_kind = _kind_of(right_spec, right_column)
                if (
                    left_kind
                    and right_kind
                    and left_kind != right_kind
                    and not ({left_kind, right_kind} <= _NUMBER_KINDS)
                ):
                    raise ReportSpecError(
                        f"cannot join {left_spec.name}.{left_column} ({left_kind}) to "
                        f"{right_spec.name}.{right_column} ({right_kind}): the columns "
                        "hold different kinds of value"
                    )
                conditions.append(
                    f"{aliases[left_spec.name]}.[{left_column}] = "
                    f"{aliases[right_spec.name]}.[{right_column}]"
                )

            if left_spec.database and right_spec.database and (
                left_spec.database != right_spec.database
            ):
                warnings.append(
                    f"cross-database join between {left_spec.database} and "
                    f"{right_spec.database}; both must be on the connected instance"
                )

            join_sql.append(
                f"{kind} {joined.qualified} AS {aliases[joined.name]} "
                f"ON {' AND '.join(conditions)}"
            )

        # --- select list ----------------------------------------------
        select_parts: list[str] = []
        columns: list[str] = []
        for ref in report.fields:
            spec = self._table(ref.table)
            if spec.name not in aliases:
                raise ReportSpecError(
                    f"field {ref.table}.{ref.column} refers to a table that is not "
                    "in this report; add a join for it first"
                )
            column = self._column(ref.table, ref.column)
            expression = f"{aliases[spec.name]}.[{column}]"

            if ref.aggregate:
                aggregate = ref.aggregate.lower()
                if aggregate not in AGGREGATES:
                    raise ReportSpecError(
                        f"unknown aggregate '{ref.aggregate}'; use one of "
                        + ", ".join(sorted(AGGREGATES))
                    )
                if aggregate == "count_distinct":
                    expression = f"COUNT(DISTINCT {expression})"
                else:
                    expression = f"{aggregate.upper()}({expression})"

            alias = _quote_alias(ref.output_name())
            select_parts.append(f"{expression} AS [{alias}]")
            columns.append(alias)

        # --- filters ---------------------------------------------------
        where_parts: list[str] = []
        parameters: dict[str, Any] = {}
        for index, spec_filter in enumerate(report.filters):
            table = self._table(spec_filter.table)
            if table.name not in aliases:
                raise ReportSpecError(
                    f"filter on {spec_filter.table}.{spec_filter.column} refers to a "
                    "table that is not in this report"
                )
            column = self._column(spec_filter.table, spec_filter.column)
            template = _OPERATORS.get(spec_filter.op)
            if template is None:
                raise ReportSpecError(
                    f"unknown filter operator '{spec_filter.op}'; use one of "
                    + ", ".join(sorted(_OPERATORS))
                )

            expression = f"{aliases[table.name]}.[{column}]"
            name = f"f{index}"

            if spec_filter.op in _NULLARY:
                where_parts.append(template.format(c=expression, p=name))
                continue

            if spec_filter.op in {"in", "not_in"}:
                values = spec_filter.values or (
                    [spec_filter.value] if spec_filter.value is not None else []
                )
                if not values:
                    raise ReportSpecError(
                        f"the '{spec_filter.op}' filter on {column} has no values"
                    )
                placeholders = []
                for value_index, value in enumerate(values[:500]):
                    key = f"{name}_{value_index}"
                    parameters[key] = value
                    placeholders.append(f":{key}")
                where_parts.append(
                    template.format(c=expression, p=", ".join(placeholders))
                )
                continue

            if spec_filter.op == "between":
                values = spec_filter.values or []
                if len(values) != 2:
                    raise ReportSpecError(
                        f"the 'between' filter on {column} needs exactly two values"
                    )
                parameters[f"{name}_lo"], parameters[f"{name}_hi"] = values
                where_parts.append(template.format(c=expression, p=name))
                continue

            value = spec_filter.value
            if spec_filter.op in _DAY_COUNT:
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise ReportSpecError(
                        f"the '{spec_filter.op}' filter on {column} needs a number "
                        "of days"
                    ) from exc
                column_kind = _kind_of(table, column)
                if column_kind and column_kind not in _DATE_KINDS:
                    warnings.append(
                        f"{column} is a {column_kind} column but is being filtered "
                        "as a date"
                    )
            elif spec_filter.op == "contains":
                value = f"%{value}%"
            elif spec_filter.op == "starts_with":
                value = f"{value}%"
            elif spec_filter.op == "ends_with":
                value = f"%{value}"

            parameters[name] = value
            where_parts.append(template.format(c=expression, p=name))

        # --- grouping and ordering -------------------------------------
        group_parts: list[str] = []
        has_aggregate = any(f.aggregate for f in report.fields)
        if report.group_by:
            for ref in report.group_by:
                table = self._table(ref.table)
                column = self._column(ref.table, ref.column)
                group_parts.append(f"{aliases[table.name]}.[{column}]")
        elif has_aggregate:
            # Every non-aggregated field must be grouped, or SQL Server
            # refuses the query. Doing it silently is friendlier than an
            # error the requester cannot act on.
            for ref in report.fields:
                if ref.aggregate:
                    continue
                table = self._table(ref.table)
                column = self._column(ref.table, ref.column)
                group_parts.append(f"{aliases[table.name]}.[{column}]")
            if group_parts:
                warnings.append(
                    "grouped automatically by the non-aggregated fields"
                )

        order_parts: list[str] = []
        for sort in report.sort:
            table = self._table(sort.table)
            if table.name not in aliases:
                raise ReportSpecError(
                    f"sort on {sort.table}.{sort.column} refers to a table that is "
                    "not in this report"
                )
            column = self._column(sort.table, sort.column)
            direction = "DESC" if sort.direction.lower().startswith("d") else "ASC"
            order_parts.append(f"{aliases[table.name]}.[{column}] {direction}")

        if not order_parts:
            # A report whose row order changes between runs is not a report.
            first = report.fields[0]
            table = self._table(first.table)
            column = self._column(first.table, first.column)
            order_parts.append(f"{aliases[table.name]}.[{column}] ASC")
            warnings.append(
                "no sort order was given, so results are ordered by the first field"
            )

        # --- assemble ---------------------------------------------------
        limit = max(1, min(int(report.row_limit or self.max_rows), self.max_rows))
        lines = [
            f"SELECT TOP ({limit})",
            "    " + ",\n    ".join(select_parts),
            f"FROM {base.qualified} AS T0",
        ]
        lines.extend(join_sql)
        if where_parts:
            joiner = " OR " if report.filter_logic == "or" else " AND "
            lines.append("WHERE " + joiner.join(where_parts))
        if group_parts:
            lines.append("GROUP BY " + ", ".join(group_parts))
        lines.append("ORDER BY " + ", ".join(order_parts))

        return CompiledQuery(
            sql="\n".join(lines),
            parameters=parameters,
            columns=columns,
            tables=[name for name in aliases],
            warnings=warnings,
        )


def compile_report(
    report: StructuredReport, catalog: NamisCatalog, *, max_rows: int = 5000
) -> CompiledQuery:
    return ReportCompiler(catalog, max_rows=max_rows).compile(report)
