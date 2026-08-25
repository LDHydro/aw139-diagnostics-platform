"""
The NAMIS field catalogue.

Exported by the existing NAMIS report generator from the decompiled client:
585 tables across four databases on one SQL Server instance, 8,000-odd
columns with their real SQL types, and 723 relationships with the join
columns already worked out.

This is the single most valuable artifact in the reporting path. Schema
introspection can recover column names, but it cannot tell you that
``MAINTACTION`` joins ``WORKREQUEST`` on two columns, or that ``AssetKey``
and ``AssetSite`` travel together as a compound key. Grounding query
authoring in this catalogue is what separates a query that runs from one
that merely looks plausible.

**The whole catalogue does not go in a prompt.** Eight thousand columns
would not fit, and if they did they would bury the ten that matter. Instead
the request is scored against table names, group names and column names, the
best few tables are selected, and their directly-related neighbours are
pulled in so the model has the joins it needs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]{3,}")

# Tokens that appear in nearly every request and carry no signal about which
# table is wanted.
_STOPWORDS = frozenset(
    {
        "all", "and", "any", "are", "but", "can", "for", "from", "get", "has",
        "have", "how", "into", "list", "many", "much", "not", "off", "out",
        "over", "show", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "was", "were", "what", "when",
        "where", "which", "who", "will", "with", "report", "reports", "data",
        "please", "give", "want", "need", "each", "per", "last", "past",
        "days", "day", "week", "month", "year", "total", "count", "number",
    }
)


@dataclass(slots=True)
class ColumnSpec:
    name: str
    sql: str = ""
    kind: str = ""
    nullable: bool = True
    pk: bool = False
    lookup: str | None = None

    def render(self) -> str:
        text = f"{self.name} {self.sql}".strip()
        if self.pk:
            text += " PK"
        return text


@dataclass(slots=True)
class TableSpec:
    name: str
    table: str = ""
    schema: str = "dbo"
    database: str = ""
    group: str = ""
    object_type: str = "TABLE"
    read_only: bool = False
    row_count: int = 0
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def qualified(self) -> str:
        """Bracket-quoted three-part name, as the generator emits."""
        table = self.table or self.name
        if self.database:
            return f"[{self.database}].[{self.schema}].[{table}]"
        return f"[{self.schema}].[{table}]"

    @property
    def plain(self) -> str:
        table = self.table or self.name
        parts = [p for p in (self.database, self.schema, table) if p]
        return ".".join(parts).lower()

    def column(self, name: str) -> ColumnSpec | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


@dataclass(slots=True)
class Relationship:
    left: str
    right: str
    on: list[tuple[str, str]] = field(default_factory=list)
    database: str = ""
    confidence: str = ""

    def render(self, catalog: NamisCatalog | None = None) -> str:
        def side(name: str) -> str:
            spec = catalog.get(name) if catalog else None
            return spec.qualified if spec else name

        conditions = " AND ".join(
            f"{self.left}.{left_column} = {self.right}.{right_column}"
            for left_column, right_column in self.on
        )
        marker = "" if self.confidence == "fk" else f"  ({self.confidence or 'inferred'})"
        return f"{side(self.left)} JOIN {side(self.right)} ON {conditions}{marker}"


class NamisCatalog:
    def __init__(
        self,
        tables: dict[str, TableSpec],
        relationships: list[Relationship],
        groups: dict[str, list[str]],
        lookups: dict[str, object] | None = None,
    ) -> None:
        self.tables = tables
        self.relationships = relationships
        self.groups = groups
        self.lookups = lookups or {}
        self._by_lower = {name.lower(): spec for name, spec in tables.items()}
        self._edges: dict[str, list[Relationship]] = {}
        for relationship in relationships:
            self._edges.setdefault(relationship.left.lower(), []).append(relationship)
            self._edges.setdefault(relationship.right.lower(), []).append(relationship)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tables)

    def get(self, name: str) -> TableSpec | None:
        spec = self._by_lower.get(name.lower())
        if spec is not None:
            return spec
        # Accept a qualified reference for an unqualified catalogue entry.
        bare = name.split(".")[-1].strip("[]").lower()
        return self._by_lower.get(bare)

    @property
    def column_count(self) -> int:
        return sum(len(t.columns) for t in self.tables.values())

    @property
    def databases(self) -> list[str]:
        return sorted({t.database for t in self.tables.values() if t.database})

    def allowed_table_names(self) -> list[str]:
        """
        Every form the guard might see a catalogued table written as.

        The catalogue is the allowlist: a table absent from it is one the
        report generator never knew about, and is not a table operations
        should be reporting on by accident.
        """
        names: set[str] = set()
        for spec in self.tables.values():
            table = (spec.table or spec.name).lower()
            names.add(table)
            if spec.schema:
                names.add(f"{spec.schema.lower()}.{table}")
            if spec.database and spec.schema:
                names.add(f"{spec.database.lower()}.{spec.schema.lower()}.{table}")
        return sorted(names)

    def related(self, name: str) -> list[Relationship]:
        return self._edges.get(name.lower(), [])

    def relationships_among(self, names: list[str]) -> list[Relationship]:
        wanted = {n.lower() for n in names}
        seen: set[tuple] = set()
        found: list[Relationship] = []
        for relationship in self.relationships:
            if (
                relationship.left.lower() in wanted
                and relationship.right.lower() in wanted
            ):
                key = (relationship.left, relationship.right, tuple(relationship.on))
                if key not in seen:
                    seen.add(key)
                    found.append(relationship)
        return found

    # ------------------------------------------------------------------
    # Selecting a workable subset
    # ------------------------------------------------------------------

    def score_table(self, spec: TableSpec, tokens: set[str]) -> float:
        """How well one table matches the words in a request."""
        if not tokens:
            return 0.0

        name_tokens = set(_WORD.findall((spec.table or spec.name).lower()))
        # Names like WORKREQUEST are one token; also match on substrings so
        # "work request" finds it.
        flat_name = (spec.table or spec.name).lower()

        score = 0.0
        for token in tokens:
            if token in name_tokens:
                score += 3.0
            elif len(token) >= 4 and token in flat_name:
                score += 2.0

        group_tokens = set(_WORD.findall(spec.group.lower()))
        score += 1.5 * len(tokens & group_tokens)

        column_hits = sum(
            1 for column in spec.columns if any(t in column.name.lower() for t in tokens)
        )
        # Column matches are weak evidence individually - PERSONNEL has a
        # Status column and so does everything else - so they saturate.
        score += min(2.0, 0.25 * column_hits)

        # A table nobody references is unlikely to be the answer.
        if spec.row_count:
            score += 0.25
        return score

    def select_for_request(
        self, request_text: str, *, limit: int = 18, expand: bool = True
    ) -> list[TableSpec]:
        """
        Pick the tables worth showing the model for this request.

        Direct matches first, then their joinable neighbours - a request
        about work requests almost always needs PERSONNEL and WORKCENTER too,
        and the model cannot ask for them once the prompt is built.
        """
        tokens = {
            token
            for token in _WORD.findall(request_text.lower())
            if token not in _STOPWORDS
        }
        if not tokens:
            return []

        scored = [
            (self.score_table(spec, tokens), spec) for spec in self.tables.values()
        ]
        scored = [(score, spec) for score, spec in scored if score > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))

        direct_limit = max(1, limit // 2) if expand else limit
        selected: list[TableSpec] = [spec for _score, spec in scored[:direct_limit]]
        chosen = {spec.name.lower() for spec in selected}

        if expand:
            for spec in list(selected):
                for relationship in self.related(spec.name):
                    for side in (relationship.left, relationship.right):
                        if len(selected) >= limit:
                            break
                        neighbour = self.get(side)
                        if neighbour and neighbour.name.lower() not in chosen:
                            chosen.add(neighbour.name.lower())
                            selected.append(neighbour)
                if len(selected) >= limit:
                    break

        return selected[:limit]

    # ------------------------------------------------------------------
    # Rendering for the prompt
    # ------------------------------------------------------------------

    def render_for_prompt(
        self,
        tables: list[TableSpec],
        *,
        max_columns_per_table: int = 40,
    ) -> str:
        """Compact schema plus the joins between the selected tables."""
        if not tables:
            return "(no matching tables were found in the NAMIS catalogue)"

        lines: list[str] = []
        for spec in tables:
            columns = spec.columns[:max_columns_per_table]
            rendered = ", ".join(c.render() for c in columns)
            if len(spec.columns) > max_columns_per_table:
                rendered += f", ... (+{len(spec.columns) - max_columns_per_table} more)"
            header = spec.qualified
            if spec.object_type and spec.object_type.upper() != "TABLE":
                header += f"  [{spec.object_type}]"
            if spec.group:
                header += f"  -- {spec.group}"
            lines.append(f"{header}\n    {rendered}")

        joins = self.relationships_among([t.name for t in tables])
        if joins:
            lines.append("\nKNOWN JOINS (use these exactly; compound keys must use every column)")
            for relationship in joins:
                lines.append(f"  {relationship.render(self)}")

        return "\n".join(lines)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _column_from(raw: dict) -> ColumnSpec:
    return ColumnSpec(
        name=str(raw.get("name", "")),
        sql=str(raw.get("sql") or raw.get("type") or ""),
        kind=str(raw.get("kind") or ""),
        nullable=bool(raw.get("nullable", True)),
        pk=bool(raw.get("pk", False)),
        lookup=raw.get("lookup"),
    )


def _table_from(name: str, raw: dict) -> TableSpec:
    columns = raw.get("columns") or raw.get("fields") or []
    return TableSpec(
        name=str(raw.get("name") or name),
        table=str(raw.get("table") or raw.get("name") or name),
        schema=str(raw.get("schema") or "dbo"),
        database=str(raw.get("database") or ""),
        group=str(raw.get("group") or ""),
        object_type=str(raw.get("objectType") or raw.get("object_type") or "TABLE"),
        read_only=bool(raw.get("readOnly", raw.get("read_only", False))),
        row_count=int(raw.get("rowCount") or raw.get("row_count") or 0),
        columns=[_column_from(c) for c in columns if isinstance(c, dict)],
    )


def _relationship_from(raw: dict) -> Relationship | None:
    left = raw.get("left") or raw.get("from_table") or raw.get("from")
    right = raw.get("right") or raw.get("to_table") or raw.get("to")
    if not left or not right:
        return None

    pairs: list[tuple[str, str]] = []
    for condition in raw.get("on") or []:
        if isinstance(condition, dict):
            left_column = condition.get("leftColumn") or condition.get("left")
            right_column = condition.get("rightColumn") or condition.get("right")
            if left_column and right_column:
                pairs.append((str(left_column), str(right_column)))
    if not pairs and raw.get("from_column") and raw.get("to_column"):
        pairs.append((str(raw["from_column"]), str(raw["to_column"])))
    if not pairs:
        return None

    return Relationship(
        left=str(left),
        right=str(right),
        on=pairs,
        database=str(raw.get("database") or ""),
        confidence=str(raw.get("confidence") or ""),
    )


def load_catalog(path: str | Path) -> NamisCatalog:
    """
    Read a catalogue exported by the NAMIS report generator.

    Tolerant of the shape differences between exports - ``columns`` or
    ``fields``, ``relationships`` or ``joins`` - because the alternative is
    a hard failure on a file that is 95% understood.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no NAMIS catalogue at {path}. Export it from the report generator "
            "(build-windows.sh regenerates it) and set ELP_REPORTS__CATALOG_PATH."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    raw_tables = data.get("tables") or {}
    tables: dict[str, TableSpec] = {}
    if isinstance(raw_tables, dict):
        for name, raw in raw_tables.items():
            if isinstance(raw, dict):
                spec = _table_from(name, raw)
                tables[spec.name] = spec
    elif isinstance(raw_tables, list):
        for raw in raw_tables:
            if isinstance(raw, dict) and raw.get("name"):
                spec = _table_from(raw["name"], raw)
                tables[spec.name] = spec

    raw_relationships = data.get("relationships") or data.get("joins") or []
    relationships = [
        rel
        for rel in (_relationship_from(r) for r in raw_relationships if isinstance(r, dict))
        if rel is not None
    ]

    groups: dict[str, list[str]] = {}
    for raw in data.get("groups") or []:
        if isinstance(raw, dict) and raw.get("name"):
            groups[str(raw["name"])] = [str(t) for t in raw.get("tables", [])]
    # Backfill each table's group from the group index when absent.
    for group_name, members in groups.items():
        for member in members:
            spec = tables.get(member)
            if spec is not None and not spec.group:
                spec.group = group_name

    catalog = NamisCatalog(
        tables=tables,
        relationships=relationships,
        groups=groups,
        lookups=data.get("lookups") or {},
    )
    log.info(
        "loaded NAMIS catalogue: %d tables, %d columns, %d relationships, %d groups "
        "across %s",
        len(catalog), catalog.column_count, len(relationships), len(groups),
        ", ".join(catalog.databases) or "one database",
    )
    return catalog


_catalog: NamisCatalog | None = None
_catalog_path: str = ""


def get_catalog(path: str | Path | None = None) -> NamisCatalog | None:
    """Load and cache the catalogue, or return ``None`` when none is configured."""
    global _catalog, _catalog_path

    from ..config import get_settings

    resolved = str(path or get_settings().reports.catalog_path)
    if _catalog is not None and _catalog_path == resolved:
        return _catalog

    try:
        _catalog = load_catalog(resolved)
        _catalog_path = resolved
        return _catalog
    except FileNotFoundError as exc:
        # Not fatal: without a catalogue the platform falls back to live
        # schema introspection, which works but loses the join knowledge.
        log.warning("%s Falling back to live schema introspection.", exc)
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        log.error("the NAMIS catalogue at %s is not readable: %s", resolved, exc)
        return None


def reset_catalog() -> None:
    global _catalog, _catalog_path
    _catalog = None
    _catalog_path = ""
