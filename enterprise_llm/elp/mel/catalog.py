"""
Importing the approved Minimum Equipment List.

An MEL export is a table with one row per item and columns that differ
between operators. As with the maintenance programme importer, columns are
matched on aliases and every rejected row is reported, because a rejected
MEL row is relief that nobody knows they have - or worse, relief the system
would deny at a moment when it exists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..maintenance.schedule_io import _to_bool, _to_int, _to_list, read_rows
from ..models import MelItem
from .dispatch import CATEGORY_INTERVAL_DAYS, parse_category_a_interval

log = logging.getLogger(__name__)

_ALIASES: dict[str, tuple[str, ...]] = {
    "item_number": (
        "itemnumber", "item", "melitem", "itemno", "melno", "number",
        "reference", "ref", "atanumber", "sequence",
    ),
    "title": ("title", "description", "itemdescription", "equipment", "name", "system2"),
    "ata_chapter": ("ata", "atachapter", "chapter"),
    "system": ("system", "subsystem", "sectionname", "group"),
    "category": ("category", "cat", "repaircategory", "rectificationcategory", "interval"),
    "number_installed": ("numberinstalled", "installed", "qtyinstalled", "no1", "fitted"),
    "number_required": (
        "numberrequired", "required", "qtyrequired", "no2",
        "numberrequiredfordispatch", "requiredfordispatch",
    ),
    "remarks": ("remarks", "remarksorexceptions", "conditions", "exceptions", "provisos", "notes"),
    "operational_procedure": ("operationalprocedure", "oprocedure", "o", "opsprocedure"),
    "maintenance_procedure": ("maintenanceprocedure", "mprocedure", "m", "maintprocedure"),
    "placard_text": ("placard", "placardtext", "placarding"),
    "performance_penalty": ("penalty", "performancepenalty", "performance", "performanceeffect"),
    "incompatible_with": ("incompatiblewith", "notwith", "exclusions", "mutuallyexclusive"),
    "prohibited_operations": ("prohibitedoperations", "restrictions", "notpermitted", "limitations"),
    "extension_permitted": ("extensionpermitted", "extendable", "extension"),
    "applicable_models": ("model", "models", "aircraftmodel", "applicablemodels"),
    "applicable_configurations": ("configuration", "configurations", "config", "effectivity"),
    "applicable_serials": ("serial", "serials", "serialnumbers", "sn"),
}

# "C", "Cat C", "Category C", "C (10 days)"
_CATEGORY_RE = re.compile(r"\b(?:cat(?:egory)?\.?\s*)?([ABCD])\b", re.IGNORECASE)
_CDL_RE = re.compile(r"\bCDL\b", re.IGNORECASE)


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_column_map(headers: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for header in headers:
        key = _normalise(header)
        if not key:
            continue
        for field_name, aliases in _ALIASES.items():
            if key == _normalise(field_name) or key in aliases:
                mapping[header] = field_name
                break
    return mapping


def parse_category(raw: Any) -> str:
    """
    Read a rectification category from whatever the operator wrote.

    Falls back to an empty string rather than guessing: an item whose
    category cannot be read has no determinable interval, and inventing one
    would put a wrong airworthiness limit into the system.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if _CDL_RE.search(text):
        return "CDL"
    match = _CATEGORY_RE.search(text)
    return match.group(1).upper() if match else ""


@dataclass
class MelImportIssue:
    row: int
    item_number: str
    problem: str


@dataclass
class MelImportResult:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    issues: list[MelImportIssue] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    # Category A items whose interval had to be read from prose.
    category_a_parsed: list[dict] = field(default_factory=list)
    category_a_unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "created": self.created,
            "updated": self.updated,
            "skipped": self.skipped,
            "unmapped_columns": self.unmapped_columns,
            "category_a_parsed": self.category_a_parsed,
            "category_a_unresolved": self.category_a_unresolved,
            "issues": [
                {"row": i.row, "item_number": i.item_number, "problem": i.problem}
                for i in self.issues
            ],
        }


def parse_row(raw: dict[str, Any], column_map: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column, field_name in column_map.items():
        if column in raw:
            values[field_name] = raw[column]

    category = parse_category(values.get("category"))
    remarks = str(values.get("remarks", "") or "").strip()

    category_a_days = None
    if category == "A":
        category_a_days = parse_category_a_interval(remarks)

    return {
        "item_number": str(values.get("item_number", "") or "").strip(),
        "title": str(values.get("title", "") or "").strip(),
        "ata_chapter": str(values.get("ata_chapter", "") or "").strip(),
        "system": str(values.get("system", "") or "").strip(),
        "category": category,
        "category_a_days": category_a_days,
        # An MEL row with no explicit counts describes a single installation
        # for which relief exists, i.e. 1 installed and 0 required.
        "number_installed": _to_int(values.get("number_installed")) or 1,
        "number_required": (
            0 if values.get("number_required") is None
            else (_to_int(values.get("number_required")) or 0)
        ),
        "remarks": remarks,
        "operational_procedure": str(values.get("operational_procedure", "") or "").strip(),
        "maintenance_procedure": str(values.get("maintenance_procedure", "") or "").strip(),
        "placard_text": str(values.get("placard_text", "") or "").strip(),
        "performance_penalty": str(values.get("performance_penalty", "") or "").strip(),
        "incompatible_with": _to_list(values.get("incompatible_with")),
        "prohibited_operations": _to_list(values.get("prohibited_operations")),
        "extension_permitted": _to_bool(values.get("extension_permitted")),
        "applicable_models": _to_list(values.get("applicable_models")),
        "applicable_configurations": _to_list(values.get("applicable_configurations")),
        "applicable_serials": _to_list(values.get("applicable_serials")),
    }


def validate(parsed: dict[str, Any]) -> str | None:
    if not parsed["item_number"]:
        return "no item number"
    if not parsed["title"]:
        return "no title or description"
    if not parsed["category"]:
        return (
            "rectification category could not be read; expected A, B, C, D or CDL"
        )
    if parsed["category"] not in CATEGORY_INTERVAL_DAYS and parsed["category"] not in {"A", "CDL"}:
        return f"unrecognised category '{parsed['category']}'"
    if parsed["number_required"] > parsed["number_installed"]:
        return (
            f"{parsed['number_required']} required exceeds "
            f"{parsed['number_installed']} installed"
        )
    return None


async def import_mel(
    session: AsyncSession,
    path: Path,
    *,
    source_document_key: str = "",
    source_revision: str = "",
    default_models: list[str] | None = None,
    replace_existing: bool = False,
    dry_run: bool = False,
) -> MelImportResult:
    """Load an MEL export into the item catalogue."""
    rows, headers = read_rows(path)
    if not rows:
        raise ValueError(f"{path.name} contains no data rows")

    column_map = _build_column_map(headers)
    if "item_number" not in column_map.values():
        raise ValueError(
            "could not find an item number column. Expected one of: "
            + ", ".join(_ALIASES["item_number"])
        )
    if "category" not in column_map.values():
        raise ValueError(
            "could not find a rectification category column. Without it no "
            "expiry can be computed, which is the point of the MEL."
        )

    result = MelImportResult(
        unmapped_columns=[h for h in headers if h and h not in column_map]
    )

    existing_rows = (
        await session.execute(
            select(MelItem).where(MelItem.source_revision == source_revision)
        )
    ).scalars().all()
    existing = {row.item_number: row for row in existing_rows}
    seen: set[str] = set()

    for index, raw in enumerate(rows, start=2):
        parsed = parse_row(raw, column_map)
        problem = validate(parsed)
        if problem:
            result.skipped += 1
            result.issues.append(
                MelImportIssue(row=index, item_number=parsed["item_number"], problem=problem)
            )
            continue

        if parsed["item_number"] in seen:
            result.skipped += 1
            result.issues.append(
                MelImportIssue(
                    row=index,
                    item_number=parsed["item_number"],
                    problem="duplicate item number in the source file",
                )
            )
            continue
        seen.add(parsed["item_number"])

        # Surface Category A items so a human can confirm the interval that
        # was read out of prose, or supply one that could not be.
        if parsed["category"] == "A":
            if parsed["category_a_days"] is not None:
                result.category_a_parsed.append(
                    {
                        "item_number": parsed["item_number"],
                        "days": parsed["category_a_days"],
                        "remarks": parsed["remarks"][:200],
                    }
                )
            else:
                result.category_a_unresolved.append(parsed["item_number"])

        if not parsed["applicable_models"] and default_models:
            parsed["applicable_models"] = list(default_models)

        if dry_run:
            result.created += parsed["item_number"] not in existing
            result.updated += parsed["item_number"] in existing
            continue

        row = existing.get(parsed["item_number"])
        if row is None:
            row = MelItem(
                item_number=parsed["item_number"],
                title=parsed["title"],
                category=parsed["category"],
                source_revision=source_revision,
            )
            session.add(row)
            result.created += 1
        else:
            result.updated += 1

        for key, value in parsed.items():
            if key != "item_number":
                setattr(row, key, value)
        row.source_document_key = source_document_key
        row.source_revision = source_revision
        row.active = True

    if replace_existing and not dry_run:
        for item_number, row in existing.items():
            if item_number not in seen:
                # Retire rather than delete: open deferrals reference it.
                row.active = False

    if not dry_run:
        await session.flush()

    log.info(
        "imported MEL %s rev %s: %d created, %d updated, %d skipped",
        path.name, source_revision or "-", result.created, result.updated, result.skipped,
    )
    return result
