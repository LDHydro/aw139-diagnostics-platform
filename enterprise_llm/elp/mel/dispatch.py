"""
Dispatch decision support against the Minimum Equipment List.

The question this answers is the one a line engineer asks several times a
week: *something is inoperative - may the aircraft be dispatched, under what
conditions, and until when?*

Three rules drive everything here, and each is a place real operations go
wrong:

1. **If it is not in the MEL, it must work.** The MEL is a list of permitted
   relief, not a list of restrictions. An item absent from it is not
   "unrestricted"; it is "no dispatch". The engine returns NOT_IN_MEL rather
   than a cautious GO, because a silent pass on an unlisted item is the
   failure mode that matters.

2. **The rectification interval excludes the day of discovery.** A Category C
   item found on the 1st runs to the end of the 11th, not the 10th. Getting
   this off by one is a real finding in a real audit.

3. **Relief is per-item quantity, not per-item.** "Two installed, one
   required for dispatch" permits exactly one to be inoperative. A second
   failure of the same item is a no-go even though the MEL lists it.

Nothing here decides anything on its own: every decision names the MEL item
and revision it came from, and a licensed engineer accepts the deferral.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

# Standard rectification intervals in consecutive calendar days, excluding
# the day of discovery. Category A has no standard interval - its limit is
# stated in the remarks column of the item itself.
CATEGORY_INTERVAL_DAYS: dict[str, int] = {
    "B": 3,
    "C": 10,
    "D": 120,
}

# Categories for which authorities commonly permit a single extension of
# equal duration, subject to operator approval. Category A never qualifies:
# its interval is item-specific and already stated in the approved document.
EXTENDABLE_CATEGORIES = frozenset({"B", "C", "D"})


class Verdict(str, enum.Enum):
    GO = "go"
    GO_WITH_CONDITIONS = "go_with_conditions"
    NO_GO = "no_go"
    NOT_IN_MEL = "not_in_mel"


class DispatchError(ValueError):
    """The request could not be evaluated as posed."""


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------

@dataclass
class MelSpec:
    """A MEL item, decoupled from the ORM so the logic stays testable."""

    id: str
    item_number: str
    title: str
    category: str
    ata_chapter: str = ""
    system: str = ""
    category_a_days: int | None = None
    number_installed: int = 1
    number_required: int = 0
    remarks: str = ""
    operational_procedure: str = ""
    maintenance_procedure: str = ""
    placard_text: str = ""
    performance_penalty: str = ""
    incompatible_with: list[str] = field(default_factory=list)
    prohibited_operations: list[str] = field(default_factory=list)
    extension_permitted: bool = False
    source_document_key: str = ""
    source_revision: str = ""
    source_reference: str = ""

    @property
    def relief_quantity(self) -> int:
        """How many of this item may be inoperative at once."""
        return max(0, self.number_installed - self.number_required)


@dataclass
class OpenDeferral:
    """An MEL item already carried as inoperative on this aircraft."""

    id: str
    item_number: str
    category: str
    defect_description: str
    discovered_on: date
    expires_on: date | None
    quantity_inoperative: int = 1
    extended: bool = False

    def days_remaining(self, today: date) -> int | None:
        if self.expires_on is None:
            return None
        return (self.expires_on - today).days

    def is_expired(self, today: date) -> bool:
        return self.expires_on is not None and self.expires_on < today


# ----------------------------------------------------------------------
# Interval arithmetic
# ----------------------------------------------------------------------

# "10 consecutive calendar days", "3 days", "120 calendar days", "24 hours"
_DAYS_RE = re.compile(
    r"(\d+)\s*(?:consecutive\s+)?(?:calendar\s+)?(day|days|dias)\b", re.IGNORECASE
)
_HOURS_RE = re.compile(r"(\d+)\s*(?:flight\s+)?(hour|hours|horas)\b", re.IGNORECASE)
_CYCLES_RE = re.compile(r"(\d+)\s*(cycle|cycles|ciclos)\b", re.IGNORECASE)


def parse_category_a_interval(remarks: str) -> int | None:
    """
    Pull a day count out of a Category A remarks column.

    Category A items state their own interval in prose, and operators write
    it a dozen different ways. Returns ``None`` when no day-based interval
    can be read - including when the interval is expressed in flight hours
    or cycles, which this engine does not convert, because converting would
    mean guessing a utilisation rate for an airworthiness limit.
    """
    if not remarks:
        return None
    match = _DAYS_RE.search(remarks)
    if match:
        return int(match.group(1))
    return None


def category_a_interval_note(remarks: str) -> str | None:
    """Explain why a Category A interval could not be read, when it could not."""
    if _HOURS_RE.search(remarks or ""):
        return (
            "the remarks state the interval in flight hours; enter the limit "
            "manually rather than converting, as the conversion would depend "
            "on an assumed utilisation rate"
        )
    if _CYCLES_RE.search(remarks or ""):
        return (
            "the remarks state the interval in cycles; enter the limit manually "
            "rather than converting"
        )
    return None


def rectification_expiry(
    category: str,
    discovered_on: date,
    *,
    category_a_days: int | None = None,
) -> date | None:
    """
    Last day the aircraft may be dispatched with the item inoperative.

    The interval runs in consecutive calendar days *excluding the day of
    discovery*, so an item found on the 1st under a 10-day category expires
    at the end of the 11th.
    """
    category = (category or "").upper()

    if category in CATEGORY_INTERVAL_DAYS:
        days = CATEGORY_INTERVAL_DAYS[category]
    elif category in {"A", "CDL"}:
        # Both take their interval from the item itself. CDL items frequently
        # have none at all, and may be carried until the next convenient
        # maintenance opportunity.
        if category_a_days is None:
            return None
        days = category_a_days
    else:
        raise DispatchError(f"unknown MEL category '{category}'")

    if days <= 0:
        raise DispatchError(f"rectification interval must be positive, got {days}")
    return discovered_on + timedelta(days=days)


def extension_expiry(deferral: OpenDeferral, item: MelSpec) -> date:
    """
    Expiry after a single extension of equal duration.

    Refuses anything the approved document does not permit, rather than
    leaving the caller to check.
    """
    category = (deferral.category or "").upper()
    if deferral.extended:
        raise DispatchError(
            f"{deferral.item_number} has already been extended once; a further "
            "extension is not permitted"
        )
    if not item.extension_permitted:
        raise DispatchError(
            f"{deferral.item_number} is not marked as extendable in the MEL"
        )
    if category not in EXTENDABLE_CATEGORIES:
        raise DispatchError(
            f"category {category} items may not be extended; the interval in the "
            "approved document is the limit"
        )
    if deferral.expires_on is None:
        raise DispatchError(
            f"{deferral.item_number} has no expiry date, so there is nothing to extend"
        )

    original_days = CATEGORY_INTERVAL_DAYS[category]
    return deferral.expires_on + timedelta(days=original_days)


# ----------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------

@dataclass
class DispatchDecision:
    verdict: Verdict
    dispatchable: bool
    summary: str
    item: MelSpec | None = None
    expires_on: date | None = None
    days_available: int | None = None
    # Things that must be done before the aircraft flies.
    conditions: list[str] = field(default_factory=list)
    # Why it cannot be dispatched.
    blocking_reasons: list[str] = field(default_factory=list)
    # Worth knowing but not blocking.
    cautions: list[str] = field(default_factory=list)
    prohibited_operations: list[str] = field(default_factory=list)
    requires_placard: bool = False
    citation: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "dispatchable": self.dispatchable,
            "summary": self.summary,
            "item": (
                {
                    "item_number": self.item.item_number,
                    "title": self.item.title,
                    "category": self.item.category,
                    "ata_chapter": self.item.ata_chapter,
                    "number_installed": self.item.number_installed,
                    "number_required": self.item.number_required,
                    "remarks": self.item.remarks,
                    "operational_procedure": self.item.operational_procedure,
                    "maintenance_procedure": self.item.maintenance_procedure,
                    "performance_penalty": self.item.performance_penalty,
                }
                if self.item
                else None
            ),
            "expires_on": self.expires_on.isoformat() if self.expires_on else None,
            "days_available": self.days_available,
            "conditions": self.conditions,
            "blocking_reasons": self.blocking_reasons,
            "cautions": self.cautions,
            "prohibited_operations": self.prohibited_operations,
            "requires_placard": self.requires_placard,
            "citation": self.citation,
        }


def _citation(item: MelSpec) -> str:
    parts = [item.source_document_key or "MEL"]
    if item.source_revision:
        parts.append(f"Rev {item.source_revision}")
    parts.append(f"item {item.item_number}")
    if item.source_reference:
        parts.append(item.source_reference)
    return ", ".join(parts)


def not_in_mel(description: str) -> DispatchDecision:
    """
    The defect matches no MEL item.

    Stated plainly rather than hedged: the absence of relief is itself the
    answer, and the aircraft is not dispatchable until the defect is fixed.
    """
    return DispatchDecision(
        verdict=Verdict.NOT_IN_MEL,
        dispatchable=False,
        summary=(
            "No MEL item covers this defect. Equipment that does not appear in "
            "the Minimum Equipment List must be serviceable for dispatch, so "
            "the aircraft may not be released until it is rectified."
        ),
        blocking_reasons=[
            f"no MEL relief exists for: {description[:200]}",
            "the MEL grants relief; it does not restrict it - an unlisted item "
            "is not optional equipment",
        ],
        cautions=[
            "if you believe relief should exist, check the item number directly "
            "rather than by description, and confirm the MEL revision indexed "
            "here is the current one",
        ],
    )


def evaluate_dispatch(
    item: MelSpec,
    *,
    discovered_on: date,
    today: date | None = None,
    open_deferrals: list[OpenDeferral] | None = None,
    quantity_inoperative: int = 1,
    intended_operation: list[str] | None = None,
) -> DispatchDecision:
    """
    Decide whether the aircraft may be dispatched with this item inoperative.

    ``open_deferrals`` is everything already carried on the aircraft, which
    matters twice: the same item may have used up its relief already, and a
    different item may be incompatible with this one.
    """
    today = today or date.today()
    open_deferrals = open_deferrals or []
    intended_operation = [op.lower() for op in (intended_operation or [])]

    conditions: list[str] = []
    blocking: list[str] = []
    cautions: list[str] = []
    category = (item.category or "").upper()

    # --- Is there any relief at all? ---------------------------------
    if item.relief_quantity <= 0:
        blocking.append(
            f"the MEL requires {item.number_required} of "
            f"{item.number_installed} installed to be serviceable, so none may "
            "be inoperative"
        )

    # --- Has the relief already been consumed? -----------------------
    already = sum(
        d.quantity_inoperative
        for d in open_deferrals
        if d.item_number == item.item_number
    )
    if already and (already + quantity_inoperative) > item.relief_quantity:
        blocking.append(
            f"{already} of {item.item_number} already inoperative; the MEL "
            f"permits at most {item.relief_quantity}"
        )
    elif quantity_inoperative > item.relief_quantity:
        blocking.append(
            f"{quantity_inoperative} inoperative exceeds the {item.relief_quantity} "
            "permitted by the MEL"
        )

    # --- Does it clash with something already carried? ---------------
    open_numbers = {d.item_number for d in open_deferrals}
    clashes = sorted(open_numbers & {n for n in item.incompatible_with})
    for clash in clashes:
        blocking.append(
            f"{item.item_number} may not be inoperative at the same time as "
            f"{clash}, which is already deferred on this aircraft"
        )

    # --- Is the aircraft already out of limits on something else? ----
    expired = [d for d in open_deferrals if d.is_expired(today)]
    for stale in expired:
        blocking.append(
            f"{stale.item_number} expired on {stale.expires_on}; the aircraft is "
            "not dispatchable until it is rectified or extended"
        )

    # --- Expiry of this item -----------------------------------------
    expires_on: date | None = None
    category_a_days = item.category_a_days
    if category == "A" and category_a_days is None:
        parsed = parse_category_a_interval(item.remarks)
        if parsed is not None:
            category_a_days = parsed
            cautions.append(
                f"Category A interval of {parsed} day(s) was read from the "
                "remarks column; confirm it against the approved MEL"
            )
        else:
            note = category_a_interval_note(item.remarks)
            blocking.append(
                "this is a Category A item and its rectification interval could "
                "not be determined automatically"
                + (f" - {note}" if note else "; read it from the remarks column")
            )

    try:
        expires_on = rectification_expiry(
            category, discovered_on, category_a_days=category_a_days
        )
    except DispatchError as exc:
        blocking.append(str(exc))

    days_available = (expires_on - today).days if expires_on else None
    if expires_on is not None and expires_on < today:
        blocking.append(
            f"the rectification interval for this item ran out on {expires_on}"
        )
    elif category == "CDL" and expires_on is None:
        cautions.append(
            "no rectification interval is recorded for this CDL item; confirm "
            "whether the approved document sets one"
        )

    # --- Conditions attached to the relief ---------------------------
    if item.operational_procedure:
        conditions.append(f"(o) operational procedure: {item.operational_procedure}")
    if item.maintenance_procedure:
        conditions.append(f"(m) maintenance procedure: {item.maintenance_procedure}")
    if item.remarks:
        conditions.append(f"MEL remarks: {item.remarks}")
    if item.placard_text:
        conditions.append(f"placard: {item.placard_text}")
    if item.performance_penalty:
        cautions.append(f"performance penalty: {item.performance_penalty}")

    # --- Operational restrictions ------------------------------------
    prohibited = list(item.prohibited_operations)
    conflicts = [
        op for op in intended_operation
        if any(op == p.lower() or op in p.lower() for p in prohibited)
    ]
    for conflict in conflicts:
        blocking.append(
            f"this item prohibits {conflict} operations, which the intended "
            "flight requires"
        )

    # --- Verdict ------------------------------------------------------
    if blocking:
        return DispatchDecision(
            verdict=Verdict.NO_GO,
            dispatchable=False,
            summary=(
                f"{item.item_number} ({item.title}) may not be deferred: "
                f"{blocking[0]}"
            ),
            item=item,
            expires_on=expires_on,
            days_available=days_available,
            conditions=conditions,
            blocking_reasons=blocking,
            cautions=cautions,
            prohibited_operations=prohibited,
            requires_placard=bool(item.placard_text),
            citation=_citation(item),
        )

    verdict = Verdict.GO_WITH_CONDITIONS if conditions else Verdict.GO
    summary = (
        f"Dispatch permitted under {item.item_number} ({item.title}), "
        f"Category {category}"
    )
    if expires_on:
        summary += (
            f". Rectify by {expires_on.isoformat()} "
            f"({days_available} day(s) from today)"
        )
    if conditions:
        summary += f". {len(conditions)} condition(s) must be satisfied first"
    summary += "."

    return DispatchDecision(
        verdict=verdict,
        dispatchable=True,
        summary=summary,
        item=item,
        expires_on=expires_on,
        days_available=days_available,
        conditions=conditions,
        blocking_reasons=[],
        cautions=cautions,
        prohibited_operations=prohibited,
        requires_placard=bool(item.placard_text),
        citation=_citation(item),
    )


# ----------------------------------------------------------------------
# Aircraft-level status
# ----------------------------------------------------------------------

@dataclass
class AirworthinessStatus:
    tail_number: str
    dispatchable: bool
    open_count: int
    expired: list[dict] = field(default_factory=list)
    expiring_soon: list[dict] = field(default_factory=list)
    open_items: list[dict] = field(default_factory=list)
    prohibited_operations: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "tail_number": self.tail_number,
            "dispatchable": self.dispatchable,
            "open_count": self.open_count,
            "expired": self.expired,
            "expiring_soon": self.expiring_soon,
            "open_items": self.open_items,
            "prohibited_operations": self.prohibited_operations,
            "summary": self.summary,
        }


def evaluate_airworthiness(
    tail_number: str,
    open_deferrals: list[OpenDeferral],
    *,
    today: date | None = None,
    warning_days: int = 3,
    prohibited_by_item: dict[str, list[str]] | None = None,
) -> AirworthinessStatus:
    """
    Whether the aircraft may be dispatched right now, given what it carries.

    A single expired item makes the aircraft undispatchable regardless of
    everything else, which is why this is computed rather than eyeballed off
    a whiteboard.
    """
    today = today or date.today()
    prohibited_by_item = prohibited_by_item or {}

    expired: list[dict] = []
    soon: list[dict] = []
    items: list[dict] = []
    prohibited: set[str] = set()

    for deferral in open_deferrals:
        remaining = deferral.days_remaining(today)
        row = {
            "id": deferral.id,
            "item_number": deferral.item_number,
            "category": deferral.category,
            "defect": deferral.defect_description,
            "discovered_on": deferral.discovered_on.isoformat(),
            "expires_on": deferral.expires_on.isoformat() if deferral.expires_on else None,
            "days_remaining": remaining,
            "extended": deferral.extended,
        }
        items.append(row)
        prohibited.update(prohibited_by_item.get(deferral.item_number, []))

        if deferral.is_expired(today):
            expired.append(row)
        elif remaining is not None and remaining <= warning_days:
            soon.append(row)

    dispatchable = not expired
    if expired:
        summary = (
            f"{tail_number} is NOT dispatchable: "
            f"{len(expired)} MEL item(s) past their rectification interval "
            f"({', '.join(e['item_number'] for e in expired)})."
        )
    elif soon:
        summary = (
            f"{tail_number} is dispatchable with {len(items)} MEL item(s) open. "
            f"{len(soon)} expire within {warning_days} day(s): "
            f"{', '.join(s['item_number'] for s in soon)}."
        )
    elif items:
        summary = f"{tail_number} is dispatchable with {len(items)} MEL item(s) open."
    else:
        summary = f"{tail_number} has no open MEL items."

    return AirworthinessStatus(
        tail_number=tail_number,
        dispatchable=dispatchable,
        open_count=len(items),
        expired=expired,
        expiring_soon=soon,
        open_items=sorted(
            items, key=lambda r: (r["days_remaining"] is None, r["days_remaining"])
        ),
        prohibited_operations=sorted(prohibited),
        summary=summary,
    )
