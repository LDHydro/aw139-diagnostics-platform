"""
Database-facing MEL operations.

The important property enforced here: **a deferral cannot be recorded that
the MEL does not permit.** ``raise_deferral`` re-runs the full dispatch
evaluation against live aircraft state and refuses a NO_GO, so the record of
what is being carried can never drift from what the approved document allows.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..maintenance.service import get_aircraft
from ..models import Aircraft, DeferralStatus, MelDeferral, MelItem
from .dispatch import (
    AirworthinessStatus,
    DispatchDecision,
    DispatchError,
    MelSpec,
    OpenDeferral,
    evaluate_airworthiness,
    evaluate_dispatch,
    extension_expiry,
    not_in_mel,
)

log = logging.getLogger(__name__)


class MelError(RuntimeError):
    """An MEL operation could not be completed."""


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

def to_spec(row: MelItem) -> MelSpec:
    return MelSpec(
        id=row.id,
        item_number=row.item_number,
        title=row.title,
        category=row.category,
        ata_chapter=row.ata_chapter,
        system=row.system,
        category_a_days=row.category_a_days,
        number_installed=row.number_installed,
        number_required=row.number_required,
        remarks=row.remarks,
        operational_procedure=row.operational_procedure,
        maintenance_procedure=row.maintenance_procedure,
        placard_text=row.placard_text,
        performance_penalty=row.performance_penalty,
        incompatible_with=list(row.incompatible_with or []),
        prohibited_operations=list(row.prohibited_operations or []),
        extension_permitted=row.extension_permitted,
        source_document_key=row.source_document_key,
        source_revision=row.source_revision,
        source_reference=row.source_reference,
    )


def to_open_deferral(row: MelDeferral) -> OpenDeferral:
    return OpenDeferral(
        id=row.id,
        item_number=row.item_number,
        category=row.category,
        defect_description=row.defect_description,
        discovered_on=row.discovered_on,
        expires_on=row.expires_on,
        quantity_inoperative=row.quantity_inoperative,
        extended=row.extended,
    )


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------

async def find_item(
    session: AsyncSession, item_number: str, *, revision: str = ""
) -> MelItem | None:
    """Exact lookup by item number, newest applicable revision by default."""
    query = select(MelItem).where(
        MelItem.item_number == item_number, MelItem.active.is_(True)
    )
    if revision:
        query = query.where(MelItem.source_revision == revision)
    rows = (await session.execute(query.order_by(MelItem.source_revision.desc()))).scalars().all()
    return rows[0] if rows else None


async def search_items(
    session: AsyncSession,
    *,
    ata_chapter: str = "",
    text: str = "",
    limit: int = 50,
) -> list[MelItem]:
    query = select(MelItem).where(MelItem.active.is_(True))
    if ata_chapter:
        query = query.where(MelItem.ata_chapter == ata_chapter)
    if text:
        pattern = f"%{text}%"
        query = query.where(
            MelItem.title.ilike(pattern)
            | MelItem.system.ilike(pattern)
            | MelItem.item_number.ilike(pattern)
        )
    return list(
        (await session.execute(query.order_by(MelItem.item_number).limit(limit)))
        .scalars()
        .all()
    )


async def open_deferrals(
    session: AsyncSession, aircraft_id: str
) -> list[MelDeferral]:
    return list(
        (
            await session.execute(
                select(MelDeferral)
                .where(
                    MelDeferral.aircraft_id == aircraft_id,
                    MelDeferral.status == DeferralStatus.OPEN.value,
                )
                .order_by(MelDeferral.expires_on)
            )
        )
        .unique()
        .scalars()
        .all()
    )


# ----------------------------------------------------------------------
# Dispatch check
# ----------------------------------------------------------------------

async def check_dispatch(
    session: AsyncSession,
    aircraft_identifier: str,
    *,
    item_number: str = "",
    description: str = "",
    discovered_on: date | None = None,
    quantity_inoperative: int = 1,
    intended_operation: list[str] | None = None,
    today: date | None = None,
) -> tuple[Aircraft, DispatchDecision]:
    """
    Evaluate whether the aircraft may be dispatched with something inoperative.

    Identify the item by number when it is known. When only a description is
    available the caller should resolve it first (the API route uses the
    document retriever against the indexed MEL), because guessing an item
    from prose is not something to do silently on an airworthiness question.
    """
    today = today or date.today()
    discovered_on = discovered_on or today
    aircraft = await get_aircraft(session, aircraft_identifier)

    if not item_number:
        return aircraft, not_in_mel(description or "unspecified defect")

    row = await find_item(session, item_number)
    if row is None:
        return aircraft, not_in_mel(
            f"{item_number}" + (f" - {description}" if description else "")
        )
    if not row.applies_to(aircraft):
        decision = not_in_mel(f"{item_number} - {row.title}")
        decision.blocking_reasons.insert(
            0,
            f"MEL item {item_number} exists but does not apply to "
            f"{aircraft.tail_number} ({aircraft.model} "
            f"{aircraft.configuration or 'no configuration'}, s/n "
            f"{aircraft.serial_number or 'unknown'})",
        )
        return aircraft, decision

    carried = [to_open_deferral(d) for d in await open_deferrals(session, aircraft.id)]
    decision = evaluate_dispatch(
        to_spec(row),
        discovered_on=discovered_on,
        today=today,
        open_deferrals=carried,
        quantity_inoperative=quantity_inoperative,
        intended_operation=intended_operation,
    )
    return aircraft, decision


# ----------------------------------------------------------------------
# Deferral lifecycle
# ----------------------------------------------------------------------

async def raise_deferral(
    session: AsyncSession,
    aircraft_identifier: str,
    item_number: str,
    *,
    defect_description: str,
    accepted_by: str,
    raised_by: str = "",
    discovered_on: date | None = None,
    quantity_inoperative: int = 1,
    work_order: str = "",
    placard_fitted: bool = False,
    operational_procedure_applied: bool = False,
    maintenance_procedure_applied: bool = False,
    notes: str = "",
    today: date | None = None,
) -> tuple[MelDeferral, DispatchDecision]:
    """
    Record a deferred defect under MEL relief.

    Refuses anything the MEL does not permit, and refuses to record a
    deferral whose required (o)/(m) procedures have not been confirmed as
    applied - a deferral is only valid once its conditions are met.
    """
    today = today or date.today()
    discovered_on = discovered_on or today

    if not accepted_by:
        raise MelError(
            "a deferral must be accepted by a named, appropriately licensed person"
        )
    if discovered_on > today:
        raise MelError("the discovery date cannot be in the future")

    aircraft, decision = await check_dispatch(
        session,
        aircraft_identifier,
        item_number=item_number,
        description=defect_description,
        discovered_on=discovered_on,
        quantity_inoperative=quantity_inoperative,
        today=today,
    )

    if not decision.dispatchable:
        raise MelError(
            "the MEL does not permit this deferral: "
            + "; ".join(decision.blocking_reasons)
        )
    if decision.item is None or decision.expires_on is None:
        raise MelError(
            "no rectification interval could be determined for this item, so a "
            "deferral cannot be recorded against it"
        )

    # The conditions are not advisory. A deferral recorded before the (m)
    # procedure is carried out is a deferral that was never valid.
    if decision.item.operational_procedure and not operational_procedure_applied:
        raise MelError(
            f"item {item_number} requires an (o) operational procedure to be "
            "applied before the deferral is valid; confirm it has been done"
        )
    if decision.item.maintenance_procedure and not maintenance_procedure_applied:
        raise MelError(
            f"item {item_number} requires an (m) maintenance procedure to be "
            "applied before the deferral is valid; confirm it has been done"
        )
    if decision.requires_placard and not placard_fitted:
        raise MelError(
            f"item {item_number} requires a placard "
            f"('{decision.item.placard_text}') to be fitted before dispatch"
        )

    record = MelDeferral(
        aircraft_id=aircraft.id,
        mel_item_id=decision.item.id,
        item_number=decision.item.item_number,
        category=decision.item.category,
        defect_description=defect_description,
        discovered_on=discovered_on,
        expires_on=decision.expires_on,
        original_expires_on=decision.expires_on,
        quantity_inoperative=quantity_inoperative,
        raised_by=raised_by or accepted_by,
        accepted_by=accepted_by,
        work_order=work_order,
        placard_fitted=placard_fitted,
        operational_procedure_applied=operational_procedure_applied,
        maintenance_procedure_applied=maintenance_procedure_applied,
        notes=notes,
        meta={"citation": decision.citation, "conditions": decision.conditions},
    )
    session.add(record)
    await session.flush()

    log.info(
        "MEL deferral raised on %s: %s (%s) until %s, accepted by %s",
        aircraft.tail_number, item_number, decision.item.category,
        decision.expires_on, accepted_by,
    )
    return record, decision


async def clear_deferral(
    session: AsyncSession,
    deferral_id: str,
    *,
    cleared_by: str,
    cleared_on: date | None = None,
    rectification_notes: str = "",
) -> MelDeferral:
    """Close a deferred defect once the equipment is serviceable again."""
    record = (
        await session.execute(select(MelDeferral).where(MelDeferral.id == deferral_id))
    ).unique().scalar_one_or_none()
    if record is None:
        raise MelError(f"no MEL deferral with id '{deferral_id}'")
    if record.status != DeferralStatus.OPEN.value:
        raise MelError(f"this deferral is already {record.status}")
    if not cleared_by:
        raise MelError("clearing a deferral requires a named person")

    record.status = DeferralStatus.CLEARED.value
    record.cleared_on = cleared_on or date.today()
    record.cleared_by = cleared_by
    record.rectification_notes = rectification_notes
    await session.flush()

    log.info("MEL deferral %s cleared by %s", record.item_number, cleared_by)
    return record


async def extend_deferral(
    session: AsyncSession,
    deferral_id: str,
    *,
    approved_by: str,
    authority_reference: str = "",
    reason: str = "",
    today: date | None = None,
) -> MelDeferral:
    """
    Apply the one-time extension, where the approved document permits it.

    Refused for Category A, for items not marked extendable, for anything
    already extended, and for anything whose interval has already run out -
    an extension is granted before the limit, not after.
    """
    today = today or date.today()
    record = (
        await session.execute(select(MelDeferral).where(MelDeferral.id == deferral_id))
    ).unique().scalar_one_or_none()
    if record is None:
        raise MelError(f"no MEL deferral with id '{deferral_id}'")
    if record.status != DeferralStatus.OPEN.value:
        raise MelError(f"this deferral is {record.status} and cannot be extended")
    if not approved_by:
        raise MelError("an extension requires a named approver")
    if record.expires_on < today:
        raise MelError(
            f"this deferral expired on {record.expires_on}; an extension must be "
            "approved before the interval runs out, not after"
        )

    item = (
        await session.execute(select(MelItem).where(MelItem.id == record.mel_item_id))
    ).scalar_one_or_none()
    if item is None:
        raise MelError("the MEL item behind this deferral is no longer in the catalogue")

    try:
        new_expiry = extension_expiry(to_open_deferral(record), to_spec(item))
    except DispatchError as exc:
        raise MelError(str(exc)) from exc

    record.expires_on = new_expiry
    record.extended = True
    record.extension_approved_by = approved_by
    record.extension_authority_reference = authority_reference
    record.extended_at = datetime.now(UTC)
    if reason:
        record.notes = f"{record.notes}\nExtension: {reason}".strip()
    await session.flush()

    log.info(
        "MEL deferral %s extended to %s by %s (%s)",
        record.item_number, new_expiry, approved_by, authority_reference or "no reference",
    )
    return record


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------

async def aircraft_status(
    session: AsyncSession,
    aircraft_identifier: str,
    *,
    today: date | None = None,
    warning_days: int = 3,
) -> tuple[Aircraft, AirworthinessStatus]:
    today = today or date.today()
    aircraft = await get_aircraft(session, aircraft_identifier)
    rows = await open_deferrals(session, aircraft.id)

    prohibited_by_item = {
        row.item_number: list(row.mel_item.prohibited_operations or [])
        for row in rows
        if row.mel_item is not None
    }
    status = evaluate_airworthiness(
        aircraft.tail_number,
        [to_open_deferral(r) for r in rows],
        today=today,
        warning_days=warning_days,
        prohibited_by_item=prohibited_by_item,
    )
    return aircraft, status


async def fleet_status(
    session: AsyncSession, *, today: date | None = None, warning_days: int = 3
) -> list[dict]:
    today = today or date.today()
    aircraft_rows = (
        await session.execute(
            select(Aircraft).where(Aircraft.in_service.is_(True))
            .order_by(Aircraft.tail_number)
        )
    ).scalars().all()

    out: list[dict] = []
    for row in aircraft_rows:
        _aircraft, status = await aircraft_status(
            session, row.id, today=today, warning_days=warning_days
        )
        out.append(status.to_dict())
    return out


async def expire_overdue(
    session: AsyncSession, *, today: date | None = None
) -> list[MelDeferral]:
    """
    Mark open deferrals whose interval has run out.

    Run nightly. The status change is a record that the limit passed; it does
    not make the aircraft airworthy again, and the item stays visible.
    """
    today = today or date.today()
    rows = (
        await session.execute(
            select(MelDeferral).where(
                MelDeferral.status == DeferralStatus.OPEN.value,
                MelDeferral.expires_on < today,
            )
        )
    ).unique().scalars().all()

    for row in rows:
        row.status = DeferralStatus.EXPIRED.value
        log.warning(
            "MEL deferral %s on aircraft %s expired on %s and is still open",
            row.item_number, row.aircraft_id, row.expires_on,
        )
    if rows:
        await session.flush()
    return list(rows)
