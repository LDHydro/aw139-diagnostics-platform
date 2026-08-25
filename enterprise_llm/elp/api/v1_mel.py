"""
Minimum Equipment List dispatch support.

Decision *support*, not decision making. Every response names the MEL item
and revision it came from, and recording a deferral requires a named,
licensed person to accept it. The platform's job is to get the item, the
interval and the conditions right, and to refuse anything the approved
document does not permit.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import audit
from ..auth.deps import require_scope
from ..auth.principal import Principal, Scope
from ..db import get_session
from ..maintenance.service import MaintenanceError
from ..mel import service as mel
from ..mel.catalog import import_mel
from ..models import Aircraft, DeferralStatus, MelDeferral, MelItem
from ..rag.retrieve import RetrievalFilter, get_retriever
from .schemas import (
    MelCandidate,
    MelCheckRequest,
    MelCheckResponse,
    MelClearRequest,
    MelDeferralRequest,
    MelDeferralResponse,
    MelExtendRequest,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mel", tags=["minimum equipment list"])


def _mel_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _item_summary(row: MelItem) -> dict:
    return {
        "item_number": row.item_number,
        "title": row.title,
        "category": row.category,
        "ata_chapter": row.ata_chapter,
        "system": row.system,
        "number_installed": row.number_installed,
        "number_required": row.number_required,
        "remarks": row.remarks,
        "operational_procedure": row.operational_procedure,
        "maintenance_procedure": row.maintenance_procedure,
        "placard_text": row.placard_text,
        "performance_penalty": row.performance_penalty,
        "prohibited_operations": list(row.prohibited_operations or []),
        "incompatible_with": list(row.incompatible_with or []),
        "extension_permitted": row.extension_permitted,
        "source": {
            "document_key": row.source_document_key,
            "revision": row.source_revision,
            "reference": row.source_reference,
        },
    }


def _deferral_response(row: MelDeferral, tail_number: str, today: date) -> MelDeferralResponse:
    return MelDeferralResponse(
        id=row.id,
        tail_number=tail_number,
        item_number=row.item_number,
        category=row.category,
        defect_description=row.defect_description,
        discovered_on=row.discovered_on,
        expires_on=row.expires_on,
        days_remaining=(row.expires_on - today).days,
        status=row.status,
        accepted_by=row.accepted_by,
        extended=row.extended,
        citation=(row.meta or {}).get("citation", ""),
        conditions=(row.meta or {}).get("conditions", []),
    )


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------

@router.get("/items")
async def list_items(
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    ata_chapter: str = "",
    q: str = "",
    limit: int = 50,
) -> list[dict]:
    rows = await mel.search_items(
        session, ata_chapter=ata_chapter, text=q, limit=min(limit, 200)
    )
    return [_item_summary(r) for r in rows]


@router.get("/items/{item_number}")
async def get_item(
    item_number: str,
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    row = await mel.find_item(session, item_number)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no MEL item '{item_number}'. Remember that equipment absent "
                "from the MEL must be serviceable for dispatch."
            ),
        )
    return _item_summary(row)


@router.post("/import")
async def import_catalogue(
    request: Request,
    file: UploadFile = File(..., description="CSV, TSV, JSON or XLSX MEL export"),
    source_document_key: str = Form(default="MEL-001"),
    source_revision: str = Form(default=""),
    default_models: str = Form(default=""),
    replace_existing: bool = Form(default=False),
    dry_run: bool = Form(default=False),
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Load the approved MEL into the item catalogue."""
    filename = Path(file.filename or "mel.csv").name
    staging = Path(tempfile.mkdtemp(prefix="elp-mel-"))
    target = staging / filename
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        try:
            result = await import_mel(
                session,
                target,
                source_document_key=source_document_key,
                source_revision=source_revision,
                default_models=[m.strip() for m in default_models.split(",") if m.strip()]
                or None,
                replace_existing=replace_existing,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    await audit.record(
        session,
        principal,
        "mel.import",
        resource=f"{filename} rev {source_revision or '-'}",
        request=request,
        detail=result.to_dict(),
    )
    return result.to_dict()


# ----------------------------------------------------------------------
# Dispatch check
# ----------------------------------------------------------------------

@router.post("/check", response_model=MelCheckResponse)
async def check(
    payload: MelCheckRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
) -> MelCheckResponse:
    """
    "Something is inoperative - may we dispatch?"

    With an item number this returns a decision. With only a description it
    returns candidate items for a human to choose from rather than guessing,
    because picking the wrong MEL item silently is how the wrong interval
    ends up on an aircraft.
    """
    today = date.today()

    if not payload.item_number and not payload.description:
        raise HTTPException(
            status_code=400, detail="supply an item_number or a description"
        )

    # Resolve a description to candidate items.
    if not payload.item_number:
        candidates = await mel.search_items(session, text=payload.description, limit=10)
        references: list[dict] = []

        # The indexed MEL document usually finds items the catalogue's simple
        # text match misses, so surface it as supporting evidence.
        try:
            passages = await get_retriever().retrieve(
                session,
                payload.description,
                principal,
                filters=RetrievalFilter(doc_types=["mel"]),
                top_k=5,
            )
            references = [p.to_reference(f"D{i}") for i, p in enumerate(passages, 1)]
        except Exception as exc:  # noqa: BLE001 - retrieval is supporting, not load-bearing
            log.info("MEL document retrieval unavailable: %s", exc)

        if len(candidates) == 1:
            payload.item_number = candidates[0].item_number
        else:
            message = (
                f"{len(candidates)} MEL items match that description. Choose the "
                "correct item number and check again."
                if candidates
                else (
                    "No MEL item matches that description. If nothing covers this "
                    "defect, the aircraft is not dispatchable until it is "
                    "rectified - but confirm by item number before acting, and "
                    "check that the indexed MEL revision is current."
                )
            )
            await audit.record(
                session,
                principal,
                "mel.check",
                resource=payload.description[:300],
                outcome="needs_clarification",
                request=request,
                detail={"tail_number": payload.tail_number, "candidates": len(candidates)},
            )
            return MelCheckResponse(
                tail_number=payload.tail_number,
                candidates=[
                    MelCandidate(
                        item_number=c.item_number,
                        title=c.title,
                        category=c.category,
                        ata_chapter=c.ata_chapter,
                        system=c.system,
                    )
                    for c in candidates
                ],
                needs_clarification=True,
                message=message,
                references=references,
            )

    try:
        aircraft, decision = await mel.check_dispatch(
            session,
            payload.tail_number,
            item_number=payload.item_number,
            description=payload.description,
            discovered_on=payload.discovered_on,
            quantity_inoperative=payload.quantity_inoperative,
            intended_operation=payload.intended_operation,
            today=today,
        )
    except MaintenanceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    await audit.record(
        session,
        principal,
        "mel.check",
        resource=f"{payload.tail_number}/{payload.item_number}",
        outcome=decision.verdict.value,
        request=request,
        detail={
            "dispatchable": decision.dispatchable,
            "expires_on": decision.expires_on.isoformat() if decision.expires_on else None,
            "citation": decision.citation,
            "blocking_reasons": decision.blocking_reasons,
        },
    )

    return MelCheckResponse(
        tail_number=aircraft.tail_number,
        decision=decision.to_dict(),
        message=decision.summary,
    )


# ----------------------------------------------------------------------
# Deferrals
# ----------------------------------------------------------------------

@router.get("/deferrals")
async def list_deferrals(
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    tail_number: str = "",
    status: str = DeferralStatus.OPEN.value,
) -> list[dict]:
    today = date.today()
    query = select(MelDeferral)
    if status:
        query = query.where(MelDeferral.status == status)
    if tail_number:
        try:
            aircraft = await mel.get_aircraft(session, tail_number)
        except MaintenanceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        query = query.where(MelDeferral.aircraft_id == aircraft.id)

    rows = (
        await session.execute(query.order_by(MelDeferral.expires_on))
    ).unique().scalars().all()

    tails = {
        a.id: a.tail_number
        for a in (await session.execute(select(Aircraft))).scalars().all()
    }
    return [
        {
            "id": r.id,
            "tail_number": tails.get(r.aircraft_id, r.aircraft_id),
            "item_number": r.item_number,
            "category": r.category,
            "defect_description": r.defect_description,
            "discovered_on": r.discovered_on.isoformat(),
            "expires_on": r.expires_on.isoformat(),
            "days_remaining": (r.expires_on - today).days,
            "status": r.status,
            "accepted_by": r.accepted_by,
            "extended": r.extended,
            "work_order": r.work_order,
        }
        for r in rows
    ]


@router.post("/deferrals", response_model=MelDeferralResponse, status_code=201)
async def raise_deferral(
    payload: MelDeferralRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> MelDeferralResponse:
    """
    Record a deferred defect under MEL relief.

    Refused outright if the MEL does not permit it, or if the item's (o)/(m)
    procedures and placarding have not been confirmed as carried out.
    """
    today = date.today()
    try:
        record, decision = await mel.raise_deferral(
            session,
            payload.tail_number,
            payload.item_number,
            defect_description=payload.defect_description,
            accepted_by=payload.accepted_by or principal.subject,
            raised_by=principal.subject,
            discovered_on=payload.discovered_on,
            quantity_inoperative=payload.quantity_inoperative,
            work_order=payload.work_order,
            placard_fitted=payload.placard_fitted,
            operational_procedure_applied=payload.operational_procedure_applied,
            maintenance_procedure_applied=payload.maintenance_procedure_applied,
            notes=payload.notes,
            today=today,
        )
    except (mel.MelError, MaintenanceError) as exc:
        await audit.record(
            session,
            principal,
            "mel.defer_refused",
            resource=f"{payload.tail_number}/{payload.item_number}",
            outcome="refused",
            request=request,
            detail={"reason": str(exc)},
        )
        raise _mel_error(exc) from exc

    await audit.record(
        session,
        principal,
        "mel.deferred",
        resource=f"{payload.tail_number}/{payload.item_number}",
        request=request,
        detail={
            "expires_on": record.expires_on.isoformat(),
            "category": record.category,
            "accepted_by": record.accepted_by,
            "citation": decision.citation,
        },
    )
    return _deferral_response(record, payload.tail_number, today)


@router.post("/deferrals/{deferral_id}/clear")
async def clear(
    deferral_id: str,
    payload: MelClearRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        record = await mel.clear_deferral(
            session,
            deferral_id,
            cleared_by=principal.subject,
            cleared_on=payload.cleared_on,
            rectification_notes=payload.rectification_notes,
        )
    except mel.MelError as exc:
        raise _mel_error(exc) from exc

    await audit.record(
        session,
        principal,
        "mel.cleared",
        resource=f"{record.item_number} ({deferral_id})",
        request=request,
        detail={"cleared_on": record.cleared_on.isoformat() if record.cleared_on else None},
    )
    return {
        "id": record.id,
        "item_number": record.item_number,
        "status": record.status,
        "cleared_on": record.cleared_on.isoformat() if record.cleared_on else None,
        "cleared_by": record.cleared_by,
    }


@router.post("/deferrals/{deferral_id}/extend")
async def extend(
    deferral_id: str,
    payload: MelExtendRequest,
    request: Request,
    principal: Principal = Depends(require_scope(Scope.MAINT_APPROVE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Apply the one-time extension where the approved document permits it.

    Requires maint:approve. Extending a rectification interval is an
    airworthiness decision, not routine planning.
    """
    try:
        record = await mel.extend_deferral(
            session,
            deferral_id,
            approved_by=principal.subject,
            authority_reference=payload.authority_reference,
            reason=payload.reason,
        )
    except mel.MelError as exc:
        await audit.record(
            session,
            principal,
            "mel.extend_refused",
            resource=deferral_id,
            outcome="refused",
            request=request,
            detail={"reason": str(exc)},
        )
        raise _mel_error(exc) from exc

    await audit.record(
        session,
        principal,
        "mel.extended",
        resource=f"{record.item_number} ({deferral_id})",
        request=request,
        detail={
            "new_expiry": record.expires_on.isoformat(),
            "original_expiry": record.original_expires_on.isoformat(),
            "authority_reference": payload.authority_reference,
            "reason": payload.reason,
        },
    )
    return {
        "id": record.id,
        "item_number": record.item_number,
        "expires_on": record.expires_on.isoformat(),
        "original_expires_on": record.original_expires_on.isoformat(),
        "extended": record.extended,
        "approved_by": record.extension_approved_by,
    }


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------

@router.get("/status")
async def fleet(
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    warning_days: int = 3,
) -> dict:
    """Dispatch status of every in-service aircraft."""
    statuses = await mel.fleet_status(session, warning_days=warning_days)
    grounded = [s["tail_number"] for s in statuses if not s["dispatchable"]]
    return {
        "aircraft": statuses,
        "summary": {
            "total": len(statuses),
            "dispatchable": len(statuses) - len(grounded),
            "not_dispatchable": len(grounded),
            "grounded_tails": grounded,
            "open_items": sum(s["open_count"] for s in statuses),
        },
    }


@router.get("/status/{tail_number}")
async def aircraft(
    tail_number: str,
    principal: Principal = Depends(require_scope(Scope.MAINT_READ)),
    session: AsyncSession = Depends(get_session),
    warning_days: int = 3,
) -> dict:
    try:
        _row, status = await mel.aircraft_status(
            session, tail_number, warning_days=warning_days
        )
    except MaintenanceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return status.to_dict()
