#!/usr/bin/env python3
"""
Nightly re-forecast of the whole fleet.

Run from the elp-scheduler systemd timer. Re-projects every task card against
the latest utilisation and reports anything that has moved towards a limit,
so a drift in flying rate is noticed before it becomes an AOG.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from elp.config import get_settings  # noqa: E402
from elp.db import close_db, get_sessionmaker  # noqa: E402
from elp.maintenance.forecast import DueStatus  # noqa: E402
from elp.maintenance.service import forecast_for_aircraft  # noqa: E402
from elp.mel.service import expire_overdue, fleet_status  # noqa: E402
from elp.models import Aircraft  # noqa: E402

log = logging.getLogger("nightly_forecast")

ATTENTION = {DueStatus.DUE_SOON, DueStatus.DUE, DueStatus.OVERDUE, DueStatus.GROUNDED}


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    settings = get_settings().maintenance
    today = date.today()

    async with get_sessionmaker()() as session:
        aircraft = (
            await session.execute(
                select(Aircraft).where(Aircraft.in_service.is_(True))
                .order_by(Aircraft.tail_number)
            )
        ).scalars().all()

        if not aircraft:
            log.warning("no in-service aircraft found; nothing to forecast")
            return 0

        log.info("re-forecasting %d aircraft", len(aircraft))
        grounded: list[str] = []
        overdue: list[str] = []
        soon: list[str] = []

        for row in aircraft:
            state, rate, forecasts, _ = await forecast_for_aircraft(
                session, row.id, today=today
            )
            attention = [f for f in forecasts if f.status in ATTENTION]
            log.info(
                "%s: %s | %d task(s) need attention within %d days",
                state.tail_number, rate.describe(), len(attention),
                settings.forecast_horizon_days,
            )
            if rate.source == "default":
                log.warning(
                    "%s has no utilisation history in the last %d days; forecasts "
                    "use the configured default rate and should not be trusted",
                    state.tail_number, settings.utilization_window_days,
                )

            for item in attention:
                line = (
                    f"{state.tail_number} {item.task_code} ({item.task_title[:50]}) "
                    f"due {item.due_on} limit {item.hard_limit_on}"
                )
                if item.status is DueStatus.GROUNDED:
                    grounded.append(line)
                    log.error("GROUNDED: %s", line)
                elif item.status is DueStatus.OVERDUE:
                    overdue.append(line)
                    log.warning("OVERDUE: %s", line)
                else:
                    soon.append(line)

        # MEL deferrals whose rectification interval has run out. These are
        # separate from scheduled-task limits and just as grounding.
        expired_mel = await expire_overdue(session, today=today)
        for row in expired_mel:
            log.error(
                "MEL EXPIRED: item %s expired %s and is still open",
                row.item_number, row.expires_on,
            )

        mel_statuses = await fleet_status(session, today=today)
        undispatchable = [s for s in mel_statuses if not s["dispatchable"]]
        for status in undispatchable:
            log.error("NOT DISPATCHABLE: %s", status["summary"])
        for status in mel_statuses:
            for soon_item in status["expiring_soon"]:
                log.warning(
                    "MEL expiring: %s %s (%s) in %s day(s)",
                    status["tail_number"], soon_item["item_number"],
                    soon_item["category"], soon_item["days_remaining"],
                )

        await session.commit()

    await close_db()

    print(
        f"\nsummary for {today.isoformat()}: "
        f"{len(grounded)} past hard limit, {len(overdue)} overdue, {len(soon)} due soon, "
        f"{len(undispatchable)} aircraft not dispatchable on MEL"
    )
    # Non-zero exit makes the systemd unit show as failed, which is what you
    # want when an aircraft is past a limit or cannot be dispatched.
    return 2 if (grounded or undispatchable) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
