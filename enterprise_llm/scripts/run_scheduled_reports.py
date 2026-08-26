#!/usr/bin/env python3
"""
Execute scheduled reports that are due.

Run from the elp-reports systemd timer, every few minutes. Reports are
matched by their own cron expression in their own timezone, so the timer
interval only bounds how promptly a due report is picked up - it does not
have to match any report's schedule.

Exits non-zero when a report fails or is blocked, so a systemd failure alert
means a report that operations expects did not arrive.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elp.db import close_db, get_sessionmaker  # noqa: E402
from elp.models import RunStatus  # noqa: E402
from elp.reports.cron import describe  # noqa: E402
from elp.reports.datasource import close_source  # noqa: E402
from elp.reports.runner import run_due  # noqa: E402
from elp.reports.service import due_definitions, prune_runs  # noqa: E402

log = logging.getLogger("scheduled_reports")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what is due without running anything",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Also delete run history past the retention window",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    now = datetime.now(UTC)

    if args.dry_run:
        async with get_sessionmaker()() as session:
            definitions = await due_definitions(session, now=now)
        if not definitions:
            print("nothing due.")
        for definition in definitions:
            print(
                f"  {definition.name}  [{definition.schedule_cron} "
                f"{definition.schedule_timezone}]  {describe(definition.schedule_cron)}"
            )
        await close_db()
        return 0

    async with get_sessionmaker()() as session:
        runs = await run_due(session, now=now)

        pruned = 0
        if args.prune:
            pruned = await prune_runs(session)
            await session.commit()

    await close_source()
    await close_db()

    if not runs:
        log.info("nothing due")
        return 0

    succeeded = [r for r in runs if r.status == RunStatus.SUCCEEDED.value]
    failed = [r for r in runs if r.status == RunStatus.FAILED.value]
    blocked = [r for r in runs if r.status == RunStatus.BLOCKED.value]

    for run in succeeded:
        log.info(
            "run %s succeeded: %d row(s) in %.0f ms, %d artifact(s)",
            run.id, run.row_count, run.duration_ms, len(run.artifacts or []),
        )
        for warning in run.warnings or []:
            log.warning("  %s", warning)
    for run in failed:
        log.error("run %s FAILED: %s", run.id, run.error)
    for run in blocked:
        log.error("run %s BLOCKED: %s", run.id, run.error)

    print(
        f"{len(runs)} scheduled report(s): {len(succeeded)} succeeded, "
        f"{len(failed)} failed, {len(blocked)} blocked"
        + (f", {pruned} old run(s) pruned" if args.prune else "")
    )
    return 1 if (failed or blocked) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
