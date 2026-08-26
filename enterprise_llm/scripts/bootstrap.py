#!/usr/bin/env python3
"""
First-run setup: create the schema and issue an administrator API key.

    python scripts/bootstrap.py --admin-key-name platform-admin
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elp.auth.apikeys import create_api_key  # noqa: E402
from elp.auth.principal import Scope  # noqa: E402
from elp.config import get_settings  # noqa: E402
from elp.db import close_db, get_sessionmaker, healthcheck, init_db  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-key-name", default="platform-admin")
    parser.add_argument(
        "--admin-groups",
        default="",
        help="Comma-separated AD groups the admin key inherits, for document ACLs",
    )
    parser.add_argument(
        "--skip-key", action="store_true", help="Only create the schema"
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"database: {settings.database_url.split('@')[-1]}")

    print("creating extensions, tables and indexes...")
    await init_db()

    health = await healthcheck()
    if health.get("status") != "ok":
        print(f"database is not healthy: {health.get('detail')}", file=sys.stderr)
        return 1
    print(f"  pgvector {health.get('pgvector')} ready")

    if not args.skip_key:
        async with get_sessionmaker()() as session:
            record, plaintext = await create_api_key(
                session,
                name=args.admin_key_name,
                scopes=list(Scope.ALL),
                groups=[g.strip() for g in args.admin_groups.split(",") if g.strip()],
                description="Created by scripts/bootstrap.py",
            )
            await session.commit()

        print("\n" + "=" * 68)
        print("ADMINISTRATOR API KEY - shown once, store it in your secret manager")
        print("=" * 68)
        print(f"  name: {record.name}")
        print(f"  key:  {plaintext}")
        print("=" * 68)
        print(
            f'\nTest it:\n  curl -H "{settings.auth.api_key_header}: {plaintext}" \\\n'
            f"       http://127.0.0.1:{settings.port}/v1/whoami\n"
        )

    await close_db()
    print("bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
