#!/usr/bin/env python3
"""
Bulk-index the department governing documents.

Point it at a directory and describe each document in a manifest, so the
access groups and revisions are version-controlled rather than typed into a
web form:

    python scripts/ingest_docs.py --manifest docs/manifest.yaml

Manifest format:

    defaults:
      department: Maintenance
      allowed_groups: [AW139-Engineering, AW139-Maint-Admins]
    documents:
      - file: MOE-Rev-C.pdf
        doc_key: MOE-001
        title: Maintenance Organisation Exposition
        revision: C
        effective_date: 2026-01-15
      - file: quality-manual.docx
        doc_key: QM-001
        title: Quality Manual
        revision: "4"
        allowed_groups: [AW139-Maint-Admins]     # overrides the default

Without a manifest, every supported file in the directory is indexed using
its filename as the document key - fine for a quick trial, not for
production, where access groups matter.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elp.db import close_db, get_sessionmaker, init_db  # noqa: E402
from elp.rag.ingest import Ingestor, IngestRequest  # noqa: E402
from elp.rag.parsers import SUPPORTED_SUFFIXES, ParseError  # noqa: E402


def _as_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def build_requests(args) -> list[IngestRequest]:
    root = Path(args.directory).resolve()
    if args.manifest:
        import yaml

        manifest_path = Path(args.manifest).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        defaults = data.get("defaults", {})
        base = manifest_path.parent if not args.directory else root

        requests = []
        for entry in data.get("documents", []):
            merged = {**defaults, **entry}
            path = Path(merged["file"])
            if not path.is_absolute():
                path = base / path
            requests.append(
                IngestRequest(
                    path=path,
                    doc_key=merged["doc_key"],
                    title=merged.get("title", ""),
                    department=merged.get("department", ""),
                    doc_type=merged.get("doc_type", "governing"),
                    revision=str(merged.get("revision", "")),
                    effective_date=_as_date(merged.get("effective_date")),
                    review_due_date=_as_date(merged.get("review_due_date")),
                    allowed_groups=list(merged.get("allowed_groups", [])),
                    classification=merged.get("classification", "internal"),
                    source_uri=str(path),
                    meta=merged.get("meta", {}),
                )
            )
        return requests

    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [
        IngestRequest(
            path=path,
            doc_key=path.stem[:128],
            title=path.stem,
            department=args.department,
            allowed_groups=[g.strip() for g in args.groups.split(",") if g.strip()],
            source_uri=str(path),
        )
        for path in files
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="", help="Directory of documents")
    parser.add_argument("--manifest", default="", help="YAML manifest describing each document")
    parser.add_argument("--department", default="", help="Department for manifest-less runs")
    parser.add_argument("--groups", default="", help="Comma-separated AD groups (manifest-less runs)")
    parser.add_argument("--force", action="store_true", help="Re-index even if unchanged")
    args = parser.parse_args()

    if not args.directory and not args.manifest:
        parser.error("supply a directory, a --manifest, or both")

    requests = build_requests(args)
    if not requests:
        print("nothing to index.")
        return 1

    print(f"indexing {len(requests)} document(s)\n")
    await init_db()
    ingestor = Ingestor()

    indexed = skipped = failed = 0
    for request in requests:
        label = f"{request.doc_key} rev {request.revision or '-'}"
        try:
            # One transaction per document: a failure on document 9 must not
            # roll back the eight that already succeeded.
            async with get_sessionmaker()() as session:
                result = await ingestor.ingest(session, request, force=args.force)
                await session.commit()
        except ParseError as exc:
            print(f"  SKIP  {label}: {exc}")
            failed += 1
            continue
        except Exception as exc:
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
            failed += 1
            continue

        if result.skipped:
            print(f"  ----  {label}: {result.reason}")
            skipped += 1
        else:
            print(
                f"  OK    {label}: {result.chunk_count} chunks, "
                f"{result.page_count} pages"
                + (f", groups={request.allowed_groups}" if request.allowed_groups else ", all staff")
            )
            indexed += 1

    await close_db()
    print(f"\nindexed {indexed}, unchanged {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
