#!/usr/bin/env python3
"""
Re-embed the corpus.

Needed when the embedding model changes (a different model means different
vectors, and old vectors become meaningless), or when chunking settings
change and you want existing documents re-split.

    python scripts/reindex.py --confirm                 # re-embed in place
    python scripts/reindex.py --confirm --rechunk       # re-parse from source too

Changing the embedding DIMENSION additionally requires recreating the column,
which this script does when --new-dim is given. That is destructive: every
existing vector is dropped before the new ones are written.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select, text  # noqa: E402

from elp.config import get_settings  # noqa: E402
from elp.db import close_db, get_engine, get_sessionmaker  # noqa: E402
from elp.llm.embeddings import get_embedding_client  # noqa: E402
from elp.models import DocChunk, Document, DocumentStatus  # noqa: E402
from elp.rag.ingest import Ingestor, IngestRequest  # noqa: E402

BATCH = 64


async def recreate_vector_column(new_dim: int) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        print(f"dropping the embedding index and column, recreating at {new_dim} dimensions")
        await conn.execute(text("DROP INDEX IF EXISTS ix_doc_chunks_embedding"))
        await conn.execute(text("ALTER TABLE doc_chunks DROP COLUMN IF EXISTS embedding"))
        await conn.execute(
            text(f"ALTER TABLE doc_chunks ADD COLUMN embedding vector({new_dim})")
        )


async def rebuild_index() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        print("rebuilding the HNSW index (this is the slow part)")
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding "
                "ON doc_chunks USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )


async def reembed() -> int:
    embedder = get_embedding_client()
    done = 0
    async with get_sessionmaker()() as session:
        total = (await session.execute(select(func.count(DocChunk.id)))).scalar_one()

    while True:
        async with get_sessionmaker()() as session:
            chunks = (
                await session.execute(
                    select(DocChunk).where(DocChunk.embedding.is_(None)).limit(BATCH)
                )
            ).unique().scalars().all()
            if not chunks:
                break

            texts = []
            for chunk in chunks:
                header = " > ".join(p for p in [chunk.section_path, chunk.heading] if p)
                texts.append(f"{header}\n\n{chunk.text}" if header else chunk.text)

            vectors = await embedder.embed(texts)
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
            await session.commit()

        done += len(chunks)
        print(f"  {done}/{total} chunks embedded", end="\r", flush=True)

    print(f"  {done}/{total} chunks embedded")
    return done


async def rechunk() -> int:
    """Re-parse every document from its original file and re-index it."""
    ingestor = Ingestor()
    async with get_sessionmaker()() as session:
        documents = (
            await session.execute(
                select(Document).where(Document.status == DocumentStatus.READY.value)
            )
        ).scalars().all()
        targets = [
            IngestRequest(
                path=Path(d.source_uri),
                doc_key=d.doc_key,
                title=d.title,
                department=d.department,
                doc_type=d.doc_type,
                revision=d.revision,
                effective_date=d.effective_date,
                review_due_date=d.review_due_date,
                allowed_groups=list(d.allowed_groups),
                classification=d.classification,
                source_uri=d.source_uri,
                meta=dict(d.meta or {}),
            )
            for d in documents
        ]

    rebuilt = 0
    for request in targets:
        if not request.path.exists():
            print(f"  SKIP {request.doc_key}: source file is gone ({request.path})")
            continue
        async with get_sessionmaker()() as session:
            result = await ingestor.ingest(session, request, force=True)
            await session.commit()
        print(f"  OK   {request.doc_key} rev {request.revision or '-'}: {result.chunk_count} chunks")
        rebuilt += 1
    return rebuilt


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", required=True)
    parser.add_argument("--rechunk", action="store_true", help="Re-parse source files as well")
    parser.add_argument("--new-dim", type=int, default=0, help="Recreate the vector column at this dimension")
    args = parser.parse_args()

    settings = get_settings()
    print(f"embedding model: {settings.inference.embed_model} ({settings.inference.embed_dim} dims)")

    if args.new_dim:
        if args.new_dim != settings.inference.embed_dim:
            print(
                f"\nrefusing to continue: --new-dim is {args.new_dim} but "
                f"ELP_INFERENCE__EMBED_DIM is {settings.inference.embed_dim}. "
                "Set the environment variable first so the ORM and the column agree.",
                file=sys.stderr,
            )
            return 1
        await recreate_vector_column(args.new_dim)

    if args.rechunk:
        print("\nre-parsing and re-indexing every document")
        count = await rechunk()
        print(f"rebuilt {count} document(s)")
    else:
        if not args.new_dim:
            # Clear existing vectors so the re-embed loop picks them all up.
            async with get_sessionmaker()() as session:
                await session.execute(text("UPDATE doc_chunks SET embedding = NULL"))
                await session.commit()
        print("\nre-embedding")
        await reembed()

    await rebuild_index()
    await close_db()
    print("reindex complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
