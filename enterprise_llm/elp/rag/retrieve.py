"""
Hybrid retrieval over the governing documents.

Three stages, each fixing a weakness of the previous one:

1. **Vector search** finds passages that mean the same thing as the question
   even when the wording differs.
2. **Keyword search** (Postgres full-text) catches the exact identifiers
   vector search is bad at - part numbers, ATA codes, clause numbers.
3. **Reciprocal rank fusion** merges the two lists, then a **cross-encoder
   reranker** scores each survivor against the question directly.

Access control is applied inside the SQL, not after: a passage the caller's
Active Directory groups do not cover is never loaded into memory, so it can
never leak through a summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import Text, bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from ..auth.principal import Principal
from ..config import RagSettings, get_settings
from ..llm.embeddings import get_embedding_client, get_rerank_client
from ..models import DocChunk, Document, DocumentStatus

log = logging.getLogger(__name__)


@dataclass
class Passage:
    """A retrieved passage plus everything needed to cite and rank it."""

    chunk_id: str
    document_id: str
    doc_key: str
    doc_title: str
    revision: str
    department: str
    section_number: str
    section_path: str
    heading: str
    page_start: int | None
    page_end: int | None
    text: str
    source_uri: str = ""
    effective_date: str = ""
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def score(self) -> float:
        return self.rerank_score or self.fusion_score

    def citation(self) -> str:
        parts = [self.doc_key]
        if self.revision:
            parts.append(f"Rev {self.revision}")
        if self.section_number:
            parts.append(f"§{self.section_number}")
        elif self.heading:
            parts.append(self.heading[:60])
        if self.page_start:
            parts.append(
                f"p. {self.page_start}"
                if not self.page_end or self.page_end == self.page_start
                else f"pp. {self.page_start}-{self.page_end}"
            )
        return ", ".join(parts)

    def to_reference(self, marker: str) -> dict:
        return {
            "marker": marker,
            "type": "document",
            "citation": self.citation(),
            "document_key": self.doc_key,
            "document_title": self.doc_title,
            "revision": self.revision,
            "department": self.department,
            "section": self.section_number,
            "section_path": self.section_path,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_uri": self.source_uri,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
        }


@dataclass
class RetrievalFilter:
    """Optional narrowing applied on top of the caller's ACL."""

    departments: list[str] = field(default_factory=list)
    doc_keys: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    # Include revisions that have been replaced by a newer issue.
    include_superseded: bool = False


def _acl_groups(principal: Principal, settings: RagSettings) -> list[str] | None:
    """
    The group list used to filter documents, or ``None`` to disable filtering.

    Returning ``None`` means "see everything" and only happens for admins
    when ``admin_bypass_acl`` is explicitly switched on.
    """
    if principal.is_admin and settings.admin_bypass_acl:
        return None
    return list(principal.groups)


class Retriever:
    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or get_settings().rag
        self.embedder = get_embedding_client()
        self.reranker = get_rerank_client()

    # ------------------------------------------------------------------

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        principal: Principal,
        *,
        filters: RetrievalFilter | None = None,
        top_k: int | None = None,
    ) -> list[Passage]:
        query = (query or "").strip()
        if not query:
            return []
        filters = filters or RetrievalFilter()
        groups = _acl_groups(principal, self.settings)

        query_vector = await self.embedder.embed_one(query)

        vector_hits = await self._vector_search(session, query_vector, groups, filters)
        keyword_hits = await self._keyword_search(session, query, groups, filters)

        fused = self._fuse(vector_hits, keyword_hits)
        if not fused:
            return []

        passages = await self._hydrate(session, fused)
        passages = await self._rerank(query, passages)

        limit = top_k or self.settings.final_top_k
        kept = [p for p in passages if p.rerank_score >= self.settings.min_rerank_score]
        # If the reranker rejected everything, keep the best fusion results
        # rather than answering with nothing at all - the caller decides
        # whether the confidence is sufficient.
        if not kept:
            kept = passages[: max(1, limit // 2)]
        return kept[:limit]

    # ------------------------------------------------------------------

    def _apply_document_filters(self, stmt, groups: list[str] | None, filters: RetrievalFilter):
        if not filters.include_superseded:
            stmt = stmt.where(Document.status == DocumentStatus.READY.value)
        else:
            stmt = stmt.where(
                Document.status.in_(
                    [DocumentStatus.READY.value, DocumentStatus.SUPERSEDED.value]
                )
            )
        if groups is not None:
            # An empty allowed_groups array means "any authenticated caller".
            stmt = stmt.where(
                text(
                    "(cardinality(documents.allowed_groups) = 0 "
                    "OR documents.allowed_groups && :acl_groups)"
                ).bindparams(bindparam("acl_groups", value=groups, type_=ARRAY(Text)))
            )
        if filters.departments:
            stmt = stmt.where(Document.department.in_(filters.departments))
        if filters.doc_keys:
            stmt = stmt.where(Document.doc_key.in_(filters.doc_keys))
        if filters.doc_types:
            stmt = stmt.where(Document.doc_type.in_(filters.doc_types))
        return stmt

    async def _vector_search(
        self,
        session: AsyncSession,
        query_vector: list[float],
        groups: list[str] | None,
        filters: RetrievalFilter,
    ) -> list[str]:
        distance = DocChunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(DocChunk.id, distance.label("distance"))
            .join(Document, Document.id == DocChunk.document_id)
            .where(DocChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(self.settings.vector_top_k)
        )
        stmt = self._apply_document_filters(stmt, groups, filters)
        rows = (await session.execute(stmt)).all()
        return [row[0] for row in rows]

    async def _keyword_search(
        self,
        session: AsyncSession,
        query: str,
        groups: list[str] | None,
        filters: RetrievalFilter,
    ) -> list[str]:
        # websearch_to_tsquery tolerates whatever a person types, including
        # quotes and OR, without raising on syntax errors.
        stmt = (
            select(
                DocChunk.id,
                text("ts_rank_cd(doc_chunks.tsv, websearch_to_tsquery('simple', :q))")
                .label("rank"),
            )
            .join(Document, Document.id == DocChunk.document_id)
            .where(text("doc_chunks.tsv @@ websearch_to_tsquery('simple', :q)"))
            .order_by(text("rank DESC"))
            .limit(self.settings.keyword_top_k)
        )
        stmt = self._apply_document_filters(stmt, groups, filters)
        try:
            rows = (await session.execute(stmt, {"q": query})).all()
        except Exception as exc:
            # A missing tsv column (schema not bootstrapped) must not take
            # down retrieval; vector search alone still answers.
            log.warning("keyword search unavailable: %s", exc)
            return []
        return [row[0] for row in rows]

    def _fuse(self, vector_hits: list[str], keyword_hits: list[str]) -> list[tuple[str, float, int | None, int | None]]:
        """Reciprocal rank fusion of the two candidate lists."""
        k = self.settings.rrf_k
        scores: dict[str, float] = {}
        vector_rank: dict[str, int] = {}
        keyword_rank: dict[str, int] = {}

        for rank, chunk_id in enumerate(vector_hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            vector_rank[chunk_id] = rank
        for rank, chunk_id in enumerate(keyword_hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            keyword_rank[chunk_id] = rank

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (chunk_id, score, vector_rank.get(chunk_id), keyword_rank.get(chunk_id))
            for chunk_id, score in ordered
        ]

    async def _hydrate(
        self,
        session: AsyncSession,
        fused: list[tuple[str, float, int | None, int | None]],
    ) -> list[Passage]:
        ids = [row[0] for row in fused]
        stmt = (
            select(DocChunk)
            .options(joinedload(DocChunk.document))
            .where(DocChunk.id.in_(ids))
        )
        chunks = {c.id: c for c in (await session.execute(stmt)).unique().scalars().all()}

        passages: list[Passage] = []
        for chunk_id, score, v_rank, k_rank in fused:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            document = chunk.document
            passages.append(
                Passage(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    doc_key=document.doc_key,
                    doc_title=document.title,
                    revision=document.revision,
                    department=document.department,
                    section_number=chunk.section_number,
                    section_path=chunk.section_path,
                    heading=chunk.heading,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    source_uri=document.source_uri,
                    effective_date=(
                        document.effective_date.isoformat()
                        if document.effective_date
                        else ""
                    ),
                    vector_rank=v_rank,
                    keyword_rank=k_rank,
                    fusion_score=score,
                )
            )
        return passages

    async def _rerank(self, query: str, passages: list[Passage]) -> list[Passage]:
        if not passages:
            return []
        documents = [
            f"{p.section_path}\n{p.heading}\n{p.text}".strip() for p in passages
        ]
        results = await self.reranker.rerank(query, documents)
        if not results:
            return passages

        for result in results:
            if 0 <= result.index < len(passages):
                passages[result.index].rerank_score = result.score

        ordered = [passages[r.index] for r in results if 0 <= r.index < len(passages)]
        # Anything the reranker did not return keeps its fusion position.
        returned = {id(p) for p in ordered}
        ordered.extend(p for p in passages if id(p) not in returned)
        return ordered


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
