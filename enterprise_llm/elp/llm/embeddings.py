"""Embedding and reranking clients (bge-m3 + bge-reranker-v2-m3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ..config import InferenceSettings, get_settings
from .client import InferenceError

log = logging.getLogger(__name__)


class EmbeddingClient:
    """OpenAI-compatible /embeddings endpoint served by vLLM or TEI."""

    def __init__(self, settings: InferenceSettings | None = None) -> None:
        self.settings = settings or get_settings().inference
        self.model = self.settings.embed_model
        self.dim = self.settings.embed_dim
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.embed_base_url.rstrip("/"),
                timeout=httpx.Timeout(120.0, connect=self.settings.connect_timeout_s),
                headers={"Authorization": "Bearer local"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching to keep VRAM use predictable."""
        if not texts:
            return []
        client = await self._http()
        batch_size = max(1, self.settings.embed_batch_size)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # Empty strings make some embedding servers error out.
            batch = [t if t.strip() else " " for t in batch]
            try:
                resp = await client.post(
                    "/embeddings", json={"model": self.model, "input": batch}
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise InferenceError(
                    f"embedding request failed against {self.settings.embed_base_url}: {exc}"
                ) from exc

            data = resp.json().get("data", [])
            if len(data) != len(batch):
                raise InferenceError(
                    f"embedding server returned {len(data)} vectors for {len(batch)} inputs"
                )
            # The server is not required to preserve input order.
            ordered = sorted(data, key=lambda d: d.get("index", 0))
            for item in ordered:
                vector = item["embedding"]
                if len(vector) != self.dim:
                    raise InferenceError(
                        f"embedding dimension mismatch: server returned {len(vector)}, "
                        f"schema expects {self.dim}. Re-index after changing the model."
                    )
                vectors.append(vector)

        return vectors

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    async def health(self) -> dict:
        try:
            vector = await self.embed_one("health check")
            return {"status": "ok", "model": self.model, "dim": len(vector)}
        except Exception as exc:
            return {"status": "error", "model": self.model, "detail": str(exc)}


@dataclass
class RerankResult:
    index: int
    score: float


class RerankClient:
    """
    Cross-encoder reranker.

    Bi-encoder retrieval is recall-oriented and noisy on governing documents,
    where many sections share boilerplate.  A cross-encoder pass over the
    top candidates is what makes the citations trustworthy.
    """

    def __init__(self, settings: InferenceSettings | None = None) -> None:
        self.settings = settings or get_settings().inference
        self.url = self.settings.rerank_url
        self.model = self.settings.rerank_model
        self.enabled = self.settings.rerank_enabled
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=self.settings.connect_timeout_s),
                headers={"Authorization": "Bearer local"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Score ``documents`` against ``query``, best first."""
        if not documents:
            return []
        if not self.enabled or not self.url:
            # Preserve the incoming order when reranking is switched off.
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
        }
        try:
            client = await self._http()
            resp = await client.post(self.url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Degrade gracefully: retrieval order is still useful.
            log.warning("reranker unavailable (%s); falling back to fusion order", exc)
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]

        rows = body.get("results", body if isinstance(body, list) else [])
        results: list[RerankResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            score = row.get("relevance_score", row.get("score"))
            if idx is None or score is None:
                continue
            results.append(RerankResult(index=int(idx), score=float(score)))

        if not results:
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def health(self) -> dict:
        if not self.enabled:
            return {"status": "disabled", "model": self.model}
        try:
            scored = await self.rerank("health check", ["health check document"])
            return {"status": "ok", "model": self.model, "scored": len(scored)}
        except Exception as exc:
            return {"status": "error", "model": self.model, "detail": str(exc)}


_embedding_client: EmbeddingClient | None = None
_rerank_client: RerankClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def get_rerank_client() -> RerankClient:
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClient()
    return _rerank_client
