"""
Deciding which internal AI systems to consult, and consulting them in parallel.

Routing is deliberately cheap-first: a capability-tag match handles the
common case without an LLM call, and the local model is only asked to choose
when the tags are ambiguous.  Every peer answer comes back attributed, so
the synthesiser can cite it as [A1], [A2], ... alongside document sources.
"""

from __future__ import annotations

import asyncio
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.principal import Principal
from ..config import FederationSettings, get_settings
from ..llm.client import ChatMessage
from ..llm.router import TaskKind, get_router
from .connectors import PeerAnswer, PeerConfig, build_connector
from .registry import PeerRegistry, get_registry

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-_]{2,}")

_ROUTER_PROMPT = """\
You route a user's question to internal AI systems that may help answer it.

AVAILABLE SYSTEMS
{catalog}

QUESTION
{question}

Reply with a comma-separated list of at most {limit} system names that are \
genuinely likely to add information the local document search cannot provide. \
Reply with the single word NONE if the question is answerable from maintenance \
documents alone. Output nothing else."""


def _tag_score(question: str, peer: PeerConfig) -> float:
    """Overlap between the question's words and the peer's capability tags."""
    words = set(_WORD_RE.findall(question.lower()))
    if not words:
        return 0.0
    score = 0.0
    for tag in peer.capabilities:
        tag_lower = tag.lower()
        tag_words = set(_WORD_RE.findall(tag_lower))
        if not tag_words:
            continue
        if tag_lower in question.lower():
            score += 1.0
        elif tag_words & words:
            score += 0.5 * len(tag_words & words) / len(tag_words)
    return score


class FederationOrchestrator:
    def __init__(
        self,
        registry: PeerRegistry | None = None,
        settings: FederationSettings | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.settings = settings or get_settings().federation

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    async def select_peers(
        self,
        question: str,
        principal: Principal,
        session: AsyncSession | None = None,
        *,
        requested: list[str] | None = None,
        use_model_router: bool = True,
    ) -> list[PeerConfig]:
        available = await self.registry.visible_to(principal, session)
        if not available:
            return []

        if requested:
            wanted = {name.lower() for name in requested}
            chosen = [p for p in available if p.name.lower() in wanted]
            missing = wanted - {p.name.lower() for p in chosen}
            if missing:
                log.info(
                    "requested peer(s) not available to %s: %s",
                    principal.subject, ", ".join(sorted(missing)),
                )
            return chosen[: self.settings.max_peers_per_query]

        if not self.settings.auto_consult:
            return []

        scored = sorted(
            ((_tag_score(question, p), p) for p in available),
            key=lambda pair: pair[0],
            reverse=True,
        )
        confident = [peer for score, peer in scored if score >= 1.0]
        if confident:
            return confident[: self.settings.max_peers_per_query]

        # Tags were inconclusive - ask the local model, which is cheap
        # compared with fanning out to every peer.
        if use_model_router and len(available) > 1:
            picked = await self._model_route(question, available)
            if picked:
                return picked[: self.settings.max_peers_per_query]

        maybe = [peer for score, peer in scored if score > 0]
        return maybe[: self.settings.max_peers_per_query]

    async def _model_route(
        self, question: str, available: list[PeerConfig]
    ) -> list[PeerConfig]:
        catalog = "\n".join(
            f"- {p.name}: {p.description or 'no description'}"
            + (f" (topics: {', '.join(p.capabilities)})" if p.capabilities else "")
            for p in available
        )
        prompt = _ROUTER_PROMPT.format(
            catalog=catalog,
            question=question,
            limit=self.settings.max_peers_per_query,
        )
        client, profile = get_router().resolve(TaskKind.ROUTING)
        try:
            completion = await client.chat(
                [ChatMessage("user", prompt)],
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                retries=0,
            )
        except Exception as exc:  # noqa: BLE001 - routing must never block an answer
            log.warning("peer routing call failed (%s); consulting no peers", exc)
            return []

        reply = completion.text.strip()
        if not reply or reply.upper().startswith("NONE"):
            return []
        names = {n.strip().lower() for n in reply.split(",") if n.strip()}
        return [p for p in available if p.name.lower() in names]

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    async def consult(
        self,
        peers: list[PeerConfig],
        question: str,
        *,
        context: str = "",
    ) -> list[PeerAnswer]:
        """Query peers concurrently.  A failing peer degrades, never blocks."""
        if not peers:
            return []

        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel))

        async def _ask(peer: PeerConfig) -> PeerAnswer:
            async with semaphore:
                connector = build_connector(peer)
                return await connector.ask(question, context=context)

        results = await asyncio.gather(
            *(_ask(peer) for peer in peers), return_exceptions=True
        )

        answers: list[PeerAnswer] = []
        for peer, result in zip(peers, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("peer '%s' raised: %s", peer.name, result)
                answers.append(
                    PeerAnswer(
                        name=peer.name,
                        display_name=peer.display_name or peer.name,
                        answer="",
                        error=f"{type(result).__name__}: {result}",
                    )
                )
            else:
                answers.append(result)
        return answers

    async def ask_peers(
        self,
        question: str,
        principal: Principal,
        session: AsyncSession | None = None,
        *,
        requested: list[str] | None = None,
        context: str = "",
    ) -> tuple[list[PeerAnswer], list[str]]:
        """
        Select and consult peers in one call.

        Returns the usable answers and a list of human-readable warnings for
        peers that failed, so the caller can surface degraded coverage.
        """
        if not self.settings.enabled:
            return [], []

        peers = await self.select_peers(
            question, principal, session, requested=requested
        )
        if not peers:
            return [], []

        answers = await self.consult(peers, question, context=context)
        usable = [a for a in answers if a.ok]
        warnings = [
            f"internal AI system '{a.display_name or a.name}' did not answer: {a.error}"
            for a in answers
            if not a.ok
        ]
        return usable, warnings

    async def health(self, session: AsyncSession | None = None) -> list[dict]:
        peers = await self.registry.all_peers(session)
        if not peers:
            return []

        async def _probe(peer: PeerConfig) -> dict:
            answer = await build_connector(peer).ask("ping")
            return {
                "name": peer.name,
                "protocol": peer.protocol,
                "enabled": peer.enabled,
                "status": "ok" if answer.ok else "error",
                "latency_ms": round(answer.latency_ms, 1),
                "detail": answer.error,
            }

        return list(await asyncio.gather(*(_probe(p) for p in peers)))


_orchestrator: FederationOrchestrator | None = None


def get_orchestrator() -> FederationOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = FederationOrchestrator()
    return _orchestrator
