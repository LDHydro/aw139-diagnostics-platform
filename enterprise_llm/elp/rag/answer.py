"""
Grounded answer synthesis with verifiable references.

The contract this module enforces: every factual sentence in an answer
carries a marker, and every marker resolves to either a passage from a
governing document or a named internal AI system.  Markers the model
invents are stripped before the answer is returned, and the confidence
score reflects what retrieval actually found - not the model's tone.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import RagSettings, get_settings
from ..llm.client import ChatMessage
from ..llm.router import TaskKind, get_router
from .retrieve import Passage

if TYPE_CHECKING:  # pragma: no cover
    from ..federation.connectors import PeerAnswer

log = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(D\d+|A\d+)\]")

SYSTEM_PROMPT = """\
You are the {app_name}, answering questions for staff of an aviation \
maintenance organisation using that organisation's own governing documents.

RULES - these are not style preferences, they are safety requirements:

1. Answer ONLY from the SOURCES below. If the sources do not contain the \
answer, say so plainly and state what document or department would hold it. \
Never fill a gap from general knowledge.
2. Cite after every factual statement using the exact markers shown, e.g. \
[D1] or [A2]. A sentence stating a limit, interval, procedure step, \
responsibility or approval requirement MUST carry a marker.
3. When sources disagree, say so explicitly, cite both, and prefer the one \
with the later effective date or higher revision.
4. Quote figures, limits, tolerances and part numbers exactly as written. \
Do not round, convert units, or infer values the source does not state.
5. Distinguish what is mandatory ("shall", "must") from what is advisory \
("should", "may"), using the source's own wording.
6. Sources marked [A#] are answers from other internal AI systems, not \
primary documents. Attribute them to the named system and treat them as \
secondary to [D#] document sources.
7. Answer in the language the question was asked in.

Format: a direct answer first, then supporting detail. Use short paragraphs \
or bullets. Do not add a reference list at the end - the platform renders \
one from your markers."""


@dataclass
class GroundedAnswer:
    answer: str
    references: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    passages_considered: int = 0
    peers_consulted: list[str] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "references": self.references,
            "confidence": round(self.confidence, 3),
            "grounded": self.grounded,
            "passages_considered": self.passages_considered,
            "peers_consulted": self.peers_consulted,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
            "latency_ms": round(self.latency_ms, 1),
            "warnings": self.warnings,
        }


def _build_context(
    passages: list[Passage],
    peer_answers: list[PeerAnswer],
    max_chars: int,
) -> tuple[str, dict[str, dict]]:
    """Render the SOURCES block and the marker -> reference lookup."""
    blocks: list[str] = []
    references: dict[str, dict] = {}
    used = 0

    for index, passage in enumerate(passages, start=1):
        marker = f"D{index}"
        header = f"[{marker}] {passage.citation()}"
        if passage.section_path:
            header += f"\n      path: {passage.section_path}"
        if passage.effective_date:
            header += f"\n      effective: {passage.effective_date}"
        body = passage.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        references[marker] = passage.to_reference(marker)
        used += len(block)

    for index, peer in enumerate(peer_answers, start=1):
        marker = f"A{index}"
        block = (
            f"[{marker}] Internal AI system: {peer.display_name or peer.name}"
            f"\n      queried at: {peer.queried_at}"
            f"\n{peer.answer.strip()}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        references[marker] = peer.to_reference(marker)
        used += len(block)

    return "\n\n---\n\n".join(blocks), references


def _confidence(passages: list[Passage], peer_answers: list[PeerAnswer]) -> float:
    """
    Confidence in the *evidence*, not in the prose.

    Driven by the best reranker score, lifted slightly when several
    independent passages agree, and floored low when only peer AI systems
    responded (a secondary source should never read as authoritative).
    """
    if not passages:
        return 0.25 if peer_answers else 0.0

    scores = sorted((p.score for p in passages), reverse=True)
    best = scores[0]
    # Reranker scores are logits in roughly [-10, 10]; squash to [0, 1].
    # Clamp first: a pathological score would overflow math.exp.
    if best > 1.0 or best < 0.0:
        best = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, best))))

    distinct_docs = len({p.doc_key for p in passages})
    corroboration = min(0.15, 0.05 * (distinct_docs - 1)) if distinct_docs > 1 else 0.0
    return max(0.0, min(1.0, best + corroboration))


def _validate_markers(
    answer: str, references: dict[str, dict]
) -> tuple[str, list[dict], list[str]]:
    """Drop markers the model invented; keep only references it actually used."""
    warnings: list[str] = []
    cited = set(_MARKER_RE.findall(answer))
    invalid = cited - set(references)

    cleaned = answer
    if invalid:
        warnings.append(
            "removed citation marker(s) that do not correspond to a source: "
            + ", ".join(sorted(invalid))
        )
        for marker in invalid:
            cleaned = cleaned.replace(f"[{marker}]", "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)

    valid_cited = cited & set(references)
    used = [references[m] for m in sorted(valid_cited, key=_marker_sort_key)]

    if references and not valid_cited:
        warnings.append(
            "the model returned no citations; the answer is not traceable to a source"
        )
    return cleaned.strip(), used, warnings


def _marker_sort_key(marker: str) -> tuple[int, int]:
    return (0 if marker.startswith("D") else 1, int(marker[1:]))


class AnswerSynthesizer:
    def __init__(self, settings: RagSettings | None = None) -> None:
        self.settings = settings or get_settings().rag
        self.app_name = get_settings().app_name

    async def answer(
        self,
        question: str,
        passages: list[Passage],
        *,
        peer_answers: list[PeerAnswer] | None = None,
        history: list[ChatMessage] | None = None,
        task: TaskKind = TaskKind.GROUNDED_ANSWER,
    ) -> GroundedAnswer:
        peer_answers = peer_answers or []
        confidence = _confidence(passages, peer_answers)
        peers = [p.name for p in peer_answers]

        if not passages and not peer_answers:
            return GroundedAnswer(
                answer=(
                    "I could not find anything in the governing documents you have "
                    "access to that addresses this question. It may live in a "
                    "document that has not been indexed yet, or in one your Active "
                    "Directory groups do not cover."
                ),
                confidence=0.0,
                grounded=False,
                passages_considered=0,
            )

        if confidence < self.settings.min_answer_confidence and not peer_answers:
            return GroundedAnswer(
                answer=(
                    "I found related material, but nothing close enough to answer "
                    "this reliably. Rather than guess on a maintenance question, "
                    "here is what came closest:\n\n"
                    + "\n".join(f"- {p.citation()}" for p in passages[:5])
                ),
                references=[p.to_reference(f"D{i}") for i, p in enumerate(passages[:5], 1)],
                confidence=confidence,
                grounded=False,
                passages_considered=len(passages),
                warnings=["retrieval confidence below the configured threshold"],
            )

        context, references = _build_context(
            passages, peer_answers, self.settings.max_context_chars
        )

        messages: list[ChatMessage] = [
            ChatMessage("system", SYSTEM_PROMPT.format(app_name=self.app_name))
        ]
        if history:
            messages.extend(history[-6:])
        messages.append(
            ChatMessage(
                "user",
                f"SOURCES\n=======\n{context}\n\n"
                f"QUESTION\n========\n{question}",
            )
        )

        client, profile = get_router().resolve(task)
        completion = await client.chat(
            messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            top_p=profile.top_p,
        )

        cleaned, used_references, warnings = _validate_markers(
            completion.text, references
        )
        if completion.finish_reason == "length":
            warnings.append("answer was truncated at the output token limit")

        return GroundedAnswer(
            answer=cleaned,
            references=used_references,
            confidence=confidence,
            grounded=bool(used_references),
            passages_considered=len(passages),
            peers_consulted=peers,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=completion.latency_ms,
            warnings=warnings,
        )


_synthesizer: AnswerSynthesizer | None = None


def get_synthesizer() -> AnswerSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = AnswerSynthesizer()
    return _synthesizer
