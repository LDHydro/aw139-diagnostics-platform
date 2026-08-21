"""
Task-based model routing.

One 32 GB card cannot host a separate specialist model per workload, so the
default deployment runs a single strong generalist for chat, document Q&A,
LaTeX and code, and varies only the sampling parameters.  Sites with a second
GPU (or enough VRAM headroom for a small code model) can point
``ELP_INFERENCE__CODE_BASE_URL`` at a dedicated endpoint and this router will
use it for code work automatically.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..config import get_settings
from .client import LlmClient


class TaskKind(str, enum.Enum):
    CHAT = "chat"                 # free-form conversation
    GROUNDED_ANSWER = "grounded"  # answer strictly from retrieved passages
    CODE = "code"                 # application development
    LATEX = "latex"               # document authoring
    ROUTING = "routing"           # short classification / peer selection
    SUMMARIZE = "summarize"
    EXPLAIN_SCHEDULE = "schedule" # narrate a maintenance plan


@dataclass(frozen=True)
class GenerationProfile:
    """Sampling parameters tuned per workload."""

    temperature: float
    max_tokens: int
    top_p: float = 0.9
    # Governing-document answers must not drift; code and LaTeX need room.
    description: str = ""


PROFILES: dict[TaskKind, GenerationProfile] = {
    # Deterministic: a maintenance answer that varies run to run is useless.
    TaskKind.GROUNDED_ANSWER: GenerationProfile(0.0, 1600, 0.9, "cited document answer"),
    TaskKind.EXPLAIN_SCHEDULE: GenerationProfile(0.1, 1200, 0.9, "schedule narration"),
    TaskKind.ROUTING: GenerationProfile(0.0, 200, 0.9, "peer/topic selection"),
    TaskKind.SUMMARIZE: GenerationProfile(0.2, 1200, 0.9, "summarisation"),
    TaskKind.CODE: GenerationProfile(0.15, 4096, 0.95, "code generation"),
    TaskKind.LATEX: GenerationProfile(0.2, 6144, 0.95, "LaTeX authoring"),
    TaskKind.CHAT: GenerationProfile(0.4, 2048, 0.95, "general conversation"),
}


class ModelRouter:
    """Hands out the right client and sampling profile for a workload."""

    def __init__(self) -> None:
        s = get_settings().inference
        self._chat = LlmClient(s.chat_base_url, s.chat_model, s.chat_api_key, s)
        if s.code_base_url and s.code_model:
            self._code = LlmClient(s.code_base_url, s.code_model, s.chat_api_key, s)
            self._dedicated_code = True
        else:
            self._code = self._chat
            self._dedicated_code = False

    @property
    def has_dedicated_code_model(self) -> bool:
        return self._dedicated_code

    def client_for(self, task: TaskKind) -> LlmClient:
        return self._code if task is TaskKind.CODE else self._chat

    def profile_for(self, task: TaskKind) -> GenerationProfile:
        return PROFILES[task]

    def resolve(self, task: TaskKind) -> tuple[LlmClient, GenerationProfile]:
        return self.client_for(task), self.profile_for(task)

    def describe(self) -> dict:
        return {
            "chat_model": self._chat.model,
            "chat_endpoint": self._chat.base_url,
            "code_model": self._code.model,
            "code_endpoint": self._code.base_url,
            "dedicated_code_model": self._dedicated_code,
        }

    async def health(self) -> dict:
        chat = await self._chat.health()
        if self._dedicated_code:
            return {"chat": chat, "code": await self._code.health()}
        return {"chat": chat}

    async def aclose(self) -> None:
        await self._chat.aclose()
        if self._dedicated_code:
            await self._code.aclose()


_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


async def close_router() -> None:
    global _router
    if _router is not None:
        await _router.aclose()
    _router = None
