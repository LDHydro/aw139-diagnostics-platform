"""
Client for the local vLLM servers running on the RTX 5090.

vLLM exposes an OpenAI-compatible surface, so this is a thin async wrapper
with the retry, timeout and streaming behaviour the rest of the platform
expects.  No data leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import InferenceSettings, get_settings

log = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """The local model server failed to answer."""


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LlmClient:
    """Async OpenAI-compatible client pointed at a local vLLM endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        settings: InferenceSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings().inference
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.settings.request_timeout_s,
                    connect=self.settings.connect_timeout_s,
                ),
                headers={"Authorization": f"Bearer {self.api_key}"},
                # A single 5090 serves every request; keeping connections
                # warm avoids TLS/TCP setup on each call.
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------

    def _payload(
        self,
        messages: list[ChatMessage] | list[dict],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalised = [
            m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalised,
            "temperature": (
                self.settings.default_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.settings.max_output_tokens,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        return payload

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        **extra: Any,
    ) -> Completion:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra=extra or None,
        )
        client = await self._http()
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            started = asyncio.get_event_loop().time()
            try:
                resp = await client.post("/chat/completions", json=payload)
                if resp.status_code >= 500:
                    raise InferenceError(
                        f"model server returned {resp.status_code}: {resp.text[:300]}"
                    )
                if resp.status_code >= 400:
                    # Client errors will not improve on retry.
                    raise InferenceError(
                        f"model server rejected the request ({resp.status_code}): "
                        f"{resp.text[:300]}"
                    )
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                usage = data.get("usage") or {}
                return Completion(
                    text=(choice.get("message") or {}).get("content", "") or "",
                    model=data.get("model", self.model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", ""),
                    latency_ms=(asyncio.get_event_loop().time() - started) * 1000,
                    raw=data,
                )
            except InferenceError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < retries:
                    backoff = 1.5 * (2**attempt)
                    log.warning(
                        "inference call failed (%s), retrying in %.1fs", exc, backoff
                    )
                    await asyncio.sleep(backoff)
                    continue

        raise InferenceError(
            f"could not reach the model server at {self.base_url}: {last_error}"
        )

    async def stream(
        self,
        messages: list[ChatMessage] | list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Yield content deltas as the model produces them."""
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra=extra or None,
        )
        client = await self._http()
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise InferenceError(
                    f"model server rejected the request ({resp.status_code}): "
                    f"{body[:300]!r}"
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = ((data.get("choices") or [{}])[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    yield content

    async def health(self) -> dict[str, Any]:
        try:
            client = await self._http()
            resp = await client.get("/models", timeout=5.0)
            resp.raise_for_status()
            models = [m.get("id") for m in resp.json().get("data", [])]
            return {"status": "ok", "endpoint": self.base_url, "models": models}
        except Exception as exc:
            return {"status": "error", "endpoint": self.base_url, "detail": str(exc)}
