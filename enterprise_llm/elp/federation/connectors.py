"""
Connectors to other internal LLM/AI systems.

Three wire formats cover essentially every internal service worth calling:
an OpenAI-compatible ``/chat/completions``, an Anthropic-style ``/messages``,
and a plain REST endpoint described by a small request/response template.

Secrets are never stored in the registry - a peer names the environment
variable that holds its credential, and the value is read at call time.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

log = logging.getLogger(__name__)


class PeerError(RuntimeError):
    """A peer system could not be reached or returned an unusable response."""


@dataclass
class PeerAnswer:
    """What another internal AI system said, in citable form."""

    name: str
    display_name: str
    answer: str
    queried_at: str = ""
    latency_ms: float = 0.0
    model: str = ""
    # References the peer itself supplied, passed through verbatim.
    peer_references: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer) and not self.error

    def to_reference(self, marker: str) -> dict:
        return {
            "marker": marker,
            "type": "ai_system",
            "citation": f"{self.display_name or self.name} (internal AI system)",
            "system": self.name,
            "display_name": self.display_name,
            "model": self.model,
            "queried_at": self.queried_at,
            "latency_ms": round(self.latency_ms, 1),
            "peer_references": self.peer_references,
        }


@dataclass
class PeerConfig:
    """Everything needed to call one peer.  Mirrors the PeerSystem table."""

    name: str
    base_url: str
    protocol: str = "openai"           # openai | anthropic | rest
    display_name: str = ""
    description: str = ""
    model: str = ""
    auth_type: str = "none"            # none | bearer | api_key | client_credentials
    auth_header: str = "Authorization"
    auth_env_var: str = ""
    token_url: str = ""
    scope: str = ""
    capabilities: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    enabled: bool = True
    citable: bool = True
    timeout_s: float = 45.0
    # REST-only: how to build the request and find the answer in the reply.
    request_template: dict[str, Any] = field(default_factory=dict)
    answer_path: str = "answer"
    references_path: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# OAuth2 client-credentials token cache
# ----------------------------------------------------------------------

_token_cache: dict[str, tuple[str, float]] = {}


async def _client_credentials_token(config: PeerConfig, client: httpx.AsyncClient) -> str:
    cached = _token_cache.get(config.name)
    if cached and cached[1] > time.time() + 30:
        return cached[0]

    secret = os.environ.get(config.auth_env_var, "")
    client_id = config.meta.get("client_id", "")
    if not (config.token_url and client_id and secret):
        raise PeerError(
            f"peer '{config.name}' uses client_credentials but token_url, "
            f"meta.client_id or ${config.auth_env_var} is missing"
        )

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": secret,
    }
    if config.scope:
        data["scope"] = config.scope

    resp = await client.post(config.token_url, data=data, timeout=20.0)
    resp.raise_for_status()
    body = resp.json()
    token = body.get("access_token", "")
    if not token:
        raise PeerError(f"peer '{config.name}' token endpoint returned no access_token")
    _token_cache[config.name] = (token, time.time() + float(body.get("expires_in", 3600)))
    return token


async def _auth_headers(config: PeerConfig, client: httpx.AsyncClient) -> dict[str, str]:
    if config.auth_type == "none":
        return {}
    if config.auth_type == "client_credentials":
        token = await _client_credentials_token(config, client)
        return {config.auth_header: f"Bearer {token}"}

    secret = os.environ.get(config.auth_env_var, "")
    if not secret:
        raise PeerError(
            f"peer '{config.name}' expects its credential in ${config.auth_env_var}, "
            "which is not set"
        )
    if config.auth_type == "bearer":
        return {config.auth_header: f"Bearer {secret}"}
    return {config.auth_header: secret}  # api_key


def _dig(payload: Any, path: str) -> Any:
    """Follow a dotted path like ``result.output.text`` through a JSON body."""
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


# ----------------------------------------------------------------------
# Connectors
# ----------------------------------------------------------------------

class PeerConnector:
    """Base connector.  Subclasses only implement the wire format."""

    def __init__(self, config: PeerConfig) -> None:
        self.config = config

    async def ask(self, question: str, *, context: str = "") -> PeerAnswer:
        started = time.monotonic()
        answer = PeerAnswer(
            name=self.config.name,
            display_name=self.config.display_name or self.config.name,
            answer="",
            queried_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                headers = await _auth_headers(self.config, client)
                text, model, references = await self._call(
                    client, headers, question, context
                )
            answer.answer = (text or "").strip()
            answer.model = model
            answer.peer_references = references
            if not answer.answer:
                answer.error = "peer returned an empty answer"
        except httpx.TimeoutException:
            answer.error = f"timed out after {self.config.timeout_s:.0f}s"
        except PeerError as exc:
            answer.error = str(exc)
        except httpx.HTTPStatusError as exc:
            answer.error = (
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            )
        except Exception as exc:  # noqa: BLE001 - one bad peer must not fail the query
            answer.error = f"{type(exc).__name__}: {exc}"

        answer.latency_ms = (time.monotonic() - started) * 1000
        if answer.error:
            log.warning("peer '%s' failed: %s", self.config.name, answer.error)
        return answer

    async def _call(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        question: str,
        context: str,
    ) -> tuple[str, str, list[dict]]:
        raise NotImplementedError


class OpenAiCompatibleConnector(PeerConnector):
    """Any service exposing ``POST {base_url}/chat/completions``."""

    async def _call(self, client, headers, question, context):
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})

        resp = await client.post(
            self.config.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json={
                "model": self.config.model or "default",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1200,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        return text, body.get("model", self.config.model), []


class AnthropicStyleConnector(PeerConnector):
    """Services exposing an Anthropic-style ``POST {base_url}/messages``."""

    async def _call(self, client, headers, question, context):
        payload: dict[str, Any] = {
            "model": self.config.model or "default",
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": question}],
        }
        if context:
            payload["system"] = context
        if "anthropic-version" not in {k.lower() for k in headers}:
            headers = {**headers, "anthropic-version": "2023-06-01"}

        resp = await client.post(
            self.config.base_url.rstrip("/") + "/messages", headers=headers, json=payload
        )
        resp.raise_for_status()
        body = resp.json()
        parts = [
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts), body.get("model", self.config.model), []


class RestConnector(PeerConnector):
    """
    A bespoke internal service.

    ``request_template`` is posted as-is with ``{question}`` and ``{context}``
    substituted into any string value; ``answer_path`` and ``references_path``
    are dotted paths into the response body.
    """

    async def _call(self, client, headers, question, context):
        template = self.config.request_template or {"query": "{question}"}
        payload = _render(template, question=question, context=context)

        resp = await client.post(self.config.base_url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()

        text = _dig(body, self.config.answer_path)
        if text is None:
            raise PeerError(
                f"no value at answer_path '{self.config.answer_path}' in the response"
            )
        references = []
        if self.config.references_path:
            found = _dig(body, self.config.references_path)
            if isinstance(found, list):
                references = [r if isinstance(r, dict) else {"text": str(r)} for r in found]
        return str(text), self.config.model, references


def _render(value: Any, **substitutions: str) -> Any:
    if isinstance(value, str):
        out = value
        for key, replacement in substitutions.items():
            out = out.replace("{" + key + "}", replacement)
        return out
    if isinstance(value, dict):
        return {k: _render(v, **substitutions) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, **substitutions) for v in value]
    return value


_CONNECTORS: dict[str, type[PeerConnector]] = {
    "openai": OpenAiCompatibleConnector,
    "anthropic": AnthropicStyleConnector,
    "rest": RestConnector,
}


def build_connector(config: PeerConfig) -> PeerConnector:
    connector = _CONNECTORS.get(config.protocol)
    if connector is None:
        raise PeerError(
            f"unknown protocol '{config.protocol}' for peer '{config.name}'. "
            f"Supported: {', '.join(sorted(_CONNECTORS))}"
        )
    return connector(config)
