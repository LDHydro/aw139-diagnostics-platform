"""
Registry of peer AI systems.

Peers are declared in ``config/peers.yaml`` (version-controlled, reviewable)
and may be added or overridden at runtime through the database, so an
operator can register a new internal service without a redeploy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.principal import Principal
from ..config import FederationSettings, get_settings
from ..models import PeerSystem
from .connectors import PeerConfig

log = logging.getLogger(__name__)


def _config_from_row(row: PeerSystem) -> PeerConfig:
    meta = row.meta or {}
    return PeerConfig(
        name=row.name,
        base_url=row.base_url,
        protocol=row.protocol,
        display_name=row.display_name,
        description=row.description,
        model=row.model,
        auth_type=row.auth_type,
        auth_header=row.auth_header,
        auth_env_var=row.auth_env_var,
        token_url=row.token_url,
        scope=row.scope,
        capabilities=list(row.capabilities or []),
        allowed_groups=list(row.allowed_groups or []),
        enabled=row.enabled,
        citable=row.citable,
        timeout_s=row.timeout_s,
        request_template=meta.get("request_template", {}),
        answer_path=meta.get("answer_path", "answer"),
        references_path=meta.get("references_path", ""),
        meta=meta,
    )


def _config_from_yaml(entry: dict) -> PeerConfig:
    return PeerConfig(
        name=entry["name"],
        base_url=entry["base_url"],
        protocol=entry.get("protocol", "openai"),
        display_name=entry.get("display_name", ""),
        description=entry.get("description", ""),
        model=entry.get("model", ""),
        auth_type=entry.get("auth_type", "none"),
        auth_header=entry.get("auth_header", "Authorization"),
        auth_env_var=entry.get("auth_env_var", ""),
        token_url=entry.get("token_url", ""),
        scope=entry.get("scope", ""),
        capabilities=list(entry.get("capabilities", [])),
        allowed_groups=list(entry.get("allowed_groups", [])),
        enabled=entry.get("enabled", True),
        citable=entry.get("citable", True),
        timeout_s=float(entry.get("timeout_s", 45.0)),
        request_template=entry.get("request_template", {}),
        answer_path=entry.get("answer_path", "answer"),
        references_path=entry.get("references_path", ""),
        meta=entry.get("meta", {}),
    )


def load_yaml_peers(path: str | Path) -> list[PeerConfig]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        import yaml
    except ImportError:  # pragma: no cover - optional dependency
        log.warning("PyYAML not installed; skipping %s", path)
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("peers", [])
    configs: list[PeerConfig] = []
    for entry in entries:
        try:
            configs.append(_config_from_yaml(entry))
        except KeyError as exc:
            log.error("peer entry in %s is missing %s; skipped", path, exc)
    return configs


class PeerRegistry:
    def __init__(self, settings: FederationSettings | None = None) -> None:
        self.settings = settings or get_settings().federation

    async def all_peers(self, session: AsyncSession | None = None) -> list[PeerConfig]:
        peers = {p.name: p for p in load_yaml_peers(self.settings.registry_path)}
        if session is not None:
            rows = (await session.execute(select(PeerSystem))).scalars().all()
            for row in rows:
                # Database entries win, so runtime changes take effect.
                peers[row.name] = _config_from_row(row)
        return list(peers.values())

    async def visible_to(
        self, principal: Principal, session: AsyncSession | None = None
    ) -> list[PeerConfig]:
        """Peers this caller's AD groups permit consulting."""
        caller_groups = {g.lower() for g in principal.groups}
        visible: list[PeerConfig] = []
        for peer in await self.all_peers(session):
            if not peer.enabled:
                continue
            if peer.allowed_groups:
                permitted = {g.lower() for g in peer.allowed_groups}
                if not (caller_groups & permitted) and not principal.is_admin:
                    continue
            visible.append(peer)
        return visible

    async def get(
        self, name: str, session: AsyncSession | None = None
    ) -> PeerConfig | None:
        for peer in await self.all_peers(session):
            if peer.name == name:
                return peer
        return None


_registry: PeerRegistry | None = None


def get_registry() -> PeerRegistry:
    global _registry
    if _registry is None:
        _registry = PeerRegistry()
    return _registry
