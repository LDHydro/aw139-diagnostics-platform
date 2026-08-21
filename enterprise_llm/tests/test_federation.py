"""Routing to, and degrading gracefully from, other internal AI systems."""

from __future__ import annotations

import pytest

from elp.auth.principal import Principal, scopes_for_roles
from elp.federation.connectors import PeerAnswer, PeerConfig, PeerError, _dig, build_connector
from elp.federation.orchestrator import FederationOrchestrator, _tag_score
from elp.federation.registry import PeerRegistry


def peer(name: str, **overrides) -> PeerConfig:
    defaults = dict(name=name, base_url=f"https://{name}.internal/v1")
    defaults.update(overrides)
    return PeerConfig(**defaults)


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------

def test_capability_tags_route_the_question():
    inventory = peer("inventory", capabilities=["stock", "part number", "lead time"])
    safety = peer("safety", capabilities=["occurrence", "hazard", "incident"])

    question = "Do we have stock of that part number at the main base?"
    assert _tag_score(question, inventory) > _tag_score(question, safety)


def test_an_unrelated_question_scores_zero():
    inventory = peer("inventory", capabilities=["stock", "lead time"])
    assert _tag_score("What torque applies to the tail rotor bolts?", inventory) == 0.0


@pytest.mark.asyncio
async def test_group_restrictions_hide_a_peer():
    """A peer holding restricted data must be invisible to those without the group."""
    class Registry(PeerRegistry):
        async def all_peers(self, session=None):
            return [
                peer("open"),
                peer("restricted", allowed_groups=["Safety-Department"]),
            ]

    registry = Registry()
    outsider = Principal(subject="a", groups=["AW139-Line"], roles=["reader"],
                         scopes=scopes_for_roles(["reader"]))
    insider = Principal(subject="b", groups=["Safety-Department"], roles=["reader"],
                        scopes=scopes_for_roles(["reader"]))

    assert {p.name for p in await registry.visible_to(outsider)} == {"open"}
    assert {p.name for p in await registry.visible_to(insider)} == {"open", "restricted"}


@pytest.mark.asyncio
async def test_disabled_peers_are_never_offered():
    class Registry(PeerRegistry):
        async def all_peers(self, session=None):
            return [peer("live"), peer("retired", enabled=False)]

    caller = Principal(subject="a", roles=["reader"], scopes=scopes_for_roles(["reader"]))
    assert {p.name for p in await Registry().visible_to(caller)} == {"live"}


@pytest.mark.asyncio
async def test_explicitly_requested_peers_are_honoured():
    class Registry(PeerRegistry):
        async def all_peers(self, session=None):
            return [peer("alpha"), peer("beta"), peer("gamma")]

    orchestrator = FederationOrchestrator(Registry())
    caller = Principal(subject="a", roles=["reader"], scopes=scopes_for_roles(["reader"]))

    chosen = await orchestrator.select_peers("anything", caller, requested=["beta"])
    assert [p.name for p in chosen] == ["beta"]


@pytest.mark.asyncio
async def test_peer_selection_respects_the_ceiling():
    class Registry(PeerRegistry):
        async def all_peers(self, session=None):
            return [peer(f"p{i}", capabilities=["stock"]) for i in range(10)]

    orchestrator = FederationOrchestrator(Registry())
    caller = Principal(subject="a", roles=["reader"], scopes=scopes_for_roles(["reader"]))

    chosen = await orchestrator.select_peers("stock levels please", caller)
    assert len(chosen) <= orchestrator.settings.max_peers_per_query


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unreachable_peer_degrades_rather_than_raising():
    """One broken peer must never take down the whole answer."""
    broken = peer("broken", base_url="http://127.0.0.1:1/v1", timeout_s=1.0)
    answer = await build_connector(broken).ask("hello")

    assert not answer.ok
    assert answer.error
    assert answer.name == "broken"


@pytest.mark.asyncio
async def test_missing_credentials_are_reported_clearly():
    configured = peer("secured", auth_type="bearer", auth_env_var="DEFINITELY_NOT_SET_12345")
    answer = await build_connector(configured).ask("hello")

    assert not answer.ok
    assert "DEFINITELY_NOT_SET_12345" in answer.error


def test_unknown_protocol_is_refused():
    with pytest.raises(PeerError, match="unknown protocol"):
        build_connector(peer("odd", protocol="carrier-pigeon"))


# ----------------------------------------------------------------------
# Response shapes
# ----------------------------------------------------------------------

def test_dotted_paths_reach_into_nested_responses():
    body = {"result": {"output": [{"text": "the answer"}]}}
    assert _dig(body, "result.output.0.text") == "the answer"
    assert _dig(body, "result.missing.text") is None
    assert _dig(body, "") == body


def test_peer_answers_render_as_attributable_citations():
    answer = PeerAnswer(
        name="reliability-analytics",
        display_name="Fleet Reliability Analytics",
        answer="That component fails roughly every 1,800 hours.",
        model="reliability-v2",
        queried_at="2026-08-21T10:00:00+00:00",
    )
    reference = answer.to_reference("A1")

    assert reference["type"] == "ai_system"
    assert reference["system"] == "reliability-analytics"
    # The reader must be able to tell a peer AI from a governing document.
    assert "internal AI system" in reference["citation"]
