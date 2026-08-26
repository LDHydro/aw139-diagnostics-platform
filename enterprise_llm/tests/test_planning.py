"""
Planning a report as a definition rather than as SQL.

This is the safer authoring path and the one the existing NAMIS generator
uses: the model is an untrusted suggester and the compiler is the gatekeeper.
A hallucinated column cannot become a query — it becomes a rejection, and the
compiler's message is precise enough to repair from.

The model is stubbed here. What is under test is the contract around it:
parsing, validation, the repair loop, and what happens when repair fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from elp.reports import authoring
from elp.reports.authoring import draft_structured, parse_structured_response


@dataclass
class _Reply:
    text: str
    model: str = "stub"
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class _StubClient:
    """Returns a scripted reply per call, and records what it was asked."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    async def chat(self, messages, **_kwargs):
        self.calls.append(list(messages))
        return _Reply(self.replies.pop(0) if self.replies else "{}")


@dataclass
class _Profile:
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9


class _StubRouter:
    def __init__(self, client: _StubClient) -> None:
        self.client = client

    def resolve(self, _task):
        return self.client, _Profile()


@pytest.fixture
def stub(monkeypatch):
    def _install(*replies: str) -> _StubClient:
        client = _StubClient(list(replies))
        monkeypatch.setattr(authoring, "get_router", lambda: _StubRouter(client))
        return client

    return _install


def definition_reply(**overrides) -> str:
    definition = {
        "base_table": "WorkRequest",
        "fields": [
            {"table": "WorkRequest", "column": "WRNo", "alias": "Work Request"}
        ],
        "sort": [{"table": "WorkRequest", "column": "WRNo", "direction": "asc"}],
        "row_limit": 200,
    }
    definition.update(overrides)
    return json.dumps(
        {
            "definition": definition,
            "explanation": "Every open work request.",
            "assumptions": ["assumed open means IsOpen = 1"],
        }
    )


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

def test_a_clean_reply_parses():
    definition, explanation, assumptions = parse_structured_response(definition_reply())
    assert definition["base_table"] == "WorkRequest"
    assert explanation.startswith("Every open")
    assert assumptions == ["assumed open means IsOpen = 1"]


def test_code_fences_and_preamble_are_tolerated():
    """Models add both however firmly they are told not to."""
    wrapped = (
        "Here is the report definition you asked for:\n\n```json\n"
        + definition_reply()
        + "\n```\n"
    )
    definition, _explanation, _assumptions = parse_structured_response(wrapped)
    assert definition["base_table"] == "WorkRequest"


def test_a_bare_definition_at_the_top_level_is_accepted():
    payload = json.dumps(
        {"base_table": "WorkRequest", "fields": [{"table": "WorkRequest", "column": "WRNo"}]}
    )
    definition, _e, _a = parse_structured_response(payload)
    assert definition["base_table"] == "WorkRequest"


def test_a_string_assumption_becomes_a_list():
    payload = json.dumps(
        {
            "definition": {"base_table": "WorkRequest", "fields": []},
            "assumptions": "just the one",
        }
    )
    _d, _e, assumptions = parse_structured_response(payload)
    assert assumptions == ["just the one"]


@pytest.mark.parametrize(
    "reply", ["no json here at all", "{not valid json}", '{"explanation": "no definition"}']
)
def test_an_unusable_reply_is_reported(reply):
    with pytest.raises(ValueError):
        parse_structured_response(reply)


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_valid_definition_compiles_to_sql(catalog, stub):
    stub(definition_reply())
    draft = await draft_structured("open work requests", catalog)

    assert draft.valid
    assert draft.report is not None
    assert "SELECT TOP (200)" in draft.sql
    assert "[NAMISNNSS].[dbo].[WorkRequest] AS T0" in draft.sql
    assert draft.columns == ["Work Request"]
    assert draft.assumptions


@pytest.mark.asyncio
async def test_the_model_is_shown_only_the_relevant_tables(catalog, stub):
    client = stub(definition_reply())
    await draft_structured("open work requests", catalog)

    system_prompt = client.calls[0][0].content
    assert "WorkRequest" in system_prompt
    # The whole catalogue would not fit in a prompt and would bury the
    # tables that matter.
    assert "FlightRecordHeaders" not in system_prompt


@pytest.mark.asyncio
async def test_a_hallucinated_column_is_repaired_from_the_compiler_message(catalog, stub):
    """
    The compiler's rejection names the problem precisely enough to fix.

    "'WorkRequest' has no column 'Status'. Did you mean: StatusCd?" is a
    better repair signal than anything a validator could invent.
    """
    client = stub(
        definition_reply(
            fields=[{"table": "WorkRequest", "column": "Status", "alias": "Status"}]
        ),
        definition_reply(
            fields=[{"table": "WorkRequest", "column": "StatusCd", "alias": "Status"}]
        ),
    )
    draft = await draft_structured("work request status", catalog)

    assert draft.valid
    assert draft.repaired
    repair_prompt = client.calls[1][-1].content
    assert "no column 'Status'" in repair_prompt
    assert "Did you mean" in repair_prompt


@pytest.mark.asyncio
async def test_a_definition_that_never_compiles_is_rejected_not_executed(catalog, stub):
    stub(
        definition_reply(fields=[{"table": "WorkRequest", "column": "Nope"}]),
        definition_reply(fields=[{"table": "WorkRequest", "column": "StillNope"}]),
    )
    draft = await draft_structured("work request numbers", catalog)

    assert not draft.valid
    assert draft.sql == ""
    assert "StillNope" in draft.rejection


@pytest.mark.asyncio
async def test_an_unanswerable_request_returns_an_explanation_not_a_guess(catalog, stub):
    """An empty field list is the model saying so; it must not be executed."""
    stub(
        json.dumps(
            {
                "definition": {"base_table": "WorkRequest", "fields": []},
                "explanation": "No table here records fuel burn.",
            }
        )
    )
    draft = await draft_structured("fuel burn per sortie", catalog)

    assert not draft.valid
    assert draft.sql == ""
    assert "fuel burn" in draft.rejection


@pytest.mark.asyncio
async def test_a_request_matching_no_table_stops_before_the_model(catalog, stub):
    client = stub(definition_reply())
    draft = await draft_structured("the and of", catalog)

    assert not draft.valid
    assert "no tables" in draft.rejection
    assert client.calls == []


@pytest.mark.asyncio
async def test_a_malformed_reply_is_repaired_once(catalog, stub):
    client = stub("I am not going to give you JSON.", definition_reply())
    draft = await draft_structured("open work requests", catalog)

    assert draft.valid
    assert draft.repaired
    assert "no JSON object" in client.calls[1][-1].content


@pytest.mark.asyncio
async def test_compiler_warnings_reach_the_reviewer(catalog, stub):
    """A silently added sort order would be a surprise at approval time."""
    stub(definition_reply(sort=[]))
    draft = await draft_structured("open work requests", catalog)

    assert draft.valid
    assert any("sort order" in w for w in draft.warnings)


@pytest.mark.asyncio
async def test_values_from_the_model_stay_bound_parameters(catalog, stub):
    """
    The model is untrusted, so even its filter values never reach the SQL.
    """
    stub(
        definition_reply(
            filters=[
                {
                    "table": "WorkRequest",
                    "column": "StatusCd",
                    "op": "eq",
                    "value": "'; DROP TABLE WorkRequest--",
                }
            ]
        )
    )
    draft = await draft_structured("work requests by status", catalog)

    assert draft.valid
    assert "DROP TABLE" not in draft.sql
    assert draft.parameters == {"f0": "'; DROP TABLE WorkRequest--"}
