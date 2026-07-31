"""End-to-end loop tests with the brain stubbed out.

These run without the cluster: no LiteLLM, no Qwen, no adapter. They pin the
parts of the contract that must hold regardless of what the models do — a valid
response shape, correct role separation, and graceful degradation — because a
malformed or missing ``answer`` scores zero for that question.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

import brain
import config
import orchestrator
from datastore import STORE
from orchestrator import answer_question


@pytest.fixture(scope="module", autouse=True)
def _data() -> None:
    STORE._load_rba()
    STORE._load_asx()
    STORE._loaded.set()


def make_brain(script: list[dict[str, Any]]):
    """Return a fake ``plan`` that replays ``script`` turn by turn."""
    calls = {"n": 0}

    async def fake_plan(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        i = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return script[i]

    fake_plan.calls = calls  # type: ignore[attr-defined]
    return fake_plan


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


DONE = {"content": "evidence gathered", "tool_calls": []}


@pytest.mark.asyncio
async def test_happy_path_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal run returns a valid response with a populated trace."""
    monkeypatch.setattr(
        brain,
        "plan",
        make_brain(
            [
                {
                    "content": "",
                    "tool_calls": [
                        tool_call("query_data", {"dataset": "rba", "metric": "count_changes"})
                    ],
                },
                DONE,
            ]
        ),
    )
    monkeypatch.setattr(orchestrator, "plan", brain.plan)

    result = await answer_question("How many RBA decisions changed the rate?")

    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert result["steps"] == 2  # one tool call + synthesis
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["tool"] == "query_data"
    # The verified numbers must reach synthesis.
    assert "41" in result["tool_trace"][0]["result"]


@pytest.mark.asyncio
async def test_brain_failure_still_returns_valid_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable brain must not produce a 500 or an empty answer."""

    async def exploding_plan(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        raise brain.BrainError("connection refused")

    monkeypatch.setattr(orchestrator, "plan", exploding_plan)

    result = await answer_question("anything")
    assert result["answer"].strip()
    assert result["tool_trace"] == []


@pytest.mark.asyncio
async def test_bad_tool_arguments_are_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected call comes back as evidence-free error text, not an exception."""
    monkeypatch.setattr(
        orchestrator,
        "plan",
        make_brain(
            [
                {
                    "content": "",
                    "tool_calls": [
                        tool_call("query_data", {"dataset": "rba", "metric": "nonsense"})
                    ],
                },
                DONE,
            ]
        ),
    )
    result = await answer_question("bad metric")
    assert result["answer"].strip()
    # The failed call is reported in the trace but excluded from evidence.
    assert len(result["tool_trace"]) == 1
    assert "error" in result["tool_trace"][0]["result"]
    assert result["diagnostics"]["tool_failures"] == 1


@pytest.mark.asyncio
async def test_unknown_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator,
        "plan",
        make_brain(
            [
                {"content": "", "tool_calls": [tool_call("make_it_up", {})]},
                DONE,
            ]
        ),
    )
    result = await answer_question("unknown tool")
    assert result["answer"].strip()
    assert "unknown tool" in result["tool_trace"][0]["result"]


@pytest.mark.asyncio
async def test_step_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A brain that never stops asking must still terminate."""
    looping = make_brain(
        [
            {
                "content": "",
                "tool_calls": [
                    tool_call("query_data", {"dataset": "rba", "metric": "coverage"})
                ],
            }
        ]
    )
    monkeypatch.setattr(orchestrator, "plan", looping)

    result = await answer_question("loop forever")
    assert len(result["tool_trace"]) <= config.MAX_AGENT_STEPS
    assert result["answer"].strip()


@pytest.mark.asyncio
async def test_mock_synthesis_is_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock mode must be visible in diagnostics, not silently indistinguishable."""
    monkeypatch.setattr(config, "DOMAIN_PREDICT_MODE", "mock")
    monkeypatch.setattr(orchestrator, "plan", make_brain([DONE]))
    result = await answer_question("anything")
    assert result["diagnostics"]["synthesis_mode"] == "mock"


@pytest.mark.asyncio
async def test_malformed_tool_arguments_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models sometimes fence their JSON; that must not cost the question."""
    monkeypatch.setattr(
        orchestrator,
        "plan",
        make_brain(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "query_data",
                                "arguments": '```json\n{"dataset":"rba","metric":"coverage"}\n```',
                            },
                        }
                    ],
                },
                DONE,
            ]
        ),
    )
    result = await answer_question("fenced json")
    assert result["diagnostics"]["tool_failures"] == 0
    assert "175" in result["tool_trace"][0]["result"]


@pytest.mark.asyncio
async def test_planner_receives_the_reserved_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning must be told to stop early enough for synthesis to run.

    The two phases are sequential, so before the reserve existed a loop that ran
    to QUERY_DEADLINE_SECONDS followed by a full-length synthesis call answered
    well past the 60s mark and lost 20% of its earned points.
    """
    seen: dict[str, Any] = {}

    async def capturing_plan(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        seen.update(kwargs)
        return DONE

    monkeypatch.setattr(orchestrator, "plan", capturing_plan)

    await answer_question("anything")

    assert "deadline" in seen, "the planner must be given a deadline"
    budget = seen["deadline"] - time.monotonic()
    assert budget <= config.plan_deadline_seconds() + 1
    # The reserve is genuinely held back, not just nominally present.
    assert budget < config.QUERY_DEADLINE_SECONDS


def test_reserve_is_large_enough_for_the_call_it_reserves_for() -> None:
    """A reserve smaller than the synthesis timeout is not a reserve.

    Planning runs to ``plan_deadline_seconds()`` and synthesis follows it, so if
    the reserve is short the worst case spills back over the request deadline by
    the difference.
    """
    assert config.SYNTHESIS_RESERVE_SECONDS >= config.SYNTHESIS_TIMEOUT_SECONDS

    worst_case = config.plan_deadline_seconds() + config.SYNTHESIS_TIMEOUT_SECONDS
    assert worst_case <= config.QUERY_DEADLINE_SECONDS
    # 60s is the hard line where the scorer starts deducting 20%.
    assert worst_case < 60


def test_synthesis_timeout_shrinks_with_the_remaining_budget() -> None:
    state = orchestrator.AgentState(question="q")
    # Fresh request: the full configured timeout is available.
    assert state.synthesis_timeout() == pytest.approx(
        float(config.SYNTHESIS_TIMEOUT_SECONDS), abs=1.0
    )
    # Nearly out of time: the call is squeezed rather than allowed to overrun.
    state.started = time.monotonic() - (config.QUERY_DEADLINE_SECONDS - 3)
    assert state.synthesis_timeout() < config.SYNTHESIS_TIMEOUT_SECONDS


def test_parse_tool_arguments_variants() -> None:
    assert brain.parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert brain.parse_tool_arguments({"a": 1}) == {"a": 1}
    assert brain.parse_tool_arguments("") == {}
    assert brain.parse_tool_arguments(None) == {}
    with pytest.raises(brain.BrainError):
        brain.parse_tool_arguments("not json at all")


def test_dataset_coverage_flags_the_2021_cutoff() -> None:
    """The coverage tool must expose the gap MHQ090 turns on."""
    import tools

    cov = tools.dataset_coverage()
    assert cov["datasets"]["asx"]["end"] == "2021-12-30"
    assert cov["datasets"]["rba"]["end"] > "2021-12-31"
    # The overlap must end in 2021, which is what makes a 2022-23 join
    # unobservable across all three datasets.
    assert cov["common_overlap"]["end"] < "2022-01-01"
