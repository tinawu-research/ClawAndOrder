"""The agent loop: Qwen plans -> runtime executes -> Nemotron synthesises.

::

    question
      -> brain.plan()        Qwen emits tool calls          (planning)
      -> tools.execute()     application code runs them     (execution)
      -> loop until the brain stops asking for tools
      -> synthesis()         fine-tuned Nemotron writes it  (synthesis)
      -> {"answer": ..., "steps": N, "tool_trace": [...]}

Responsibilities are kept strictly separate because the split is itself scored:
the brain never touches a dataset, the runtime never decides what to call, and
the domain model never selects a tool.

Every request builds its own :class:`AgentState`; nothing mutable is shared, so
the three concurrent requests the harness sends cannot interfere. Tool execution
is CPU-bound (regex over 800 MB of article text), so it is dispatched to a
bounded thread pool rather than run on the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import config
import tools
from brain import BrainError, initial_messages, parse_tool_arguments, plan
from compact import compact_payload
from synthesis import synthesize

logger = logging.getLogger(__name__)

#: Bounded so N concurrent requests cannot spawn unbounded threads.
_TOOL_SEMAPHORE = asyncio.Semaphore(config.TOOL_WORKERS)

#: Tool results are truncated before going back to the brain. A full ranking of
#: 17 tickers with six fields each will otherwise crowd out the conversation.
MAX_RESULT_CHARS = 6000


@dataclass
class AgentState:
    """Per-request state. One instance per ``/query`` call, never shared."""

    question: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    brain_calls: int = 0
    started: float = field(default_factory=time.monotonic)
    notes: list[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def plan_deadline(self) -> float:
        """Monotonic instant at which planning must stop.

        Held back from the request deadline so synthesis still has room; the two
        phases run in sequence, so sharing one deadline let them add up.
        """
        return self.started + config.plan_deadline_seconds()

    @property
    def out_of_time(self) -> bool:
        return time.monotonic() >= self.plan_deadline

    def remaining(self) -> float:
        """Seconds left in the whole request, synthesis included."""
        return max(0.0, config.QUERY_DEADLINE_SECONDS - self.elapsed)

    def synthesis_timeout(self) -> float:
        """Timeout for the final call: whatever is left, capped by the setting."""
        return max(
            float(config.BRAIN_MIN_ATTEMPT_SECONDS),
            min(float(config.SYNTHESIS_TIMEOUT_SECONDS), self.remaining()),
        )


def _render_result(payload: dict[str, Any]) -> str:
    """JSON rendering, kept for the organizer-facing ``tool_trace`` only."""
    text = json.dumps(payload, default=str, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + ' ...", "_truncated": true}'
    return text


def _render_for_model(payload: dict[str, Any]) -> str:
    """Compact rendering for anything that costs context.

    Both servers cap ``max_model_len`` at 4096. Feeding raw JSON back into the
    planning loop overflowed that on the multi-tool questions and killed
    planning outright; compaction is lossless for values and roughly halves the
    token cost. See ``compact.py``.
    """
    return compact_payload(payload)


async def _run_tool(name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    async with _TOOL_SEMAPHORE:
        return await asyncio.to_thread(tools.execute, name, arguments)


async def answer_question(question: str) -> dict[str, Any]:
    """Run one question end to end. Always returns a valid response object.

    Never raises: the scoring rules give zero for a malformed response, so every
    failure path still produces a non-empty ``answer``.
    """
    state = AgentState(question=question, messages=initial_messages(question))

    while state.steps < config.MAX_AGENT_STEPS:
        if state.out_of_time:
            state.notes.append(
                f"stopped planning after {state.elapsed:.1f}s to leave "
                f"{config.SYNTHESIS_RESERVE_SECONDS}s of the "
                f"{config.QUERY_DEADLINE_SECONDS}s budget for synthesis"
            )
            break

        try:
            # The planner retries internally; the deadline stops those retries
            # from eating the synthesis reserve.
            message = await plan(state.messages, deadline=state.plan_deadline)
            state.brain_calls += 1
        except BrainError as exc:
            # Without a planner there is nothing to gather. Synthesise from
            # whatever was already collected instead of returning an error.
            logger.error("brain failed on step %d: %s", state.steps, exc)
            state.notes.append(f"brain unavailable: {exc}")
            break

        tool_calls = message.get("tool_calls") or []
        state.messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )

        if not tool_calls:
            # The brain considers the evidence complete.
            break

        for call in tool_calls:
            state.steps += 1
            function = call.get("function") or {}
            name = function.get("name") or "unknown"
            try:
                arguments = parse_tool_arguments(function.get("arguments"))
            except BrainError as exc:
                payload, ok = {"error": str(exc)}, False
                arguments = {}
            else:
                payload, ok = await _run_tool(name, arguments)

            state.tool_trace.append(
                {
                    "tool": name,
                    "args": arguments,
                    # JSON for the organizers' diagnostics; compact for anything
                    # that has to fit in a context window.
                    "result": _render_result(payload),
                    "compact": _render_for_model(payload),
                    "ok": ok,
                }
            )
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{state.steps}",
                    "content": _render_for_model(payload),
                }
            )

    if state.steps >= config.MAX_AGENT_STEPS:
        state.notes.append(
            f"reached MAX_AGENT_STEPS={config.MAX_AGENT_STEPS}"
        )

    # Only successful calls are offered as evidence; a failed call's error text
    # would otherwise read to the synthesiser as a finding.
    verified = [entry for entry in state.tool_trace if entry.get("ok")]
    try:
        answer, mode = await synthesize(
            question, verified, timeout=state.synthesis_timeout()
        )
    except Exception as exc:  # noqa: BLE001 - a valid response is mandatory
        logger.exception("synthesis failed outright")
        answer, mode = (
            "The agent could not complete synthesis for this question, so no "
            "grounded answer can be given.",
            f"failed: {type(exc).__name__}",
        )

    response: dict[str, Any] = {
        "answer": answer,
        # steps = tool calls executed + the synthesis step, per the response
        # contract's "tool calls plus synthesis steps".
        "steps": len(state.tool_trace) + 1,
        "tool_trace": [
            {"tool": e["tool"], "args": e["args"], "result": e["result"]}
            for e in state.tool_trace
        ],
    }
    # Private diagnostics. Only `answer` is graded; `steps` and `tool_trace` are
    # kept for the organizers' report, and these extras help us debug our own.
    response["diagnostics"] = {
        "latency_seconds": round(state.elapsed, 3),
        "brain_calls": state.brain_calls,
        "tool_calls": len(state.tool_trace),
        "tool_failures": sum(1 for e in state.tool_trace if not e.get("ok")),
        "synthesis_mode": mode,
        "notes": state.notes,
    }
    return response
