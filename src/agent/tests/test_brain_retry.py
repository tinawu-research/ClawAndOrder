"""Tests for planning-turn retries and the request time budget.

Under the harness's three concurrent questions the shared 35B brain routinely
takes longer than one attempt's timeout. A planning turn that returns nothing
gathers no evidence, so the question scores zero — a single transient read
timeout used to cost the whole question. These tests pin the retry policy, the
deadline that keeps retries from eating the synthesis reserve, and the two
silent-failure modes worth guarding: an exception that names no cause, and a
generation truncated before it emitted its tool call.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx
import pytest

import brain
import config


def completion(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


NATIVE_CALL = [
    {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "query_data",
            "arguments": json.dumps({"dataset": "rba", "metric": "count_changes"}),
        },
    }
]

#: Verbatim shape from the cluster, which emits calls as text rather than in the
#: native field. See test_brain_parsing.py.
TEXT_CALL = (
    "<tool_call>\n<function=query_data>\n<parameter=dataset>\nrba\n</parameter>\n"
    "<parameter=metric>\ncount_changes\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic retry settings, with no real sleeping."""
    monkeypatch.setattr(config, "BRAIN_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "BRAIN_TIMEOUT_SECONDS", 22)
    monkeypatch.setattr(config, "BRAIN_MIN_ATTEMPT_SECONDS", 6)
    monkeypatch.setattr(config, "BRAIN_RETRY_BACKOFF_SECONDS", 0.0)


def stub_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Route the brain's HTTP calls to ``handler``. Returns the request log."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(timeout_seconds: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="http://brain.test",
            transport=httpx.MockTransport(recording),
            timeout=timeout_seconds,
        )

    monkeypatch.setattr(brain, "_client", factory)
    return seen


def body_of(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


MESSAGES = [{"role": "user", "content": "How many RBA decisions changed the rate?"}]


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transient_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure that cost MHQ084 its whole score must now be survivable."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("")
        return httpx.Response(200, json=completion(tool_calls=NATIVE_CALL))

    stub_transport(monkeypatch, handler)

    message = await brain.plan(MESSAGES)
    assert calls["n"] == 2
    assert message["tool_calls"][0]["function"]["name"] == "query_data"


@pytest.mark.asyncio
async def test_retry_after_a_timeout_asks_for_brevity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identical retry reproduces the timeout that caused it.

    Under concurrency a read timeout means the model is generating more than the
    clock allows, so the second attempt has to ask for something shorter.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("")
        return httpx.Response(200, json=completion(tool_calls=NATIVE_CALL))

    seen = stub_transport(monkeypatch, handler)

    await brain.plan(MESSAGES)

    first = body_of(seen[0])["messages"]
    retry = body_of(seen[1])["messages"]
    assert not any("did not finish in time" in m["content"] for m in first)
    assert any("did not finish in time" in m["content"] for m in retry)
    # The nudge is appended to the original turn, not stacked on a previous one.
    assert len(retry) == len(first) + 1


@pytest.mark.asyncio
async def test_attempts_are_front_loaded_but_leave_room_for_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first attempt gets the most time; a retry slot survives regardless.

    Splitting the budget evenly starved the first attempt — 15s, 11s and 9s
    against a brain that needs 15-25s on a hard question — so both attempts
    failed where one longer attempt would have completed.
    """
    # Scaled down so the attempts can burn real wall clock quickly. A timed-out
    # attempt consumes its whole timeout in production, and it is that elapsed
    # time which has to shrink the next attempt's slice.
    monkeypatch.setattr(config, "BRAIN_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "BRAIN_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(config, "BRAIN_MIN_ATTEMPT_SECONDS", 0.5)
    timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(1.5)
        raise httpx.ReadTimeout("")

    def factory(timeout_seconds: float) -> httpx.AsyncClient:
        timeouts.append(timeout_seconds)
        return httpx.AsyncClient(
            base_url="http://brain.test",
            transport=httpx.MockTransport(handler),
            timeout=timeout_seconds,
        )

    monkeypatch.setattr(brain, "_client", factory)

    started = time.monotonic()
    with pytest.raises(brain.BrainError):
        await brain.plan(MESSAGES, deadline=started + 3)

    assert len(timeouts) == 2
    # Attempt 1 takes the full configured timeout when the budget allows it.
    assert timeouts[0] == pytest.approx(float(config.BRAIN_TIMEOUT_SECONDS), abs=0.1)
    # Attempt 2 gets what is genuinely left, which is less — and never so little
    # that it was not worth starting.
    assert config.BRAIN_MIN_ATTEMPT_SECONDS <= timeouts[1] < timeouts[0]
    # Neither attempt was allowed to run past the deadline.
    assert time.monotonic() <= started + 3.5


@pytest.mark.asyncio
async def test_error_reports_the_attempts_actually_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count must be what happened, not the configured maximum."""
    monkeypatch.setattr(config, "BRAIN_MIN_ATTEMPT_SECONDS", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.7)
        raise httpx.ReadTimeout("")

    stub_transport(monkeypatch, handler)

    # Room for one attempt only, though three are configured.
    with pytest.raises(brain.BrainError) as exc:
        await brain.plan(MESSAGES, deadline=time.monotonic() + 1.2)

    assert "1 attempt(s)" in str(exc.value)


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_transport(
        monkeypatch, lambda request: (_ for _ in ()).throw(httpx.ReadTimeout(""))
    )

    with pytest.raises(brain.BrainError) as exc:
        await brain.plan(MESSAGES)

    assert len(seen) == config.BRAIN_MAX_ATTEMPTS
    assert "3 attempt(s)" in str(exc.value)


@pytest.mark.asyncio
async def test_overload_status_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """503 from the proxy is worth another attempt; the model is just busy."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(200, json=completion(tool_calls=NATIVE_CALL))

    stub_transport(monkeypatch, handler)

    message = await brain.plan(MESSAGES)
    assert calls["n"] == 2
    assert message["tool_calls"]


@pytest.mark.asyncio
async def test_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 is deterministic — retrying it just burns the time budget."""
    seen = stub_transport(
        monkeypatch, lambda request: httpx.Response(400, text="context length exceeded")
    )

    with pytest.raises(brain.BrainError) as exc:
        await brain.plan(MESSAGES)

    assert len(seen) == 1
    assert "400" in str(exc.value)


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_expired_deadline_makes_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no time left, fail immediately so synthesis still gets its reserve."""
    seen = stub_transport(
        monkeypatch, lambda request: httpx.Response(200, json=completion())
    )

    with pytest.raises(brain.BrainError):
        await brain.plan(MESSAGES, deadline=time.monotonic() - 1)

    assert seen == []


@pytest.mark.asyncio
async def test_deadline_caps_the_attempt_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt must not be allowed to run past the planning deadline."""
    timeouts: list[float] = []

    def factory(timeout_seconds: float) -> httpx.AsyncClient:
        timeouts.append(timeout_seconds)
        return httpx.AsyncClient(
            base_url="http://brain.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=completion(tool_calls=NATIVE_CALL))
            ),
            timeout=timeout_seconds,
        )

    monkeypatch.setattr(brain, "_client", factory)

    await brain.plan(MESSAGES, deadline=time.monotonic() + 9)
    assert timeouts and timeouts[0] <= 9.0


@pytest.mark.asyncio
async def test_deadline_stops_further_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries stop once too little time remains to be worth an attempt.

    Scaled down to keep the test quick. The handler burns real wall clock,
    because a timed-out attempt consumes its whole timeout in production and it
    is that elapsed time which closes the window on the next attempt.
    """
    monkeypatch.setattr(config, "BRAIN_MIN_ATTEMPT_SECONDS", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.7)
        raise httpx.ReadTimeout("")

    seen = stub_transport(monkeypatch, handler)

    # Room for one 1.2s attempt; the 0.5s left afterwards is below the minimum.
    with pytest.raises(brain.BrainError):
        await brain.plan(MESSAGES, deadline=time.monotonic() + 1.2)

    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Output cap and truncated generations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_output_cap_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncapped, vLLM generates until the window is full and the turn times out."""
    seen = stub_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json=completion(tool_calls=NATIVE_CALL)),
    )

    await brain.plan(MESSAGES)
    assert body_of(seen[0])["max_tokens"] == config.BRAIN_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_thinking_is_disabled_on_the_planning_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the turn narrates its deliberation until the cap truncates it.

    Measured at 3x concurrency: 11.6s and a `length` finish with the tool call
    lost, versus 4.8s and a clean `stop` once thinking is off. The server's
    --override-generation-config does not reach the chat template, so the kwarg
    has to travel on each request.
    """
    monkeypatch.setattr(config, "BRAIN_DISABLE_THINKING", True)
    seen = stub_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json=completion(tool_calls=NATIVE_CALL)),
    )

    await brain.plan(MESSAGES)
    body = body_of(seen[0])
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # "required" would strip the brain of its only way to end the loop.
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_thinking_kwarg_is_omitted_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escape hatch: a brain build that rejects the kwarg can turn it off."""
    monkeypatch.setattr(config, "BRAIN_DISABLE_THINKING", False)
    seen = stub_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json=completion(tool_calls=NATIVE_CALL)),
    )

    await brain.plan(MESSAGES)
    assert "chat_template_kwargs" not in body_of(seen[0])


@pytest.mark.asyncio
async def test_truncated_reasoning_is_retried_with_a_nudge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn cut off mid-reasoning has no tool call, and must not read as "done".

    The tool call is emitted after the reasoning, so hitting the output cap loses
    it. Accepting that silently is the handout's 0% "no tool use" answer.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json=completion(
                    content="Let me think about which metric applies",
                    finish_reason="length",
                ),
            )
        return httpx.Response(200, json=completion(content=TEXT_CALL))

    seen = stub_transport(monkeypatch, handler)

    message = await brain.plan(MESSAGES)
    assert message["tool_calls"][0]["function"]["name"] == "query_data"
    # The retry carried the brevity instruction, not just the same prompt again.
    retry_messages = body_of(seen[1])["messages"]
    assert any("did not finish in time" in m["content"] for m in retry_messages)


@pytest.mark.asyncio
async def test_truncation_with_a_recoverable_call_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncation only matters when it cost us the call. Here it did not."""
    seen = stub_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=completion(content=TEXT_CALL, finish_reason="length")
        ),
    )

    message = await brain.plan(MESSAGES)
    assert len(seen) == 1
    assert message["tool_calls"]


@pytest.mark.asyncio
async def test_completed_turn_without_tool_calls_is_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine "evidence gathered" turn must not be retried into oblivion."""
    seen = stub_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=completion(content="I have gathered all the evidence.")
        ),
    )

    message = await brain.plan(MESSAGES)
    assert len(seen) == 1
    assert not message.get("tool_calls")


# ---------------------------------------------------------------------------
# Diagnosability
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_error_names_the_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx timeouts stringify to ''. The log line must still name a cause."""
    stub_transport(
        monkeypatch, lambda request: (_ for _ in ()).throw(httpx.ReadTimeout(""))
    )

    with pytest.raises(brain.BrainError) as exc:
        await brain.plan(MESSAGES)

    assert "ReadTimeout" in str(exc.value)


def test_describe_handles_empty_exception_text() -> None:
    assert brain._describe(httpx.ReadTimeout("")) == "ReadTimeout"
    assert "boom" in brain._describe(httpx.ConnectError("boom"))


@pytest.mark.asyncio
async def test_reasoning_is_still_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry handling must not have bypassed the <think> strip."""
    stub_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=completion(content="<think>deliberating</think>done")
        ),
    )

    message = await brain.plan(MESSAGES)
    assert message["content"] == "done"
