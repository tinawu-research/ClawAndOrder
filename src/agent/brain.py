"""The reasoning brain: supplied Qwen3.6-35B-A3B-FP8 via the ``agent-brain`` alias.

The brain owns planning, tool selection and tool-call generation, and nothing
else. It never writes the final answer — that is the fine-tuned Nemotron's job
(see :mod:`synthesis`) — and it never executes a tool, which is
:mod:`tools`'s job. Participants do not fine-tune this model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

import config
from tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Textual tool-call recovery
# ---------------------------------------------------------------------------
# The supplied vLLM server behind `agent-brain` *is* started with
# `--enable-auto-tool-choice --tool-call-parser hermes`, but the native
# `tool_calls` field still comes back empty on every `tool_choice: "auto"` turn.
# The hermes parser expects a JSON body inside `<tool_call>`, whereas this build
# emits nested XML, so the parser never matches. (Forcing
# `tool_choice: "required"` does produce native calls, via a different decoding
# path — but that removes the brain's ability to end the loop by answering
# without a call, so it is not usable here.)
#
# Qwen still *decides* correctly — it picks the right tool and arguments — but
# emits the call as text in its own markup:
#
#     <tool_call>
#     <function=query_data>
#     <parameter=dataset>
#     rba
#     </parameter>
#     ...
#     </function>
#     </tool_call>
#
# Left unparsed, every question degrades to "no tool use", which the handout
# documents as a 0% answer. We therefore recover tool calls from the text as a
# fallback. This is additive: if the server is later restarted with
# `--enable-auto-tool-choice`, the native field wins and none of this runs.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAIL_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE
)
_XML_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z0-9_.-]+)\s*>\s*(.*?)\s*(?:</function>|$)",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.-]+)\s*>\s*(.*?)\s*(?:</parameter>|$)",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(content: str) -> str:
    """Remove ``<think>`` reasoning so it is never mistaken for an answer."""
    text = _THINK_BLOCK_RE.sub("", content or "")
    # Some builds emit the closing tag without the opening one.
    if "</think>" in text.lower():
        text = _THINK_TAIL_RE.sub("", text)
    return text.strip()


def _coerce(value: str) -> Any:
    """Best-effort typing of an XML parameter value.

    ``2019`` becomes an int and ``["TAH.AX"]`` a list, while a bare word such as
    ``rba`` stays a string. The tool layer normalises further, so an
    under-converted value is harmless; a mis-converted one would not be.
    """
    text = value.strip()
    if not text:
        return text
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def extract_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Recover tool calls embedded in assistant text.

    Handles the two shapes seen in the wild: a JSON object inside
    ``<tool_call>`` (Hermes style) and the nested ``<function=>`` /
    ``<parameter=>`` markup above.
    """
    if not content:
        return []
    blocks = _TOOL_CALL_BLOCK_RE.findall(content)
    if not blocks:
        # Some responses omit the <tool_call> wrapper entirely.
        if "<function=" in content:
            blocks = [content]
        else:
            return []

    calls: list[dict[str, Any]] = []
    for block in blocks:
        stripped = block.strip()
        # Hermes style: {"name": ..., "arguments": {...}}
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("name"):
                arguments = parsed.get("arguments", parsed.get("parameters", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                calls.append(
                    {
                        "id": f"text_call_{len(calls) + 1}",
                        "type": "function",
                        "function": {
                            "name": str(parsed["name"]),
                            "arguments": json.dumps(arguments or {}),
                        },
                    }
                )
                continue

        for name, body in _XML_FUNCTION_RE.findall(block):
            arguments = {
                key: _coerce(value)
                for key, value in _XML_PARAMETER_RE.findall(body)
            }
            calls.append(
                {
                    "id": f"text_call_{len(calls) + 1}",
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "arguments": json.dumps(arguments),
                    },
                }
            )
    return calls


SYSTEM_PROMPT = """\
You are the planning brain of a financial-market question answering agent. You \
have three approved local datasets, reachable only through your tools:

  RBA  cash-rate decisions,      175 records, 2010-02-03 .. 2026-06-17
  ASX  18 tickers x 1774 bars each,           2015-01-02 .. 2021-12-30
  AFR  219,538 news articles,                 2015-01-02 .. 2021-12-29

Note the asymmetry: RBA extends years beyond ASX and AFR, which both stop in
2021. Any question about market or news behaviour after 2021 is unobservable.

YOUR JOB is to gather exact evidence by calling tools. You do NOT write the
final answer — a separate domain model does that from the evidence you collect.
So do not compose prose, do not summarise, and do not stop early because you
could phrase something plausibly. Stop only when every part of the question is
backed by a tool result.

RULES

1. Never compute, estimate, recall or guess a number, date, count, rate,
   return, ranking or streak. Every such value must come from a tool result.
   A plausible sentence with an unverified number scores zero.

2. Re-read the question and list every component it asks for. A question
   asking for a count AND a cumulative change AND two endpoint rates needs all
   four in evidence. Each component is scored independently, so a missing one
   is lost points even when the rest are right.

3. Prefer ONE precise call over several vague ones. Most questions need 1-3
   calls; hard multi-part ones may need 4-5. There is a strict time budget, so
   never enumerate raw rows to inspect them by eye — pick the metric that
   computes the answer directly.

4. When the question says "non-Tabcorp" or "excluding Tabcorp", pass
   exclude_tickers=["TAH.AX"]. Forgetting it changes the answer.

5. For "the cash rate in force on <date>", use rba/lookup_rate with
   date_from=<date>. It resolves on-or-before, which is what "in force" means.

6. For AFR counts, prefer terms=["word"] — it applies the required word
   boundaries for you. Use pattern= only when the question dictates an exact
   regex, and then include the \\b anchors yourself.

6a. A one-week move after an event is asx/event_window with calendar_days=7,
   NOT sessions=5. "the week after the cut", "the one-week return" and a span
   written as two dates a week apart ("5-12 Jun") are all calendar_days=7.
   The two disagree whenever a holiday falls inside the week — after
   2019-06-05 sessions=5 runs to 13 Jun and calendar_days=7 to 12 Jun — so
   the wrong one returns a plausible number for the wrong window. Use
   sessions=N only when the question counts sessions itself.

7. If the question spans datasets, or names a period that might fall outside
   one of them, call dataset_coverage first. Do not substitute a nearby period
   and do not estimate values the data does not cover. Insufficient coverage is
   a finding, not a reason to stop: still gather the part of the question that
   IS observable in whichever dataset covers the period.

8. If a tool returns an error, read it and retry with corrected arguments.
   Do not repeat an identical failing call.

When the evidence is complete, reply with a short plain-text note of what you
gathered and make no further tool calls. That note is not the answer; it just
ends your turn.

OUTPUT DISCIPLINE

Do not write a "Thinking Process" section, do not number your deliberation and
do not narrate what you are about to do. Emit the tool call as your first
output. When you are done gathering, one short sentence ends the turn.

This is a hard latency constraint, not a style preference: you are shared
between three questions at once, and a question that spends its budget watching
you deliberate scores nothing at all.
"""


class BrainError(RuntimeError):
    """The brain could not be reached or returned an unusable response."""


class _Retryable(Exception):
    """Internal: this attempt failed in a way another attempt might survive."""

    def __init__(self, reason: str, *, backoff: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.backoff = backoff


#: Statuses worth another attempt: overload and proxy/upstream hiccups. A 400 is
#: deterministic — the same request would fail the same way — so it is not here.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Appended when a turn ran out of clock or hit the output cap. Qwen is not ours
#: to fine-tune, so brevity has to be asked for.
#:
#: Deliberately does NOT demand a tool call. An earlier version said "emit the
#: tool call now", which forced a call on turns whose correct action was to stop:
#: the model duly re-issued the previous call verbatim and burned the rest of the
#: question's budget. Both outcomes have to stay available.
_BREVITY_NUDGE = (
    "Your previous reply did not finish in time. Answer immediately and "
    "briefly, with no preamble and no narrated deliberation. If you still need "
    "data, emit only the tool call. If the evidence you already have is "
    "complete, reply with one short sentence and no tool call."
)


def _describe(exc: Exception) -> str:
    """Render an exception with its type.

    httpx timeout and connect exceptions frequently stringify to the empty
    string, which produced log lines that named no cause at all.
    """
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _client(timeout_seconds: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=config.LITELLM_URL.rstrip("/"),
        headers={
            "Authorization": f"Bearer {config.LITELLM_KEY}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(timeout_seconds),
    )


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a tool-call ``arguments`` field into a dict.

    Models emit this as a JSON string, occasionally as a dict, and sometimes as
    a JSON string wrapped in markdown fences. Recovering here is much cheaper
    than spending a step on a retry.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BrainError(f"tool arguments were not valid JSON: {raw!r}") from exc
    return parsed if isinstance(parsed, dict) else {"value": parsed}


async def _attempt(
    messages: list[dict[str, Any]], timeout_seconds: float
) -> tuple[dict[str, Any], str]:
    """One HTTP round trip. Returns ``(assistant_message, finish_reason)``.

    Raises :class:`_Retryable` for failures another attempt might survive and
    :class:`BrainError` for those it would not.

    ``temperature=0`` because planning must be reproducible: the architecture
    score rewards reproducibility, and a re-run that picks a different metric is
    not diagnosable.
    """
    payload: dict[str, Any] = {
        "model": config.BRAIN_MODEL,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": config.BRAIN_MAX_OUTPUT_TOKENS,
    }
    if config.BRAIN_DISABLE_THINKING:
        # Suppresses the narrated preamble that would otherwise consume the
        # output cap before a tool call is emitted. See the measurements on
        # config.BRAIN_DISABLE_THINKING. `tool_choice` stays "auto": the loop
        # ends when the brain answers *without* a call, so "required" would
        # remove its only way to say "the evidence is complete".
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    async with _client(timeout_seconds) as client:
        try:
            response = await client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Retryable(
                f"{_describe(exc)} after {timeout_seconds:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise BrainError(
                f"brain request to {config.LITELLM_URL} failed: {_describe(exc)}"
            ) from exc

    if response.status_code in _RETRYABLE_STATUS:
        raise _Retryable(
            f"HTTP {response.status_code}: {response.text[:200]}", backoff=True
        )
    if response.status_code >= 400:
        raise BrainError(
            f"brain returned HTTP {response.status_code}: {response.text[:400]}"
        )
    try:
        body = response.json()
        choice = body["choices"][0]
        message = dict(choice["message"])
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise _Retryable(
            f"unparseable brain response: {response.text[:200]}"
        ) from exc
    return message, str(choice.get("finish_reason") or "")


async def plan(
    messages: list[dict[str, Any]], *, deadline: float | None = None
) -> dict[str, Any]:
    """One brain turn, retried within the request's remaining time.

    ``deadline`` is a :func:`time.monotonic` value past which planning must not
    continue. Attempts are front-loaded: the first gets as much of the remaining
    budget as it can while still leaving ``BRAIN_MIN_ATTEMPT_SECONDS`` for each
    attempt behind it. Dividing the budget evenly instead starved the first
    attempt — measured at 15s, 11s and 9s against a brain that needs 15-25s for a
    hard question — so both attempts failed where one longer one would have run.

    A retry after a timeout also carries the brevity nudge. Under concurrency a
    read timeout and a truncated generation have the same cause — the model is
    producing more reasoning than the clock allows — so repeating the request
    unchanged tends to reproduce the timeout exactly.
    """
    attempts = max(1, config.BRAIN_MAX_ATTEMPTS)
    turn = list(messages)
    last_error = "no attempt was made"
    made = 0

    for attempt in range(1, attempts + 1):
        timeout = float(config.BRAIN_TIMEOUT_SECONDS)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining < config.BRAIN_MIN_ATTEMPT_SECONDS:
                last_error = (
                    f"{last_error}; gave up with {remaining:.1f}s left, below the "
                    f"{config.BRAIN_MIN_ATTEMPT_SECONDS}s minimum attempt"
                )
                break
            # Hold back a minimum-length slot for each attempt still to come.
            held_back = config.BRAIN_MIN_ATTEMPT_SECONDS * (attempts - attempt)
            timeout = min(
                timeout,
                remaining,
                max(remaining - held_back, float(config.BRAIN_MIN_ATTEMPT_SECONDS)),
            )

        try:
            made += 1
            message, finish_reason = await _attempt(turn, timeout)
        except _Retryable as exc:
            last_error = exc.reason
            logger.warning(
                "brain attempt %d/%d failed: %s", attempt, attempts, last_error
            )
            # Ask for brevity next time; the clock, not the request, is what ran
            # out, and an identical retry burns the rest of the budget.
            turn = [*messages, {"role": "user", "content": _BREVITY_NUDGE}]
            if exc.backoff and attempt < attempts:
                await asyncio.sleep(config.BRAIN_RETRY_BACKOFF_SECONDS)
            continue

        content = message.get("content") or ""
        if not message.get("tool_calls"):
            recovered = extract_text_tool_calls(content)
            if recovered:
                logger.info(
                    "recovered %d tool call(s) from assistant text — the brain "
                    "endpoint is not emitting native tool_calls",
                    len(recovered),
                )
                message["tool_calls"] = recovered

        if finish_reason == "length" and not message.get("tool_calls"):
            # Truncated mid-reasoning, so the call never got emitted. Accepting
            # this would silently look like "the brain is done".
            last_error = (
                f"generation hit the {config.BRAIN_MAX_OUTPUT_TOKENS}-token cap "
                "before emitting a tool call"
            )
            logger.warning(
                "brain attempt %d/%d: %s — retrying with a brevity nudge",
                attempt,
                attempts,
                last_error,
            )
            turn = [*messages, {"role": "user", "content": _BREVITY_NUDGE}]
            continue

        # Reasoning must never survive into the transcript: it would be re-read
        # as gathered evidence on the next turn, and the handout's 0% example is
        # precisely a model narrating its plan instead of acting on it.
        message["content"] = strip_reasoning(content)
        return message

    raise BrainError(f"brain failed after {made} attempt(s): {last_error}")


def initial_messages(question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
