"""Tests for recovering tool calls from assistant text.

The supplied `agent-brain` endpoint is not started with a tool-call parser, so it
never populates the OpenAI ``tool_calls`` field. Qwen still chooses correctly but
emits the call as markup in ``content``. Without recovery every question degrades
to "no tool use", which the handout documents as a 0% answer — so these tests
guard the difference between scoring and not scoring.

The captured fixture below is a verbatim response from the live endpoint.
"""

from __future__ import annotations

import json

from brain import extract_text_tool_calls, strip_reasoning

# Verbatim from the cluster's agent-brain endpoint.
LIVE_CAPTURE = (
    "<tool_call>\n<function=query_data>\n<parameter=dataset>\nrba\n</parameter>\n"
    "<parameter=metric>\ncount_changes\n</parameter>\n"
    "<parameter=date_from>\n2010-02-03\n</parameter>\n"
    "<parameter=date_to>\n2026-06-17\n</parameter>\n</function>\n</tool_call>"
)


def args_of(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


def test_live_capture_is_recovered() -> None:
    calls = extract_text_tool_calls(LIVE_CAPTURE)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "query_data"
    assert args_of(calls[0]) == {
        "dataset": "rba",
        "metric": "count_changes",
        "date_from": "2010-02-03",
        "date_to": "2026-06-17",
    }


def test_hermes_json_style() -> None:
    text = (
        '<tool_call>{"name": "query_data", "arguments": '
        '{"dataset": "asx", "metric": "annual_return", "year": 2019}}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert len(calls) == 1
    assert args_of(calls[0])["year"] == 2019


def test_hermes_json_with_stringified_arguments() -> None:
    text = (
        '<tool_call>{"name": "retrieve", "arguments": '
        '"{\\"headline\\": \\"Travel stocks take off\\"}"}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert args_of(calls[0])["headline"] == "Travel stocks take off"


def test_multiple_calls_in_one_message() -> None:
    text = LIVE_CAPTURE + "\n" + (
        "<tool_call>\n<function=dataset_coverage>\n</function>\n</tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == [
        "query_data",
        "dataset_coverage",
    ]
    assert args_of(calls[1]) == {}


def test_missing_tool_call_wrapper() -> None:
    """Some responses emit <function=...> with no <tool_call> around it."""
    text = (
        "<function=query_data>\n<parameter=dataset>\nafr\n</parameter>\n"
        "<parameter=metric>\ncount\n</parameter>\n</function>"
    )
    calls = extract_text_tool_calls(text)
    assert args_of(calls[0]) == {"dataset": "afr", "metric": "count"}


def test_unterminated_block_is_still_recovered() -> None:
    """A truncated generation must not silently lose the call."""
    text = (
        "<tool_call>\n<function=query_data>\n<parameter=dataset>\nrba\n"
        "</parameter>\n<parameter=metric>\nextremes\n</parameter>"
    )
    calls = extract_text_tool_calls(text)
    assert calls and args_of(calls[0])["metric"] == "extremes"


def test_argument_type_coercion() -> None:
    text = (
        "<function=query_data>\n"
        "<parameter=dataset>\nasx\n</parameter>\n"
        "<parameter=metric>\nrank_annual_returns\n</parameter>\n"
        "<parameter=year>\n2018\n</parameter>\n"
        '<parameter=exclude_tickers>\n["TAH.AX"]\n</parameter>\n'
        "<parameter=annualised>\nfalse\n</parameter>\n"
        "</function>"
    )
    args = args_of(extract_text_tool_calls(text)[0])
    assert args["year"] == 2018                    # int, not "2018"
    assert args["exclude_tickers"] == ["TAH.AX"]   # list, not a string
    assert args["annualised"] is False             # bool, not "false"
    assert args["dataset"] == "asx"                # bare word stays a string


def test_regex_pattern_argument_survives_intact() -> None:
    """AFR patterns contain backslashes that must not be mangled."""
    text = (
        "<function=query_data>\n<parameter=dataset>\nafr\n</parameter>\n"
        "<parameter=metric>\ncount\n</parameter>\n"
        "<parameter=pattern>\n\\bunemployment\\b\n</parameter>\n</function>"
    )
    args = args_of(extract_text_tool_calls(text)[0])
    assert args["pattern"] == r"\bunemployment\b"


def test_plain_prose_yields_nothing() -> None:
    """A genuine "I'm done" message must not be misread as a tool call."""
    assert extract_text_tool_calls("I have gathered all the evidence.") == []
    assert extract_text_tool_calls("") == []


def test_strip_reasoning_removes_think_blocks() -> None:
    assert strip_reasoning("<think>plan plan</think>answer") == "answer"
    # Closing tag without an opening one, which this endpoint actually emits.
    assert strip_reasoning("long reasoning...</think>\nthe rest") == "the rest"
    assert strip_reasoning("no tags here") == "no tags here"


def test_reasoning_is_stripped_but_call_survives() -> None:
    """Recovery reads the raw content, so ordering with <think> cannot matter."""
    text = "I need the change counts.</think>\n" + LIVE_CAPTURE
    assert len(extract_text_tool_calls(text)) == 1
    assert "<tool_call>" in text and "</think>" not in strip_reasoning(text)
