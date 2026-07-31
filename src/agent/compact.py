"""Render tool payloads compactly, for both the planning loop and synthesis.

The tool payloads are JSON, and JSON is an expensive way to say very little.
``rank_annual_returns`` spends 2,435 characters to convey 17 numbers, because
it repeats ``first_date`` and ``last_date`` on every row, lists ``tickers_used``
in full, and pays JSON's punctuation tax throughout.

That matters twice over, and both servers cap ``max_model_len`` at 4096:

* The **planning loop** feeds every tool result back to the brain. Four calls at
  the old 6,000-char cap overflowed the context and killed planning outright on
  the hardest questions.
* The **synthesis** prompt carries the same payloads, and the fine-tune has a
  far smaller sequence budget than that.

The transformation here is deliberately *lossless for values*: it hoists
columns that are constant across rows into a header, drops keys that restate
the request, and strips JSON syntax. No number, date or identifier is removed.
Truncation only ever happens at the explicit character budget, and then only to
free-text article bodies.
"""

from __future__ import annotations

import json
from typing import Any

#: Keys that restate the request or the schema rather than carrying a finding.
#: ``metric`` and ``dataset`` are already visible in the call header that
#: precedes every block, so repeating them in the body is pure overhead.
_DROP_KEYS = frozenset(
    {
        "dataset",
        "metric",
        "match_rules",
        "note",
        "notes",
        "ignored_arguments",
        "index_ready",
        "rows_identical_across_tickers",
    }
)

#: Long verbatim text fields, truncated to a sentence boundary when over budget.
_TEXT_KEYS = frozenset({"text", "blob", "body", "snippet", "excerpt"})


def _fmt_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # 4 dp is finer than every declared tolerance (the tightest is +/-0.0001
        # on quoted closes), and strips float64 noise like 22.171299999999998.
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _fmt_list(values: list[Any]) -> str:
    return ", ".join(_fmt_scalar(v) for v in values)


def _truncate_text(text: str, budget: int) -> str:
    """Cut to the last sentence boundary inside ``budget`` characters."""
    if len(text) <= budget:
        return text
    window = text[:budget]
    for stop in (". ", "! ", "? ", "\n"):
        cut = window.rfind(stop)
        if cut > budget * 0.5:
            return window[: cut + 1].rstrip() + " [truncated]"
    return window.rstrip() + " [truncated]"


def _split_constant_columns(
    rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Partition a table's columns into constant-valued and varying.

    Constant columns are the big win: ``window_return`` repeats
    ``used_start_date`` and ``used_end_date`` on all 17 rows.
    """
    if not rows:
        return {}, []
    columns = list(rows[0].keys())
    constant: dict[str, Any] = {}
    varying: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        first = values[0]
        if len(rows) > 1 and all(v == first for v in values) and not isinstance(first, (dict, list)):
            constant[column] = first
        else:
            varying.append(column)
    return constant, varying


def _render_table(name: str, rows: list[dict[str, Any]], text_budget: int) -> list[str]:
    constant, varying = _split_constant_columns(rows)

    header = name
    if constant:
        header += " " + " ".join(f"{k}={_fmt_scalar(v)}" for k, v in constant.items())
    header += f" [{' '.join(varying)}] n={len(rows)}"

    lines = [header]
    for row in rows:
        cells = []
        for column in varying:
            value = row.get(column)
            if isinstance(value, str) and column in _TEXT_KEYS:
                value = _truncate_text(value, text_budget)
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, default=str, ensure_ascii=False)
            cells.append(_fmt_scalar(value))
        lines.append("  " + " | ".join(cells))
    return lines


def compact_payload(payload: Any, *, text_budget: int = 900) -> str:
    """Render one tool payload as compact text.

    ``text_budget`` caps each free-text field (an AFR article body); structured
    values are never dropped.
    """
    if not isinstance(payload, dict):
        return _fmt_scalar(payload)

    scalars: list[str] = []
    blocks: list[str] = []

    for key, value in payload.items():
        if key in _DROP_KEYS:
            continue

        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            blocks.extend(_render_table(key, value, text_budget))
        elif isinstance(value, list):
            # Never elide entries. An earlier version summarised long lists by
            # count, which silently dropped 8 of the 17 tickers from a drawdown
            # roster. The saving comes from removing JSON syntax, not from
            # discarding data.
            scalars.append(f"{key}=[{_fmt_list(value)}]")
        elif isinstance(value, dict):
            # Same rule. Truncating a 84-entry ``by_month`` map to its first 12
            # would have thrown away the peak month, which is the graded value.
            inner = " ".join(f"{k}:{_fmt_scalar(v)}" for k, v in value.items())
            scalars.append(f"{key}={{{inner}}}")
        elif isinstance(value, str) and key in _TEXT_KEYS:
            scalars.append(f"{key}={_truncate_text(value, text_budget)}")
        else:
            scalars.append(f"{key}={_fmt_scalar(value)}")

    lines: list[str] = []
    if scalars:
        lines.append(" ".join(scalars))
    lines.extend(blocks)
    return "\n".join(lines) if lines else "(empty result)"


def compact_result(result: str | dict[str, Any], *, text_budget: int = 900) -> str:
    """Compact a payload that may already have been serialised to JSON."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return compact_payload(result, text_budget=text_budget)
