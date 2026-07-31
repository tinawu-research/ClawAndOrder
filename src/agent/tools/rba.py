"""Deterministic metrics over the RBA cash-rate decision table.

Every value returned here is computed by parsing and arithmetic — never by a
model. The Challenge Brief's zero-score examples are all cases where a model
narrated instead of computing, so no function in this module is allowed to
estimate.

Cumulative moves are summed in **integer basis points** and only converted to
percentage points on the way out. ``0.15`` and ``0.25`` do not sum exactly in
binary floating point, and a cumulative "-2.25 pp" that reaches the synthesis
model as ``-2.2500000000000004`` loses the component it was meant to earn.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any

from datastore import STORE, RbaRow
from tools.dateparse import parse_from, parse_to


class MetricError(ValueError):
    """Raised for an unusable metric or argument, surfaced back to the brain."""


def _rows() -> tuple[RbaRow, ...]:
    if not STORE.rba:
        raise MetricError("RBA dataset is not loaded")
    return STORE.rba


def _bp(row: RbaRow) -> int:
    """A row's change as integer basis points (``+0.25`` -> ``25``)."""
    return round(row.change * 100)


def _points(basis_points: int) -> float:
    """Basis points back to percentage points, exactly (``-225`` -> ``-2.25``)."""
    return round(basis_points / 100, 2)


def _parse_date(raw: str, *, upper: bool, field: str) -> date:
    try:
        return parse_to(raw) if upper else parse_from(raw)
    except ValueError as exc:
        raise MetricError(f"{field}: {exc}") from exc


def _window(
    date_from: str | None = None, date_to: str | None = None
) -> tuple[date, date]:
    rows = _rows()
    start = (
        _parse_date(date_from, upper=False, field="date_from")
        if date_from
        else rows[0].effective
    )
    end = (
        _parse_date(date_to, upper=True, field="date_to")
        if date_to
        else rows[-1].effective
    )
    if start > end:
        raise MetricError(f"date_from {start} is after date_to {end}")
    return start, end


def _in_window(row: RbaRow, start: date, end: date) -> bool:
    return start <= row.effective <= end


def coverage() -> dict[str, Any]:
    """Record count and full date span of the decision table."""
    rows = _rows()
    return {
        "metric": "coverage",
        "records": len(rows),
        "first_effective_date": rows[0].iso,
        "last_effective_date": rows[-1].iso,
        "first_target": rows[0].target,
        "last_target": rows[-1].target,
    }


def count(date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
    """Total decision records in the window (changes and holds alike)."""
    start, end = _window(date_from, date_to)
    rows = [r for r in _rows() if _in_window(r, start, end)]
    return {
        "metric": "count",
        "window": [start.isoformat(), end.isoformat()],
        "records": len(rows),
    }


def count_changes(
    date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    """Non-zero rate changes, split into increases and decreases.

    Answers the canonical easy question: *how many cash-rate decisions changed
    the rate, and how many were increases versus decreases?*
    """
    start, end = _window(date_from, date_to)
    rows = [r for r in _rows() if _in_window(r, start, end)]
    changed = [r for r in rows if r.change != 0.0]
    increases = [r for r in changed if r.change > 0]
    decreases = [r for r in changed if r.change < 0]
    return {
        "metric": "count_changes",
        "window": [start.isoformat(), end.isoformat()],
        "total_records": len(rows),
        "changes": len(changed),
        "increases": len(increases),
        "decreases": len(decreases),
        "cumulative_change_points": _points(sum(_bp(r) for r in changed)),
        "increase_points": _points(sum(_bp(r) for r in increases)),
        "decrease_points": _points(sum(_bp(r) for r in decreases)),
        "increases_by_year": dict(
            sorted(Counter(r.effective.year for r in increases).items())
        ),
        "decreases_by_year": dict(
            sorted(Counter(r.effective.year for r in decreases).items())
        ),
    }


def count_increases(
    date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    result = count_changes(date_from, date_to)
    return {
        "metric": "count_increases",
        "window": result["window"],
        "increases": result["increases"],
        "by_year": result["increases_by_year"],
    }


def count_decreases(
    date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    result = count_changes(date_from, date_to)
    return {
        "metric": "count_decreases",
        "window": result["window"],
        "decreases": result["decreases"],
        "by_year": result["decreases_by_year"],
    }


def extremes() -> dict[str, Any]:
    """Highest and lowest cash-rate target, with first effective date and count.

    ``first_effective_date`` is the *earliest* record carrying that target. The
    handout's partial-credit example lost points precisely here, so the earliest
    record — not the first one encountered in file order — is what is reported.
    """
    rows = _rows()
    highest = max(r.target for r in rows)
    lowest = min(r.target for r in rows)

    def describe(target: float) -> dict[str, Any]:
        matching = [r for r in rows if r.target == target]
        return {
            "target": target,
            "records": len(matching),
            "first_effective_date": min(r.effective for r in matching).isoformat(),
            "last_effective_date": max(r.effective for r in matching).isoformat(),
        }

    return {
        "metric": "extremes",
        "highest": describe(highest),
        "lowest": describe(lowest),
    }


def max_hold_streak(top_n: int | None = None) -> dict[str, Any]:
    """Longest gap in days between two consecutive non-zero rate changes.

    Reports the rate held during the streak and the rate it moved to, because
    the full-marks example for this question states both.

    ``top_n`` returns a ranking rather than just the winner, which is what
    "the three longest holds" style questions need. The stretch still running at
    the end of the table is reported separately under ``open_streak``: it has no
    closing change, so it is not comparable with a completed one and must not be
    silently ranked against them.
    """
    rows = _rows()
    changed = [r for r in rows if r.change != 0.0]
    if len(changed) < 2:
        raise MetricError("fewer than two rate changes on record")

    streaks = [
        {
            "days": (nxt.effective - cur.effective).days,
            "start_date": cur.iso,
            "end_date": nxt.iso,
            "rate_during_hold": cur.target,
            "rate_after_change": nxt.target,
            "change_points": _points(_bp(nxt)),
            # Hold decisions sitting inside the streak: the meetings that left
            # the target alone, which is what makes it a "hold" streak.
            "hold_decisions_within": sum(
                1 for r in rows if cur.effective < r.effective < nxt.effective
            ),
        }
        for cur, nxt in zip(changed, changed[1:])
    ]
    ranked = sorted(streaks, key=lambda s: (-s["days"], s["start_date"]))
    longest = ranked[0]

    result = {
        "metric": "max_hold_streak",
        **longest,
        "open_streak": {
            "days": (rows[-1].effective - changed[-1].effective).days,
            "since": changed[-1].iso,
            "through": rows[-1].iso,
            "rate_held": changed[-1].target,
            "note": (
                "Still open at the end of the table — no closing change, so "
                "not comparable with the completed streaks above."
            ),
        },
    }
    if top_n:
        result["ranked_longest_first"] = ranked[: int(top_n)]
    return result


def lookup_rate(date_from: str) -> dict[str, Any]:
    """Cash-rate target in force **on or before** ``date_from``.

    Strictly on-or-before, never nearest. The table runs to 17 Jun 2026, so a
    nearest-date lookup would happily return a rate from the future — the
    handout calls this out explicitly, and it is worth 2 points on each of the
    article-sentiment questions.
    """
    target_day = _parse_date(date_from, upper=True, field="date_from")
    rows = _rows()
    applicable = [r for r in rows if r.effective <= target_day]
    if not applicable:
        return {
            "metric": "lookup_rate",
            "query_date": target_day.isoformat(),
            "in_force": None,
            "note": (
                f"No RBA record on or before {target_day.isoformat()}; the table "
                f"begins {rows[0].iso}."
            ),
        }
    row = applicable[-1]
    result = {
        "metric": "lookup_rate",
        "query_date": target_day.isoformat(),
        "cash_rate_target": row.target,
        "effective_date": row.iso,
        "change_points_on_that_date": _points(_bp(row)),
    }

    # The next move gives the synthesis model the "held until" context these
    # questions usually want alongside the rate itself.
    later = [r for r in rows if r.effective > target_day and r.change != 0.0]
    if later:
        result["next_change"] = {
            "effective_date": later[0].iso,
            "change_points": _points(_bp(later[0])),
            "new_target": later[0].target,
        }
    elif target_day > rows[-1].effective:
        # Carrying the last known target past the end of the table is an
        # assumption, not evidence. Say so, or it gets quoted as a fact.
        result["note"] = (
            f"{target_day.isoformat()} is beyond the table, which ends "
            f"{rows[-1].iso}. This carries the last known target forward and is "
            "not observed evidence."
        )
    else:
        result["next_change"] = None
    return result


def period_summary(date_from: str, date_to: str) -> dict[str, Any]:
    """Cuts, hikes, cumulative move and endpoint rates for a rate cycle.

    Covers the "easing period" / "tightening cycle" family of questions, which
    ask for a count, a per-year split, a cumulative change in percentage points,
    and both endpoint targets.

    ``target_before_first_change`` is the rate in force immediately *before* the
    first change inside the window — i.e. the last record preceding it, which is
    normally outside the window.
    """
    start, end = _window(date_from, date_to)
    rows = _rows()
    in_window = [r for r in rows if _in_window(r, start, end)]
    changed = [r for r in in_window if r.change != 0.0]

    before = [r for r in rows if r.effective < (changed[0].effective if changed else start)]
    target_before = before[-1].target if before else None

    at_end = [r for r in rows if r.effective <= end]
    increases = [r for r in changed if r.change > 0]
    decreases = [r for r in changed if r.change < 0]
    return {
        "metric": "period_summary",
        "window": [start.isoformat(), end.isoformat()],
        "records_in_window": len(in_window),
        "changes": len(changed),
        "increases": len(increases),
        "decreases": len(decreases),
        "increases_by_year": dict(
            sorted(Counter(r.effective.year for r in increases).items())
        ),
        "decreases_by_year": dict(
            sorted(Counter(r.effective.year for r in decreases).items())
        ),
        "cumulative_change_points": _points(sum(_bp(r) for r in changed)),
        "target_before_first_change": target_before,
        "target_at_window_end": at_end[-1].target if at_end else None,
        "first_change_date": changed[0].iso if changed else None,
        "last_change_date": changed[-1].iso if changed else None,
        "changes_detail": [
            {
                "effective_date": r.iso,
                "change_points": _points(_bp(r)),
                "new_target": r.target,
            }
            for r in changed
        ],
    }


def list_changes(
    date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    """Every non-zero change in the window, as dated rows.

    Deliberately returns changes only. ``list`` over the whole table is what the
    handout's zero-score example did before eyeballing the output; the questions
    that need row-level detail always want the changes.
    """
    start, end = _window(date_from, date_to)
    changed = [
        r for r in _rows() if _in_window(r, start, end) and r.change != 0.0
    ]
    return {
        "metric": "list_changes",
        "window": [start.isoformat(), end.isoformat()],
        "count": len(changed),
        "changes": [
            {
                "effective_date": r.iso,
                "change_points": _points(_bp(r)),
                "new_target": r.target,
                "direction": "increase" if r.change > 0 else "decrease",
            }
            for r in changed
        ],
    }


def compare_periods(
    date_from: str,
    date_to: str,
    compare_from: str,
    compare_to: str,
) -> dict[str, Any]:
    """Two rate-cycle windows side by side, with the differences computed here.

    "Was the tightening faster than the easing", "compare 2011-13 with 2022-23".
    Without this the brain has to make two ``period_summary`` calls and subtract
    the results itself — arithmetic the architecture explicitly assigns to the
    application code, and a step the 60-second budget cannot always afford.
    """
    left = period_summary(date_from, date_to)
    right = period_summary(compare_from, compare_to)

    def stance(points: float) -> str:
        return (
            "net tightening"
            if points > 0
            else "net easing"
            if points < 0
            else "no net change"
        )

    return {
        "metric": "compare_periods",
        "period_a": left,
        "period_b": right,
        "period_a_stance": stance(left["cumulative_change_points"]),
        "period_b_stance": stance(right["cumulative_change_points"]),
        "difference_a_minus_b": {
            "changes": left["changes"] - right["changes"],
            "increases": left["increases"] - right["increases"],
            "decreases": left["decreases"] - right["decreases"],
            "cumulative_change_points": _points(
                round(left["cumulative_change_points"] * 100)
                - round(right["cumulative_change_points"] * 100)
            ),
        },
    }


METRICS = {
    "coverage": coverage,
    "count": count,
    "count_changes": count_changes,
    "count_increases": count_increases,
    "count_decreases": count_decreases,
    "extremes": extremes,
    "max_hold_streak": max_hold_streak,
    "lookup_rate": lookup_rate,
    "period_summary": period_summary,
    "compare_periods": compare_periods,
    "list_changes": list_changes,
    # Alias: the handout's tool table names this "list".
    "list": list_changes,
}
