"""Formatting rules for gold answers, matched to the reference answers.

The grading judge accepts equivalent formats, but the reference answers are the
safest target because they are what the expected facts were written from. Every
convention here was read off ``public_questions.jsonl`` rather than invented:

* returns and drawdowns are always signed to two decimals -- ``+22.17%``
* RBA rates carry two decimals and no sign -- ``0.10%``, ``4.75%``
* cumulative change is in signed percentage points -- ``-2.25 percentage points``
* counts are comma-grouped -- ``1,774``
* dates read ``5 Jun 2019`` (12 of the 15 references use this form)
* rankings are one line of ``1) ... ; 2) ... ; 3) ...``

Two decimals on returns is safe because the declared tolerance is +/-0.02
percentage points, which is wider than the rounding error.
"""

from __future__ import annotations

from datetime import date, datetime

_MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def pct(value: float, *, signed: bool = True, dp: int = 2) -> str:
    """A calculated percentage: return, drawdown, volatility, share."""
    return f"{value:+.{dp}f}%" if signed else f"{value:.{dp}f}%"


def rate(value: float) -> str:
    """An RBA cash-rate target. Never signed."""
    return f"{value:.2f}%"


def points(value: float) -> str:
    """A cumulative rate change in percentage points. Always signed."""
    return f"{value:+.2f} percentage points"


def count(value: int | float) -> str:
    return f"{int(value):,}"


def volume(value: float) -> str:
    return f"{value:,.2f}"


def correlation(value: float) -> str:
    return f"{value:.3f}"


def close(value: float) -> str:
    return f"{value:.4f}"


def day(value: str | date | datetime) -> str:
    """Render a date as ``5 Jun 2019``."""
    if isinstance(value, str):
        value = datetime.strptime(value[:10], "%Y-%m-%d").date()
    if isinstance(value, datetime):
        value = value.date()
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def month(year: int, month_number: int) -> str:
    """Render a month as ``May 2020``."""
    return f"{_MONTHS[month_number - 1]} {year}"


def ticker(value: str) -> str:
    """Full ticker form, as the references use (``BHP.AX``, not ``BHP``)."""
    return value if value.endswith(".AX") else f"{value}.AX"


def direction_verb(value: float) -> str:
    """``rose`` / ``fell`` for a signed move."""
    return "rose" if value >= 0 else "fell"


def ranking(entries: list[str]) -> str:
    """One-line ranking, matching the MHQ055 reference exactly."""
    return "; ".join(f"{i}) {text}" for i, text in enumerate(entries, start=1))


def join_clauses(parts: list[str]) -> str:
    """Join independent answer components into flowing prose.

    Components are graded separately, so every one must survive intact; this
    only decides the connective tissue between them.
    """
    cleaned = []
    for part in parts:
        part = part.strip().rstrip(".")
        if not part:
            continue
        # Sentence-initial capital. Only touch a leading lowercase letter, so
        # tickers and already-capitalised openings are left alone.
        if part[0].islower():
            part = part[0].upper() + part[1:]
        cleaned.append(part)
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."


def oxford(items: list[str]) -> str:
    """``a, b and c``."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"
