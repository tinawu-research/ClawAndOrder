"""Deterministic numeric and date checking, parsed from each case's tolerance note.

This is the second of the three signals in the scorer. The LLM judge is the
headline metric because the organizers' grader is an LLM, but an LLM cannot
reliably decide whether 22.19 is within +/-0.02 of 22.17. This module can, and
the disagreements between the two are the most informative rows in the report.

Tolerance classes are parsed from the ``grading.tolerance_note`` text rather
than hard-coded per question, because hidden questions carry the same schema
with the same wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

EXACT = 0.0
PCT_LIKE = 0.02
CORRELATION = 0.001
CLOSE = 0.0001
VOLUME = 1.0

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# "5 Jun 2019", "20 March 2020"
_RE_DMY = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"
)
# "Jun 5, 2019", "June 2019"
_RE_MDY = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b"
)
_RE_MY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b")
# "2019-06-05", "2019-06"
_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")

# A number, optionally signed, optionally comma-grouped, optionally decimal.
_RE_NUMBER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")

_HEDGE_WORDS = (
    "approximately", "roughly", "about", "around", "circa", "approx",
    "nearly", "an estimated", "in the region of", "or so", "~",
)
_RE_HEDGE = re.compile("|".join(re.escape(w) for w in _HEDGE_WORDS), re.I)


@dataclass(frozen=True)
class ToleranceProfile:
    """Which tolerance classes this case enables."""

    pct_like: float = EXACT
    correlation: float = EXACT
    close: float = EXACT
    volume: float = EXACT

    @property
    def any_relaxed(self) -> bool:
        return any((self.pct_like, self.correlation, self.close, self.volume))


def parse_tolerance_note(note: str) -> ToleranceProfile:
    """Read the declared numeric tolerances out of a case's tolerance note."""
    if not note:
        return ToleranceProfile()
    text = note.lower()
    return ToleranceProfile(
        pct_like=PCT_LIKE if re.search(r"\+/-\s*0\.02", text) else EXACT,
        correlation=CORRELATION if re.search(r"correlations?\s+allow\s*\+/-\s*0\.001", text) else EXACT,
        close=CLOSE if re.search(r"closes?\s+allow\s*\+/-\s*0\.0001", text) else EXACT,
        volume=VOLUME if re.search(r"volume\s+allows?\s*\+/-\s*1\s+share", text) else EXACT,
    )


def _month(token: str) -> int | None:
    return _MONTHS.get(token[:3].lower())


def extract_dates(text: str) -> set[str]:
    """Return every date in ``text`` normalised to ISO.

    Month-precision dates normalise to ``YYYY-MM`` so that "Jan 2024" and
    "2024-01" compare equal, which the scoring handout requires.
    """
    found: set[str] = set()

    for day, mon, year in _RE_DMY.findall(text):
        m = _month(mon)
        if m:
            found.add(f"{int(year):04d}-{m:02d}-{int(day):02d}")

    for mon, day, year in _RE_MDY.findall(text):
        m = _month(mon)
        if m:
            found.add(f"{int(year):04d}-{m:02d}-{int(day):02d}")

    for year, mon, day in _RE_ISO.findall(text):
        if day:
            found.add(f"{year}-{mon}-{day}")
        else:
            found.add(f"{year}-{mon}")

    # Month-year only, when not already captured as a full date.
    for mon, year in _RE_MY.findall(text):
        m = _month(mon)
        if m and not any(d.startswith(f"{year}-{m:02d}-") for d in found):
            found.add(f"{int(year):04d}-{m:02d}")

    return found


def _strip_dates(text: str) -> str:
    """Blank out date spans so their digits are not read as bare numbers."""
    out = text
    for pattern in (_RE_DMY, _RE_MDY, _RE_ISO, _RE_MY):
        out = pattern.sub(" ", out)
    return out


def classify(raw: str, context: str, profile: ToleranceProfile) -> float:
    """Return the absolute tolerance for one numeric literal.

    ``context`` is the surrounding text, used to decide which declared class a
    number belongs to. Counts, years and rankings always stay exact.
    """
    lowered = context.lower()

    if "correlation" in lowered:
        return profile.correlation
    if re.search(r"\bclos(e|ing)\b", lowered):
        return profile.close
    if re.search(r"\b(share|volume)s?\b", lowered) and "percentage" not in lowered:
        return profile.volume

    is_percent = "%" in context or "per cent" in lowered
    if not is_percent:
        return EXACT

    # The note declares RBA values exact but returns/drawdowns/volatility
    # relaxed, and both are percentages. Expected facts rarely name the
    # quantity ("BHP.AX was best at +22.17%"), so key off the RBA vocabulary
    # instead: anything rate-flavoured or expressed in percentage points is an
    # RBA value and stays exact; every other percentage is a calculated one.
    if "percentage point" in lowered or re.search(r"\b(cash[- ]rate|target|rate)\b", lowered):
        return EXACT
    return profile.pct_like


def _numbers_with_context(text: str, window: int = 45) -> list[tuple[float, str, str]]:
    """Extract ``(value, raw, surrounding_context)`` for each numeric literal."""
    stripped = _strip_dates(text)
    out: list[tuple[float, str, str]] = []
    for match in _RE_NUMBER.finditer(stripped):
        raw = match.group(0)
        try:
            value = float(raw.replace(",", "").lstrip("+"))
        except ValueError:
            continue
        start = max(0, match.start() - window)
        end = min(len(stripped), match.end() + window)
        out.append((value, raw, stripped[start:end]))
    return out


def numeric_check(expected_fact: str, answer: str, tolerance_note: str = "") -> str:
    """Return ``PASS``, ``FAIL`` or ``N/A`` for the values in one expected fact.

    ``N/A`` means the fact carries no dates or numbers, so this signal has
    nothing to say and the LLM verdict stands alone.
    """
    profile = parse_tolerance_note(tolerance_note)

    expected_dates = extract_dates(expected_fact)
    answer_dates = extract_dates(answer)
    expected_numbers = _numbers_with_context(expected_fact)

    if not expected_dates and not expected_numbers:
        return "N/A"

    for wanted in expected_dates:
        if wanted in answer_dates:
            continue
        # A month-precision expectation is satisfied by any day in that month.
        if len(wanted) == 7 and any(d.startswith(wanted + "-") for d in answer_dates):
            continue
        return "FAIL"

    answer_numbers = [value for value, _, _ in _numbers_with_context(answer)]
    for value, raw, context in expected_numbers:
        tolerance = classify(raw, context, profile)
        if not any(abs(value - candidate) <= tolerance for candidate in answer_numbers):
            return "FAIL"

    return "PASS"


def hedge_flag(answer: str, near_digits: int = 40) -> bool:
    """True when a hedging word sits within ``near_digits`` chars of a value.

    The handout rejects "approximately 41" while leaving ordinary prose alone,
    so proximity to a number is what matters, not the word in isolation.
    """
    for match in _RE_HEDGE.finditer(answer):
        window = answer[match.end(): match.end() + near_digits]
        if re.search(r"\d", window):
            return True
    return False


def iso_to_reference(value: str | date) -> str:
    """Render a date the way the reference answers do: ``5 Jun 2019``."""
    if isinstance(value, str):
        parts = value.split("-")
        value = date(int(parts[0]), int(parts[1]), int(parts[2]))
    month = [k for k, v in _MONTHS.items() if v == value.month][0]
    return f"{value.day} {month.capitalize()} {value.year}"
