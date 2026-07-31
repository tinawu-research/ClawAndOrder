"""Date parsing for tool arguments, shared by the three dataset modules.

The brain passes dates through in whatever form the question used, so tool
arguments are parsed liberally: ``2019``, ``2019-06``, ``Jun 2019``,
``2019-06-05``, ``5 Jun 2019``, ``05/06/2019`` and ``20190605`` all resolve.
Tool *output* is always ISO ``YYYY-MM-DD``, because date precision is graded.

The part that matters for scoring is **bound-aware widening**. A partial date
resolves to the period it names, in the direction it is used: as a lower bound
``"2019"`` is 2019-01-01, as an upper bound it is 2019-12-31. That makes
"across 2011-2013" or "during 2019" a single call with the arguments the
question already supplied. ``datastore.parse_flexible_date`` handles only fully
specified days, so before this module ``date_from="2011"`` was a rejected call
and a wasted step.

``strptime`` with ``%b``/``%B`` is deliberately avoided for the month names:
those are locale-dependent, and the evaluation cluster's locale is not ours.
"""

from __future__ import annotations

import calendar
import re
from datetime import date

from datastore import parse_flexible_date

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_DAY = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_YEAR = re.compile(r"^(\d{4})$")
_TEXT_DAY = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})$")
_MONTH_TEXT_DAY = re.compile(r"^([A-Za-z]{3,9})\.?,?\s+(\d{1,2}),?\s+(\d{4})$")
_TEXT_MONTH = re.compile(r"^([A-Za-z]{3,9})\.?,?\s+(\d{4})$")


def iso(day: date) -> str:
    """Render a date the way every tool must report it."""
    return day.isoformat()


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def parse_bound(raw: str | date, *, upper: bool) -> date:
    """Parse a model-supplied date, widening partials to the period.

    Args:
        raw: The date text, or an already-parsed ``date``.
        upper: True when this is the inclusive end of a range (or an "as of"
            date), so a partial resolves to the last day of the period rather
            than the first.

    Raises:
        ValueError: If the text is empty or in no recognised form.
    """
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    if not text:
        raise ValueError("date is empty")

    if match := _ISO_DAY.match(text):
        year, month, day = (int(g) for g in match.groups())
        return date(year, month, day)

    if match := _ISO_MONTH.match(text):
        year, month = (int(g) for g in match.groups())
        return month_end(year, month) if upper else date(year, month, 1)

    if match := _YEAR.match(text):
        year = int(match.group(1))
        return date(year, 12, 31) if upper else date(year, 1, 1)

    if match := _TEXT_DAY.match(text):
        day, month_name, year = match.groups()
        if (key := month_name[:3].lower()) in MONTHS:
            return date(int(year), MONTHS[key], int(day))

    if match := _MONTH_TEXT_DAY.match(text):
        month_name, day, year = match.groups()
        if (key := month_name[:3].lower()) in MONTHS:
            return date(int(year), MONTHS[key], int(day))

    if match := _TEXT_MONTH.match(text):
        month_name, year = match.groups()
        if (key := month_name[:3].lower()) in MONTHS:
            month, y = MONTHS[key], int(year)
            return month_end(y, month) if upper else date(y, month, 1)

    # Compact and slash-separated day forms (20190605, 05/06/2019) are
    # unambiguous once we know it is a full day, so hand them to the datastore
    # parser rather than duplicating its format table here.
    try:
        return parse_flexible_date(text)
    except ValueError:
        pass

    raise ValueError(
        f"could not read {raw!r} as a date. Use YYYY-MM-DD, YYYY-MM, YYYY, "
        "or '5 Jun 2019'."
    )


def parse_from(raw: str | date) -> date:
    """Parse an inclusive window start (partials widen to the period start)."""
    return parse_bound(raw, upper=False)


def parse_to(raw: str | date) -> date:
    """Parse an inclusive window end (partials widen to the period end)."""
    return parse_bound(raw, upper=True)
