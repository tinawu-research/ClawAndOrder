"""Deterministic metrics over the 18 ASX price series.

Conventions fixed by the reference answers:

* Returns are **close-to-close, first-to-last** over the requested span.
* A "basket" return is the *arithmetic mean of constituent returns*, not the
  return of a value-weighted index.
* "non-Tabcorp" means the 17 tickers left after excluding ``TAH.AX``. Several
  questions are unanswerable without applying that exclusion — the highest
  average-volume ticker flips from ``TAH.AX`` to ``AMP.AX`` because of it.
* Drawdown is running-peak on close, evaluated on every row.
"""

from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Iterable, Sequence

import config
from datastore import STORE, AsxSeries, normalise_ticker
from tools.dateparse import parse_from, parse_to
from tools.rba import MetricError

#: ``tickers="all"`` means the full 18, Tabcorp included.
_ALL_ALIASES = {"all", "*", "everything", "all tickers", "every ticker"}
#: ``tickers="basket"`` means the 17 the graded questions keep asking for.
_BASKET_ALIASES = {
    "basket",
    "non-tabcorp",
    "non tabcorp",
    "non-tabcorp basket",
    "nontabcorp",
    "ex-tabcorp",
    "excluding tabcorp",
}


def _series() -> dict[str, AsxSeries]:
    if not STORE.asx:
        raise MetricError("ASX dataset is not loaded")
    return STORE.asx


@lru_cache(maxsize=1)
def _company_aliases() -> dict[str, str]:
    """Lowercased company name -> canonical ticker, read from the filenames.

    The company name appears only in the filename (``Qantas-ASX-2015-2021``),
    never in the rows, so without this a question that says "Qantas" or
    "Transurban" cannot be routed to a ticker at all — ``normalise_ticker``
    would turn it into ``QANTAS.AX`` and the call would fail. Built once by
    reading the first line of each price file.
    """
    aliases: dict[str, str] = {}
    directory = config.ASX_DIR
    try:
        files = sorted(directory.glob("*.jsonl"))
    except OSError:
        files = []
    for file in files:
        stem = file.name.split("-")[0].strip().lower()
        if not stem:
            continue
        try:
            with file.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    ticker = str(json.loads(line)["ticker"]).strip().upper()
                    break
                else:
                    continue
        except (OSError, ValueError, KeyError):
            continue
        if ticker in _series():
            aliases.setdefault(stem, ticker)
    return aliases


def _canonical(name: str) -> str | None:
    """Resolve one ticker, bare symbol or company name, or None if unknown."""
    universe = _series()
    text = str(name).strip()
    if not text:
        return None
    direct = normalise_ticker(text)
    if direct in universe:
        return direct
    return _company_aliases().get(text.lower())


def _resolve(
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> list[str]:
    """Normalise the requested ticker universe.

    Accepts a list or a comma-separated string, because the brain emits both,
    and resolves company names as well as symbols. ``"all"`` and ``"basket"``
    are accepted as whole-universe shorthands.

    The default remains **all 18 tickers**. Defaulting to the non-Tabcorp
    basket would silently drop a constituent from any question that did not ask
    for the exclusion, and a silently wrong basket is indistinguishable from a
    right one in the answer text. Exclusion stays explicit.
    """

    def as_list(value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [
                p for p in (s.strip() for s in value.replace(";", ",").split(",")) if p
            ]
        return [str(v) for v in value]

    universe = _series()
    tabcorp = normalise_ticker(config.TABCORP_TICKER)
    requested = as_list(tickers)

    # Whole-universe shorthands, only meaningful as the sole argument.
    if len(requested) == 1:
        key = requested[0].strip().lower()
        if key in _ALL_ALIASES:
            requested = []
        elif key in _BASKET_ALIASES:
            requested = [t for t in sorted(universe) if t != tabcorp]

    wanted: list[str] = []
    unknown: list[str] = []
    for name in requested:
        canonical = _canonical(name)
        if canonical is None:
            unknown.append(str(name).strip())
        elif canonical not in wanted:
            wanted.append(canonical)
    if unknown:
        raise MetricError(
            f"unknown ticker(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(universe))}"
        )

    dropped = set()
    for name in as_list(exclude_tickers):
        canonical = _canonical(name)
        # An unknown exclusion is not fatal — it excludes nothing — but
        # normalising it keeps "Tabcorp" and "TAH" both working.
        dropped.add(canonical or normalise_ticker(name))

    selected = wanted or sorted(universe)
    result = [t for t in selected if t not in dropped]
    if not result:
        raise MetricError("ticker selection is empty after exclusions")
    return result


@lru_cache(maxsize=1)
def _sessions() -> tuple[date, ...]:
    """Every trading day in the dataset, ascending.

    Needed to walk N sessions forward from an event: the market is closed on
    weekends *and* public holidays, so "five trading days later" cannot be
    derived by adding days to a calendar date.
    """
    return tuple(sorted({b.day for s in _series().values() for b in s.bars}))


def _window_bounds(
    date_from: str | None, date_to: str | None
) -> tuple[date, date]:
    """Resolve an optional date window, defaulting to the full sample."""
    sessions = _sessions()
    try:
        start = parse_from(date_from) if date_from else sessions[0]
        end = parse_to(date_to) if date_to else sessions[-1]
    except ValueError as exc:
        raise MetricError(str(exc)) from exc
    if start > end:
        raise MetricError(f"date_from {start} is after date_to {end}")
    return start, end


def _pct(new: float, old: float) -> float:
    if old == 0:
        raise MetricError("cannot compute a return from a zero base price")
    return (new / old - 1.0) * 100.0


def _first_last_in_year(series: AsxSeries, year: int) -> tuple[Any, Any]:
    bars = [b for b in series.bars if b.day.year == year]
    if len(bars) < 2:
        raise MetricError(f"{series.ticker} has too few {year} bars")
    return bars[0], bars[-1]


def coverage() -> dict[str, Any]:
    """Dimensions and common date range of the price dataset."""
    series = _series()
    per_ticker = {t: len(s.bars) for t, s in series.items()}
    starts = [s.bars[0].day for s in series.values()]
    ends = [s.bars[-1].day for s in series.values()]
    row_counts = sorted(set(per_ticker.values()))
    return {
        "metric": "coverage",
        "tickers": len(series),
        "ticker_list": sorted(series),
        "rows_per_ticker": row_counts[0] if len(row_counts) == 1 else row_counts,
        "rows_identical_across_tickers": len(row_counts) == 1,
        # "common" range = the span every ticker covers.
        "common_start": max(starts).isoformat(),
        "common_end": min(ends).isoformat(),
        "earliest": min(starts).isoformat(),
        "latest": max(ends).isoformat(),
    }


def annual_return(
    year: int,
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """First-to-last close return within a calendar year, per ticker."""
    year = int(year)
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    rows = []
    for ticker in selected:
        first, last = _first_last_in_year(series[ticker], year)
        rows.append(
            {
                "ticker": ticker,
                "return_pct": round(_pct(last.close, first.close), 4),
                "first_date": first.day.isoformat(),
                "last_date": last.day.isoformat(),
            }
        )
    returns = [r["return_pct"] for r in rows]
    return {
        "metric": "annual_return",
        "year": year,
        "tickers_used": selected,
        "results": rows,
        "basket_average_return_pct": round(statistics.fmean(returns), 4),
    }


def rank_annual_returns(
    year: int,
    exclude_tickers: Sequence[str] | str | None = None,
    tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Annual returns ranked best to worst, with best/worst called out."""
    result = annual_return(year, tickers, exclude_tickers)
    ranked = sorted(result["results"], key=lambda r: r["return_pct"], reverse=True)
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return {
        "metric": "rank_annual_returns",
        "year": result["year"],
        "tickers_used": result["tickers_used"],
        "ranked": ranked,
        "best": ranked[0],
        "worst": ranked[-1],
        "basket_average_return_pct": result["basket_average_return_pct"],
    }


def full_sample_return(
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """First-to-last close return across each ticker's whole history."""
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    rows = []
    for ticker in selected:
        bars = series[ticker].bars
        rows.append(
            {
                "ticker": ticker,
                "return_pct": round(_pct(bars[-1].close, bars[0].close), 4),
                "first_date": bars[0].day.isoformat(),
                "last_date": bars[-1].day.isoformat(),
            }
        )
    ranked = sorted(rows, key=lambda r: r["return_pct"], reverse=True)
    return {
        "metric": "full_sample_return",
        "tickers_used": selected,
        "ranked": ranked,
        "best": ranked[0],
        "worst": ranked[-1],
        "basket_average_return_pct": round(
            statistics.fmean(r["return_pct"] for r in rows), 4
        ),
    }


def window_return(
    date_from: str,
    date_to: str,
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Close-to-close return between two explicit dates, per ticker + basket.

    Used by every "return in the N days after an RBA decision" question. The
    reference derivations use *exact-date* closes, so a date that was not a
    trading day is reported rather than silently shifted; ``used_date`` records
    the fallback actually applied when the requested date was a market holiday.
    """
    try:
        start = parse_from(date_from)
        end = parse_to(date_to)
    except ValueError as exc:
        raise MetricError(str(exc)) from exc
    if start >= end:
        raise MetricError(f"date_from {start} must precede date_to {end}")
    selected = _resolve(tickers, exclude_tickers)
    series = _series()

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for ticker in selected:
        s = series[ticker]
        exact_start, exact_end = s.close_on(start), s.close_on(end)
        used_start, used_end = start, end
        if exact_start is None:
            fallback = s.close_on_or_before(start)
            if fallback is None:
                raise MetricError(f"{ticker} has no data on or before {start}")
            used_start, exact_start = fallback
            notes.append(f"{ticker}: {start} not a trading day, used {used_start}")
        if exact_end is None:
            fallback = s.close_on_or_before(end)
            if fallback is None:
                raise MetricError(f"{ticker} has no data on or before {end}")
            used_end, exact_end = fallback
            notes.append(f"{ticker}: {end} not a trading day, used {used_end}")
        rows.append(
            {
                "ticker": ticker,
                "return_pct": round(_pct(exact_end, exact_start), 4),
                "start_close": round(exact_start, 6),
                "end_close": round(exact_end, 6),
                "used_start_date": used_start.isoformat(),
                "used_end_date": used_end.isoformat(),
            }
        )
    basket = round(statistics.fmean(r["return_pct"] for r in rows), 4)
    out: dict[str, Any] = {
        "metric": "window_return",
        "window": [start.isoformat(), end.isoformat()],
        "tickers_used": selected,
        "constituents": len(rows),
        "results": rows,
        "basket_average_return_pct": basket,
        "direction": "up" if basket > 0 else "down" if basket < 0 else "flat",
    }
    if notes:
        out["trading_day_adjustments"] = notes
    return out


def avg_volume(
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Mean daily volume per ticker, ranked descending.

    Full sample unless a window is given — "the most heavily traded stock in
    2020" needs the window, and computing it over the whole history instead is
    a wrong answer that looks right.
    """
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    windowed = bool(date_from or date_to)
    start, end = _window_bounds(date_from, date_to)

    rows = []
    for t in selected:
        bars = series[t].slice_between(start, end) if windowed else series[t].bars
        if not bars:
            continue
        rows.append(
            {
                "ticker": t,
                "avg_daily_volume": round(
                    statistics.fmean(b.volume for b in bars), 2
                ),
                "total_volume": round(sum(b.volume for b in bars), 2),
                "trading_days": len(bars),
            }
        )
    if not rows:
        raise MetricError(
            f"no ASX bars between {start.isoformat()} and {end.isoformat()}"
        )
    ranked = sorted(rows, key=lambda r: r["avg_daily_volume"], reverse=True)
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return {
        "metric": "avg_volume",
        "window": [start.isoformat(), end.isoformat()] if windowed else None,
        "tickers_used": selected,
        "ranked": ranked,
        "highest": ranked[0],
        "lowest": ranked[-1],
    }


#: The handout's metric table calls this ``rank_avg_volume``.
rank_avg_volume = avg_volume


def max_drawdown(
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
    top_n: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Worst running-peak-to-trough decline per ticker, with endpoint dates.

    The peak reported is the running maximum that produced the worst trough, so
    the pair of dates is the actual peak/trough pair, not the global maximum
    paired with the global minimum. Restricted to a date window when one is
    given, which is what "the worst drawdown during the 2020 crash" means.
    """
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    windowed = bool(date_from or date_to)
    start, end = _window_bounds(date_from, date_to)
    rows = []
    for ticker in selected:
        bars = (
            series[ticker].slice_between(start, end)
            if windowed
            else series[ticker].bars
        )
        if len(bars) < 2:
            continue
        peak_close = bars[0].close
        peak_day = bars[0].day
        worst = 0.0
        worst_peak_day: date = bars[0].day
        worst_trough_day: date = bars[0].day
        worst_peak_close = peak_close
        worst_trough_close = bars[0].close
        for bar in bars:
            if bar.close > peak_close:
                peak_close = bar.close
                peak_day = bar.day
            drop = (bar.close / peak_close - 1.0) * 100.0
            if drop < worst:
                worst = drop
                worst_peak_day, worst_trough_day = peak_day, bar.day
                worst_peak_close, worst_trough_close = peak_close, bar.close
        rows.append(
            {
                "ticker": ticker,
                "max_drawdown_pct": round(worst, 4),
                "peak_date": worst_peak_day.isoformat(),
                "trough_date": worst_trough_day.isoformat(),
                "peak_close": round(worst_peak_close, 6),
                "trough_close": round(worst_trough_close, 6),
            }
        )
    if not rows:
        raise MetricError(
            f"no ASX bars between {start.isoformat()} and {end.isoformat()}"
        )
    ranked = sorted(rows, key=lambda r: r["max_drawdown_pct"])
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    if top_n:
        ranked = ranked[: int(top_n)]
    return {
        "metric": "max_drawdown",
        "window": [start.isoformat(), end.isoformat()] if windowed else None,
        "tickers_used": selected,
        "ranked_worst_first": ranked,
        "worst": ranked[0],
    }


def volatility(
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
    year: int | None = None,
    annualised: bool = True,
) -> dict[str, Any]:
    """Standard deviation of daily close-to-close returns, in percent.

    Annualised with sqrt(252) by default; ``annualised=false`` gives the raw
    daily figure.
    """
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    rows = []
    for ticker in selected:
        bars = series[ticker].bars
        if year:
            bars = tuple(b for b in bars if b.day.year == int(year))
        if len(bars) < 3:
            raise MetricError(f"{ticker} has too few bars for volatility")
        daily = [
            (b.close / a.close - 1.0) * 100.0 for a, b in zip(bars, bars[1:])
        ]
        sd = statistics.stdev(daily)
        rows.append(
            {
                "ticker": ticker,
                "daily_volatility_pct": round(sd, 4),
                "annualised_volatility_pct": round(sd * (252**0.5), 4),
                "observations": len(daily),
            }
        )
    key = "annualised_volatility_pct" if annualised else "daily_volatility_pct"
    ranked = sorted(rows, key=lambda r: r[key], reverse=True)
    return {
        "metric": "volatility",
        "year": int(year) if year else None,
        "annualised": bool(annualised),
        "tickers_used": selected,
        "ranked": ranked,
        "highest": ranked[0],
        "lowest": ranked[-1],
    }


def correlation(
    tickers: Sequence[str] | str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Pearson correlation of daily returns between exactly two tickers."""
    selected = _resolve(tickers)
    if len(selected) != 2:
        raise MetricError(
            f"correlation needs exactly 2 tickers, got {len(selected)}"
        )
    series = _series()
    try:
        start = parse_from(date_from) if date_from else None
        end = parse_to(date_to) if date_to else None
    except ValueError as exc:
        raise MetricError(str(exc)) from exc

    def daily(ticker: str) -> dict[date, float]:
        bars = series[ticker].bars
        if start:
            bars = tuple(b for b in bars if b.day >= start)
        if end:
            bars = tuple(b for b in bars if b.day <= end)
        return {
            b.day: (b.close / a.close - 1.0)
            for a, b in zip(bars, bars[1:])
        }

    left, right = daily(selected[0]), daily(selected[1])
    shared = sorted(set(left) & set(right))
    if len(shared) < 3:
        raise MetricError("too few overlapping trading days for correlation")
    coeff = statistics.correlation(
        [left[d] for d in shared], [right[d] for d in shared]
    )
    return {
        "metric": "correlation",
        "tickers": selected,
        "correlation": round(coeff, 6),
        "overlapping_days": len(shared),
        "window": [shared[0].isoformat(), shared[-1].isoformat()],
    }


def close_on(ticker: str, day: str) -> dict[str, Any]:
    """Closing price for one ticker on one date (with on-or-before fallback)."""
    selected = _resolve([ticker])[0]
    series = _series()[selected]
    try:
        target = parse_to(day)
    except ValueError as exc:
        raise MetricError(str(exc)) from exc
    exact = series.close_on(target)
    if exact is not None:
        return {
            "metric": "close_on",
            "ticker": selected,
            "date": target.isoformat(),
            "close": round(exact, 6),
            "exact_trading_day": True,
        }
    fallback = series.close_on_or_before(target)
    if fallback is None:
        raise MetricError(f"{selected} has no data on or before {target}")
    used, value = fallback
    return {
        "metric": "close_on",
        "ticker": selected,
        "date": target.isoformat(),
        "close": round(value, 6),
        "exact_trading_day": False,
        "used_date": used.isoformat(),
        "note": f"{target} was not a trading day; used {used}",
    }


def event_window(
    date_from: str,
    sessions: int | None = None,
    calendar_days: int | None = None,
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Return over a window that starts at an event and runs forward.

    The "how did prices react in the N days after the RBA decision / the
    article" family. Give exactly one of:

    * ``sessions=5`` — five *trading* days after the event date. The brain
      cannot compute this itself: the market closes for public holidays as well
      as weekends, so adding 5 to a date lands on the wrong bar.
    * ``calendar_days=7`` — the same date one week later.

    The end date is resolved here and reported, so the synthesis model can state
    the window it actually measured.
    """
    if (sessions is None) == (calendar_days is None):
        raise MetricError(
            "give exactly one of sessions or calendar_days (sessions=5 for five "
            "trading days after the event, calendar_days=7 for one week)"
        )
    try:
        start = parse_from(date_from)
    except ValueError as exc:
        raise MetricError(str(exc)) from exc

    all_sessions = _sessions()
    if start > all_sessions[-1] or start < all_sessions[0]:
        raise MetricError(
            f"event date {start.isoformat()} is outside the price data, which "
            f"covers {all_sessions[0].isoformat()} to "
            f"{all_sessions[-1].isoformat()}"
        )

    if sessions is not None:
        count = int(sessions)
        if count < 1:
            raise MetricError("sessions must be 1 or more")
        ahead = [d for d in all_sessions if d > start]
        if len(ahead) < count:
            raise MetricError(
                f"only {len(ahead)} trading session(s) exist after "
                f"{start.isoformat()}; the data ends "
                f"{all_sessions[-1].isoformat()}"
            )
        end = ahead[count - 1]
        basis = f"{count} trading session(s) after the event"
    else:
        days = int(calendar_days)
        if days < 1:
            raise MetricError("calendar_days must be 1 or more")
        end = start + timedelta(days=days)
        if end > all_sessions[-1]:
            raise MetricError(
                f"window end {end.isoformat()} is beyond the price data, which "
                f"ends {all_sessions[-1].isoformat()}"
            )
        basis = f"{days} calendar day(s) after the event"

    result = window_return(
        start.isoformat(), end.isoformat(), tickers, exclude_tickers
    )
    result["metric"] = "event_window"
    result["event_date"] = start.isoformat()
    result["window_end"] = end.isoformat()
    result["window_basis"] = basis
    result["sessions_in_window"] = sum(1 for d in all_sessions if start < d <= end)
    return result


def price_on_date(
    day: str,
    tickers: Sequence[str] | str | None = None,
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Full OHLCV bars for one date, for one or more tickers.

    ``close_on`` answers "what did it close at"; this answers the questions that
    also want the day's range or the volume traded on it.
    """
    selected = _resolve(tickers, exclude_tickers)
    series = _series()
    try:
        target = parse_to(day)
    except ValueError as exc:
        raise MetricError(str(exc)) from exc

    rows = []
    adjustments = []
    for ticker in selected:
        bars = series[ticker].slice_between(target, target)
        used = target
        if not bars:
            fallback = series[ticker].close_on_or_before(target)
            if fallback is None:
                continue
            used = fallback[0]
            bars = series[ticker].slice_between(used, used)
            adjustments.append(
                f"{ticker}: {target.isoformat()} not a trading day, used "
                f"{used.isoformat()}"
            )
        bar = bars[0]
        rows.append(
            {
                "ticker": ticker,
                "date": used.isoformat(),
                "open": round(bar.open, 6),
                "high": round(bar.high, 6),
                "low": round(bar.low, 6),
                "close": round(bar.close, 6),
                "volume": bar.volume,
            }
        )
    if not rows:
        raise MetricError(
            f"no ASX bars on or before {target.isoformat()} for the selection"
        )
    out: dict[str, Any] = {
        "metric": "price_on_date",
        "requested_date": target.isoformat(),
        "tickers_used": selected,
        "bars": rows,
    }
    if adjustments:
        out["trading_day_adjustments"] = adjustments
    return out


def basket_tickers(
    exclude_tickers: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """The ticker universe, with the standard non-Tabcorp basket spelled out."""
    universe = sorted(_series())
    non_tabcorp = [
        t for t in universe if t != normalise_ticker(config.TABCORP_TICKER)
    ]
    return {
        "metric": "basket_tickers",
        "all_tickers": universe,
        "count": len(universe),
        "non_tabcorp_basket": non_tabcorp,
        "non_tabcorp_count": len(non_tabcorp),
        "excluded": [normalise_ticker(config.TABCORP_TICKER)],
    }


METRICS = {
    "coverage": coverage,
    "dimensions": coverage,
    "annual_return": annual_return,
    "rank_annual_returns": rank_annual_returns,
    "full_sample_return": full_sample_return,
    "window_return": window_return,
    "event_window": event_window,
    "price_on_date": price_on_date,
    "avg_volume": avg_volume,
    "rank_avg_volume": rank_avg_volume,
    "max_drawdown": max_drawdown,
    "volatility": volatility,
    "correlation": correlation,
    "close_on": close_on,
    "basket_tickers": basket_tickers,
}
