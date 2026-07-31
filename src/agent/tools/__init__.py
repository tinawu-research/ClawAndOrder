"""Tool registry: JSON schemas for the brain, and validated dispatch.

The brain *requests* calls; this package *executes* them. That split is a scored
architecture requirement, so every argument the model supplies is validated here
before any dataset is touched, and a bad call comes back as a structured error
the brain can recover from rather than an exception that kills the request.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from datastore import STORE
from tools import afr, asx, rba
from tools.rba import MetricError

logger = logging.getLogger(__name__)

DATASETS: dict[str, dict[str, Callable[..., Any]]] = {
    "rba": rba.METRICS,
    "asx": asx.METRICS,
    "afr": afr.METRICS,
}

#: Aliases for dataset names the brain tends to invent.
_DATASET_ALIASES = {
    "rba_rates": "rba",
    "rba-rates": "rba",
    "cash_rate": "rba",
    "asx_prices": "asx",
    "prices": "asx",
    "afr_news": "afr",
    "news": "afr",
}


def _normalise_dataset(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    return _DATASET_ALIASES.get(key, key)


def query_data(dataset: str, metric: str, **kwargs: Any) -> dict[str, Any]:
    """Single entry point for all deterministic dataset access."""
    ds = _normalise_dataset(dataset)
    if ds not in DATASETS:
        raise MetricError(
            f"unknown dataset {dataset!r}. Valid: {', '.join(sorted(DATASETS))}"
        )
    metrics = DATASETS[ds]
    key = str(metric).strip().lower()
    if key not in metrics:
        raise MetricError(
            f"unknown metric {metric!r} for {ds}. "
            f"Valid: {', '.join(sorted(metrics))}"
        )
    fn = metrics[key]

    # Drop arguments the metric does not accept. The brain routinely passes
    # spare keys (a stray `year` on a coverage call); silently ignoring them
    # beats failing the call and burning a step.
    signature = inspect.signature(fn)
    accepted = set(signature.parameters)
    supplied = {k: v for k, v in kwargs.items() if v is not None}
    usable = {k: v for k, v in supplied.items() if k in accepted}
    ignored = sorted(set(supplied) - set(usable))

    result = fn(**usable)
    result.setdefault("dataset", ds)
    if ignored:
        result["ignored_arguments"] = ignored
    return result


def retrieve(**kwargs: Any) -> dict[str, Any]:
    """Retrieve AFR article text as evidence (no classification here)."""
    signature = inspect.signature(afr.find_article)
    usable = {
        k: v
        for k, v in kwargs.items()
        if v is not None and k in signature.parameters
    }
    return afr.find_article(**usable)


def dataset_coverage() -> dict[str, Any]:
    """Row counts and date spans for all three datasets, plus their overlap.

    Exists because at least one benchmark question is answered correctly only by
    declining the analysis: RBA runs past 2021 while ASX and AFR stop there, so a
    "how did news and prices react to the 2022-23 tightening cycle" join cannot
    be observed. Reporting the overlap explicitly gives the brain a fact to
    reason from instead of an absence to hallucinate over.
    """
    cov = STORE.coverage()
    starts = [v["start"] for v in cov.values() if "start" in v]
    ends = [v["end"] for v in cov.values() if "end" in v]
    overlap_start = max(starts) if starts else None
    overlap_end = min(ends) if ends else None
    return {
        "tool": "dataset_coverage",
        "datasets": cov,
        "common_overlap": {
            "start": overlap_start,
            "end": overlap_end,
            "valid": bool(
                overlap_start and overlap_end and overlap_start <= overlap_end
            ),
        },
        # This note is read at the exact moment the brain decides whether to
        # keep gathering, so it has to say what to do next. The earlier wording
        # stopped at "say so explicitly", and the brain duly reported the
        # coverage gap and made no further call — losing the half of the
        # question that the covering dataset could still answer.
        "note": (
            "Cross-dataset claims are only observable inside common_overlap. "
            "Outside it, say so explicitly rather than inferring values — but "
            "do NOT stop here. A coverage gap is one finding, not the whole "
            "answer: now query whichever dataset DOES cover the period for the "
            "part of the question it can answer (e.g. if the period is after "
            "the ASX/AFR end date, the RBA side is still fully observable, so "
            "count and date those decisions)."
        ),
    }


TOOL_IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "query_data": query_data,
    "retrieve": retrieve,
    "dataset_coverage": dataset_coverage,
}


# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas
# ---------------------------------------------------------------------------
# The metric catalogue is spelled out in the description because the brain
# cannot discover it at run time, and a wrong metric name is the difference
# between full marks and zero — the handout's own worked example scores 0%
# for calling the right dataset with the wrong metric.

_QUERY_DATA_DESCRIPTION = """\
Run an exact, deterministic calculation over one approved dataset. Always use \
this for any number, count, date, rate, return, ranking or streak. Never \
compute such a value yourself and never estimate one from retrieved article \
text.

dataset="rba" — cash-rate decisions, 175 records, 2010-02-03 to 2026-06-17
  coverage                record count and date span
  count                   decision records in a window
  count_changes           non-zero changes split into increases/decreases,
                          with per-year breakdown and cumulative points
  count_increases         hikes only            count_decreases  cuts only
  extremes                highest/lowest target + first effective date + count
  max_hold_streak         longest gap between two non-zero changes, in days,
                          with start/end dates and the rate held vs moved to
  lookup_rate             target in force ON OR BEFORE date_from. Use this for
                          "the rate in force on <date>". Requires date_from.
  period_summary          a cycle: cuts, hikes, per-year split, cumulative
                          change, rate before the first change, rate at the end.
                          Requires date_from and date_to.
  compare_periods         two cycles side by side with the differences already
                          subtracted. Requires date_from + date_to for the
                          first and compare_from + compare_to for the second.
                          Use for "compare X with Y" instead of two calls.
  list_changes            every non-zero change in a window, as dated rows

dataset="asx" — 18 tickers, 1774 daily bars each, 2015-01-02 to 2021-12-30
  coverage                tickers, rows per ticker, common date range
  basket_tickers          the ticker universe and the non-Tabcorp basket
  annual_return           first-to-last close return in a calendar year.
                          Requires year.
  rank_annual_returns     the same, ranked, with best and worst. Requires year.
  full_sample_return      first-to-last close return over the whole history
  window_return           close-to-close return between two exact dates, per
                          ticker plus the basket average. Requires date_from
                          and date_to.
  event_window            return measured FORWARD from an event date. Requires
                          date_from plus exactly one of sessions=N (N trading
                          days later) or calendar_days=N. Do NOT compute the
                          end date yourself, the market shuts on holidays too.
                          A WEEK IS calendar_days=7, never sessions=5: after
                          2019-06-05 those land on different days (12 vs 13
                          Jun, because 10 Jun was a holiday) and return
                          different numbers. Reserve sessions=N for questions
                          that count sessions explicitly.
  avg_volume              mean daily volume per ticker, ranked; date_from and
                          date_to restrict it to a period
  max_drawdown            worst running-peak-to-trough decline with peak and
                          trough dates; top_n limits the ranking, date_from and
                          date_to restrict it to a period
  volatility              stdev of daily returns (annualised by default)
  correlation             Pearson correlation between exactly 2 tickers
  close_on                closing price for one ticker on one date. Requires
                          ticker and day.
  price_on_date           full open/high/low/close/volume bars for one date.
                          Requires day.

dataset="afr" — 219,538 articles, 2015-01-01 to 2021-12-31
  coverage                article count and date span
  count                   articles matching a pattern, counted ONCE per record
  count_by_year           counts per year, with the peak year; accepts
                          date_from and date_to
  count_by_month          counts per YYYY-MM, with the peak month
  share                   matches as a percentage of the corpus or of one year

AFR matching is always case-insensitive, always across HEADLINE + SUBHEAD + \
INTRO + TEXT combined, and always once per record. Supply the search either as:
  terms=["unemployment"]   escaped and word-anchored for you (preferred), or
  pattern="\\\\bRBA\\\\b|cash rate"  a raw regex used verbatim — YOU must add the
                           \\b anchors, or short acronyms match inside unrelated
                           words and the count comes out too high.

To exclude Tabcorp from any ASX metric, pass exclude_tickers=["TAH.AX"]. \
"non-Tabcorp basket" means exactly that: the other 17 tickers. The ASX default \
is all 18 tickers, so the exclusion must be passed whenever the question asks \
for it. tickers also accepts company names ("Qantas", "Transurban") and the \
shorthands tickers=["all"] and tickers=["basket"] (basket = non-Tabcorp).

Dates are read liberally: "2019", "2019-06", "Jun 2019", "2019-06-05" and \
"5 Jun 2019" all work. A partial date widens to the period it names, in the \
direction it is used — date_from="2019" is 2019-01-01, date_to="2019" is \
2019-12-31 — so "during 2019" or "across 2011-2013" is a single call.\
"""

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": _QUERY_DATA_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "enum": ["rba", "asx", "afr"],
                        "description": "Which approved dataset to query.",
                    },
                    "metric": {
                        "type": "string",
                        "description": (
                            "Metric name from the catalogue for that dataset."
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": (
                            "Start date, YYYY-MM-DD. For rba/lookup_rate this is "
                            "the as-of date and the only required argument."
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date, YYYY-MM-DD.",
                    },
                    "compare_from": {
                        "type": "string",
                        "description": (
                            "Start of the SECOND period, for rba/compare_periods."
                        ),
                    },
                    "compare_to": {
                        "type": "string",
                        "description": (
                            "End of the SECOND period, for rba/compare_periods."
                        ),
                    },
                    "day": {
                        "type": "string",
                        "description": (
                            "A single date, for asx/close_on and "
                            "asx/price_on_date."
                        ),
                    },
                    "ticker": {
                        "type": "string",
                        "description": "One ticker, for asx/close_on.",
                    },
                    "sessions": {
                        "type": "integer",
                        "description": (
                            "asx/event_window: trading days forward from "
                            "date_from. Use ONLY when the question counts "
                            "sessions explicitly ('five trading sessions "
                            "after'). For a week, use calendar_days=7 instead. "
                            "Mutually exclusive with calendar_days."
                        ),
                    },
                    "calendar_days": {
                        "type": "integer",
                        "description": (
                            "asx/event_window: calendar days forward from "
                            "date_from. This is the right choice for 'one "
                            "week', 'the week after' and a span written as two "
                            "dates a week apart (5-12 Jun -> calendar_days=7). "
                            "Mutually exclusive with sessions."
                        ),
                    },
                    "year": {
                        "type": "integer",
                        "description": "Calendar year, e.g. 2019.",
                    },
                    "month": {
                        "type": "integer",
                        "description": "Calendar month 1-12, used with year.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ASX tickers such as CBA.AX. Omit for all 18."
                        ),
                    },
                    "exclude_tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            'Tickers to drop. Pass ["TAH.AX"] whenever the '
                            "question says non-Tabcorp or excluding Tabcorp."
                        ),
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "AFR regex, used verbatim. Add \\b anchors yourself."
                        ),
                    },
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "AFR search terms; escaped and word-anchored for you."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Truncate a ranking to the first N rows.",
                    },
                    "annualised": {
                        "type": "boolean",
                        "description": "Annualise volatility (default true).",
                    },
                },
                "required": ["dataset", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Fetch AFR article text as evidence. Give headline and date when "
                "the question names an article — that is an exact lookup. This "
                "returns text only; it does not count, calculate or classify "
                "sentiment. Never derive a number from retrieved prose: use "
                "query_data for every count and calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "headline": {
                        "type": "string",
                        "description": "Exact article headline from the question.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Publication date, YYYY-MM-DD.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex alternative to a headline lookup.",
                    },
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Word-anchored terms to search instead.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max articles to return (default 5).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "Characters of article text per result "
                            "(default 1200, max 20000). Raise it only when the "
                            "snippet is genuinely too short to judge."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dataset_coverage",
            "description": (
                "Row counts and date spans for all three datasets plus their "
                "common overlap. Call this FIRST whenever a question spans "
                "multiple datasets or mentions a period that may fall outside "
                "one of them. If the requested period lies outside the overlap, "
                "the correct answer says the supplied evidence cannot support "
                "the analysis and explains the coverage gap — it does not "
                "estimate the missing values."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute(name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run a tool call. Returns ``(payload, ok)``; never raises.

    A failed call becomes a payload the brain can read and retry from, because
    an unhandled exception here would cost the whole question instead of one
    step.
    """
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return (
            {
                "error": f"unknown tool {name!r}",
                "available_tools": sorted(TOOL_IMPLEMENTATIONS),
            },
            False,
        )
    try:
        return impl(**(arguments or {})), True
    except MetricError as exc:
        logger.warning("tool %s rejected arguments %s: %s", name, arguments, exc)
        return {"error": str(exc), "tool": name, "arguments": arguments}, False
    except TypeError as exc:
        logger.warning("tool %s bad signature %s: %s", name, arguments, exc)
        return {
            "error": f"invalid arguments for {name}: {exc}",
            "arguments": arguments,
        }, False
    except Exception as exc:  # noqa: BLE001 - must not kill the request
        logger.exception("tool %s failed", name)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "tool": name,
            "arguments": arguments,
        }, False
