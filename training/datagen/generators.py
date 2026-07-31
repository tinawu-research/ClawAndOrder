"""Deterministic example generators, one per question category.

Every gold answer is derived from a real tool payload by code, never written
from knowledge. That makes targets exact by construction and satisfies the
rule that dataset-derived facts come from the data tools rather than model
memory.

The important design choice is that the *requested component subset* is
sampled, not fixed. A generator that always maps ``rank_annual_returns`` to
"X best, Y worst" teaches a metric-to-sentence reflex, and a hidden question
asking for three components would lose a third of its marks. Sampling subsets
teaches the model to answer exactly what was asked, which is what
component-based grading actually rewards.
"""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterator

import answers as A
from spec import Example

AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tools import afr, asx, rba  # noqa: E402

TAH = "TAH.AX"
EXCLUDE_TAH = [TAH]

#: Held out from training entirely, so the test split measures generalisation
#: across entities and periods rather than recall.
HOLDOUT_TICKERS = {"SUN.AX", "TPG.AX", "CMW.AX"}
HOLDOUT_YEAR = 2017

TRAIN_YEARS = [2015, 2016, 2018, 2019, 2020, 2021]

_REGISTRY: dict[str, Callable[[random.Random, bool], Iterator[Example]]] = {}


def generator(name: str):
    def wrap(fn):
        _REGISTRY[name] = fn
        return fn

    return wrap


def registry() -> dict[str, Callable[[random.Random, bool], Iterator[Example]]]:
    return dict(_REGISTRY)


def _subsets(items: list[str], rng: random.Random, *, low: int = 1, high: int = 4) -> list[tuple[str, ...]]:
    """All component subsets of the requested sizes, shuffled."""
    out: list[tuple[str, ...]] = []
    for size in range(low, min(high, len(items)) + 1):
        out.extend(combinations(items, size))
    rng.shuffle(out)
    return out


def _qd(dataset: str, metric: str, **kwargs) -> tuple[str, dict[str, Any], Any]:
    """Build a ``query_data`` call the way the brain emits it, and run it."""
    args: dict[str, Any] = {"dataset": dataset, "metric": metric}
    args.update({k: v for k, v in kwargs.items() if v is not None})
    module = {"rba": rba, "asx": asx, "afr": afr}[dataset]
    payload = module.METRICS[metric](**kwargs)
    return ("query_data", args, payload)


# --------------------------------------------------------------------------
# 1. RBA counting and cycle summaries
# --------------------------------------------------------------------------

_RBA_CYCLES = [
    ("the 2011-2013 easing period", "2011-01-01", "2013-12-31"),
    ("2015", "2015-01-01", "2015-12-31"),
    ("2016", "2016-01-01", "2016-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("the 2022-2023 tightening cycle", "2022-01-01", "2023-12-31"),
    ("the 2015-2016 easing stretch", "2015-01-01", "2016-12-31"),
    ("the 2019-2020 easing run", "2019-01-01", "2020-12-31"),
]


@generator("rba_counts")
def gen_rba_counts(rng: random.Random, held_out: bool) -> Iterator[Example]:
    # Whole-of-dataset counts.
    tool = _qd("rba", "count_changes")
    payload = tool[2]
    facts = {
        "changes": f"{A.count(payload['changes'])} of the {A.count(payload['total_records'])} "
                   f"decision records changed the rate",
        "split": f"{A.count(payload['increases'])} increases and "
                 f"{A.count(payload['decreases'])} decreases",
        "cumulative": f"the cumulative change was {A.points(payload['cumulative_change_points'])}",
    }
    prompts = {
        ("changes", "split"): "From the first RBA record to the last, how many cash-rate "
                              "decisions changed the rate, and how many were increases versus decreases?",
        ("changes",): "Across the whole RBA dataset, how many decisions changed the cash rate?",
        ("split",): "Across the whole RBA dataset, how many rate changes were increases and how many were decreases?",
        ("changes", "split", "cumulative"): "Across the whole RBA dataset, how many decisions changed the "
                                            "rate, what was the increase/decrease split, and what was the "
                                            "cumulative change?",
    }
    for subset, question in prompts.items():
        parts = [facts[k] for k in subset]
        yield Example(
            category="rba_counts",
            template_id=f"rba_counts.whole.{'_'.join(subset)}",
            question=question,
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[tool],
            param_key=f"whole|{','.join(subset)}",
            split_keys={"years": [], "tickers": []},
        )

    # Windowed cycle summaries.
    for label, start, end in _RBA_CYCLES:
        year = int(start[:4])
        if (year == HOLDOUT_YEAR) != held_out:
            continue
        tool = _qd("rba", "period_summary", date_from=start, date_to=end)
        payload = tool[2]
        if payload.get("changes", 0) == 0:
            continue

        moves = "cuts" if payload["decreases"] >= payload["increases"] else "hikes"
        n_moves = max(payload["decreases"], payload["increases"])
        by_year = payload["decreases_by_year"] if moves == "cuts" else payload["increases_by_year"]

        facts = {
            "count": f"{A.count(n_moves)} {moves} occurred across {label}",
            "by_year": ", ".join(f"{A.count(v)} in {k}" for k, v in sorted(by_year.items())),
            "cumulative": f"they totalled {A.points(payload['cumulative_change_points'])}",
            "endpoints": f"taking the target from {A.rate(payload['target_before_first_change'])} "
                         f"before the first change to {A.rate(payload['target_at_window_end'])}",
            "first": f"the first change took effect on {A.day(payload['first_change_date'])}",
        }
        available = ["count", "by_year", "cumulative", "endpoints", "first"]
        for subset in _subsets(available, rng, low=2, high=3)[:4]:
            asked = A.oxford(
                [
                    {"count": "how many changes occurred", "by_year": "the yearly breakdown",
                     "cumulative": "the cumulative change", "endpoints": "the start and end targets",
                     "first": "the first effective date"}[k]
                    for k in subset
                ]
            )
            parts = [facts[k] for k in subset]
            yield Example(
                category="rba_counts",
                template_id=f"rba_counts.cycle.{'_'.join(subset)}",
                question=f"Across {label}, report {asked}.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[tool],
                param_key=f"{start}:{end}|{','.join(subset)}",
                split_keys={"years": [year], "tickers": []},
            )

    # Extremes and hold streaks.
    tool = _qd("rba", "extremes")
    payload = tool[2]
    for which, label in (("highest", "highest"), ("lowest", "lowest")):
        node = payload[which]
        parts = [
            f"the {label} cash-rate target in the RBA dataset was {A.rate(node['target'])}",
            f"it first took effect on {A.day(node['first_effective_date'])}"
            if "first_effective_date" in node else None,
            f"{A.count(node['records'])} decision records show that rate",
        ]
        parts = [p for p in parts if p]
        yield Example(
            category="rba_counts",
            template_id=f"rba_counts.extremes.{which}",
            question=f"What is the {label} cash-rate target in the RBA dataset, when did it first "
                     f"take effect, and how many decision records show that rate?",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[tool],
            param_key=f"extremes|{which}",
            split_keys={"years": [], "tickers": []},
        )

    tool = _qd("rba", "max_hold_streak")
    payload = tool[2]
    parts = [
        f"the longest stretch between two non-zero RBA rate changes was {A.count(payload['days'])} days",
        f"lasting from {A.day(payload['start_date'])} to {A.day(payload['end_date'])}",
        f"during which the rate held at {A.rate(payload['rate_during_hold'])} before changing to "
        f"{A.rate(payload['rate_after_change'])}",
    ]
    yield Example(
        category="rba_counts",
        template_id="rba_counts.hold_streak",
        question="What was the longest stretch between two non-zero RBA rate changes?",
        answer=A.join_clauses(parts),
        components=parts,
        tool_calls=[tool],
        param_key="hold_streak",
        split_keys={"years": [], "tickers": []},
    )


# --------------------------------------------------------------------------
# 2. Dataset shape and coverage
# --------------------------------------------------------------------------

@generator("coverage")
def gen_coverage(rng: random.Random, held_out: bool) -> Iterator[Example]:
    if held_out:
        return

    asx_tool = _qd("asx", "coverage")
    afr_tool = _qd("afr", "coverage")
    rba_tool = _qd("rba", "coverage")
    asx_p, afr_p, rba_p = asx_tool[2], afr_tool[2], rba_tool[2]

    specs = [
        (
            "asx",
            asx_tool,
            {
                "tickers": f"there are {A.count(asx_p['tickers'])} ticker files",
                "rows": f"each containing {A.count(asx_p['rows_per_ticker'])} rows",
                "range": f"covering {A.day(asx_p['common_start'])} through "
                         f"{A.day(asx_p['common_end'])}",
            },
            {"tickers": "how many ticker files there are", "rows": "the rows per ticker",
             "range": "the common date range"},
            "the ASX dataset",
        ),
        (
            "afr",
            afr_tool,
            {
                "articles": f"the AFR corpus holds {A.count(afr_p['articles'])} articles",
                "dated": f"{A.count(afr_p['dated_articles'])} carry a publication date and "
                         f"{A.count(afr_p['undated_articles'])} do not",
                "range": f"running from {A.day(afr_p['start'])} to {A.day(afr_p['end'])}",
            },
            {"articles": "the article count", "dated": "how many are dated",
             "range": "the period covered"},
            "the AFR corpus",
        ),
        (
            "rba",
            rba_tool,
            {
                "records": f"the RBA dataset holds {A.count(rba_p['records'])} decision records",
                "range": f"spanning {A.day(rba_p['first_effective_date'])} to "
                         f"{A.day(rba_p['last_effective_date'])}",
                "targets": f"the target starts at {A.rate(rba_p['first_target'])} and ends at "
                           f"{A.rate(rba_p['last_target'])}",
            },
            {"records": "the record count", "range": "the period spanned",
             "targets": "the first and last targets"},
            "the RBA dataset",
        ),
    ]

    for name, tool, facts, asked, subject in specs:
        for subset in _subsets(list(facts), rng, low=1, high=3):
            parts = [facts[k] for k in subset]
            yield Example(
                category="coverage",
                template_id=f"coverage.{name}.{'_'.join(subset)}",
                question=f"For {subject}, report {A.oxford([asked[k] for k in subset])}.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[tool],
                param_key=f"coverage|{name}|{','.join(subset)}",
                split_keys={"years": [], "tickers": []},
            )

    # Cross-dataset overlap: the reasoning the refusal category depends on.
    parts = [
        f"the ASX data runs {A.day(asx_p['common_start'])} to {A.day(asx_p['common_end'])}",
        f"the AFR corpus runs {A.day(afr_p['start'])} to {A.day(afr_p['end'])}",
        f"the RBA data runs {A.day(rba_p['first_effective_date'])} to "
        f"{A.day(rba_p['last_effective_date'])}",
    ]
    yield Example(
        category="coverage",
        template_id="coverage.cross",
        question="What period does each of the three supplied datasets cover?",
        answer=A.join_clauses(parts),
        components=parts,
        tool_calls=[asx_tool, afr_tool, rba_tool],
        param_key="coverage|cross",
        split_keys={"years": [], "tickers": []},
    )


# --------------------------------------------------------------------------
# 3. ASX ranked annual returns
# --------------------------------------------------------------------------

@generator("asx_returns")
def gen_asx_returns(rng: random.Random, held_out: bool) -> Iterator[Example]:
    years = [HOLDOUT_YEAR] if held_out else TRAIN_YEARS
    for year in years:
        tool = _qd("asx", "rank_annual_returns", year=year, exclude_tickers=EXCLUDE_TAH)
        payload = tool[2]
        best, worst = payload["best"], payload["worst"]
        ranked = payload["ranked"]

        subject_pool = [
            row for row in ranked
            if (row["ticker"] in HOLDOUT_TICKERS) == held_out
        ]
        named = rng.choice(subject_pool) if subject_pool else ranked[0]

        facts = {
            "best": f"{A.ticker(best['ticker'])} was best at {A.pct(best['return_pct'])}",
            "worst": f"{A.ticker(worst['ticker'])} was worst at {A.pct(worst['return_pct'])}",
            "basket": f"the non-Tabcorp basket averaged {A.pct(payload['basket_average_return_pct'])}",
            "named": f"{A.ticker(named['ticker'])} returned {A.pct(named['return_pct'])}, "
                     f"ranking {A.count(named['rank'])} of {A.count(len(ranked))}",
        }
        asked_text = {
            "best": "the best performer",
            "worst": "the worst performer",
            "basket": "the basket average",
            "named": f"where {A.ticker(named['ticker'])} ranked",
        }
        for subset in _subsets(list(facts), rng, low=1, high=3)[:5]:
            parts = [facts[k] for k in subset]
            yield Example(
                category="asx_returns",
                template_id=f"asx_returns.{'_'.join(subset)}",
                question=f"Excluding Tabcorp, for {year} report "
                         f"{A.oxford([asked_text[k] for k in subset])}.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[tool],
                param_key=f"returns|{year}|{','.join(subset)}",
                split_keys={"years": [year], "tickers": [named["ticker"]]},
            )


# --------------------------------------------------------------------------
# 4. ASX average volume
# --------------------------------------------------------------------------

@generator("asx_volume")
def gen_asx_volume(rng: random.Random, held_out: bool) -> Iterator[Example]:
    # No blanket early return on held_out. The basket-wide facts below (highest,
    # lowest, volatility) are properties of the whole sample and have no held-out
    # variant, so they are skipped -- but the named-ticker rank lookups do have
    # one, and gating the whole generator would leave that branch unreachable.
    tool = _qd("asx", "avg_volume", exclude_tickers=EXCLUDE_TAH)
    payload = tool[2]
    high, low = payload["highest"], payload["lowest"]
    ranked = payload["ranked"]

    facts = {
        "highest": f"{A.ticker(high['ticker'])} has the highest average daily volume at "
                   f"{A.volume(high['avg_daily_volume'])} shares per trading day",
        "lowest": f"{A.ticker(low['ticker'])} has the lowest at "
                  f"{A.volume(low['avg_daily_volume'])} shares per trading day",
        "days": f"each series covers {A.count(high['trading_days'])} trading days",
    }
    asked = {"highest": "the highest", "lowest": "the lowest", "days": "the number of trading days"}
    for subset in [] if held_out else _subsets(list(facts), rng, low=1, high=3):
        parts = [facts[k] for k in subset]
        yield Example(
            category="asx_volume",
            template_id=f"asx_volume.{'_'.join(subset)}",
            question=f"Excluding Tabcorp, which ticker has {A.oxford([asked[k] for k in subset])} "
                     f"average daily volume over the full sample?",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[tool],
            param_key=f"volume|{','.join(subset)}",
            split_keys={"years": [], "tickers": [high["ticker"], low["ticker"]]},
        )

    # Named-ticker rank lookups: same payload, a different requested component.
    for row in ranked[:8]:
        if (row["ticker"] in HOLDOUT_TICKERS) != held_out:
            continue
        parts = [
            f"{A.ticker(row['ticker'])} averaged {A.volume(row['avg_daily_volume'])} shares per "
            f"trading day, ranking {A.count(row['rank'])} of {A.count(len(ranked))} "
            f"non-Tabcorp constituents"
        ]
        yield Example(
            category="asx_volume",
            template_id="asx_volume.named_rank",
            question=f"Excluding Tabcorp, what is {A.ticker(row['ticker'])}'s average daily volume "
                     f"over the full sample and where does it rank?",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[tool],
            param_key=f"volume|named|{row['ticker']}",
            split_keys={"years": [], "tickers": [row["ticker"]]},
        )

    # Volatility shares the shape and is a metric the public set never uses,
    # which makes it useful coverage against unseen hidden-question metrics.
    for year in [] if held_out else rng.sample(TRAIN_YEARS, 3):
        vol_tool = _qd("asx", "volatility", exclude_tickers=EXCLUDE_TAH, year=year, annualised=True)
        vol = vol_tool[2]
        hi, lo = vol["highest"], vol["lowest"]
        parts = [
            f"{A.ticker(hi['ticker'])} was the most volatile in {year} at "
            f"{A.pct(hi['annualised_volatility_pct'], signed=False)} annualised",
            f"{A.ticker(lo['ticker'])} the least at "
            f"{A.pct(lo['annualised_volatility_pct'], signed=False)}",
        ]
        yield Example(
            category="asx_volume",
            template_id="asx_volume.volatility",
            question=f"Excluding Tabcorp, which ticker had the highest and lowest annualised "
                     f"volatility in {year}?",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[vol_tool],
            param_key=f"volatility|{year}",
            split_keys={"years": [year], "tickers": [hi["ticker"], lo["ticker"]]},
        )


# --------------------------------------------------------------------------
# 5. ASX drawdown rankings
# --------------------------------------------------------------------------

_WORDS = ["", "single", "two", "three", "four", "five", "six"]


@generator("asx_drawdown")
def gen_asx_drawdown(rng: random.Random, held_out: bool) -> Iterator[Example]:
    if held_out:
        return

    for top_n in (2, 3, 4, 5):
        tool = _qd("asx", "max_drawdown", exclude_tickers=EXCLUDE_TAH, top_n=top_n)
        payload = tool[2]
        rows = payload["ranked_worst_first"]

        # Full ranking, with dates -- the MHQ055 shape.
        entries = [
            f"{A.ticker(r['ticker'])} {A.pct(r['max_drawdown_pct'])}, "
            f"{A.day(r['peak_date'])} to {A.day(r['trough_date'])}"
            for r in rows
        ]
        yield Example(
            category="asx_drawdown",
            template_id=f"asx_drawdown.dated.top{top_n}",
            question=f"Rank the {_WORDS[top_n]} worst non-Tabcorp full-sample maximum drawdowns "
                     f"and identify each peak and trough date.",
            answer=A.ranking(entries) + ".",
            components=entries,
            tool_calls=[tool],
            param_key=f"drawdown|dated|top{top_n}",
            split_keys={"years": [], "tickers": [r["ticker"] for r in rows]},
        )

        # Magnitudes only: the same evidence, a narrower request. This is the
        # component-subsetting case that catches template lock.
        bare = [f"{A.ticker(r['ticker'])} {A.pct(r['max_drawdown_pct'])}" for r in rows]
        yield Example(
            category="asx_drawdown",
            template_id=f"asx_drawdown.bare.top{top_n}",
            question=f"Rank the {_WORDS[top_n]} worst non-Tabcorp full-sample maximum drawdowns "
                     f"by magnitude.",
            answer=A.ranking(bare) + ".",
            components=bare,
            tool_calls=[tool],
            param_key=f"drawdown|bare|top{top_n}",
            split_keys={"years": [], "tickers": [r["ticker"] for r in rows]},
        )

    # Single-worst, with the peak/trough closes as an extra component.
    tool = _qd("asx", "max_drawdown", exclude_tickers=EXCLUDE_TAH, top_n=1)
    worst = tool[2]["worst"]
    variants = {
        "plain": [
            f"{A.ticker(worst['ticker'])} had the worst non-Tabcorp maximum drawdown at "
            f"{A.pct(worst['max_drawdown_pct'])}"
        ],
        "dated": [
            f"{A.ticker(worst['ticker'])} had the worst non-Tabcorp maximum drawdown at "
            f"{A.pct(worst['max_drawdown_pct'])}",
            f"running from {A.day(worst['peak_date'])} to {A.day(worst['trough_date'])}",
        ],
        "closes": [
            f"{A.ticker(worst['ticker'])} had the worst non-Tabcorp maximum drawdown at "
            f"{A.pct(worst['max_drawdown_pct'])}",
            f"falling from a peak close of {A.close(worst['peak_close'])} to a trough close of "
            f"{A.close(worst['trough_close'])}",
        ],
    }
    questions = {
        "plain": "Excluding Tabcorp, which ticker had the worst full-sample maximum drawdown?",
        "dated": "Excluding Tabcorp, which ticker had the worst full-sample maximum drawdown, "
                 "and over what dates?",
        "closes": "Excluding Tabcorp, which ticker had the worst full-sample maximum drawdown, "
                  "and what were the peak and trough closes?",
    }
    for key, parts in variants.items():
        yield Example(
            category="asx_drawdown",
            template_id=f"asx_drawdown.worst.{key}",
            question=questions[key],
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[tool],
            param_key=f"drawdown|worst|{key}",
            split_keys={"years": [], "tickers": [worst["ticker"]]},
        )


# --------------------------------------------------------------------------
# 6. AFR counting
# --------------------------------------------------------------------------

_AFR_TERMS = [
    ("unemployment", ["unemployment"]),
    ("inflation", ["inflation"]),
    ("recession", ["recession"]),
    ("dividend", ["dividend"]),
    ("QBE", ["QBE"]),
    ("BHP", ["BHP"]),
    ("iron ore", ["iron ore"]),
    ("housing", ["housing"]),
    ("superannuation", ["superannuation"]),
]


@generator("afr_counts")
def gen_afr_counts(rng: random.Random, held_out: bool) -> Iterator[Example]:
    terms = _AFR_TERMS[-2:] if held_out else _AFR_TERMS[:-2]

    for label, term_list in terms:
        tool = _qd("afr", "count_by_year", terms=term_list)
        payload = tool[2]
        by_year = payload["by_year"]
        peak_year = payload["peak_year"]

        facts = {
            "peak_year": f"it peaked in {peak_year} with {A.count(payload['peak_year_count'])} matching records",
            "total": f"the corpus holds {A.count(payload['total'])} matching records in total",
        }
        for subset in _subsets(list(facts), rng, low=1, high=2):
            parts = [facts[k] for k in subset]
            yield Example(
                category="afr_counts",
                template_id=f"afr_counts.year.{'_'.join(subset)}",
                question=f"Using a case-insensitive once-per-record whole-word {label} search of the "
                         f"AFR corpus, which year has the highest count"
                         f"{' and what is the overall total' if 'total' in subset else ''}?",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[tool],
                param_key=f"afr_year|{label}|{','.join(subset)}",
                split_keys={"years": [], "terms": [label]},
            )

        # Year + month peak, the MHQ061 shape.
        month_tool = _qd("afr", "count_by_month", terms=term_list, year=peak_year)
        month_payload = month_tool[2]
        by_month = month_payload.get("by_month", {})
        if by_month:
            peak_month = max(by_month, key=lambda k: by_month[k])
            month_number = int(str(peak_month).split("-")[-1])
            parts = [
                f"it peaked in {peak_year} with {A.count(payload['peak_year_count'])} matching records",
                f"{A.month(peak_year, month_number)} is the peak month with "
                f"{A.count(by_month[peak_month])}",
            ]
            yield Example(
                category="afr_counts",
                template_id="afr_counts.year_month",
                question=f"Using a case-insensitive once-per-record whole-word {label} search, "
                         f"which year and which month have the highest AFR counts?",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[tool, month_tool],
                param_key=f"afr_year_month|{label}",
                split_keys={"years": [peak_year], "terms": [label]},
            )

        # Single-year count with share.
        for year in ([HOLDOUT_YEAR] if held_out else rng.sample(TRAIN_YEARS, 2)):
            count_tool = _qd("afr", "count", terms=term_list, year=year)
            count_payload = count_tool[2]
            parts = [
                f"there are {A.count(count_payload['matching_records'])} AFR records matching "
                f"whole-word {label} in {year}"
            ]
            yield Example(
                category="afr_counts",
                template_id="afr_counts.single_year",
                question=f"For {year}, report the once-per-record whole-word {label} AFR count.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[count_tool],
                param_key=f"afr_count|{label}|{year}",
                split_keys={"years": [year], "terms": [label]},
            )


# --------------------------------------------------------------------------
# 8. RBA event to ASX window return
# --------------------------------------------------------------------------

_EVENT_WINDOWS = [
    ("2019-06-05", "2019-06-12"),
    ("2019-07-03", "2019-07-10"),
    ("2019-10-02", "2019-10-09"),
    ("2020-03-04", "2020-03-11"),
    ("2016-05-04", "2016-05-11"),
    ("2016-08-03", "2016-08-10"),
    ("2015-02-04", "2015-02-11"),
    ("2015-05-06", "2015-05-13"),
    ("2020-11-04", "2020-11-11"),
]

_FOCUS_TICKERS = ["CBA.AX", "NAB.AX", "ANZ.AX", "BHP.AX", "RIO.AX"]


@generator("rba_asx_event")
def gen_rba_asx_event(rng: random.Random, held_out: bool) -> Iterator[Example]:
    for start, end in _EVENT_WINDOWS:
        year = int(start[:4])
        if (year == HOLDOUT_YEAR) != held_out:
            continue

        rate_tool = _qd("rba", "lookup_rate", date_from=start)
        rate_payload = rate_tool[2]
        window_tool = _qd(
            "asx", "window_return", date_from=start, date_to=end, exclude_tickers=EXCLUDE_TAH
        )
        window_payload = window_tool[2]
        by_ticker = {r["ticker"]: r for r in window_payload["results"]}
        basket = window_payload["basket_average_return_pct"]

        facts = {
            "rate": f"the RBA target moved to {A.rate(rate_payload['cash_rate_target'])}",
            "basket": f"from {A.day(start)} to {A.day(end)} the non-Tabcorp basket "
                      f"{A.direction_verb(basket)} {A.pct(basket)}",
            "focus": ", ".join(
                f"{t.split('.')[0]} {A.pct(by_ticker[t]['return_pct'])}"
                for t in _FOCUS_TICKERS if t in by_ticker
            ),
        }
        asked = {
            "rate": "the new target",
            "basket": "the non-Tabcorp basket return",
            "focus": "the returns for CBA.AX, NAB.AX, ANZ.AX, BHP.AX and RIO.AX",
        }
        for subset in _subsets(list(facts), rng, low=2, high=3)[:3]:
            parts = [facts[k] for k in subset]
            yield Example(
                category="rba_asx_event",
                template_id=f"rba_asx_event.{'_'.join(subset)}",
                question=f"After the {A.day(start)} RBA decision, report "
                         f"{A.oxford([asked[k] for k in subset])} over {A.day(start)} to {A.day(end)}.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=[rate_tool, window_tool],
                param_key=f"event|{start}|{','.join(subset)}",
                split_keys={"years": [year], "tickers": _FOCUS_TICKERS},
            )


# --------------------------------------------------------------------------
# 9. Multi-dataset composite
# --------------------------------------------------------------------------

@generator("composite")
def gen_composite(rng: random.Random, held_out: bool) -> Iterator[Example]:
    years = [HOLDOUT_YEAR] if held_out else TRAIN_YEARS
    for year in years:
        rba_tool = _qd("rba", "period_summary", date_from=f"{year}-01-01", date_to=f"{year}-12-31")
        rba_payload = rba_tool[2]
        asx_tool = _qd("asx", "rank_annual_returns", year=year, exclude_tickers=EXCLUDE_TAH)
        asx_payload = asx_tool[2]
        afr_tool = _qd("afr", "count", terms=["cash rate", "rate cut", "RBA"], year=year)
        afr_payload = afr_tool[2]

        if rba_payload["changes"]:
            moves = "cut" if rba_payload["decreases"] >= rba_payload["increases"] else "raised"
            n = max(rba_payload["decreases"], rba_payload["increases"])
            rba_fact = (
                f"the RBA {moves} {A.count(n)} times for "
                f"{A.points(rba_payload['cumulative_change_points'])}, ending at "
                f"{A.rate(rba_payload['target_at_window_end'])}"
            )
        else:
            rba_fact = (
                f"the RBA held the target at "
                f"{A.rate(rba_payload['target_at_window_end'])} all year"
            )

        facts = {
            "rba": rba_fact,
            "afr": f"AFR contains {A.count(afr_payload['matching_records'])} matching records",
            "basket": f"the non-Tabcorp ASX average return was "
                      f"{A.pct(asx_payload['basket_average_return_pct'])}",
            "best": f"{A.ticker(asx_payload['best']['ticker'])} led at "
                    f"{A.pct(asx_payload['best']['return_pct'])}",
        }
        asked = {
            "rba": "the RBA rate activity and year-end target",
            "afr": "the AFR cash-rate pattern count",
            "basket": "the non-Tabcorp ASX basket's average annual return",
            "best": "the best-performing non-Tabcorp ticker",
        }
        tools_for = {
            "rba": rba_tool, "afr": afr_tool, "basket": asx_tool, "best": asx_tool,
        }

        for subset in _subsets(list(facts), rng, low=2, high=4)[:4]:
            parts = [facts[k] for k in subset]
            calls, seen = [], set()
            for key in subset:
                call = tools_for[key]
                marker = (call[0], json.dumps(call[1], sort_keys=True))
                if marker not in seen:
                    seen.add(marker)
                    calls.append(call)
            yield Example(
                category="composite",
                template_id=f"composite.{'_'.join(subset)}",
                question=f"For {year}, report {A.oxford([asked[k] for k in subset])}.",
                answer=A.join_clauses(parts),
                components=parts,
                tool_calls=calls,
                param_key=f"composite|{year}|{','.join(subset)}",
                split_keys={"years": [year], "tickers": []},
            )


# --------------------------------------------------------------------------
# 10. Justified refusal on coverage mismatch
# --------------------------------------------------------------------------

_OUT_OF_RANGE = [
    ("the 2022-2023 RBA tightening cycle", "2022-01-01", "2023-12-31"),
    ("the 2024 rate decisions", "2024-01-01", "2024-12-31"),
    ("the 2025 easing cycle", "2025-01-01", "2025-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
]


@generator("refusal")
def gen_refusal(rng: random.Random, held_out: bool) -> Iterator[Example]:
    if held_out:
        return

    asx_cov = _qd("asx", "coverage")
    afr_cov = _qd("afr", "coverage")

    for label, start, end in _OUT_OF_RANGE:
        rba_tool = _qd("rba", "period_summary", date_from=start, date_to=end)
        payload = rba_tool[2]
        if not payload["changes"]:
            continue

        moves = "hikes" if payload["increases"] >= payload["decreases"] else "cuts"
        n = max(payload["increases"], payload["decreases"])
        asx_end = asx_cov[2]["common_end"]
        afr_end = afr_cov[2]["end"]

        parts = [
            "No",
            f"the RBA data covers the {A.count(n)} {moves} from "
            f"{A.day(payload['first_change_date'])} to {A.day(payload['last_change_date'])}, "
            f"but the ASX data ends {A.day(asx_end)} and the AFR corpus ends {A.day(afr_end)}",
            "a three-dataset reaction analysis is therefore unsupported by the supplied evidence",
        ]
        yield Example(
            category="refusal",
            template_id="refusal.coverage_mismatch",
            question=f"Can the three supplied datasets support a fully observed analysis of how AFR "
                     f"news and ASX prices reacted to {label}?",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[rba_tool, asx_cov, afr_cov],
            param_key=f"refusal|{start}",
            split_keys={"years": [int(start[:4])], "tickers": []},
        )

        # ASX-only variant: the market leg alone is already unsupported.
        asx_parts = [
            "No",
            f"the ASX price data ends {A.day(asx_end)}, so it contains no observations for {label}",
            f"the RBA record does cover the period, showing {A.count(n)} {moves}, but the market "
            f"reaction cannot be measured from the supplied data",
        ]
        yield Example(
            category="refusal",
            template_id="refusal.asx_out_of_range",
            question=f"Using the supplied data, what was the ASX basket's reaction to {label}?",
            answer=A.join_clauses(asx_parts),
            components=asx_parts,
            tool_calls=[asx_cov, rba_tool],
            param_key=f"refusal|asx|{start}",
            split_keys={"years": [int(start[:4])], "tickers": []},
        )

        # AFR-only variant.
        afr_parts = [
            "No",
            f"the AFR corpus ends {A.day(afr_end)} and holds no articles from {label}",
            "the supplied news evidence therefore cannot support that comparison",
        ]
        yield Example(
            category="refusal",
            template_id="refusal.afr_out_of_range",
            question=f"How did AFR coverage of the cash rate change during {label}?",
            answer=A.join_clauses(afr_parts),
            components=afr_parts,
            tool_calls=[afr_cov],
            param_key=f"refusal|afr|{start}",
            split_keys={"years": [int(start[:4])], "tickers": []},
        )

    # The other edge of the window: data requested from before coverage starts.
    asx_start = asx_cov[2]["common_start"]
    afr_start = afr_cov[2]["start"]
    for label, year in (("2012", 2012), ("2013", 2013), ("2014", 2014)):
        parts = [
            "No",
            f"the ASX data begins {A.day(asx_start)} and the AFR corpus begins "
            f"{A.day(afr_start)}, so neither covers {label}",
            "that period cannot be analysed from the supplied datasets",
        ]
        yield Example(
            category="refusal",
            template_id="refusal.before_coverage",
            question=f"Report the non-Tabcorp ASX basket return and AFR article count for {label}.",
            answer=A.join_clauses(parts),
            components=parts,
            tool_calls=[asx_cov, afr_cov],
            param_key=f"refusal|before|{year}",
            split_keys={"years": [year], "tickers": []},
        )
