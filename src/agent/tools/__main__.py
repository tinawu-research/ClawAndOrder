"""Drive the tool layer from the terminal, without a model in the loop.

Spot-checking a reference answer should not require standing up the agent, the
brain and the synthesiser. This runs exactly the dispatch path the brain uses —
``tools.execute`` — so what you see here is what the model would have seen::

    cd src/agent
    python -m tools query_data rba count_changes
    python -m tools query_data rba compare_periods \\
        --date-from 2011 --date-to 2013 --compare-from 2022 --compare-to 2023
    python -m tools query_data asx event_window \\
        --date-from 2019-06-05 --sessions 5 --exclude-tickers TAH.AX
    python -m tools retrieve --headline "..." --date 2020-03-02
    python -m tools dataset_coverage
    python -m tools metrics                 # list every dataset and metric

Only the datasets a call needs are loaded: RBA and ASX are near-instant, AFR
takes ~25s because it builds the token index.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from datastore import STORE

import tools


def _load(dataset: str | None) -> None:
    """Load only what the requested call touches."""
    wanted = {dataset} if dataset else {"rba", "asx", "afr"}
    if "rba" in wanted:
        STORE._load_rba()
    if "asx" in wanted:
        STORE._load_asx()
    if "afr" in wanted:
        print("loading the AFR corpus and index (~25s)...", file=sys.stderr)
        STORE._load_afr()
        STORE.afr_index.build(STORE.afr)


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools",
        description="Run the agent's tools directly against the local datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metrics", help="list every dataset and its metric names")
    sub.add_parser("dataset_coverage", help="row counts, spans and the overlap")

    query = sub.add_parser("query_data", help="run one metric")
    query.add_argument("dataset", choices=["rba", "asx", "afr"])
    query.add_argument("metric")
    query.add_argument("--date-from")
    query.add_argument("--date-to")
    query.add_argument("--compare-from")
    query.add_argument("--compare-to")
    query.add_argument("--day")
    query.add_argument("--ticker")
    query.add_argument("--year", type=int)
    query.add_argument("--month", type=int)
    query.add_argument("--sessions", type=int)
    query.add_argument("--calendar-days", type=int)
    query.add_argument("--top-n", type=int)
    query.add_argument("--tickers", help="comma-separated")
    query.add_argument("--exclude-tickers", help="comma-separated")
    query.add_argument("--pattern", help="AFR regex, used verbatim")
    query.add_argument("--terms", help="comma-separated AFR terms")
    query.add_argument(
        "--no-annualise", action="store_true", help="raw daily volatility"
    )

    get = sub.add_parser("retrieve", help="fetch AFR article text")
    get.add_argument("--headline")
    get.add_argument("--date")
    get.add_argument("--pattern")
    get.add_argument("--terms", help="comma-separated")
    get.add_argument("--limit", type=int)
    get.add_argument("--max-chars", type=int)

    args = parser.parse_args(argv)

    if args.command == "metrics":
        _print(
            {
                dataset: sorted(metrics)
                for dataset, metrics in tools.DATASETS.items()
            }
        )
        return 0

    if args.command == "dataset_coverage":
        _load(None)
        payload, ok = tools.execute("dataset_coverage", {})
    elif args.command == "query_data":
        _load(args.dataset)
        arguments = {
            "dataset": args.dataset,
            "metric": args.metric,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "compare_from": args.compare_from,
            "compare_to": args.compare_to,
            "day": args.day,
            "ticker": args.ticker,
            "year": args.year,
            "month": args.month,
            "sessions": args.sessions,
            "calendar_days": args.calendar_days,
            "top_n": args.top_n,
            "tickers": _split(args.tickers),
            "exclude_tickers": _split(args.exclude_tickers),
            "pattern": args.pattern,
            "terms": _split(args.terms),
        }
        if args.no_annualise:
            arguments["annualised"] = False
        payload, ok = tools.execute(
            "query_data", {k: v for k, v in arguments.items() if v is not None}
        )
    else:
        _load("afr")
        arguments = {
            "headline": args.headline,
            "date": args.date,
            "pattern": args.pattern,
            "terms": _split(args.terms),
            "limit": args.limit,
            "max_chars": args.max_chars,
        }
        payload, ok = tools.execute(
            "retrieve", {k: v for k, v in arguments.items() if v is not None}
        )

    _print(payload)
    # A rejected call is a failed run here, even though the agent treats it as a
    # recoverable step and hands the error back to the brain.
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
