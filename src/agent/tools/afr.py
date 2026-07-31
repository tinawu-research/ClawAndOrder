"""Pattern counting and article retrieval over the AFR news corpus.

Setup_Instructions.md marks four rules non-negotiable, because scores are
computed by running the same tool calls against the same data:

1. Match across ``HEADLINE`` + ``SUBHEAD`` + ``INTRO`` + ``TEXT`` **combined**.
2. Case-insensitive.
3. **Once per record** — an article counts once no matter how often the term
   appears, in however many fields.
4. Whole-word searches need boundary anchors (``\\bNAB\\b``, not ``NAB``), or
   short acronyms match inside unrelated words and inflate the count.

Rules 1–3 are structural here: the search surface is a single pre-lowercased
blob per article and every count is a boolean ``search`` over it, so there is no
way for a caller to get them wrong. Rule 4 is the caller's to get right, so this
module offers ``terms=`` — which anchors each term for you — alongside the raw
``pattern=`` the organizer's reference derivations use.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any, Sequence

from datastore import (
    STORE,
    AfrArticle,
    _literal_screens as build_literal_screens,
)
from tools.dateparse import parse_from, parse_to
from tools.rba import MetricError

#: Cap on returned article bodies, to keep a tool result inside the brain's
#: context window. Counting is never capped.
MAX_RETURNED_ARTICLES = 5
SNIPPET_CHARS = 1200


def _articles() -> tuple[AfrArticle, ...]:
    if not STORE.afr:
        raise MetricError("AFR dataset is not loaded")
    return STORE.afr


def build_pattern(
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    whole_word: bool = True,
) -> str:
    """Resolve the caller's intent into a single regex string.

    ``pattern`` is used verbatim — that is what the reference derivations
    specify, e.g. ``interest rates?|cash rate|rate cut|rate hike|\\bRBA\\b``.
    ``terms`` is the safe path: each term is escaped and, by default, wrapped in
    word boundaries before being joined into an alternation.
    """
    if pattern and terms:
        raise MetricError("pass either pattern or terms, not both")
    if pattern:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise MetricError(f"invalid regex {pattern!r}: {exc}") from exc
        return pattern
    if not terms:
        raise MetricError("one of pattern or terms is required")
    items = (
        [t.strip() for t in terms.split(",")]
        if isinstance(terms, str)
        else [str(t).strip() for t in terms]
    )
    items = [t for t in items if t]
    if not items:
        raise MetricError("terms is empty")
    escaped = [re.escape(t) for t in items]
    if not whole_word:
        return "|".join(escaped)
    # Anchor each alternative separately rather than wrapping the group:
    # \ba\b|\bb\b is equivalent to \b(?:a|b)\b but keeps every branch free of
    # groups, which is what lets the token prefilter analyse it. The grouped
    # form forces a full-corpus scan.
    return "|".join(rf"\b{term}\b" for term in escaped)


@lru_cache(maxsize=256)
def _matching_ids(pattern: str) -> tuple[int, ...]:
    """Record ids whose combined text matches ``pattern``, once per record.

    Cached: a multi-part question often re-uses the same pattern across a total,
    a per-year split and a per-month split, and the harness may send related
    questions concurrently.
    """
    articles = _articles()
    compiled = re.compile(pattern, re.IGNORECASE)
    search = compiled.search

    # Strategy 1: the token index proved a candidate superset. Verify each with
    # the real regex, so the result is identical to a full scan.
    prefilter = STORE.afr_index.prefilter(pattern)
    if prefilter.candidates is not None:
        return tuple(
            sorted(i for i in prefilter.candidates if search(articles[i].blob))
        )

    # Strategy 2: no safe whole-word token, but every branch has a guaranteed
    # literal (typical of phrase alternations). Screen on those with plain
    # substring search, then verify with the regex.
    screens = build_literal_screens(pattern)
    if screens:
        candidates = STORE.afr_index.screen(screens)
        if candidates is not None:
            return tuple(
                sorted(i for i in candidates if search(articles[i].blob))
            )

    # Strategy 3: nothing can be screened. One regex pass over the contiguous
    # corpus, which is still exact.
    scanned = STORE.afr_index.scan(pattern)
    if scanned is not None:
        return scanned

    # Strategy 3: index unavailable — correct, just slowest.
    return tuple(i for i, art in enumerate(articles) if search(art.blob))


def _filtered(
    pattern: str,
    year: int | None = None,
    month: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[int]:
    ids = _matching_ids(pattern)
    articles = _articles()
    try:
        start = parse_from(date_from) if date_from else None
        end = parse_to(date_to) if date_to else None
    except ValueError as exc:
        raise MetricError(str(exc)) from exc
    date_filtered = bool(year or month or start or end)
    out = []
    for i in ids:
        day = articles[i].publication
        if day is None:
            # Undated records cannot satisfy a date filter. They still appear in
            # unfiltered totals, so an undated match is never lost — only
            # excluded from the bucket it cannot be assigned to.
            if date_filtered:
                continue
            out.append(i)
            continue
        if year and day.year != int(year):
            continue
        if month and day.month != int(month):
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        out.append(i)
    return out


def coverage() -> dict[str, Any]:
    """Article count and publication-date span of the corpus."""
    articles = _articles()
    days = [a.publication for a in articles if a.publication]
    return {
        "metric": "coverage",
        "articles": len(articles),
        "dated_articles": len(days),
        "undated_articles": len(articles) - len(days),
        "start": min(days).isoformat(),
        "end": max(days).isoformat(),
        "index_ready": STORE.afr_index.ready,
    }


def count(
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    whole_word: bool = True,
    year: int | None = None,
    month: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Number of articles matching the pattern, counted once per record."""
    resolved = build_pattern(pattern, terms, whole_word)
    ids = _filtered(resolved, year, month, date_from, date_to)
    total = len(_articles())
    return {
        "metric": "count",
        "pattern": resolved,
        "matching_records": len(ids),
        "corpus_articles": total,
        "share_pct": round(100.0 * len(ids) / total, 4) if total else 0.0,
        "filters": {
            "year": int(year) if year else None,
            "month": int(month) if month else None,
            "date_from": date_from,
            "date_to": date_to,
        },
        "match_rules": (
            "case-insensitive; HEADLINE+SUBHEAD+INTRO+TEXT combined; "
            "once per record"
        ),
    }


def count_by_year(
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    whole_word: bool = True,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Matching-article counts grouped by publication year, with the peak.

    Accepts an optional date window, so "coverage by year between the two rate
    cycles" is one call rather than a per-year fan-out.
    """
    resolved = build_pattern(pattern, terms, whole_word)
    articles = _articles()
    ids = (
        _filtered(resolved, date_from=date_from, date_to=date_to)
        if (date_from or date_to)
        else _matching_ids(resolved)
    )
    tally = Counter(
        articles[i].publication.year for i in ids if articles[i].publication
    )
    ordered = dict(sorted(tally.items()))
    peak = max(tally.items(), key=lambda kv: (kv[1], -kv[0])) if tally else None
    return {
        "metric": "count_by_year",
        "pattern": resolved,
        "window": [date_from, date_to] if (date_from or date_to) else None,
        "by_year": ordered,
        "total": sum(tally.values()),
        "peak_year": peak[0] if peak else None,
        "peak_year_count": peak[1] if peak else None,
    }


def count_by_month(
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    whole_word: bool = True,
    year: int | None = None,
) -> dict[str, Any]:
    """Counts grouped by ``YYYY-MM``, with the peak month called out."""
    resolved = build_pattern(pattern, terms, whole_word)
    articles = _articles()
    ids = _filtered(resolved, year=year)
    tally = Counter(
        f"{articles[i].publication.year:04d}-{articles[i].publication.month:02d}"
        for i in ids
        if articles[i].publication
    )
    ordered = dict(sorted(tally.items()))
    peak = max(tally.items(), key=lambda kv: (kv[1], kv[0])) if tally else None
    return {
        "metric": "count_by_month",
        "pattern": resolved,
        "year": int(year) if year else None,
        "by_month": ordered,
        "total": sum(tally.values()),
        "peak_month": peak[0] if peak else None,
        "peak_month_count": peak[1] if peak else None,
    }


def share(
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    whole_word: bool = True,
    year: int | None = None,
) -> dict[str, Any]:
    """Matching articles as a percentage of the corpus (or of one year)."""
    resolved = build_pattern(pattern, terms, whole_word)
    articles = _articles()
    matching = _filtered(resolved, year=year)
    denominator = (
        sum(1 for a in articles if a.publication and a.publication.year == int(year))
        if year
        else len(articles)
    )
    return {
        "metric": "share",
        "pattern": resolved,
        "year": int(year) if year else None,
        "matching_records": len(matching),
        "denominator": denominator,
        "share_pct": (
            round(100.0 * len(matching) / denominator, 4) if denominator else 0.0
        ),
    }


def _normalise_headline(text: str) -> str:
    """Collapse case, punctuation and whitespace for headline comparison.

    Headlines round-trip through the model, which changes curly quotes to
    straight ones and drops the odd hyphen. Comparing on a normalised form keeps
    an exact-article lookup from failing on cosmetics.
    """
    lowered = str(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def find_article(
    headline: str | None = None,
    date: str | None = None,
    pattern: str | None = None,
    terms: Sequence[str] | str | None = None,
    limit: int = MAX_RETURNED_ARTICLES,
    max_chars: int = SNIPPET_CHARS,
) -> dict[str, Any]:
    """Retrieve article text — by exact headline+date, or by pattern.

    The article-sentiment questions name a headline and a publication date, so
    the primary path is an exact lookup. Sentiment classification is *not* done
    here: this tool returns evidence, and the fine-tuned domain model classifies
    it during synthesis.
    """
    articles = _articles()
    try:
        day = parse_to(date) if date else None
    except ValueError as exc:
        raise MetricError(str(exc)) from exc
    # One article is short; a whole page of them is not. Widening the snippet is
    # allowed, but the total returned text stays bounded or it crowds the tool
    # results already sitting in the brain's context.
    width = max(200, min(int(max_chars), 20_000))
    hits: list[int] = []

    if headline:
        needle = _normalise_headline(headline)
        exact = [
            i
            for i, a in enumerate(articles)
            if _normalise_headline(a.headline) == needle
            and (day is None or a.publication == day)
        ]
        if not exact and day is not None:
            # Retry without the date before falling back to substring matching:
            # a date mismatch is more likely than a headline mismatch.
            exact = [
                i
                for i, a in enumerate(articles)
                if _normalise_headline(a.headline) == needle
            ]
        if not exact:
            exact = [
                i
                for i, a in enumerate(articles)
                if needle and needle in _normalise_headline(a.headline)
                and (day is None or a.publication == day)
            ]
        hits = exact
    elif pattern or terms:
        resolved = build_pattern(pattern, terms)
        hits = _filtered(
            resolved,
            date_from=day.isoformat() if day else None,
            date_to=day.isoformat() if day else None,
        )
    elif day is not None:
        hits = [i for i, a in enumerate(articles) if a.publication == day]
    else:
        raise MetricError("find_article needs a headline, a pattern or a date")

    capped = hits[: max(1, int(limit))]
    return {
        "tool": "retrieve",
        "query": {"headline": headline, "date": date, "pattern": pattern},
        "matches": len(hits),
        "returned": len(capped),
        "articles": [
            {
                "headline": articles[i].headline,
                "publication_date": (
                    articles[i].publication.isoformat()
                    if articles[i].publication
                    else None
                ),
                # blob is the lowercased HEADLINE/SUBHEAD/INTRO/TEXT surface;
                # truncated so a long feature cannot crowd out the tool results
                # already in the brain's context.
                "text": articles[i].blob[:width],
                "truncated": len(articles[i].blob) > width,
            }
            for i in capped
        ],
    }


METRICS = {
    "coverage": coverage,
    "count": count,
    "count_by_year": count_by_year,
    "count_by_month": count_by_month,
    "share": share,
}
