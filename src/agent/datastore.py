"""Eager, read-only in-memory store for the three approved datasets.

Loaded once at process start; every tool reads immutable structures afterwards,
so three concurrent ``/query`` requests share the data with no locking and no
possibility of cross-request state bleed.

Dataset facts verified against the supplied corpus:

===========  ==================================================================
RBA          ``RBA-rates.csv`` — 175 rows, ``Effective Date`` formatted
             ``3 Feb 2010`` (**not** ISO). 41 rows carry a non-zero change:
             20 increases, 21 decreases. The companion ``RBA-rates.jsonl``
             carries a UTF-8 BOM, so both files are opened ``utf-8-sig``.
             The table runs to ``17 Jun 2026``, i.e. **beyond** the ASX and AFR
             coverage, which is what makes coverage checks load-bearing.
ASX          18 files, 1,774 rows each, 2015-01-02 .. 2021-12-30. Filenames are
             company names but every row carries the real ``ticker``
             (``Aurizon-ASX-*.jsonl`` -> ``AZJ.AX``), so tickers are always read
             from the data, never derived from the filename.
AFR          85 ``AFR_*.jsonl`` files, 219,538 articles. ``PUBLICATIONDATE`` is
             compact ``YYYYMMDD``. ``_conversion_summary.json`` sits in the same
             directory and is **not** article data, hence the ``AFR_*.jsonl``
             glob rather than ``*.json*``.
===========  ==================================================================
"""

from __future__ import annotations

import array
import csv
import glob
import json
import logging
import os
import re
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RbaRow:
    """One RBA cash-rate decision record."""

    effective: date
    change: float
    target: float

    @property
    def iso(self) -> str:
        return self.effective.isoformat()


@dataclass(frozen=True, slots=True)
class AsxBar:
    """One daily OHLCV bar for one ticker."""

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class AsxSeries:
    """A ticker's full price history, ascending by date."""

    ticker: str
    bars: tuple[AsxBar, ...]
    _days: tuple[date, ...]

    def close_on(self, day: date) -> float | None:
        """Close on an exact trading date, or None if the market was shut."""
        i = bisect_right(self._days, day) - 1
        if i < 0 or self._days[i] != day:
            return None
        return self.bars[i].close

    def close_on_or_before(self, day: date) -> tuple[date, float] | None:
        """Most recent close on or before ``day`` — for non-trading dates."""
        i = bisect_right(self._days, day) - 1
        if i < 0:
            return None
        return self._days[i], self.bars[i].close

    def slice_between(self, start: date, end: date) -> tuple[AsxBar, ...]:
        lo = bisect_right(self._days, start) - 1
        lo = lo if lo >= 0 and self._days[lo] == start else lo + 1
        hi = bisect_right(self._days, end)
        return self.bars[max(lo, 0) : hi]


@dataclass(frozen=True, slots=True)
class AfrArticle:
    """One AFR article, with a precomputed search surface.

    ``blob`` is the lowercased concatenation of HEADLINE, SUBHEAD, INTRO and
    TEXT separated by newlines. Setup_Instructions.md makes searching those four
    fields *combined* non-negotiable: searching only the headline or only the
    body yields counts that will not match the reference answers. Matching is
    once-per-record, which falls out of running one ``search`` over ``blob``.
    """

    headline: str
    #: None for the 92 records that ship with an empty PUBLICATIONDATE. They are
    #: kept so the corpus total still matches the organizer's conversion summary
    #: (219,538), but they are excluded from any date-filtered query rather than
    #: being silently bucketed into a wrong year.
    publication: date | None
    blob: str


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_RBA_DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%Y-%m-%d")


def parse_rba_date(raw: str) -> date:
    """Parse ``3 Feb 2010`` (and tolerate ISO, in case the corpus changes)."""
    text = raw.strip()
    for fmt in _RBA_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised RBA date: {raw!r}")


def parse_flexible_date(raw: str | date) -> date:
    """Parse any date shape the brain might emit into a ``date``.

    The brain is a language model, so it will not reliably pick one format.
    Accepting ISO, compact, and human forms here is cheaper than trying to
    constrain it through the prompt, and a rejected date costs a whole question.
    """
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    if not text:
        raise ValueError("empty date")
    formats = (
        "%Y-%m-%d",
        "%Y%m%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text.replace(",", ""), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {raw!r}")


# ---------------------------------------------------------------------------
# AFR regex prefilter
# ---------------------------------------------------------------------------
# A naive implementation re-scans all 219,538 article blobs (~800 MB of text)
# for every AFR question. That is several seconds per call single-threaded, and
# the harness sends three questions at once against a 60-second penalty
# threshold. So we build a token -> postings index at start-up and use it to
# shrink the candidate set before running the real regex.
#
# The filter must never drop a true match. For a top-level alternation, a record
# can only match branch B if it contains every literal token B requires; so
#     candidates = union over branches of (intersection of that branch's tokens)
# is a guaranteed superset. The real regex then confirms each candidate, which
# keeps results exactly identical to a full scan.
#
# Tokens governed by ``?`` or ``*`` are optional and therefore excluded from a
# branch's required set (``rates?`` contributes nothing, not ``rates``). If any
# branch yields no usable token, we fall back to scanning everything.

# Maximal runs of alphanumerics, and nothing else. Deliberately NOT
# apostrophe-aware: `\bQBE\b` matches inside "QBE's", so indexing "qbe's" as one
# token would hide that record from the prefilter and undercount. Splitting on
# every non-alphanumeric guarantees the index is a superset of what any
# `\b`-anchored pattern can match — if `\bfoo\b` matches, "foo" is by definition
# a complete alphanumeric run, hence a token.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LITERAL_RUN_RE = re.compile(r"[A-Za-z0-9]+")
#: Regex metacharacters that make literal extraction from a branch unsafe.
_UNSAFE_META = set("[](){}\\.*+?^$|")


def _split_top_level_alternation(pattern: str) -> list[str] | None:
    """Split ``a|b|c`` on top-level ``|`` only, respecting groups and classes."""
    branches: list[str] = []
    depth = 0
    in_class = False
    current: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            current.append(pattern[i : i + 2])
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
            current.append(ch)
        elif ch == "[":
            in_class = True
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "|" and depth == 0:
            branches.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if in_class or depth != 0:
        return None
    branches.append("".join(current))
    return branches


def _required_tokens(branch: str) -> set[str] | None:
    """Literal lowercase tokens a record must contain to match ``branch``.

    Returns None when nothing can be required safely.

    A literal is only a valid requirement when the pattern forces it to align
    with a whole word on *both* sides. Otherwise it can match a fragment of a
    longer word, and the fragment is not what got indexed:

        branch "rate cut"  matches the text "rate cuts"
        -> the indexed token is "cuts", so requiring "cut" would drop that
           record even though it is a true match

    The same applies on the left ("prorate cut" contains "rate cut"), so a
    literal sitting at a branch edge is never safe — a branch boundary is an
    alternation, not a word boundary. Only ``\\b`` or a literal non-alphanumeric
    character (a space, a hyphen) delimits a token, and a trailing quantifier
    makes it optional. Being conservative here costs speed; being wrong costs
    correctness.
    """
    required: set[str] = set()
    i = 0
    n = len(branch)
    while i < n:
        ch = branch[i]
        if ch == "\\":
            i += 2
            continue
        if ch in _UNSAFE_META:
            # Groups and classes: give up rather than risk dropping a match.
            if ch in "([":
                return None
            i += 1
            continue
        run = _LITERAL_RUN_RE.match(branch, i)
        if not run:
            i += 1
            continue
        token = run.group(0).lower()
        start, end = run.start(), run.end()

        left_delimited = start >= 2 and branch[start - 2 : start] == "\\b"
        if not left_delimited and start >= 1:
            prev = branch[start - 1]
            left_delimited = not prev.isalnum() and prev not in _UNSAFE_META

        right_delimited = branch[end : end + 2] == "\\b"
        if not right_delimited and end < n:
            nxt = branch[end]
            right_delimited = not nxt.isalnum() and nxt not in _UNSAFE_META

        if left_delimited and right_delimited and len(token) > 1:
            required.add(token)
        i = end
    return required or None


def _literal_screens(pattern: str) -> list[str] | None:
    """One guaranteed-literal substring per top-level branch, or None.

    Where :func:`_required_tokens` needs whole words, this needs only a literal
    run, so it succeeds on the phrase patterns that defeat the token index:

        "rate cut"        -> "rate cut"
        "interest rates?" -> "interest rate"   (the optional 's' is dropped)
        "\\bRBA\\b"         -> "rba"             (assertions carry no text)

    Any record matching a branch must contain that branch's literal, so
    screening on them is a guaranteed superset. It matters because plain
    substring search is C-level and roughly an order of magnitude faster than
    running the alternation as a regex over the whole corpus.

    Returns None if any branch has no usable literal, since a branch we cannot
    screen could match anything.
    """
    branches = _split_top_level_alternation(pattern)
    if not branches:
        return None
    screens: list[str] = []
    for branch in branches:
        literal: list[str] = []
        i = 0
        n = len(branch)
        while i < n:
            ch = branch[i]
            if ch == "\\":
                nxt = branch[i + 1 : i + 2]
                if nxt and not nxt.isalnum():
                    # An escaped literal such as \. or \- contributes its char.
                    literal.append(nxt)
                # \b, \d, \w and friends contribute no text.
                i += 2
                continue
            if ch in _UNSAFE_META:
                break  # a group, class or quantifier ends the guaranteed run
            if i + 1 < n and branch[i + 1] in "?*":
                # This character is optional, so it is not guaranteed. Anything
                # after it is still literal, but no longer contiguous with what
                # we have, so stop here and keep the prefix.
                break
            literal.append(ch)
            i += 1
        text = "".join(literal).strip().lower()
        if len(text) < 2:
            return None
        screens.append(text)
    return screens or None


@dataclass(frozen=True, slots=True)
class Prefilter:
    """Candidate record ids for a pattern, or None to mean 'scan everything'."""

    candidates: frozenset[int] | None
    exact: bool


#: Separator joining article blobs into one contiguous corpus. NUL cannot occur
#: in the source text, so it cannot appear inside a legitimate match.
_CORPUS_SEP = "\x00"


class AfrIndex:
    """Token postings plus a contiguous corpus, for two search strategies."""

    def __init__(self) -> None:
        self._postings: dict[str, array.array] = {}
        self._corpus: str = ""
        self._starts: array.array = array.array("q")
        self._ends: array.array = array.array("q")
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def build(self, articles: Sequence[AfrArticle]) -> None:
        started = time.monotonic()
        # Accumulate in plain lists (cheap append) then freeze to array('i') to
        # keep the resident size down — postings dominate this structure.
        staging: dict[str, list[int]] = {}
        for idx, art in enumerate(articles):
            for token in set(_TOKEN_RE.findall(art.blob)):
                staging.setdefault(token, []).append(idx)
        self._postings = {t: array.array("i", ids) for t, ids in staging.items()}

        # One big string, plus each record's span within it. A single finditer
        # over this beats 219k individual search calls by roughly 4x — the
        # per-call overhead dominates at that count.
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for art in articles:
            starts.append(cursor)
            cursor += len(art.blob)
            ends.append(cursor)
            cursor += len(_CORPUS_SEP)
        self._corpus = _CORPUS_SEP.join(art.blob for art in articles)
        self._starts = array.array("q", starts)
        self._ends = array.array("q", ends)

        self._ready = True
        logger.info(
            "AFR index built: %d tokens, %.0f MB corpus, %d articles in %.1fs",
            len(self._postings),
            len(self._corpus) / 1e6,
            len(articles),
            time.monotonic() - started,
        )

    def screen(self, literals: Sequence[str]) -> frozenset[int] | None:
        """Record ids containing at least one of ``literals``.

        Uses ``str.find`` rather than a regex. After a hit, the cursor jumps to
        the end of that record, so the loop runs once per *matching record*
        rather than once per occurrence.
        """
        if not self._ready:
            return None
        corpus, starts, ends = self._corpus, self._starts, self._ends
        hits: set[int] = set()
        for literal in literals:
            pos = corpus.find(literal)
            while pos != -1:
                idx = bisect_right(starts, pos) - 1
                if idx >= 0:
                    hits.add(idx)
                    pos = corpus.find(literal, max(pos + 1, ends[idx]))
                else:
                    pos = corpus.find(literal, pos + 1)
        return frozenset(hits)

    def scan(self, pattern: str) -> tuple[int, ...] | None:
        """Exact once-per-record match ids via one pass over the corpus.

        Returns None when the index has not been built, so the caller can fall
        back to a per-record loop. A match straddling the NUL separator is
        discarded: it belongs to no single record and would be a false positive.
        """
        if not self._ready:
            return None
        starts, ends = self._starts, self._ends
        hits: set[int] = set()
        for match in re.finditer(pattern, self._corpus, re.IGNORECASE):
            idx = bisect_right(starts, match.start()) - 1
            if idx < 0:
                continue
            if match.end() <= ends[idx]:
                hits.add(idx)
        return tuple(sorted(hits))

    def prefilter(self, pattern: str) -> Prefilter:
        """Shrink the candidate set for ``pattern``, or admit defeat."""
        if not self._ready:
            return Prefilter(None, exact=False)
        branches = _split_top_level_alternation(pattern)
        if not branches:
            return Prefilter(None, exact=False)

        union: set[int] = set()
        for branch in branches:
            tokens = _required_tokens(branch)
            if not tokens:
                # This branch could match anything we cannot predict.
                return Prefilter(None, exact=False)
            postings = [self._postings.get(t) for t in tokens]
            if any(p is None for p in postings):
                continue  # a required token is absent corpus-wide
            postings.sort(key=len)  # intersect smallest-first
            acc = set(postings[0])
            for p in postings[1:]:
                acc &= set(p)
                if not acc:
                    break
            union |= acc
        return Prefilter(frozenset(union), exact=True)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class DataStore:
    """Holds all three datasets. Call :meth:`load` once before serving."""

    def __init__(self) -> None:
        self.rba: tuple[RbaRow, ...] = ()
        self.asx: dict[str, AsxSeries] = {}
        self.afr: tuple[AfrArticle, ...] = ()
        self.afr_index = AfrIndex()
        self._loaded = threading.Event()
        self._load_error: str | None = None
        self._stats: dict[str, object] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._loaded.is_set() and self._load_error is None

    @property
    def error(self) -> str | None:
        return self._load_error

    @property
    def stats(self) -> dict[str, object]:
        return dict(self._stats)

    def load(self) -> None:
        """Load every dataset. Idempotent; safe to call from one thread."""
        with self._lock:
            if self._loaded.is_set():
                return
            started = time.monotonic()
            try:
                self._load_rba()
                self._load_asx()
                self._load_afr()
                if config.AFR_BUILD_INDEX and self.afr:
                    self.afr_index.build(self.afr)
            except Exception as exc:  # noqa: BLE001 - surfaced through /health
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("dataset load failed")
            self._stats["load_seconds"] = round(time.monotonic() - started, 1)
            self._loaded.set()

    # -- RBA ---------------------------------------------------------------
    def _load_rba(self) -> None:
        path = config.RBA_DIR / "RBA-rates.csv"
        if not path.exists():
            raise FileNotFoundError(f"RBA CSV not found at {path}")
        rows: list[RbaRow] = []
        # utf-8-sig: harmless when no BOM is present (the CSV), required when
        # one is (the companion JSONL).
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                effective = raw.get("Effective Date")
                if not effective or not effective.strip():
                    continue
                rows.append(
                    RbaRow(
                        effective=parse_rba_date(effective),
                        change=float(raw["Change % points"]),
                        target=float(raw["Cash rate target%"]),
                    )
                )
        rows.sort(key=lambda r: r.effective)
        self.rba = tuple(rows)
        self._stats["rba_records"] = len(rows)
        self._stats["rba_coverage"] = (
            f"{rows[0].iso}..{rows[-1].iso}" if rows else "empty"
        )
        logger.info("RBA loaded: %d records", len(rows))

    # -- ASX ---------------------------------------------------------------
    def _load_asx(self) -> None:
        paths = sorted(glob.glob(str(config.ASX_DIR / "*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"no ASX jsonl files under {config.ASX_DIR}")
        series: dict[str, list[AsxBar]] = {}
        for path in paths:
            with open(path, encoding="utf-8-sig") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    # The ticker always comes from the record, never the
                    # filename: Aurizon-ASX-*.jsonl holds AZJ.AX.
                    series.setdefault(rec["ticker"], []).append(
                        AsxBar(
                            day=parse_flexible_date(rec["date"]),
                            open=float(rec["open"]),
                            high=float(rec["high"]),
                            low=float(rec["low"]),
                            close=float(rec["close"]),
                            volume=float(rec["volume"]),
                        )
                    )
        self.asx = {}
        for ticker, bars in series.items():
            bars.sort(key=lambda b: b.day)
            self.asx[ticker] = AsxSeries(
                ticker=ticker,
                bars=tuple(bars),
                _days=tuple(b.day for b in bars),
            )
        counts = {len(s.bars) for s in self.asx.values()}
        self._stats["asx_tickers"] = len(self.asx)
        self._stats["asx_rows_per_ticker"] = (
            counts.pop() if len(counts) == 1 else sorted(counts)
        )
        logger.info("ASX loaded: %d tickers", len(self.asx))

    # -- AFR ---------------------------------------------------------------
    def _load_afr(self) -> None:
        # AFR_*.jsonl, never *.json* — _conversion_summary.json lives here too
        # and is metadata, not article data.
        paths = sorted(glob.glob(str(config.AFR_DIR / "AFR_*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"no AFR jsonl files under {config.AFR_DIR}")
        if config.AFR_MAX_FILES:
            paths = paths[: config.AFR_MAX_FILES]
            logger.warning(
                "AFR_MAX_FILES=%d — loading a SUBSET of the corpus. Counts will "
                "not match reference answers. Unset before evaluation.",
                config.AFR_MAX_FILES,
            )
        articles: list[AfrArticle] = []
        undated = 0
        for path in paths:
            with open(path, encoding="utf-8-sig") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    blob = "\n".join(
                        (
                            rec.get("HEADLINE") or "",
                            rec.get("SUBHEAD") or "",
                            rec.get("INTRO") or "",
                            rec.get("TEXT") or "",
                        )
                    ).lower()
                    raw_date = str(rec.get("PUBLICATIONDATE") or "").strip()
                    try:
                        published = (
                            parse_flexible_date(raw_date) if raw_date else None
                        )
                    except ValueError:
                        published = None
                    if published is None:
                        undated += 1
                    articles.append(
                        AfrArticle(
                            headline=rec.get("HEADLINE") or "",
                            publication=published,
                            blob=blob,
                        )
                    )
        self.afr = tuple(articles)
        self._stats["afr_articles"] = len(articles)
        self._stats["afr_files"] = len(paths)
        self._stats["afr_undated"] = undated
        if undated:
            logger.warning(
                "%d AFR articles have no PUBLICATIONDATE; they count toward "
                "corpus-wide totals but are excluded from year/month filters",
                undated,
            )
        if articles:
            days = [a.publication for a in articles if a.publication]
            self._stats["afr_coverage"] = (
                f"{min(days).isoformat()}..{max(days).isoformat()}"
            )
        # Cross-check against the organizer's own conversion summary when it is
        # present: a mismatch means we dropped or double-counted articles, which
        # would silently corrupt every AFR count.
        summary = config.AFR_DIR / "_conversion_summary.json"
        if summary.exists() and not config.AFR_MAX_FILES:
            try:
                expected = json.loads(summary.read_text(encoding="utf-8-sig"))
                total = int(expected.get("totalArticles", 0))
                self._stats["afr_expected_articles"] = total
                if total and total != len(articles):
                    logger.error(
                        "AFR count mismatch: loaded %d, summary says %d",
                        len(articles),
                        total,
                    )
                    self._stats["afr_count_mismatch"] = True
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("could not read %s", summary)
        logger.info("AFR loaded: %d articles", len(articles))

    # -- coverage ----------------------------------------------------------
    def coverage(self) -> dict[str, dict[str, str | int]]:
        """Per-dataset date coverage.

        This is load-bearing, not decorative: RBA extends to 2026 while ASX and
        AFR stop at 2021, and the highest-value hard question in the public set
        (MHQ090) is answered correctly only by *refusing* a three-dataset join
        on that basis.
        """
        out: dict[str, dict[str, str | int]] = {}
        if self.rba:
            out["rba"] = {
                "records": len(self.rba),
                "start": self.rba[0].iso,
                "end": self.rba[-1].iso,
            }
        if self.asx:
            days = [d for s in self.asx.values() for d in (s._days[0], s._days[-1])]
            out["asx"] = {
                "tickers": len(self.asx),
                "rows_per_ticker": len(next(iter(self.asx.values())).bars),
                "start": min(days).isoformat(),
                "end": max(days).isoformat(),
            }
        if self.afr:
            days = [a.publication for a in self.afr if a.publication]
            out["afr"] = {
                "articles": len(self.afr),
                "start": min(days).isoformat(),
                "end": max(days).isoformat(),
            }
        return out

    def tickers(self, exclude: Iterable[str] | None = None) -> list[str]:
        """Sorted tickers, optionally excluding some (case/suffix tolerant)."""
        drop = {normalise_ticker(t) for t in (exclude or ())}
        return sorted(t for t in self.asx if normalise_ticker(t) not in drop)


def normalise_ticker(raw: str) -> str:
    """``bhp`` / ``BHP.AX`` / ``bhp.ax`` -> ``BHP.AX``.

    The brain writes tickers inconsistently; the reference answers use the
    ``.AX`` suffix. Normalising on the way in means a formatting slip does not
    silently drop a constituent from a basket.
    """
    text = str(raw).strip().upper()
    if not text:
        return text
    return text if text.endswith(".AX") else f"{text}.AX"


#: Process-wide singleton. Populated by :func:`load_store` at server start-up.
STORE = DataStore()


def load_store() -> DataStore:
    STORE.load()
    return STORE
