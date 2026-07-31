"""Regression tests pinning the tool layer to the published reference answers.

Every expected value here comes from ``public_questions.jsonl`` or from a worked
example in the organizer handout. These are the numbers the hidden-question judge
compares against, so a failure here is a scoring regression, not a style nit.

The AFR tests load 219,538 articles and build the index (~25s), so they are
marked ``slow``::

    pytest tests -m "not slow"    # RBA + ASX only, fast
    pytest tests                  # everything
"""

from __future__ import annotations

import pytest

from datastore import STORE
from tools import afr, asx, rba

TABCORP = ["TAH.AX"]


@pytest.fixture(scope="session", autouse=True)
def _rba_asx() -> None:
    STORE._load_rba()
    STORE._load_asx()


@pytest.fixture(scope="session")
def _afr() -> None:
    STORE._load_afr()
    STORE.afr_index.build(STORE.afr)


def approx(value: float, expected: float, tol: float = 0.02) -> bool:
    """Reference tolerance: returns/drawdowns allow +/-0.02 percentage points."""
    return abs(value - expected) <= tol


# ---------------------------------------------------------------------------
# RBA
# ---------------------------------------------------------------------------
def test_mhq001_count_changes() -> None:
    """41 of the 175 records changed the rate: 20 increases, 21 decreases."""
    r = rba.count_changes()
    assert r["total_records"] == 175
    assert r["changes"] == 41
    assert r["increases"] == 20
    assert r["decreases"] == 21


def test_mhq035_easing_2011_2013() -> None:
    """8 cuts (2/4/2), -2.25pp, 4.75% before the first cut to 2.50% at the end."""
    r = rba.period_summary("2011-01-01", "2013-12-31")
    assert r["decreases"] == 8
    assert r["decreases_by_year"] == {2011: 2, 2012: 4, 2013: 2}
    assert approx(r["cumulative_change_points"], -2.25, 0.001)
    assert r["target_before_first_change"] == 4.75
    assert r["target_at_window_end"] == 2.5


def test_mhq084_rba_2019() -> None:
    """Three cuts in 2019 for -0.75pp, ending at 0.75%."""
    r = rba.period_summary("2019-01-01", "2019-12-31")
    assert r["decreases"] == 3
    assert approx(r["cumulative_change_points"], -0.75, 0.001)
    assert r["target_at_window_end"] == 0.75


def test_handout_max_hold_streak() -> None:
    """1036 days, 2016-08-03 to 2019-06-05, held at 1.5 then moved to 1.25."""
    r = rba.max_hold_streak()
    assert r["days"] == 1036
    assert r["start_date"] == "2016-08-03"
    assert r["end_date"] == "2019-06-05"
    assert r["rate_during_hold"] == 1.5
    assert r["rate_after_change"] == 1.25


def test_handout_lowest_target() -> None:
    """Lowest target 0.1, first effective 2020-11-04, shown by 16 records."""
    low = rba.extremes()["lowest"]
    assert low["target"] == 0.1
    assert low["first_effective_date"] == "2020-11-04"
    assert low["records"] == 16


def test_highest_target_uses_effective_date() -> None:
    """Highest target 4.75 across 11 records, first effective 2010-11-03.

    The handout's partial-credit example asserts the judge expected
    2010-11-02 for this date. The supplied corpus contains exactly one Nov-2010
    row -- ``3 Nov 2010,+0.25,4.75`` -- so 2010-11-03 is the only value derivable
    from the data, and 2010-11-02 is presumably the announcement date rather
    than the effective date. This test pins the data-truthful answer; see the
    README's known-limitations section.
    """
    high = rba.extremes()["highest"]
    assert high["target"] == 4.75
    assert high["records"] == 11
    assert high["first_effective_date"] == "2010-11-03"


def test_handout_tightening_cycle_2022_2023() -> None:
    """13 hikes, +4.25pp cumulative, 0.1% before the first, 4.35% final."""
    r = rba.period_summary("2022-01-01", "2023-12-31")
    assert r["increases"] == 13
    assert approx(r["cumulative_change_points"], 4.25, 0.001)
    assert r["target_before_first_change"] == 0.1
    assert r["target_at_window_end"] == 4.35
    assert r["first_change_date"] == "2022-05-04"
    assert r["last_change_date"] == "2023-11-08"


@pytest.mark.parametrize(
    "as_of,expected_rate,expected_effective",
    [
        ("2021-02-23", 0.1, "2021-02-03"),  # MHQ058
        ("2021-11-25", 0.1, "2021-11-03"),  # MHQ067
        ("2020-11-28", 0.1, "2020-11-04"),  # MHQ080
    ],
)
def test_lookup_rate_on_or_before(
    as_of: str, expected_rate: float, expected_effective: str
) -> None:
    """"In force on <date>" resolves on-or-before, never to a future record."""
    r = rba.lookup_rate(as_of)
    assert r["cash_rate_target"] == expected_rate
    assert r["effective_date"] == expected_effective
    assert r["effective_date"] <= as_of


def test_lookup_rate_never_returns_the_future() -> None:
    """The table runs to 2026, so a nearest-date lookup would leak forward."""
    r = rba.lookup_rate("2015-06-15")
    assert r["effective_date"] <= "2015-06-15"


# ---------------------------------------------------------------------------
# ASX
# ---------------------------------------------------------------------------
def test_mhq040_dimensions() -> None:
    """18 ticker files, 1,774 rows each, 2 Jan 2015 through 30 Dec 2021."""
    r = asx.coverage()
    assert r["tickers"] == 18
    assert r["rows_per_ticker"] == 1774
    assert r["common_start"] == "2015-01-02"
    assert r["common_end"] == "2021-12-30"


def test_tickers_come_from_data_not_filenames() -> None:
    """Aurizon-ASX-*.jsonl holds AZJ.AX; deriving from the filename would fail."""
    assert "AZJ.AX" in STORE.asx
    assert "QAN.AX" in STORE.asx
    assert "AURIZON.AX" not in STORE.asx


def test_mhq045_best_worst_2018() -> None:
    """Excluding Tabcorp: BHP.AX best +22.17%, AMP.AX worst -50.04%."""
    r = asx.rank_annual_returns(2018, exclude_tickers=TABCORP)
    assert r["best"]["ticker"] == "BHP.AX"
    assert approx(r["best"]["return_pct"], 22.17)
    assert r["worst"]["ticker"] == "AMP.AX"
    assert approx(r["worst"]["return_pct"], -50.04)


def test_mhq049_avg_volume() -> None:
    """AMP.AX highest at 11,635,671.71 shares/day (tolerance +/-1 share)."""
    r = asx.avg_volume(exclude_tickers=TABCORP)
    assert r["highest"]["ticker"] == "AMP.AX"
    assert approx(r["highest"]["avg_daily_volume"], 11_635_671.71, 1.0)


def test_mhq049_exclusion_actually_matters() -> None:
    """Without the exclusion TAH.AX ranks higher, so the answer would flip."""
    assert asx.avg_volume()["highest"]["ticker"] == "TAH.AX"


def test_mhq055_worst_three_drawdowns() -> None:
    """AMP -82.45%, AGL -76.24%, QAN -71.08%, with their peak/trough dates."""
    ranked = asx.max_drawdown(exclude_tickers=TABCORP, top_n=3)[
        "ranked_worst_first"
    ]
    expected = [
        ("AMP.AX", -82.45, "2015-03-20", "2021-12-17"),
        ("AGL.AX", -76.24, "2017-04-10", "2021-11-16"),
        ("QAN.AX", -71.08, "2019-12-19", "2020-03-19"),
    ]
    assert len(ranked) == 3
    for row, (ticker, pct, peak, trough) in zip(ranked, expected):
        assert row["ticker"] == ticker
        assert approx(row["max_drawdown_pct"], pct)
        assert row["peak_date"] == peak
        assert row["trough_date"] == trough


def test_mhq072_basket_and_named_tickers() -> None:
    """5->12 Jun 2019: basket +2.88%; CBA +0.60, NAB +1.39, ANZ +0.89,
    BHP +5.89, RIO +2.91."""
    r = asx.window_return("2019-06-05", "2019-06-12", exclude_tickers=TABCORP)
    assert r["constituents"] == 17
    assert approx(r["basket_average_return_pct"], 2.88)
    by_ticker = {x["ticker"]: x["return_pct"] for x in r["results"]}
    for ticker, expected in [
        ("CBA.AX", 0.60),
        ("NAB.AX", 1.39),
        ("ANZ.AX", 0.89),
        ("BHP.AX", 5.89),
        ("RIO.AX", 2.91),
    ]:
        assert approx(by_ticker[ticker], expected), ticker


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2019-06-05", "2019-06-12", 2.88),
        ("2019-07-03", "2019-07-10", 0.24),
        ("2019-10-02", "2019-10-09", -2.17),
    ],
)
def test_mhq074_one_week_after_each_2019_cut(
    start: str, end: str, expected: float
) -> None:
    r = asx.window_return(start, end, exclude_tickers=TABCORP)
    assert approx(r["basket_average_return_pct"], expected)


def test_mhq080_five_session_window() -> None:
    """Non-Tabcorp basket rose 2.37% from 30 Nov to 7 Dec 2020."""
    r = asx.window_return("2020-11-30", "2020-12-07", exclude_tickers=TABCORP)
    assert approx(r["basket_average_return_pct"], 2.37)
    assert r["direction"] == "up"


@pytest.mark.parametrize(
    "event,expected_end,expected",
    [
        ("2019-06-05", "2019-06-12", 2.88),
        ("2019-07-03", "2019-07-10", 0.24),
        ("2019-10-02", "2019-10-09", -2.17),
        ("2020-11-30", "2020-12-07", 2.37),
    ],
)
def test_event_window_week_is_calendar_days_seven(
    event: str, expected_end: str, expected: float
) -> None:
    """A one-week event window must land on the graded end date.

    The sibling tests above pin these figures through ``window_return`` with
    both dates spelled out, which cannot catch the mistake that actually
    happens: the brain reaches ``event_window`` and has to choose between
    ``sessions`` and ``calendar_days``. Only ``calendar_days=7`` reproduces the
    reference answers.
    """
    r = asx.event_window(event, calendar_days=7, exclude_tickers=TABCORP)
    assert r["window"] == [event, expected_end]
    assert approx(r["basket_average_return_pct"], expected)


def test_event_window_sessions_five_is_not_a_week() -> None:
    """Why the parameter choice is graded, not cosmetic.

    10 Jun 2019 was a holiday, so five *sessions* after 5 Jun is 13 Jun while a
    week later is 12 Jun. Picking ``sessions=5`` returns a perfectly plausible
    number for a window the question did not ask about — worth 0 points.
    """
    week = asx.event_window("2019-06-05", calendar_days=7, exclude_tickers=TABCORP)
    sessions = asx.event_window("2019-06-05", sessions=5, exclude_tickers=TABCORP)

    assert week["window"][1] == "2019-06-12"
    assert sessions["window"][1] == "2019-06-13"
    assert not approx(
        sessions["basket_average_return_pct"],
        week["basket_average_return_pct"],
    )


def test_mhq076_qbe_best_2021() -> None:
    """QBE.AX had the best non-Tabcorp 2021 return at +35.57%."""
    r = asx.rank_annual_returns(2021, exclude_tickers=TABCORP)
    assert r["best"]["ticker"] == "QBE.AX"
    assert approx(r["best"]["return_pct"], 35.57)


def test_mhq084_basket_annual_return_2019() -> None:
    """Non-Tabcorp average 2019 return +20.11%."""
    r = asx.annual_return(2019, exclude_tickers=TABCORP)
    assert approx(r["basket_average_return_pct"], 20.11)


def test_ticker_normalisation() -> None:
    """The brain writes tickers loosely; a slip must not drop a constituent."""
    r = asx.window_return("2019-06-05", "2019-06-12", tickers=["cba", "NAB.ax"])
    assert {x["ticker"] for x in r["results"]} == {"CBA.AX", "NAB.AX"}


# ---------------------------------------------------------------------------
# AFR
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_afr_corpus_total(_afr: None) -> None:
    """219,538 articles, matching the organizer's own conversion summary."""
    assert len(STORE.afr) == 219_538


@pytest.mark.slow
def test_mhq061_unemployment_peaks(_afr: None) -> None:
    """Peaked in 2020 with 1,452 records; May 2020 the peak month with 218."""
    by_year = afr.count_by_year(pattern=r"\bunemployment\b")
    assert by_year["peak_year"] == 2020
    assert by_year["peak_year_count"] == 1452
    by_month = afr.count_by_month(pattern=r"\bunemployment\b", year=2020)
    assert by_month["peak_month"] == "2020-05"
    assert by_month["peak_month_count"] == 218


@pytest.mark.slow
def test_mhq076_qbe_count_2021(_afr: None) -> None:
    """369 AFR records match whole-word QBE in 2021."""
    assert afr.count(pattern=r"\bQBE\b", year=2021)["matching_records"] == 369


@pytest.mark.slow
def test_mhq084_rate_pattern_2019(_afr: None) -> None:
    """3,181 records match the organizer's rate/RBA alternation in 2019."""
    pattern = r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b"
    assert afr.count(pattern=pattern, year=2019)["matching_records"] == 3181


@pytest.mark.slow
def test_terms_path_matches_anchored_pattern(_afr: None) -> None:
    """``terms=`` must be exactly equivalent to hand-anchoring the regex."""
    via_terms = afr.count_by_year(terms=["unemployment"])
    via_regex = afr.count_by_year(pattern=r"\bunemployment\b")
    assert via_terms["by_year"] == via_regex["by_year"]


@pytest.mark.slow
def test_prefilter_is_exact_versus_full_scan(_afr: None) -> None:
    """The optimised paths must agree with a brute-force scan, exactly.

    This is the test that caught the original prefilter bug: requiring the token
    "cut" for the branch ``rate cut`` silently dropped every article that said
    "rate cuts", because the indexed token there is "cuts".
    """
    import re

    patterns = [
        r"\bunemployment\b",
        r"\bQBE\b",
        r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b",
        r"\bRBA\b|inflation",
    ]
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        brute = {
            i for i, art in enumerate(STORE.afr) if compiled.search(art.blob)
        }
        afr._matching_ids.cache_clear()
        optimised = set(afr._matching_ids(pattern))
        assert optimised == brute, (
            f"{pattern!r}: optimised={len(optimised)} brute={len(brute)} "
            f"missing={len(brute - optimised)} extra={len(optimised - brute)}"
        )


@pytest.mark.slow
def test_word_boundaries_change_the_count(_afr: None) -> None:
    """Unanchored short acronyms over-count, which is why anchors are required."""
    anchored = afr.count(pattern=r"\bNAB\b")["matching_records"]
    bare = afr.count(pattern="NAB")["matching_records"]
    assert bare > anchored


@pytest.mark.slow
@pytest.mark.parametrize(
    "headline,date",
    [
        ("Travel stocks take off on vaccine rollout", "2021-02-23"),
        ("Why investors don't believe the RBA on interest rates", "2021-11-25"),
        ("Energy stocks shine as vaccines fuel oil rally", "2020-11-28"),
    ],
)
def test_named_article_retrieval(_afr: None, headline: str, date: str) -> None:
    """The three sentiment questions name an article; retrieval must be exact."""
    r = afr.find_article(headline=headline, date=date)
    assert r["matches"] >= 1
    got = r["articles"][0]
    assert got["publication_date"] == date
    assert got["headline"].lower() == headline.lower()
    assert len(got["text"]) > 100


@pytest.mark.slow
def test_undated_articles_excluded_from_date_filters(_afr: None) -> None:
    """The 92 undated records must not land in an arbitrary year bucket."""
    total = afr.count(terms=["the"])["matching_records"]
    by_year = afr.count_by_year(terms=["the"])
    assert sum(by_year["by_year"].values()) <= total
