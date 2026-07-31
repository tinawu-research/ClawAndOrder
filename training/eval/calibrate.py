"""Validate the judge against known labels before trusting any model delta.

The judge here is a reimplementation of the organizers' grader. Reporting a
base-vs-fine-tuned improvement measured with an uncalibrated instrument is
worth nothing, so this runs first and gates everything downstream.

Three sources of free ground truth:

Reference positives
    Every public question's own ``reference_answer`` graded against its own
    components must score 10/10. A miss is a harness bug, full stop.
Handout negatives
    The scoring guide publishes worked examples it labels as zero -- hedged
    figures, a correct number in the wrong context, a date off by one day,
    the base model thinking out loud. All must come back NO.
Formatting-equivalence positives
    Cases the handout says are accepted: 1,774 == 1774, ISO == "2 Jan 2015".

    python training/eval/calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION, judge_many  # noqa: E402
from scorer import aggregate, load_questions, score_batch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"
OUT = REPO_ROOT / "training" / "results" / "judge_calibration.md"

NOTE_EXACT = (
    "Exact values are required for dates, counts, labels, rankings, rates, and "
    "stated numeric results. Commas, trailing zeros, and equivalent date formats "
    "are accepted."
)

#: (name, question, answer, expected_fact, tolerance_note, gold_verdict)
ADVERSARIAL: list[tuple[str, str, str, str, str, bool]] = [
    (
        "hedged-count",
        "From the first RBA record to the last, how many cash-rate decisions changed the rate?",
        "There were approximately 41 decisions that changed the rate.",
        "41 total changes",
        NOTE_EXACT,
        False,
    ),
    (
        "wrong-context",
        "From the first RBA record to the last, how many cash-rate decisions changed the rate, "
        "and how many were increases versus decreases?",
        "There are 41 records in total, 20 of which are holds.",
        "41 of the 175 decision records changed the rate: 20 increases and 21 decreases.",
        NOTE_EXACT,
        False,
    ),
    (
        "no-number",
        "What was the longest period the RBA held rates unchanged?",
        "Based on my search of the RBA data, the cash rate was held steady for an extended "
        "period during 2016-2019.",
        "1036 days, from 2016-08-03 to 2019-06-05",
        NOTE_EXACT,
        False,
    ),
    (
        "off-by-one-date",
        "What is the highest cash-rate target ever in the RBA dataset, and when did it first "
        "take effect?",
        "The highest cash-rate target is 4.75, which first appeared on the effective date "
        "2010-11-03.",
        "2010-11-02",
        NOTE_EXACT,
        False,
    ),
    (
        "thinking-out-loud",
        "From the first RBA record to the last, how many cash-rate decisions changed the rate?",
        "The user is asking for statistics about RBA cash rate decisions. I need to find the "
        "RBA records in the team's corpora. I will use the retrieve tool to search for 'RBA "
        "cash rate decisions' and then analyze the retrieved documents to count them.",
        "41 total changes",
        NOTE_EXACT,
        False,
    ),
    (
        "refusal",
        "What is the lowest cash-rate target in the RBA dataset?",
        "I do not have enough information to answer that question.",
        "0.1",
        NOTE_EXACT,
        False,
    ),
    (
        "comma-equivalence",
        "What are the dimensions of the ASX dataset?",
        "There are 18 ticker files, each containing 1774 rows.",
        "each containing 1,774 rows",
        NOTE_EXACT,
        True,
    ),
    (
        "iso-date-equivalence",
        "When did the lowest cash-rate target first take effect?",
        "It first took effect on 2020-11-04.",
        "4 Nov 2020",
        NOTE_EXACT,
        True,
    ),
    (
        "reference-date-equivalence",
        "What is the common date range of the ASX dataset?",
        "The data covers 2 Jan 2015 through 30 Dec 2021.",
        "covering 2015-01-02 through 2021-12-30",
        NOTE_EXACT,
        True,
    ),
    (
        "percent-prose-equivalence",
        "Excluding Tabcorp, which ticker had the best 2018 return?",
        "BHP.AX performed best, posting a 22.17 per cent gain.",
        "BHP.AX was best at +22.17%",
        NOTE_EXACT,
        True,
    ),
    (
        "trailing-zero-equivalence",
        "What did the cash rate target fall to by the end of 2013?",
        "The target fell to 2.5%.",
        "2.50%",
        NOTE_EXACT,
        True,
    ),
    (
        "word-number-equivalence",
        "How many cuts occurred across the 2011-2013 easing period?",
        "Eight cuts occurred.",
        "8 cuts occurred",
        NOTE_EXACT,
        True,
    ),
]


def main() -> int:
    records = load_questions(QUESTIONS)
    print(f"loaded {len(records)} public questions\n")

    # --- Gate 1: reference answers must score full marks -------------------
    results = score_batch(
        [(record, record["reference_answer"], None) for record in records]
    )
    summary = aggregate(results)

    print("=== Gate 1: reference answers vs their own components ===")
    failures = []
    for result in results:
        perfect = abs(result.points_earned - result.points_max) < 1e-6
        if not perfect:
            failures.append(result)
        flag = "ok " if perfect else "MISS"
        print(
            f"{flag} {result.qid}  {result.points_earned:5.2f}/{result.points_max:5.2f}"
            f"  ({result.difficulty})"
        )
        for component in result.components:
            if not component.llm_yes:
                print(
                    f"      -> {component.component_id} NO "
                    f"(p_yes={component.p_yes:.3f}, numeric={component.numeric})"
                )
                print(f"         expected: {component.expected_fact[:110]}")

    print(
        f"\nreference score: {summary['component_score_llm']:.1%} "
        f"({summary['points_earned']}/{summary['points_max']} pts), "
        f"{len(failures)} question(s) below full marks"
    )

    # --- Gate 2: adversarial triples ---------------------------------------
    print("\n=== Gate 2: adversarial and equivalence triples ===")
    verdicts = judge_many(
        [(q, a, f, n) for _, q, a, f, n, _ in ADVERSARIAL]
    )
    agree = 0
    for (name, _q, _a, _f, _n, gold), verdict in zip(ADVERSARIAL, verdicts):
        ok = verdict.yes == gold
        agree += ok
        print(
            f"{'ok ' if ok else 'FAIL'} {name:28} "
            f"judge={'YES' if verdict.yes else 'NO ':<3} gold={'YES' if gold else 'NO'} "
            f"p_yes={verdict.p_yes:.3f}"
        )
    rate = agree / len(ADVERSARIAL)
    print(f"\nadversarial agreement: {agree}/{len(ADVERSARIAL)} = {rate:.1%}")

    # --- Report -------------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Judge calibration",
        "",
        f"- judge model: `{JUDGE_MODEL}`",
        f"- prompt version: `{JUDGE_PROMPT_VERSION}`",
        "- protocol: one expected fact per call, verdict read from YES/NO token logprobs,",
        "  `temperature=0`, `seed=0`, thinking disabled.",
        "",
        "## Gate 1 - reference answers score full marks",
        "",
        f"Score on the 15 public reference answers: **{summary['component_score_llm']:.1%}** "
        f"({summary['points_earned']}/{summary['points_max']} points). "
        f"Questions below full marks: **{len(failures)}**.",
        "",
        "| question | earned | max | difficulty |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.qid} | {result.points_earned:.2f} | {result.points_max:.2f} "
            f"| {result.difficulty} |"
        )
    lines += [
        "",
        "## Gate 2 - adversarial and equivalence triples",
        "",
        f"Agreement with published labels: **{agree}/{len(ADVERSARIAL)} = {rate:.1%}**.",
        "",
        "| case | judge | gold | p(yes) |",
        "|---|---|---|---:|",
    ]
    for (name, _q, _a, _f, _n, gold), verdict in zip(ADVERSARIAL, verdicts):
        lines.append(
            f"| {name} | {'YES' if verdict.yes else 'NO'} | {'YES' if gold else 'NO'} "
            f"| {verdict.p_yes:.3f} |"
        )
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    return 0 if not failures and rate >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
