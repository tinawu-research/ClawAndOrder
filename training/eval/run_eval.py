"""Base-vs-fine-tuned sweep: replay every arm, grade, compare, report.

This produces the comparison the 30% "fine-tuned model quality" criterion asks
for. It is deliberately one command so the result is reproducible by someone who
did not watch it being built:

    python training/eval/run_eval.py --arms base ck20 ck40
    python training/eval/run_eval.py --arms base ck20 --conditions clean noisy \\
        insufficient shuffled --prompt-2x2

Every arm answers the same questions from byte-identical frozen evidence, so the
only thing varying between arms is the weights. Scoring reuses the calibrated
component judge (15/15 on reference answers, 12/12 on adversarial negatives),
and the headline number is the same component score the organizers compute.

The report never states a delta without its paired interval. A +12pp with a CI
that crosses zero is not an improvement, and saying so plainly is worth more
than a large unqualified number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import replay  # noqa: E402
import secondary  # noqa: E402
import stats  # noqa: E402
from scorer import aggregate, load_questions, score_batch  # noqa: E402

QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"
RESULTS = REPO_ROOT / "training" / "results"


def per_question_scores(results: list[Any]) -> dict[str, float]:
    """Fraction of available points earned, keyed by question id.

    A fraction rather than raw points: questions carry different totals, and an
    unweighted mean of raw points would let the highest-scoring questions
    dominate the paired comparison.
    """
    out = {}
    for result in results:
        out[result.qid] = (
            result.points_earned / result.points_max if result.points_max else 0.0
        )
    return out


def per_question_strict(results: list[Any]) -> dict[str, float]:
    """Same, but requiring the numeric check to agree with the judge.

    Reported alongside the headline because the two can move in opposite
    directions: a verbose answer can win an LLM verdict while stating the value
    in a form the tolerance arithmetic rejects.
    """
    return {
        r.qid: (r.points_earned_strict / r.points_max if r.points_max else 0.0)
        for r in results
    }


def evaluate_cell(
    records: list[dict[str, Any]],
    by_qid: dict[str, dict[str, Any]],
    arm: str,
    condition: str,
    *,
    long_prompt: bool,
    workers: int,
) -> dict[str, Any]:
    """Generate, grade, and measure one (arm, condition) cell."""
    arm_id = replay.BASE_MODEL_ID if arm == "base" else arm

    started = time.monotonic()
    generations = replay.replay_arm(
        records, arm_id, condition, long_prompt=long_prompt, workers=workers
    )
    gen_seconds = time.monotonic() - started

    errors = [g for g in generations if g.error]
    if errors:
        print(f"      ! {len(errors)} generation errors; first: {errors[0].error[:160]}")

    # Grade only what the harness has a question record for.
    items = [
        (by_qid[g.qid], g.answer, g.latency_seconds)
        for g in generations
        if g.qid in by_qid
    ]
    results = score_batch(items)

    metrics = [
        secondary.measure(
            g.qid, arm, condition, g.answer, g.trace, by_qid.get(g.qid, {}).get("prompt", "")
        )
        for g in generations
    ]

    summary = aggregate(results)
    summary["secondary"] = secondary.summarise(metrics)
    summary["generation_seconds"] = round(gen_seconds, 1)
    summary["error_rate"] = round(len(errors) / len(generations), 4) if generations else 0.0
    summary["served_models"] = sorted({g.served_model for g in generations if g.served_model})

    return {
        "arm": arm,
        "condition": condition,
        "long_prompt": long_prompt,
        "summary": summary,
        "per_question": per_question_scores(results),
        "per_question_strict": per_question_strict(results),
        "generations": [g.to_record() for g in generations],
        "secondary_rows": [m.to_record() for m in metrics],
    }


def format_report(cells: list[dict[str, Any]], baseline: str, conditions: list[str]) -> str:
    """Markdown report: headline table, paired deltas, secondary metrics."""
    lines: list[str] = []
    add = lines.append

    add("# Base vs fine-tuned Nemotron — controlled comparison\n")
    add(
        "Every arm answers the same questions from byte-identical frozen tool "
        "evidence. Qwen routing and tool execution are held fixed, so the only "
        "variable between arms is the synthesis model's weights.\n"
    )
    add(
        f"- Judge: calibrated component judge (see `judge_calibration.md`)\n"
        f"- Baseline arm: `{baseline}`\n"
        f"- Decoding: greedy (`temperature=0`, `seed=0`)\n"
    )

    indexed = {(c["arm"], c["condition"], c["long_prompt"]): c for c in cells}
    arms: list[str] = []
    for cell in cells:
        if cell["arm"] not in arms:
            arms.append(cell["arm"])

    # ---- headline ----
    add("\n## Component score\n")
    header = "| arm | " + " | ".join(conditions) + " |"
    add(header)
    add("|" + "---|" * (len(conditions) + 1))
    for arm in arms:
        row = [f"`{arm}`"]
        for condition in conditions:
            cell = indexed.get((arm, condition, False))
            row.append(
                f"{cell['summary']['component_score_llm']:.1%}" if cell else "—"
            )
        add("| " + " | ".join(row) + " |")

    # ---- paired deltas ----
    add(f"\n## Paired delta vs `{baseline}`\n")
    add(
        "Bootstrap over questions (2,000 resamples), not components — components "
        "within a question are correlated. `p` is an exact sign-flip permutation "
        "test on paired differences.\n"
    )
    add("| arm | condition | base | arm | delta | 95% CI | p | W/L/T |")
    add("|---|---|---|---|---|---|---|---|")
    for condition in conditions:
        base_cell = indexed.get((baseline, condition, False))
        if not base_cell:
            continue
        for arm in arms:
            if arm == baseline:
                continue
            cell = indexed.get((arm, condition, False))
            if not cell:
                continue
            qids = sorted(set(base_cell["per_question"]) & set(cell["per_question"]))
            delta = stats.compare(
                [base_cell["per_question"][q] for q in qids],
                [cell["per_question"][q] for q in qids],
            )
            flag = "" if delta.significant else " ⚠"
            add(
                f"| `{arm}` | {condition} | {delta.mean_a:.1%} | {delta.mean_b:.1%} | "
                f"**{delta.delta:+.1%}**{flag} | "
                f"[{delta.ci_low:+.1%}, {delta.ci_high:+.1%}] | {delta.p_value:.4f} | "
                f"{delta.n_better}/{delta.n_worse}/{delta.n_tied} |"
            )
    add("\n⚠ = 95% CI includes zero; not a demonstrated improvement.\n")

    add(f"\n### Strict score (judge **and** numeric tolerance must agree)\n")
    add(
        "The headline metric follows the organizers' LLM judge. This one also "
        "requires the stated value to pass tolerance arithmetic, so it catches "
        "answers a judge accepts but whose figures are mis-stated or "
        "mis-formatted.\n"
    )
    add("| arm | condition | base | arm | delta | 95% CI | p | W/L/T |")
    add("|---|---|---|---|---|---|---|---|")
    for condition in conditions:
        base_cell = indexed.get((baseline, condition, False))
        if not base_cell:
            continue
        for arm in arms:
            if arm == baseline:
                continue
            cell = indexed.get((arm, condition, False))
            if not cell or "per_question_strict" not in cell:
                continue
            qids = sorted(set(base_cell["per_question_strict"]) & set(cell["per_question_strict"]))
            delta = stats.compare(
                [base_cell["per_question_strict"][q] for q in qids],
                [cell["per_question_strict"][q] for q in qids],
            )
            flag = "" if delta.significant else " ⚠"
            add(
                f"| `{arm}` | {condition} | {delta.mean_a:.1%} | {delta.mean_b:.1%} | "
                f"**{delta.delta:+.1%}**{flag} | "
                f"[{delta.ci_low:+.1%}, {delta.ci_high:+.1%}] | {delta.p_value:.4f} | "
                f"{delta.n_better}/{delta.n_worse}/{delta.n_tied} |"
            )

    # ---- secondary ----
    add("\n## Secondary metrics (deterministic, no judge)\n")
    add(
        "`hallucinated_number_rate` counts numeric literals asserted in the "
        "answer that appear in neither the evidence nor the question, allowing "
        "for rounding. It is computable only because the evidence is frozen.\n"
    )
    add("| arm | condition | hedge | halluc. num | leaked reasoning | format viol. | empty | words p50 |")
    add("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for condition in conditions:
            cell = indexed.get((arm, condition, False))
            if not cell:
                continue
            s = cell["summary"]["secondary"]
            add(
                f"| `{arm}` | {condition} | {s['hedge_rate']:.1%} | "
                f"{s['hallucinated_number_rate']:.1%} | {s['leaked_reasoning_rate']:.1%} | "
                f"{s['format_violation_rate']:.1%} | {s['empty_rate']:.1%} | "
                f"{s['answer_words_p50']} |"
            )

    # ---- prompt 2x2 ----
    long_cells = [c for c in cells if c["long_prompt"]]
    if long_cells:
        add("\n## Prompt fairness control (2×2)\n")
        add(
            "The system prompt was shortened (332 → 145 tokens) to fit the "
            "sequence budget, so the fine-tuned arm was trained on the short "
            "prompt. Without this control the comparison would silently be "
            "\"base + long prompt vs FT + short prompt\". Both arms are run "
            "against both prompts.\n"
        )
        add("| arm | prompt | condition | component score |")
        add("|---|---|---|---|")
        for cell in sorted(cells, key=lambda c: (c["arm"], c["long_prompt"])):
            if cell["condition"] != "clean":
                continue
            prompt = "long (pre-FT)" if cell["long_prompt"] else "short (shipped)"
            add(
                f"| `{cell['arm']}` | {prompt} | {cell['condition']} | "
                f"{cell['summary']['component_score_llm']:.1%} |"
            )

    # ---- provenance ----
    add("\n## Provenance\n")
    add("| arm | prompt | condition | served model id | errors | gen seconds |")
    add("|---|---|---|---|---|---|")
    for cell in cells:
        s = cell["summary"]
        served = ", ".join(f"`{m}`" for m in s["served_models"]) or "—"
        prompt = "long" if cell["long_prompt"] else "short"
        add(
            f"| `{cell['arm']}` | {prompt} | {cell['condition']} | {served} | "
            f"{s['error_rate']:.1%} | {s['generation_seconds']} |"
        )
    add(
        "\nThe served model id is echoed by vLLM per request. A LoRA arm "
        "reporting the base id would mean the adapter was not applied.\n"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["base"])
    parser.add_argument("--conditions", nargs="+", default=["clean"], choices=replay.CONDITIONS)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--prompt-2x2", action="store_true",
                        help="also run every arm against the pre-fine-tune long prompt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="training/results/base_vs_ft.md")
    parser.add_argument("--questions", default=str(QUESTIONS),
                        help="question file in the organizers' schema")
    parser.add_argument("--frozen", default=str(replay.FROZEN_DIR),
                        help="directory of *.clean.json frozen evidence")
    args = parser.parse_args()

    records = replay.load_frozen(Path(args.frozen))
    # Question records key on "id" (the organizers' schema); frozen-evidence
    # files key on "qid". Held-out sets built by build_heldout.py carry "id" too.
    by_qid = {r["id"]: r for r in load_questions(args.questions)}
    print(f"{len(records)} frozen questions, {len(by_qid)} question records")

    missing = [r["qid"] for r in records if r["qid"] not in by_qid]
    if missing:
        print(f"  ! no question record for: {', '.join(missing)} (excluded from scoring)")

    cells: list[dict[str, Any]] = []
    for arm in args.arms:
        for condition in args.conditions:
            print(f"  {arm:8} {condition:13} ...", flush=True)
            cell = evaluate_cell(
                records, by_qid, arm, condition,
                long_prompt=False, workers=args.workers,
            )
            s = cell["summary"]
            print(
                f"      score {s['component_score_llm']:.1%} "
                f"(strict {s['component_score_strict']:.1%}) "
                f"halluc {s['secondary']['hallucinated_number_rate']:.1%} "
                f"in {s['generation_seconds']}s"
            )
            cells.append(cell)

        if args.prompt_2x2:
            print(f"  {arm:8} {'clean (long)':13} ...", flush=True)
            cell = evaluate_cell(
                records, by_qid, arm, "clean",
                long_prompt=True, workers=args.workers,
            )
            print(f"      score {cell['summary']['component_score_llm']:.1%}")
            cells.append(cell)

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = format_report(cells, args.baseline, args.conditions)
    (REPO_ROOT / args.out).write_text(report, encoding="utf-8")

    # Full detail alongside the prose, so a reviewer can recompute any number.
    raw = [
        {k: v for k, v in cell.items() if k != "generations"} | {"generations": cell["generations"]}
        for cell in cells
    ]
    raw_name = Path(args.out).stem + "_raw.json"
    (RESULTS / raw_name).write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"wrote training/results/{raw_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
