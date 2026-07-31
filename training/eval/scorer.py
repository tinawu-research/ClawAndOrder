"""Score an answer against a question's grading components.

``hidden_question_score = sum(earned) / sum(max)``, with per-component partial
credit. Three independent signals are recorded per component:

``llm_verdict``
    The headline. The organizers grade with an LLM, so predicting the hidden
    set means grading with an LLM too.
``numeric_check``
    Deterministic tolerance arithmetic. Cannot judge prose, but never gets the
    arithmetic wrong.
``hedge_flag``
    A hedging word next to a value, which the handout scores as zero.

Reporting the strict score alongside the headline, plus their disagreement
rate, is what turns a number into evidence: the disagreements are exactly the
rows where our judge is most likely miscalibrated against the organizers'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from judge import Verdict, judge_many
from tolerances import hedge_flag, numeric_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ComponentResult:
    component_id: str
    expected_fact: str
    points_max: float
    llm_yes: bool
    p_yes: float
    numeric: str  # PASS | FAIL | N/A
    points_earned: float

    @property
    def strict_yes(self) -> bool:
        """Both signals agree the fact is present."""
        return self.llm_yes and self.numeric != "FAIL"

    @property
    def disagreement(self) -> bool:
        return self.numeric != "N/A" and self.llm_yes != (self.numeric == "PASS")


@dataclass
class QuestionResult:
    qid: str
    question: str
    answer: str
    difficulty: str
    dataset_scope: str
    components: list[ComponentResult] = field(default_factory=list)
    hedged: bool = False
    latency_seconds: float | None = None

    @property
    def points_max(self) -> float:
        return sum(c.points_max for c in self.components)

    @property
    def points_earned(self) -> float:
        return sum(c.points_earned for c in self.components)

    @property
    def points_earned_strict(self) -> float:
        return sum(c.points_max for c in self.components if c.strict_yes)

    @property
    def score(self) -> float:
        return self.points_earned / self.points_max if self.points_max else 0.0

    def with_latency_penalty(self) -> float:
        """Earned points after the response-time rule.

        <=60s full, >60s and <=300s loses 20% of earned, >300s zero.
        """
        if self.latency_seconds is None:
            return self.points_earned
        if self.latency_seconds > 300:
            return 0.0
        if self.latency_seconds > 60:
            return self.points_earned * 0.8
        return self.points_earned


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    """Read a question file in the organizers' public_questions.jsonl schema."""
    records = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_answer(
    record: dict[str, Any],
    answer: str,
    *,
    latency_seconds: float | None = None,
    use_cache: bool = True,
) -> QuestionResult:
    """Grade one answer against one question record."""
    return score_batch([(record, answer, latency_seconds)], use_cache=use_cache)[0]


def score_batch(
    items: list[tuple[dict[str, Any], str, float | None]],
    *,
    use_cache: bool = True,
    workers: int = 16,
) -> list[QuestionResult]:
    """Grade many (record, answer, latency) triples, judging all components concurrently.

    Batching matters: judging component-by-component serially over six model
    arms is the difference between a 2-minute sweep and a 30-minute one.
    """
    calls: list[tuple[str, str, str, str]] = []
    index: list[tuple[int, dict[str, Any]]] = []

    for position, (record, answer, _) in enumerate(items):
        grading = record.get("grading", {})
        note = grading.get("tolerance_note", "")
        for component in grading.get("components", []):
            calls.append((record["prompt"], answer, component["expected_fact"], note))
            index.append((position, component))

    verdicts: list[Verdict] = judge_many(calls, workers=workers, use_cache=use_cache)

    results = [
        QuestionResult(
            qid=record.get("id", f"Q{position:03d}"),
            question=record["prompt"],
            answer=answer,
            difficulty=record.get("difficulty", "unknown"),
            dataset_scope=record.get("dataset_scope", "unknown"),
            hedged=hedge_flag(answer),
            latency_seconds=latency,
        )
        for position, (record, answer, latency) in enumerate(items)
    ]

    for (position, component), verdict, call in zip(index, verdicts, calls):
        _record, answer, expected_fact, note = call
        numeric = numeric_check(expected_fact, answer, note)
        points_max = float(component.get("points", 0))
        results[position].components.append(
            ComponentResult(
                component_id=component.get("component_id", "C??"),
                expected_fact=expected_fact,
                points_max=points_max,
                llm_yes=verdict.yes,
                p_yes=verdict.p_yes,
                numeric=numeric,
                points_earned=points_max if verdict.yes else 0.0,
            )
        )

    return results


def aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    """Roll per-question results into the reported summary."""
    total_max = sum(r.points_max for r in results)
    total_earned = sum(r.points_earned for r in results)
    total_strict = sum(r.points_earned_strict for r in results)
    total_penalised = sum(r.with_latency_penalty() for r in results)

    all_components = [c for r in results for c in r.components]
    disagreements = [c for c in all_components if c.disagreement]

    by_difficulty: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = by_difficulty.setdefault(
            result.difficulty, {"earned": 0.0, "max": 0.0, "n": 0}
        )
        bucket["earned"] += result.points_earned
        bucket["max"] += result.points_max
        bucket["n"] += 1

    by_scope: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = by_scope.setdefault(
            result.dataset_scope, {"earned": 0.0, "max": 0.0, "n": 0}
        )
        bucket["earned"] += result.points_earned
        bucket["max"] += result.points_max
        bucket["n"] += 1

    return {
        "n_questions": len(results),
        "n_components": len(all_components),
        "points_max": round(total_max, 2),
        "points_earned": round(total_earned, 2),
        "component_score_llm": round(total_earned / total_max, 4) if total_max else 0.0,
        "component_score_strict": round(total_strict / total_max, 4) if total_max else 0.0,
        "component_score_after_latency": (
            round(total_penalised / total_max, 4) if total_max else 0.0
        ),
        "disagreement_rate": (
            round(len(disagreements) / len(all_components), 4) if all_components else 0.0
        ),
        "hedged_answers": sum(1 for r in results if r.hedged),
        "by_difficulty": {
            key: {
                "score": round(v["earned"] / v["max"], 4) if v["max"] else 0.0,
                "n": int(v["n"]),
            }
            for key, v in sorted(by_difficulty.items())
        },
        "by_scope": {
            key: {
                "score": round(v["earned"] / v["max"], 4) if v["max"] else 0.0,
                "n": int(v["n"]),
            }
            for key, v in sorted(by_scope.items())
        },
    }
