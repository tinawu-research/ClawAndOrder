"""Score the running agent against the 15 public calibration questions.

Mirrors the organizer harness closely enough to be useful before eval day: it
POSTs each ``prompt`` to ``/query``, times the response, applies the same
slow-response penalty, and scores each grading component.

The difference that matters: the official scorer uses an LLM judge that accepts
synonyms and equivalent formatting, whereas this checks that each expected fact's
numbers and dates appear in the answer. So a component this marks failed may
still pass officially, and a sentiment component may pass here for the wrong
reason. Read it as a triage list, not a score.

    python eval/run_public_eval.py
    python eval/run_public_eval.py --url http://172.20.0.5:5000 --workers 3
    python eval/run_public_eval.py --only MHQ061,MHQ084 --show-answers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parents[3]
    / "Participant_Package"
    / "public_questions.jsonl"
)

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_STOPWORDS = {
    "the", "and", "with", "that", "from", "were", "this", "there", "their",
    "was", "for", "are", "its", "than", "then", "each", "both", "over",
}


def numbers_in(text: str) -> set[str]:
    """Numeric tokens, normalised so 1,452 == 1452 and +22.17 == 22.17."""
    out = set()
    for raw in _NUMBER_RE.findall(text or ""):
        cleaned = raw.replace(",", "").lstrip("+").rstrip(".")
        out.add(cleaned)
        # 2.50 and 2.5 are the same rate; 0.10 and 0.1 the same target.
        if "." in cleaned:
            out.add(cleaned.rstrip("0").rstrip("."))
    return out


def score_component(answer: str, expected_fact: str) -> tuple[bool, list[str]]:
    """Heuristic component check. Returns ``(likely_pass, missing_values)``."""
    wanted = numbers_in(expected_fact)
    have = numbers_in(answer)
    missing = sorted(w for w in wanted if w not in have)
    if wanted:
        return not missing, missing
    # No numbers (sentiment / direction facts): fall back to keyword overlap.
    words = [
        w for w in re.findall(r"[a-z]{4,}", expected_fact.lower())
        if w not in _STOPWORDS
    ]
    if not words:
        return True, []
    lowered = (answer or "").lower()
    hits = sum(1 for w in words if w in lowered)
    return hits >= max(1, len(words) // 3), []


def ask(url: str, question: str, timeout: float) -> tuple[dict[str, Any], float]:
    payload = json.dumps({"question": question}).encode()
    req = request.Request(
        f"{url.rstrip('/')}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except error.HTTPError as exc:
        body = {"answer": "", "_error": f"HTTP {exc.code}: {exc.read()[:200]!r}"}
    except Exception as exc:  # noqa: BLE001 - report, do not abort the run
        body = {"answer": "", "_error": f"{type(exc).__name__}: {exc}"}
    return body, time.monotonic() - started


def evaluate_case(url: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    grading = case.get("grading", {})
    components = grading.get("components", [])
    body, elapsed = ask(url, case["prompt"], timeout)
    answer = body.get("answer") or ""

    earned = 0.0
    max_score = 0.0
    detail = []
    for component in components:
        points = float(component.get("points", 0))
        max_score += points
        passed, missing = score_component(answer, component.get("expected_fact", ""))
        earned += points if passed else 0.0
        detail.append(
            {
                "component_id": component.get("component_id"),
                "points": points,
                "passed": passed,
                "missing_values": missing,
                "expected_fact": component.get("expected_fact"),
            }
        )

    # Same penalty schedule as the official scorer.
    penalty = ""
    if elapsed > 300:
        earned, penalty = 0.0, "TIMEOUT >300s -> zero"
    elif elapsed > 60:
        earned, penalty = earned * 0.8, "slow >60s -> -20%"

    return {
        "id": case.get("id"),
        "difficulty": case.get("difficulty"),
        "datasets": "+".join(case.get("datasets", [])),
        "scope": case.get("dataset_scope"),
        "earned": round(earned, 2),
        "max": round(max_score, 2),
        "latency": round(elapsed, 2),
        "penalty": penalty,
        "answer": answer,
        "reference": case.get("reference_answer", ""),
        "steps": body.get("steps"),
        "tools": [t.get("tool") for t in body.get("tool_trace", [])],
        "error": body.get("_error"),
        "components": detail,
        "diagnostics": body.get("diagnostics", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:5000")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="concurrent requests; the official harness uses 3",
    )
    parser.add_argument("--timeout", type=float, default=310.0)
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"questions file not found: {args.questions}", file=sys.stderr)
        return 2

    cases = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    try:
        with request.urlopen(f"{args.url.rstrip('/')}/health", timeout=30) as resp:
            health_ok = resp.status == 200
    except Exception as exc:  # noqa: BLE001
        print(f"/health unreachable at {args.url}: {exc}", file=sys.stderr)
        print("The official harness skips the team entirely when this happens.",
              file=sys.stderr)
        return 2
    if not health_ok:
        print("/health did not return 200 — the harness would skip this team.",
              file=sys.stderr)
        return 2

    print(f"{len(cases)} case(s) against {args.url}, {args.workers} concurrent\n")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(
            pool.map(lambda c: evaluate_case(args.url, c, args.timeout), cases)
        )
    wall = time.monotonic() - started

    header = f"{'ID':<8}{'DIFF':<8}{'DATA':<12}{'SCORE':>10}{'SEC':>8}  NOTES"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["id"] or ""):
        notes = []
        if r["error"]:
            notes.append(r["error"])
        if r["penalty"]:
            notes.append(r["penalty"])
        failed = [c["component_id"] for c in r["components"] if not c["passed"]]
        if failed:
            notes.append("failed " + ",".join(failed))
        print(
            f"{r['id']:<8}{r['difficulty']:<8}{r['datasets']:<12}"
            f"{r['earned']:>5}/{r['max']:<4}{r['latency']:>8}  "
            f"{'; '.join(notes)}"
        )

    earned = sum(r["earned"] for r in results)
    maximum = sum(r["max"] for r in results)
    pct = 100.0 * earned / maximum if maximum else 0.0
    slow = sum(1 for r in results if r["latency"] > 60)
    print("-" * len(header))
    print(f"hidden-question proxy score: {earned:.2f}/{maximum:.2f} = {pct:.1f}%")
    print(f"wall clock {wall:.1f}s | slow responses (>60s): {slow}")
    print(
        "\nNOTE: heuristic scoring. The official judge is an LLM that accepts "
        "synonyms and\nequivalent formatting, so treat failures as items to "
        "inspect rather than as a verdict."
    )

    if args.show_answers:
        for r in sorted(results, key=lambda r: r["id"] or ""):
            print(f"\n=== {r['id']} [{r['difficulty']}] {r['latency']}s "
                  f"tools={r['tools']}")
            print(f"  ANSWER   : {r['answer']}")
            print(f"  REFERENCE: {r['reference']}")
            for c in r["components"]:
                mark = "PASS" if c["passed"] else "FAIL"
                extra = (
                    f"  missing {c['missing_values']}" if c["missing_values"] else ""
                )
                print(f"  [{mark}] {c['component_id']} ({c['points']}) "
                      f"{c['expected_fact']}{extra}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "earned": earned,
                    "max": maximum,
                    "percent": pct,
                    "wall_seconds": wall,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
