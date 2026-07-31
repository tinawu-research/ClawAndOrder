"""Capture real tool traces once, so every model arm is graded on identical evidence.

The base-vs-fine-tuned comparison has to isolate *synthesis* quality. If each
arm ran its own planning loop, Qwen's tool-call variance would leak into the
delta -- and Qwen at temperature 0 through vLLM is not bit-reproducible. So the
planning loop runs once here, and the resulting evidence is replayed verbatim
to base and to every checkpoint.

Runs the orchestrator in-process with ``DOMAIN_PREDICT_MODE=mock``: the real
brain plans and the real tools execute, but synthesis is skipped, which is both
faster and avoids burning the domain endpoint on traces we are about to discard.

    python training/eval/freeze_evidence.py [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# mock synthesis: we want the trace, not an answer. Must be set before the
# agent package is imported, because config reads the environment at import.
os.environ.setdefault("DOMAIN_PREDICT_MODE", "mock")

sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datastore  # noqa: E402
from orchestrator import answer_question  # noqa: E402
from scorer import load_questions  # noqa: E402

QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"
OUT_DIR = REPO_ROOT / "training" / "eval_sets" / "frozen_evidence"


async def _capture(record: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        started = time.monotonic()
        response = await answer_question(record["prompt"])
        elapsed = time.monotonic() - started

    diagnostics = response.get("diagnostics", {})
    return {
        "qid": record["id"],
        "prompt": record["prompt"],
        "difficulty": record.get("difficulty"),
        "datasets": record.get("datasets"),
        "dataset_scope": record.get("dataset_scope"),
        # Only successful calls reach synthesis in production, so the frozen
        # trace must be filtered the same way or it is off-distribution.
        "tool_trace": response.get("tool_trace", []),
        "capture": {
            "latency_seconds": round(elapsed, 2),
            "planning_seconds": diagnostics.get("latency_seconds"),
            "brain_calls": diagnostics.get("brain_calls"),
            "tool_calls": diagnostics.get("tool_calls"),
            "tool_failures": diagnostics.get("tool_failures"),
            "notes": diagnostics.get("notes", []),
        },
    }


async def main_async(limit: int | None, workers: int) -> int:
    records = load_questions(QUESTIONS)
    if limit:
        records = records[:limit]

    print(f"loading datasets ...", flush=True)
    started = time.monotonic()
    await asyncio.to_thread(datastore.STORE.load)
    print(f"datasets loaded in {time.monotonic() - started:.1f}s\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(workers)

    print(f"capturing {len(records)} traces ({workers}-way) ...\n", flush=True)
    captures = await asyncio.gather(*(_capture(r, semaphore) for r in records))

    empty = []
    for capture in captures:
        path = OUT_DIR / f"{capture['qid']}.clean.json"
        path.write_text(json.dumps(capture, indent=2, ensure_ascii=False), encoding="utf-8")
        stats = capture["capture"]
        n_calls = len(capture["tool_trace"])
        if n_calls == 0:
            empty.append(capture["qid"])
        print(
            f"{capture['qid']}  {n_calls} call(s)  "
            f"brain={stats['brain_calls']}  fail={stats['tool_failures']}  "
            f"{stats['latency_seconds']:6.1f}s  "
            f"{'  ' + ' | '.join(stats['notes']) if stats['notes'] else ''}"
        )

    total = sum(len(c["tool_trace"]) for c in captures)
    latencies = sorted(c["capture"]["latency_seconds"] for c in captures)
    print(
        f"\n{len(captures)} traces, {total} tool calls "
        f"(mean {total / len(captures):.1f}/question)"
    )
    print(
        f"planning latency: p50={latencies[len(latencies) // 2]:.1f}s  "
        f"max={latencies[-1]:.1f}s"
    )
    if empty:
        print(f"\nWARNING: {len(empty)} question(s) produced no evidence: {', '.join(empty)}")
    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 1 if empty else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    return asyncio.run(main_async(args.limit, args.workers))


if __name__ == "__main__":
    raise SystemExit(main())
