"""Build-time checks on the generated corpus.

The parity check is the important one. Everything else in the pipeline can be
wrong in ways that merely cost accuracy; a train/serve prompt mismatch costs the
entire adapter, silently, and does not show up in the loss curve. So the first
check reconstructs a prompt from a captured live trace and asserts it is
byte-identical to what the generator produces from the same trace.

    python training/datagen/verify.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))

import render  # noqa: E402
import tokens  # noqa: E402

DATA_DIR = REPO_ROOT / "training" / "data"
FROZEN_DIR = REPO_ROOT / "training" / "eval_sets" / "frozen_evidence"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_parity() -> list[str]:
    """Generated prompts must match what the live agent would send."""
    problems: list[str] = []
    captures = sorted(FROZEN_DIR.glob("*.clean.json"))
    if not captures:
        return ["no frozen evidence captured; run training/eval/freeze_evidence.py"]

    import synthesis

    checked = 0
    for path in captures:
        capture = json.loads(path.read_text(encoding="utf-8"))
        trace = capture["tool_trace"]
        if not trace:
            continue

        # What the served agent would send.
        served = (
            f"Question:\n{capture['prompt']}\n\n"
            f"Verified tool results:\n{synthesis.format_evidence(trace)}\n\n"
            "Write the final answer."
        )
        # What the generator builds from the same trace.
        generated = render.user_content(capture["prompt"], trace)

        if served != generated:
            problems.append(f"parity mismatch on {capture['qid']}")
        checked += 1

    if checked == 0:
        problems.append("no non-empty traces available to check parity against")
    return problems


def check_system_prompt() -> list[str]:
    """The generator must use the served constant, not a copy of it."""
    import synthesis

    if render.SYNTHESIS_SYSTEM_PROMPT is not synthesis.SYNTHESIS_SYSTEM_PROMPT:
        return ["render.SYNTHESIS_SYSTEM_PROMPT is not the served object"]
    return []


def check_splits(train: list[dict], val: list[dict]) -> list[str]:
    """No fact set may appear in both splits, in any surface form."""
    problems = []

    train_gold = {row["meta"]["gold_key"] for row in train}
    val_gold = {row["meta"]["gold_key"] for row in val}
    overlap = train_gold & val_gold
    if overlap:
        problems.append(f"{len(overlap)} gold_key(s) appear in both train and val")

    def fingerprint(row):
        return json.dumps(row["messages"], sort_keys=True)

    train_prints = {fingerprint(r) for r in train}
    collisions = sum(1 for r in val if fingerprint(r) in train_prints)
    if collisions:
        problems.append(f"{collisions} val row(s) are byte-identical to a train row")

    return problems


def check_prefix_balance(train: list[dict], window: int = 160, minimum: int = 6) -> list[str]:
    """Any prefix must be category-balanced, because step 20 only sees 160 rows."""
    prefix = collections.Counter(r["meta"]["category"] for r in train[:window])
    everything = {r["meta"]["category"] for r in train}
    thin = sorted(c for c in everything if prefix.get(c, 0) < minimum)
    if thin:
        return [f"categories under {minimum} in the first {window} rows: {', '.join(thin)}"]
    return []


def check_tokens(train: list[dict], budget: int) -> list[str]:
    """Nothing may exceed the budget: SFT truncates the label away, silently."""
    over = []
    for row in train:
        n = row["meta"].get("tokens")
        if n is None:
            n = tokens.count_messages(row["messages"])
        if n > budget:
            over.append((row["meta"]["template_id"], n))
    if over:
        worst = sorted(over, key=lambda x: -x[1])[:3]
        return [
            f"{len(over)} row(s) over the {budget}-token budget; worst: "
            + ", ".join(f"{t} at {n}" for t, n in worst)
        ]
    return []


def check_answers(train: list[dict]) -> list[str]:
    """Cheap sanity checks on the targets themselves."""
    problems = []
    hedges = ("approximately", "roughly", "about ", "around ")
    hedged = [
        r for r in train
        if any(h in r["messages"][-1]["content"].lower() for h in hedges)
    ]
    if hedged:
        problems.append(f"{len(hedged)} target answer(s) contain a hedging word")

    empty = [r for r in train if not r["messages"][-1]["content"].strip()]
    if empty:
        problems.append(f"{len(empty)} target answer(s) are empty")

    leaked = [
        r for r in train
        if "query_data(" in r["messages"][-1]["content"]
        or "tool result" in r["messages"][-1]["content"].lower()
    ]
    if leaked:
        problems.append(f"{len(leaked)} target answer(s) mention tools")

    return problems


def main() -> int:
    train = _read(DATA_DIR / "train.jsonl")
    val = _read(DATA_DIR / "val.jsonl")
    if not train:
        print("no train.jsonl; run build.py first")
        return 1

    report = json.loads((DATA_DIR / "build_report.json").read_text()) if (
        DATA_DIR / "build_report.json"
    ).exists() else {}
    budget = report.get("budget", 1024)

    checks = [
        ("prompt parity with the served agent", check_parity()),
        ("system prompt is imported, not copied", check_system_prompt()),
        ("train/val leakage", check_splits(train, val)),
        ("prefix category balance", check_prefix_balance(train)),
        ("token budget", check_tokens(train, budget)),
        ("target answer hygiene", check_answers(train)),
    ]

    failures = 0
    for name, problems in checks:
        if problems:
            failures += 1
            print(f"FAIL  {name}")
            for problem in problems:
                print(f"        {problem}")
        else:
            print(f"ok    {name}")

    print(f"\ntrain={len(train)} val={len(val)} budget={budget}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
