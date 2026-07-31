"""Author the held-out evaluation sets from held-out parameters.

``public15`` is only 15 questions and, worse, it stops being held-out the moment
it picks a checkpoint. This builds two sets the training corpus has never seen,
in the organizers' own question schema so the same judge grades all of them.

``heldout40``
    Same generators as the training corpus, run with ``held_out=True``: the
    reserved tickers (SUN.AX, TPG.AX, CMW.AX) as *subjects*, and the reserved
    year 2017. Distribution matches training; only the parameters are new. This
    is the reported comparison.

``probe20``
    Deliberately shifted. Component subsets the templates never emit together,
    evidence with a block removed, and reordered payloads. This is the
    overfitting detector — if ``heldout40`` improves while ``probe20`` does not,
    the model has learned our templates rather than the task.

Both carry their own frozen evidence, generated from the same real tool payloads
that produce the gold answer, so no live agent run is needed.

    python training/eval_sets/build_heldout.py

**Limitation, stated plainly:** this evidence is generator-produced, not captured
from a live Qwen trace. ``public15`` remains the only set whose evidence came off
the real planner, so the two are reported separately and never pooled.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "training" / "datagen"))
sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))

import datastore  # noqa: E402
import generators  # noqa: E402
import perturb  # noqa: E402
from spec import Example  # noqa: E402

TOLERANCE_EXACT = (
    "Exact values are required for dates, counts, labels, rankings, rates, and "
    "stated numeric results. Commas, trailing zeros, and equivalent date formats "
    "are accepted."
)

#: Which datasets each category touches, for the ``dataset_scope`` slice.
CATEGORY_DATASETS = {
    "rba_counts": ["RBA"],
    "coverage": ["RBA"],
    "asx_returns": ["ASX"],
    "asx_volume": ["ASX"],
    "asx_drawdown": ["ASX"],
    "afr_counts": ["AFR"],
    "sentiment": ["AFR", "RBA"],
    "rba_asx_event": ["RBA", "ASX"],
    "composite": ["RBA", "ASX", "AFR"],
    "refusal": ["RBA", "ASX", "AFR"],
}

#: Rough difficulty, used only to slice the report.
CATEGORY_DIFFICULTY = {
    "coverage": "easy",
    "rba_counts": "easy",
    "asx_volume": "easy",
    "afr_counts": "medium",
    "asx_returns": "medium",
    "asx_drawdown": "medium",
    "sentiment": "medium",
    "rba_asx_event": "hard",
    "composite": "hard",
    "refusal": "hard",
}


def to_question_record(example: Example, qid: str) -> dict:
    """Render a generated Example in the organizers' public-question schema.

    The 10 points are split evenly across components, matching how the public
    set distributes credit when a question asks for several facts.
    """
    components = example.components or [example.answer]
    points = round(10.0 / len(components), 4)
    datasets = CATEGORY_DATASETS.get(example.category, ["RBA"])
    return {
        "schema_version": "2.0",
        "id": qid,
        "visibility": "heldout",
        "difficulty": CATEGORY_DIFFICULTY.get(example.category, "medium"),
        "datasets": datasets,
        "dataset_scope": "single" if len(datasets) == 1 else "multi",
        "prompt": example.question,
        "reference_answer": example.answer,
        "required_facts": list(components),
        "grading": {
            "method": "component_based",
            "max_score": 10,
            "components": [
                {
                    "component_id": f"C{i + 1:02d}",
                    "expected_fact": fact,
                    "points": points,
                }
                for i, fact in enumerate(components)
            ],
            "tolerance_note": TOLERANCE_EXACT,
        },
        "meta": {
            "category": example.category,
            "template_id": example.template_id,
            "param_key": example.param_key,
            "gold_key": example.gold_key,
            "split_keys": example.split_keys,
            "perturbations": example.perturbations,
        },
    }


def to_frozen(example: Example, qid: str) -> dict:
    """Frozen-evidence file in the shape ``replay.load_frozen`` expects."""
    record = to_question_record(example, qid)
    return {
        "qid": qid,
        "prompt": example.question,
        "difficulty": record["difficulty"],
        "datasets": record["datasets"],
        "dataset_scope": record["dataset_scope"],
        "tool_trace": example.trace(),
        "capture": {"source": "generated", "category": example.category},
    }


def training_gold_keys() -> set[str]:
    """Gold keys present in the shipped training corpus.

    The leakage assertion below is the point of this function: a held-out set
    that shares a fact set with training measures memorisation.
    """
    keys: set[str] = set()
    path = REPO_ROOT / "training" / "data" / "train.jsonl"
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["meta"]["gold_key"])
    return keys


def stratified_sample(
    pool: list[Example], n: int, rng: random.Random
) -> list[Example]:
    """Take round-robin across categories so no category dominates."""
    buckets: dict[str, list[Example]] = collections.defaultdict(list)
    for example in pool:
        buckets[example.category].append(example)
    for items in buckets.values():
        rng.shuffle(items)

    order = sorted(buckets, key=lambda c: -len(buckets[c]))
    out: list[Example] = []
    while len(out) < n and any(buckets[c] for c in order):
        for category in order:
            if len(out) >= n:
                break
            if buckets[category]:
                out.append(buckets[category].pop(0))
    return out


def write_set(
    examples: list[Example], name: str, prefix: str, out_dir: Path
) -> list[dict]:
    """Write questions jsonl plus one frozen-evidence file per question."""
    frozen_dir = out_dir / f"frozen_{name}"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, example in enumerate(examples, start=1):
        qid = f"{prefix}{i:03d}"
        records.append(to_question_record(example, qid))
        (frozen_dir / f"{qid}.clean.json").write_text(
            json.dumps(to_frozen(example, qid), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    path = out_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    mix = collections.Counter(r["meta"]["category"] for r in records)
    print(f"\n{name}: {len(records)} questions -> {path.relative_to(REPO_ROOT)}")
    print(f"  frozen evidence -> {frozen_dir.relative_to(REPO_ROOT)}/")
    for category, count in sorted(mix.items()):
        print(f"    {category:16} {count:3}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=int, default=40)
    parser.add_argument("--probe", type=int, default=20)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("loading datasets ...", flush=True)
    datastore.STORE.load()

    print("generating held-out pool (reserved tickers + year 2017) ...", flush=True)
    pool: list[Example] = []
    for name, fn in generators.registry().items():
        try:
            pool.extend(fn(rng, True))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! generator {name} failed: {type(exc).__name__}: {exc}")
    print(f"  {len(pool)} candidates")

    trained = training_gold_keys()
    fresh = [e for e in pool if e.gold_key not in trained]
    print(f"  {len(pool) - len(fresh)} dropped as seen in training, {len(fresh)} remain")
    if not fresh:
        raise SystemExit("no unseen examples generated; check held_out=True support")

    rng.shuffle(fresh)
    heldout = stratified_sample(fresh, args.heldout, rng)

    # The probe set is a *stress variant of the same fact sets*, not an
    # independent sample. The generators yield only ~20 genuinely held-out fact
    # sets, so carving a second disjoint set out of them would leave both too
    # small to say anything.
    #
    # That is not merely a fallback. What the probe tests is robustness to
    # presentation: identical facts, evidence reshaped by a distractor block,
    # reordered payload keys, or a truncated block. If the score holds on
    # heldout but drops here, the model learned our evidence layout rather than
    # to read evidence -- which is exactly the template-lock signal wanted. Held
    # constant across the pair, the facts are the control, not the variable.
    probe_source = stratified_sample(fresh, min(args.probe, len(fresh)), rng)
    probe = perturb.apply_perturbations(probe_source, rng, rate=1.0)

    out_dir = HERE
    heldout_records = write_set(heldout, f"heldout{len(heldout)}", "HO", out_dir)
    probe_records = write_set(probe, f"probe{len(probe)}", "PR", out_dir)

    # ---- gates ----
    print("\nchecks:")
    ho_keys = {r["meta"]["gold_key"] for r in heldout_records}
    pr_keys = {r["meta"]["gold_key"] for r in probe_records}

    leaked = (ho_keys | pr_keys) & trained
    print(f"  {'ok  ' if not leaked else 'FAIL'}  no gold key shared with train.jsonl"
          f" ({len(trained)} train keys checked)")

    # Deliberately NOT asserting the two sets are disjoint -- the probe set is a
    # perturbed restatement of the same facts, so overlap is the design. What
    # must hold is that neither shares a fact set with training.
    print(f"  note  probe shares {len(ho_keys & pr_keys)} fact sets with heldout "
          f"(by design: same facts, reshaped evidence)")

    empty = [r["id"] for r in heldout_records + probe_records
             if not r["grading"]["components"]]
    print(f"  {'ok  ' if not empty else 'FAIL'}  every question has components")

    bad_points = [
        r["id"] for r in heldout_records + probe_records
        if abs(sum(c["points"] for c in r["grading"]["components"]) - 10.0) > 0.05
    ]
    print(f"  {'ok  ' if not bad_points else 'FAIL'}  components sum to 10 points")

    if leaked or empty or bad_points:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
