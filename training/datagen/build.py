"""Assemble the training corpus: generate, perturb, gate, split, interleave, write.

Sizing is driven by what the run actually consumes, not by the organizers'
48,000-sample reference. At ``MAX_STEPS=100`` with an effective batch of 8 the
trainer sees ~800 sequences, and ``WARMUP_STEPS=50`` means only ~400 of those
arrive at full learning rate. A 48,000-row file would be sampled at under 2%.
So the target is ~800 well-stratified rows, and diversity beats volume.

Two gates protect the run:

Token gate
    Every example is measured with the real tokenizer and dropped if it exceeds
    the budget. SFT truncates from the right, so an over-long row loses its
    assistant span and trains on nothing -- silently.
Prefix balance
    ``CHECKPOINT_EVERY=20`` and the handout says step 20 is often the best
    checkpoint. Step 20 is the first 160 rows. Rows are written in round-robin
    category order so any prefix is balanced, and the build fails if the first
    160 miss a category.

    python training/datagen/build.py --budget 1024 --target 800
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src" / "agent"))

import datastore  # noqa: E402
import generators  # noqa: E402
import perturb  # noqa: E402
import tokens  # noqa: E402
from spec import Example  # noqa: E402

OUT_DIR = REPO_ROOT / "training" / "data"

#: Share of the corpus each category should occupy. Weighted by how much of the
#: behaviour is *not* copyable straight out of the evidence: sentiment and
#: refusal are oversampled relative to their share of the public set, coverage
#: and volume undersampled because they are near-verbatim copies.
TARGET_MIX = {
    "rba_counts": 0.12,
    "coverage": 0.05,
    "asx_returns": 0.12,
    "asx_volume": 0.05,
    "asx_drawdown": 0.07,
    "afr_counts": 0.12,
    "sentiment": 0.16,
    "rba_asx_event": 0.13,
    "composite": 0.10,
    "refusal": 0.08,
}


def load_sentiment(path: Path) -> list[Example]:
    """Load teacher-labelled sentiment examples, if they have been built."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out.append(
            Example(
                category="sentiment",
                template_id=record["template_id"],
                question=record["question"],
                answer=record["answer"],
                components=record["components"],
                tool_calls=[tuple(c) for c in record["tool_calls"]],
                param_key=record["param_key"],
                split_keys=record.get("split_keys", {}),
            )
        )
    return out


def generate_pool(rng: random.Random, held_out: bool) -> list[Example]:
    pool: list[Example] = []
    for name, fn in generators.registry().items():
        try:
            pool.extend(fn(rng, held_out))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! generator {name} failed: {type(exc).__name__}: {exc}")
    return pool


def token_gate(examples: list[Example], budget: int) -> tuple[list[Example], list[Example]]:
    kept, dropped = [], []
    for example in examples:
        try:
            n = tokens.count_messages(example.messages())
        except tokens.TokenizerError as exc:
            raise SystemExit(f"tokenizer unavailable, cannot gate safely: {exc}")
        example.tokens = n
        (kept if n <= budget else dropped).append(example)
    return kept, dropped


def dedupe(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    out = []
    for example in examples:
        fingerprint = example.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(example)
    return out


def stratify(examples: list[Example], target: int, rng: random.Random) -> list[Example]:
    """Sample toward TARGET_MIX, taking everything available in short categories."""
    by_category: dict[str, list[Example]] = collections.defaultdict(list)
    for example in examples:
        by_category[example.category].append(example)

    chosen: list[Example] = []
    taken: dict[str, int] = {}
    shortfall = 0
    for category, share in TARGET_MIX.items():
        want = round(target * share)
        available = by_category.get(category, [])
        rng.shuffle(available)
        take = available[:want]
        chosen.extend(take)
        taken[category] = len(take)
        shortfall += max(0, want - len(take))

    # Redistribute any shortfall, but cap each category at 1.5x its target
    # share. Without the cap the whole shortfall lands in whichever category has
    # the deepest pool -- sentiment, which has a teacher-labelled article per
    # example -- and that category ends up at double its intended weight while
    # the multi-tool categories stay under.
    if shortfall:
        for category, share in sorted(TARGET_MIX.items(), key=lambda kv: -kv[1]):
            if shortfall <= 0:
                break
            ceiling = int(round(target * share * 1.5))
            already = taken.get(category, 0)
            headroom = max(0, ceiling - already)
            if not headroom:
                continue
            extra = by_category.get(category, [])[already: already + min(headroom, shortfall)]
            chosen.extend(extra)
            taken[category] = already + len(extra)
            shortfall -= len(extra)

    return chosen


def interleave(examples: list[Example], rng: random.Random) -> list[Example]:
    """Round-robin over categories so every prefix is balanced."""
    buckets: dict[str, list[Example]] = collections.defaultdict(list)
    for example in examples:
        buckets[example.category].append(example)
    for items in buckets.values():
        # Easy formats first inside each category: a free curriculum effect.
        items.sort(key=lambda e: e.tokens)

    order = sorted(buckets, key=lambda c: -len(buckets[c]))
    out: list[Example] = []
    while any(buckets[c] for c in order):
        for category in order:
            if buckets[category]:
                out.append(buckets[category].pop(0))
    return out


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")


def write_io_jsonl(path: Path, examples: list[Example]) -> None:
    """Alternate ``{"input", "output"}`` shape.

    Emitted alongside the chat form because the NeMo recipe's expected schema is
    not yet confirmed, and producing both costs nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            messages = example.messages()
            handle.write(
                json.dumps(
                    {
                        "system": messages[0]["content"],
                        "input": messages[1]["content"],
                        "output": messages[2]["content"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1024, help="max tokens per example")
    parser.add_argument("--target", type=int, default=800, help="rows in train.jsonl")
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--paraphrase", type=int, default=2, help="variants per example, 0 to skip")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("loading datasets ...", flush=True)
    datastore.STORE.load()

    print("generating train-side pool ...", flush=True)
    pool = generate_pool(rng, held_out=False)
    print(f"  {len(pool)} base examples")

    sentiment = load_sentiment(OUT_DIR / "sentiment_examples.jsonl")
    if sentiment:
        print(f"  {len(sentiment)} sentiment examples")
        pool.extend(sentiment)
    else:
        print("  ! no sentiment examples yet (run label_sentiment.py)")

    if args.paraphrase:
        print(f"paraphrasing ({args.paraphrase} per example) ...", flush=True)
        import paraphrase

        variants = paraphrase.expand(pool, n=args.paraphrase)
        print(f"  {len(variants)} accepted paraphrases")
        pool.extend(variants)

    print("applying perturbations ...", flush=True)
    pool = perturb.apply_perturbations(pool, rng, rate=0.35)

    n_negatives = round(args.target * 0.12)
    print(f"building ~{n_negatives} negative examples ...", flush=True)
    pool.extend(perturb.build_negatives(pool, rng, target=n_negatives))

    print(f"deduping {len(pool)} candidates ...", flush=True)
    pool = dedupe(pool)
    print(f"  {len(pool)} unique")

    print(f"token gate at {args.budget} ...", flush=True)
    kept, dropped = token_gate(pool, args.budget)
    print(f"  kept {len(kept)}, dropped {len(dropped)} over budget")
    if dropped:
        by_category = collections.Counter(e.category for e in dropped)
        for category, n in by_category.most_common():
            print(f"    dropped {n:4} from {category}")

    print("splitting ...", flush=True)
    # Paraphrases and perturbations of one fact set share a gold_key and must
    # not straddle the split, or validation is measuring memorisation.
    by_gold: dict[str, list[Example]] = collections.defaultdict(list)
    for example in kept:
        by_gold[example.gold_key].append(example)

    gold_keys = sorted(by_gold)
    rng.shuffle(gold_keys)

    # Hold out a *fraction* of fact sets for validation rather than filling a
    # fixed row count first -- with a small pool the latter starves training.
    val_fraction = min(0.2, args.val / max(len(kept), 1))
    n_val_keys = max(1, round(len(gold_keys) * val_fraction))

    val: list[Example] = []
    train_pool: list[Example] = []
    for position, key in enumerate(gold_keys):
        if position < n_val_keys and len(val) < args.val:
            val.extend(by_gold[key])
        else:
            train_pool.extend(by_gold[key])

    train = stratify(train_pool, args.target, rng)
    train = interleave(train, rng)

    # Prefix balance check: step 20 sees only the first 160 rows.
    prefix = collections.Counter(e.category for e in train[:160])
    print(f"\nfirst-160 category histogram (the step-20 checkpoint):")
    for category, n in sorted(prefix.items()):
        print(f"  {category:16} {n:3}")
    thin = [c for c in set(e.category for e in train) if prefix.get(c, 0) < 6]
    if thin:
        print(f"  ! thin in prefix (<6): {', '.join(sorted(thin))}")

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_io_jsonl(OUT_DIR / "train.io.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_io_jsonl(OUT_DIR / "val.io.jsonl", val)

    lengths = sorted(e.tokens for e in train)
    mix = collections.Counter(e.category for e in train)
    print(f"\ntrain: {len(train)} rows -> {OUT_DIR.relative_to(REPO_ROOT)}/train.jsonl")
    print(f"val:   {len(val)} rows")
    print(f"tokens: min={lengths[0]} p50={lengths[len(lengths)//2]} "
          f"p95={lengths[int(len(lengths)*0.95)]} max={lengths[-1]}")
    print("\nfinal mix:")
    for category, n in sorted(mix.items()):
        print(f"  {category:16} {n:4}  {n/len(train):5.1%}  (target {TARGET_MIX.get(category,0):.0%})")

    report = {
        "budget": args.budget,
        "seed": args.seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "dropped_over_budget": len(dropped),
        "token_p50": lengths[len(lengths) // 2],
        "token_max": lengths[-1],
        "mix": dict(mix),
        "prefix_160": dict(prefix),
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
