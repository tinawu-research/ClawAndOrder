"""Prove the LoRA adapter is actually applied, rather than assumed.

The submission rules require demonstrating that the fine-tuned model is genuinely
used by the solution. Three failure modes are silent — nothing errors, and the
answers look plausible in all of them:

1. `DOMAIN_PREDICT_MODE` is still `mock`.
2. Mode is `llm`, but synthesis raised and degraded to the mock fallback,
   which is logged at ERROR and still returns HTTP 200.
3. Mode is `llm`, nothing errors, but the `domain-ft` alias still resolves to the
   **base** weights. This was the actual configuration for most of the build.

Only the third is hard to catch by reading logs, so this is a behavioural test:
send the identical prompt at temperature 0 to the base id and to each adapter id
and compare. Greedy decoding is deterministic, so if an adapter is genuinely
applied the outputs must differ somewhere. Byte-identical text across every
question means the adapter is not being applied, whatever the config claims.

    python training/eval/fingerprint.py --arms ck20 ck40 ck60 ck80 ck100
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import replay  # noqa: E402

OUT = REPO_ROOT / "logs" / "eval" / "adapter_fingerprint.json"


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, help="adapter model ids")
    parser.add_argument("--n", type=int, default=5, help="questions to probe")
    args = parser.parse_args()

    records = replay.load_frozen()[: args.n]
    print(f"probing {len(records)} questions at temperature 0\n")

    base = replay.replay_arm(records, replay.BASE_MODEL_ID, "clean", workers=2)
    base_by_qid = {g.qid: g for g in base}

    report = {
        "base_model_id": replay.BASE_MODEL_ID,
        "questions": [r["qid"] for r in records],
        "arms": {},
    }

    all_pass = True
    for arm in args.arms:
        generations = replay.replay_arm(records, arm, "clean", workers=2)
        rows = []
        identical = 0
        for generation in generations:
            reference = base_by_qid.get(generation.qid)
            same = bool(reference) and generation.answer == reference.answer
            identical += same
            rows.append({
                "qid": generation.qid,
                "base_sha": digest(reference.answer) if reference else "",
                "arm_sha": digest(generation.answer),
                "identical": same,
                "served_model": generation.served_model,
            })

        differs = len(rows) - identical
        verdict = "APPLIED" if differs else "NOT APPLIED"
        all_pass &= differs > 0
        report["arms"][arm] = {
            "verdict": verdict,
            "differs_from_base": differs,
            "identical_to_base": identical,
            "served_models": sorted({r["served_model"] for r in rows if r["served_model"]}),
            "rows": rows,
        }
        print(f"  {arm:8} {verdict:12} differs on {differs}/{len(rows)} questions "
              f"(served: {', '.join(report['arms'][arm]['served_models']) or '—'})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    if not all_pass:
        print("\nFAIL: at least one arm produced byte-identical output to base.")
        return 1
    print("\nAll arms differ from base — adapters are applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
