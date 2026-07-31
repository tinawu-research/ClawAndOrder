# `training/` — fine-tuning evidence

LoRA fine-tune of `Llama-3.1-Nemotron-Nano-8B-v1` for grounded answer synthesis.
Qwen3.6-35B (`agent-brain`) plans and calls tools and is **not** modified; this
model writes the final `answer` field from the verified tool results.

## Start here

| | |
|---|---|
| **What was done and why** | [`MODEL_CARD.md`](MODEL_CARD.md) |
| **Did it beat the base model?** | [`results/base_vs_ft.md`](results/base_vs_ft.md) |
| **How the data was built** | [`data_prep/DATA_CARD.md`](data_prep/DATA_CARD.md) |
| **What happened during training** | [`logs/README.md`](logs/README.md) |

**Headline, stated plainly:** the fine-tune did **not** improve component
accuracy on the organizers' questions (all confidence intervals cross zero). It
did roughly halve latency (24.3s → 11.3s), improve the strict score by +10.9pp
without losing a single question, and match the reference answer format closely.
It also regressed badly on insufficient evidence, inventing figures where the
base model correctly declines. All of that is documented rather than buried, with
the cause identified and the fix specified.

## Contents against the submission checklist

| required | where |
|---|---|
| Training / fine-tuning scripts | [`scripts/train.sh`](scripts/train.sh), [`scripts/serve_ft.sh`](scripts/serve_ft.sh), [`trainer.py`](trainer.py) |
| Data-preparation scripts | [`datagen/`](datagen/) — 11 modules, entry point [`datagen/build.py`](datagen/build.py) |
| Configuration and hyperparameters | [`configs/lora_nemotron_r32.yaml`](configs/lora_nemotron_r32.yaml), fully commented with the measurements behind each deviation |
| Training logs and metrics | [`logs/`](logs/) — all five runs including the three failures, indexed in [`logs/README.md`](logs/README.md) |
| Model card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Held-out results vs base | [`results/base_vs_ft.md`](results/base_vs_ft.md), [`results/public15.md`](results/public15.md), [`results/heldout19_full.md`](results/heldout19_full.md) |

## Reproducing end to end

```bash
# 1. Build the corpus (790 train / 144 val). Gates fail loudly.
python training/datagen/label_sentiment.py      # Qwen teacher labels, cached
python training/datagen/build.py --budget 1024 --target 800
python training/datagen/verify.py

# 2. Train (~14 min on one GB10)
bash training/scripts/train.sh --dry-run        # validate config + data first
bash training/scripts/train.sh

# 3. Serve base + every checkpoint as separate model ids
bash training/scripts/serve_ft.sh --with-adapters

# 4. Prove the adapters are actually applied, then measure
python training/eval/fingerprint.py --arms ck20 ck40 ck60 ck80 ck100
python training/eval/run_eval.py --arms base ck20 ck40 ck60 ck80 ck100 \
    --conditions clean --out training/results/public15.md
```

## How the evaluation works

Tool traces are captured once and replayed **byte-identically** to every arm, so
Qwen routing and tool execution are held fixed and the only variable is the
synthesis weights. Arms are served from one vLLM process as separate model ids,
so switching arm is a different `model` field, not a restart.

Four evidence conditions, because a clean-only result measures only the best
case: `clean`, `noisy` (irrelevant block prepended), `insufficient` (a block
removed), `shuffled` (block and key order reversed, no values changed).

Grading reuses the calibrated component judge — 15/15 on the reference answers
and 12/12 on adversarial negatives before any model delta was believed
([`results/judge_calibration.md`](results/judge_calibration.md)). Deltas are
reported with a paired bootstrap over questions and an exact sign-flip
permutation p-value; an interval crossing zero is marked as not demonstrated.

Two fairness controls are included because they change the conclusion:

- **Prompt 2×2** — the system prompt was shortened 332 → 145 tokens for training,
  so both arms are run against both prompts. Base scores 9.6pp *higher* on the
  long prompt, meaning a short-prompt-only comparison would have overstated the
  fine-tune.
- **Reasoning stripped** — Nemotron-Nano is a toggleable reasoning model;
  `<think>` blocks are removed identically from both arms before judging.

We also report that **our own held-out set turned out to be a biased instrument**
— its expected facts come from the same verbalizers that wrote the training
targets, so it rewards reproducing our phrasing rather than being correct. The
accuracy claim therefore rests on the organizers' `public15` alone. This was
found by cross-checking the two sets against each other.

## Layout

```
configs/     LoRA recipe (commented with the evidence for each choice)
data/        train/val jsonl in both chat and input/output form, build report
data_prep/   DATA_CARD.md
datagen/     generators, perturbations, sentiment teacher, build, verify
eval/        judge, tolerances, scorer, replay, secondary, stats, fingerprint
eval_sets/   frozen evidence for public15 + generated held-out sets
logs/        every training run, indexed
results/     comparison reports and raw generations
scripts/     train.sh, serve_ft.sh
```

`datagen/render.py` imports `SYNTHESIS_SYSTEM_PROMPT` and `format_evidence` from
the served agent rather than reimplementing them, so train/serve prompt drift is
structurally impossible; `verify.py` asserts byte-identity against live traces.
