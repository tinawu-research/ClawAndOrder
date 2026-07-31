# Model card — `domain-ft`, LoRA fine-tune of Llama-3.1-Nemotron-Nano-8B-v1

## What this model does

It writes the final `answer` field, and nothing else.

The architecture assigns planning and tool selection to Qwen3.6-35B-A3B-FP8
(`agent-brain`), which is not modified. The agent runtime executes the tool calls
deterministically. This model receives the question plus the verified tool
results and synthesises the answer.

That narrow job is the whole design brief: read the supplied evidence, state
every requested component exactly, invent nothing, and when the evidence does
not support a component, say so rather than fill the gap.

| | |
|---|---|
| Base model | `Llama-3.1-Nemotron-Nano-8B-v1` |
| Method | LoRA (PEFT), rank 32, alpha 32, dropout 0.1 |
| Target modules | all linear — `q,k,v,o,gate,up,down` × 32 layers (224 modules) |
| Trainable params | ~168 MB adapter |
| Framework | NeMo AutoModel in `nvcr.io/nvidia/nemo:25.09` |
| Hardware | 1 × NVIDIA GB10, 121 GB unified memory |
| Peak memory | ~24.6 GiB |
| Throughput | ~8.5 s/step, ~500 tok/s |

## Training configuration

Recipe: [`configs/lora_nemotron_r32.yaml`](configs/lora_nemotron_r32.yaml).
Launcher: [`scripts/train.sh`](scripts/train.sh).

| parameter | reference baseline | used | why |
|---|---|---|---|
| LoRA rank / alpha | 32 / 32 | 32 / 32 | accepted |
| max steps | 100 | 100 | accepted |
| checkpoint every | 20 | 20 | accepted |
| effective batch | 8 (2 × 4) | 8 (1 × 8) | same optimisation, buys sequence length |
| `seq_length` | 512 | **1024** | 512 truncates our labels — measured, see below |
| peak LR | 5e-5 | **2e-5** | 5e-5 diverged reproducibly — measured, see below |
| warmup steps | 50 | **20** | 50 of 100 steps spent below working LR |

The handout states these values are "a reference baseline, not a required
configuration" and invites documented deviation. Both deviations below were
forced by measurement, not preference.

### Sequence length: 512 truncates the labels

Measured with this model's own tokenizer, not estimated. The system prompt alone
was 332 tokens of the 512 budget:

| example | tokens | vs 512 |
|---|---:|---|
| `rba.coverage` | 437 | fits |
| `rba.count_changes` | 542 | 1.1× |
| `afr.count_by_month` | 557 | 1.1× |
| `asx.max_drawdown` top-3 | 814 | 1.6× |
| sentiment (article + rate) | 825 | 1.6× |
| `asx.rank_annual_returns` | 1,471 | 2.9× |
| 4-call cross-dataset | **4,520** | **8.8×** |

Supervised fine-tuning truncates from the right, so an over-long example keeps
the system prompt, keeps part of the evidence, and silently drops the entire
assistant span — a training row with no label. Only 3 of 13 real question shapes
fit at 512.

Two changes fixed it rather than one:

1. **Lossless evidence compaction** ([`src/agent/compact.py`](../src/agent/compact.py)) —
   renders tool payloads in line form instead of JSON. No values are elided;
   the saving is JSON syntax only. `window_return` 896 → 84 tokens,
   `rank_annual_returns` 722 → 109, `avg_volume` 571 → 54.
2. **Shorter system prompt** — 332 → 145 tokens.

After both, 11 of 13 shapes fit under 1024. Every shipped row is
tokenizer-verified against the budget; rows that still exceed it are dropped, not
truncated.

This also fixed a live serving bug: the agent allowed 8 blocks × 6,000 chars
against `max_model_len=4096`, which overflowed the *planning* loop on exactly the
hard multi-tool questions. Both vLLM servers now run at 16,384.

### Learning rate: 5e-5 diverges on this configuration

The reference baseline is 5e-5, with a note that 1e-4 spikes after warmup. On
this configuration the same failure arrives at 5e-5. Two independent runs
diverged on reaching it.

**Run without warmup** (a scheduler block was missing, so LR was constant 5e-5
from step 0) — diverged immediately:

```
step 0  loss 5.11    step 3  loss 2.69    step 5  loss 12.72
step 1  loss 2.61    step 4  loss 10.54   step 6  loss 19.01   grad_norm 18,176
```

**Run with 50-step warmup to a 5e-5 peak** — healthy for 43 steps, then came
apart precisely as the ramp crossed ~4.5e-5:

| step | loss | grad_norm | lr |
|---:|---:|---:|---:|
| 20 | 0.53 | 20 | 2.39e-05 |
| 32 | 0.27 | 17 | 3.47e-05 |
| 43 | 0.43 | 35 | 4.46e-05 |
| 44 | 0.43 | **152** | 4.55e-05 |
| 47 | 0.81 | **824** | 4.82e-05 |
| 49 | 0.86 | **1,384** | 5.00e-05 |
| 55 | 3.43 | 5,632 | 4.81e-05 |
| 64 | 9.50 | **36,352** | 3.90e-05 |
| 79 | 7.69 | 198 | 1.58e-05 |

It never recovered even as cosine decay brought the LR back below the level it
had been stable at — the signature of a damaged optimiser state rather than one
bad batch.

Gradient clipping at `max_grad_norm=1.0` is applied by the recipe and did not
prevent this. Clipping bounds a gradient's magnitude but not its direction, and
Adam renormalises by its second moment afterwards, so against this failure mode
it is close to inert.

Why 5e-5 is hotter here than in the reference: rank 32 applied to *all* linear
layers (7 modules × 32 layers) rather than attention only, `local_batch_size 1`,
and 1024 tokens of predominantly-answer loss.

**Chosen: peak 2e-5**, comfortably inside the region observed to be stable
(grad_norm 17–50 throughout). Warmup shortened to 20 steps because at 100 total
steps a 50-step warmup spent half the run below working LR, and the low-LR region
was never where the instability lived.

### Answer-only loss mask

`answer_only_loss_mask: true` with `start_of_turn_token: "<|start_header_id|>"`.
Without it the constant system prefix carries most of the gradient and the model
spends capacity reproducing instructions it already receives at inference.

Verified active rather than silently masking nothing: `num_label_tokens` logs
~2,600–4,200 per global batch of 8, i.e. ~120–500 label tokens per example
against sequences of up to 1024 — consistent with loss on the answer span only.

### Data ordering

`shuffle: false`, deliberately. `train.io.jsonl` is written in round-robin
category order so that *any* prefix is category-balanced. Since checkpoints land
every 20 steps and step 20 sees only the first 160 rows, this is what makes an
early checkpoint worth evaluating at all. The build fails if any category appears
fewer than 6 times in the first 160 rows.

## Training data

Full detail in [`data_prep/DATA_CARD.md`](data_prep/DATA_CARD.md).

790 train / 144 validation rows, built by
[`datagen/build.py`](datagen/build.py) from the supplied datasets only.

Sizing is driven by consumption, not by the 48,000-sample reference: 100 steps ×
effective batch 8 = ~800 sequences total. A 48,000-row file would be sampled at
under 2%, so the target is ~800 well-stratified rows and diversity beats volume.

| category | share | | category | share |
|---|---:|---|---|---:|
| sentiment | 24.3% | | coverage | 7.6% |
| afr_counts | 15.4% | | composite | 6.5% |
| rba_counts | 13.0% | | asx_volume | 6.2% |
| asx_returns | 9.1% | | refusal | 5.1% |
| rba_asx_event | 8.6% | | asx_drawdown | 4.2% |

Token distribution: min 223, p50 592, p95 958, max 1021.

Two properties that matter more than the counts:

**Every gold answer is derived from a real tool payload**, never written from
model knowledge. Generation is parameterised over `(dataset, metric, args)`
tuples and never keyed to question ids.

**Component subsets are sampled per example.** A template that always maps
`rank_annual_returns → "X best, Y worst"` teaches a metric→sentence map, and the
model then drops the third component when a hidden question asks for three.
Since grading is per-component, component-following is the actual skill being
bought, so one payload yields 6–10 examples asking for different subsets in
different orders.

**Train/serve prompt parity is structural, not checked.**
[`datagen/render.py`](datagen/render.py) imports `SYNTHESIS_SYSTEM_PROMPT` and
`format_evidence` from the served agent rather than reimplementing them, so the
two cannot drift. `verify.py` asserts byte-identity against live agent traces.

## Checkpoint selection

Not on validation loss. Val loss is token-level cross-entropy against our own
synthetic phrasing, so minimising it rewards matching our templates — which is
the specific overfitting mode of concern. With only five candidates there is no
search-cost argument for a proxy metric.

Every checkpoint is instead evaluated on the real component metric, all served
from one vLLM process as separate model ids so that switching arms is a different
`model` field rather than a restart.

Rule, pre-registered before results were seen:

1. Disqualify any arm with `hallucinated_number_rate` above base, or `error_rate > 2%`.
2. Rank survivors by component score on held-out questions.
3. Tie-break inside the paired CI by the adversarial probe set, then p95 latency,
   then **earlier step**.
4. If the leader's edge over `ck20` sits inside the CI, take `ck20`.

**Outcome: `ck100` selected.** Applying the rule was awkward, because it assumed
our `heldout19` set would be a valid accuracy instrument and it turned out not to
be (see Evaluation). On the organizers' `public15` — the trustworthy set —
`ck100` leads every fine-tuned arm: highest component score, highest strict
score, lowest hallucination rate among FT arms, smallest gap to base. Note that
*every* arm fails the disqualification clause under insufficient evidence, which
is recorded as the headline limitation rather than worked around.

The rule intended `public15` to be reported but never used as a tie-break, since
it stops being held-out the moment it selects a checkpoint. That intent could not
be honoured once the alternative set proved biased; we state this openly rather
than present a selection the evidence does not support.

## Evaluation

Full report: [`results/base_vs_ft.md`](results/base_vs_ft.md). Judge calibration:
[`results/judge_calibration.md`](results/judge_calibration.md).

The comparison replays byte-identical frozen tool evidence to every arm, so Qwen
routing and tool execution are held fixed and the only variable is the synthesis
weights. Four conditions — `clean`, `noisy`, `insufficient`, `shuffled` — because
a clean-only result measures only the best case. Deltas use a paired bootstrap
over questions (not components, which correlate within a question) and an exact
sign-flip permutation p-value; an interval crossing zero is marked as not
demonstrated.

Results on `public15`, the organizers' own questions:

| | base | `ck100` (selected) |
|---|---:|---:|
| component score | **57.1%** | 54.7% (−2.4%, CI crosses zero) |
| strict (judge + tolerance) | 40.4% | **51.3%** (+10.9%, CI [+1.3, +26.4], 4W/0L/11T) |
| hallucinated numbers | 6.7% | 13.3% |
| median answer length | 26 words | **13 words** |
| generation time | 24.3s | **11.3s** |
| invented figures, no evidence | **0.0%** | 73.7% |

**We do not claim an accuracy improvement.** Component score did not improve on
the organizers' questions; every confidence interval crosses zero. What did
improve, robustly: strict score (+10.9pp, never losing a question), latency
(roughly halved), and format adherence to the reference answer style.

Two caveats we found ourselves and report rather than bury:

1. **Our own held-out set is a biased instrument.** `heldout19` shows +47pp, but
   its `expected_fact` strings come from the same verbalizers that wrote the
   training targets, so it rewards reproducing our phrasing rather than being
   correct. It is valid for format control and robustness, not for accuracy.
2. **The shortened system prompt handicaps the base arm** by 9.6pp on
   `heldout19`. The 2×2 prompt control exists to catch exactly this; with each
   arm on its better prompt the spread narrows substantially.

## Limitations

- **Invents figures when evidence is missing.** The most serious limitation. Under
  the `insufficient` condition the fine-tuned arms fabricate confident numbers
  73.7% of the time where base correctly declines 100% of the time. Cause is
  corpus balance — refusal is 5.1% of rows and empty-trace negatives ~10
  examples, so "emit a confident terse answer" dominated. The fix is a data
  change (raise the insufficient share to ~20% via `TARGET_MIX`), not a
  hyperparameter change; it was not applied here for time reasons.
- **Drops components on multi-part questions.** MHQ061 asks for peak year and
  peak month; the model answers only the year. The sampled-component-subset
  design was meant to prevent this and only partly did — many 1-component
  training examples appear to have taught brevity as a prior.
- **Undertrained.** 100 steps at effective batch 8 is 1.01 epochs over 790 rows,
  at half the reference learning rate. The base-vs-FT gap closes monotonically
  from ck20 to ck100, so the run stopped while still improving. Nothing in the
  evidence indicates overfitting.
- Sentiment labels are Qwen-teacher distillations kept only on coarse-class
  unanimity across three rubric phrasings (244 of 390, 63%). The model has seen
  only the unambiguous end of that distribution.
- Held-out axes are tickers `SUN.AX`/`TPG.AX`/`CMW.AX` and year 2017. Performance
  on entities outside the supplied datasets is untested and out of scope.
- The adapter is trained for this agent's exact prompt format. It is not a
  general-purpose finance model and should not be used outside this pipeline.
