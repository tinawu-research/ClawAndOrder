# Data card — synthesis training corpus

790 training rows, 144 validation rows. Built entirely from the supplied
datasets by [`../datagen/build.py`](../datagen/build.py). No hidden evaluation
data is used, and no answer is keyed to a question id.

Reproduce with:

```bash
python training/datagen/label_sentiment.py     # Qwen teacher labels (optional, cached)
python training/datagen/build.py --budget 1024 --target 800
python training/datagen/verify.py              # all gates must pass
```

## Provenance

| source | path | used for |
|---|---|---|
| RBA cash rate | `data set/RBA/` | rate lookups, change counts, cycle summaries |
| ASX daily bars | `data set/ASX/*.jsonl` | returns, volume, drawdown, correlation |
| AFR articles | `data set/AFR/` | article counts by month/year, sentiment snippets |

Every gold answer is **derived from a real tool payload**, never written from
model knowledge. The generator calls the same `src/agent/tools/{rba,asx,afr}.py`
functions the live agent calls, keeps the returned payload, and renders the
answer from it with a verbalizer. Targets are therefore grounded by construction
and numerically exact for free.

Generation is parameterised over `(dataset, metric, args)` tuples. Nothing is
keyed to a question id, and the 15 public questions are not used as templates.

## Composition

| category | rows | share | target | what it teaches |
|---|---:|---:|---:|---|
| sentiment | 192 | 24.3% | 16% | the only non-derivable behaviour |
| afr_counts | 122 | 15.4% | 12% | month/year peaks, count formatting |
| rba_counts | 103 | 13.0% | 12% | change counts, cycle direction |
| asx_returns | 72 | 9.1% | 12% | ranked returns, signed percentages |
| rba_asx_event | 68 | 8.6% | 13% | two-tool composition |
| coverage | 60 | 7.6% | 5% | dataset shape (near-verbatim copy) |
| composite | 51 | 6.5% | 10% | multi-dataset, most components |
| asx_volume | 49 | 6.2% | 5% | volume formatting |
| refusal | 40 | 5.1% | 8% | state the limitation, don't invent |
| asx_drawdown | 33 | 4.2% | 7% | the `1) … 2) … 3) …` ranking format |

Sentiment and refusal are oversampled relative to their share of the public
question set because they are the behaviours least copyable straight out of the
evidence. Coverage and volume are undersampled because they are near-verbatim
copies once the format is learned.

Shares diverge from targets where a category's pool was exhausted; the shortfall
redistribution is capped at 1.5× a category's target share, without which the
entire shortfall lands in whichever pool is deepest (sentiment reached 30.6%
before the cap was added).

Token distribution: min 223, p50 592, p95 958, max 1021. 35 candidates were
dropped for exceeding the 1024 budget.

## Sampled component subsets

The highest-value design decision in the corpus, and it costs nothing.

A template that always renders `rank_annual_returns → "X was best, Y was worst"`
teaches a metric→sentence mapping. The model then drops the third component when
a hidden question asks for three. Because grading is per-component, the skill
actually being bought is *component-following*, not metric-verbalizing.

So each payload is decomposed into its available components, a subset of 1–4 is
sampled, and both the question and the answer are rendered over exactly that
subset, in the order asked:

```
params → tool payload → available_components{best, worst, basket_avg, span, rank_of(X), n}
                      → sample requested_subset (1–4)
                      → question renders exactly that subset
                      → answer verbalizes exactly that subset, in the asked order
```

One payload yields 6–10 distinct examples.

## Answer formatting

Tools return `22.1712`, `0.1`, `2019-06-05`; the reference answers want
`+22.17%`, `0.10%`, `5 Jun 2019`. Verbalizers in
[`../datagen/answers.py`](../datagen/answers.py) match the references exactly:

| kind | format | example |
|---|---|---|
| returns, drawdowns, volatility | `f"{v:+.2f}%"` — always signed | `+22.17%` |
| RBA rates | `f"{v:.2f}%"` | `0.10%` |
| cumulative rate change | `f"{v:+.2f} percentage points"` | `-2.25 percentage points` |
| counts | `f"{n:,}"` | `1,774` |
| average volume | `f"{v:,.2f} shares per trading day"` | |
| correlations | 3 dp | |
| closes | 4 dp | |
| dates | `d Mon YYYY` (12 of 15 references use it) | `5 Jun 2019` |
| months | `Mon YYYY` | `May 2020` |
| rankings | one line, semicolon-separated | `1) AMP.AX -82.45%, 20 Mar 2015 to 17 Dec 2021; 2) …` |

## Realism: closing the sim-to-real gap

Evidence blocks are rendered by the **production** `format_evidence`, imported
rather than reimplemented (see below). The tool names in the header are the three
the brain actually emits — `query_data`, `retrieve`, `dataset_coverage` — with
the arguments Qwen would emit (`{"dataset": "asx", "metric": …}`), not
`asx.rank_annual_returns`. Getting this wrong would put every example
off-distribution.

**Perturbations** ([`../datagen/perturb.py`](../datagen/perturb.py)) applied to
35% of examples, all modelled on real Qwen habits:

| perturbation | effect |
|---|---|
| coverage probe | prepends a `dataset_coverage` block |
| distractor | inserts an irrelevant but real block |
| shuffle | permutes block order |
| duplicate with variant | re-runs the metric without an exclusion, so two blocks disagree |
| truncate block | clips a payload mid-way |

Error blocks are deliberately **not** in the corpus: the orchestrator filters
`ok=False` before synthesis, so that distribution never occurs at inference.

## Negative and robustness examples

~12% of the corpus. Every one of these **answers and then states the gap** —
never an empty response and never a bare refusal.

| type | behaviour taught |
|---|---|
| N1 missing component | state what is supported, then note the unsupported part |
| N2 empty trace | state that the supplied data does not contain the evidence |
| N5 distractor blocks | ignore them, answer only what was asked |
| N6 contradictory duplicates | pick the block matching the question's exclusion |
| N7 truncated block | state what is visible, do not extrapolate |

This follows the rules directly: return a response for every question, state the
limitation in the `answer` field, do not invent a figure. A bare "I cannot
answer" scores zero.

**Discrimination guard:** for every N1/N2 example there are ~2 with the same
question shape but *complete* evidence and a confident full answer. Without it
the model learns "question looks like this → hedge", which is worse than the
problem being fixed.

Target answers never mention tools, `tool_trace`, or the evidence format — an
early draft phrased the N2 text as "The supplied tool results…", which leaks
pipeline internals into a graded field.

## Sentiment labels — Qwen teacher distillation

Only the 5-way sentiment label needs a teacher; the rate is copied from
`lookup_rate` and the direction is a function of the label.

Qwen labels articles offline and the labels are written into **Nemotron's**
training targets. Qwen's own weights are never modified — the competition permits
fine-tuning only the Nemotron model.

Protocol ([`../datagen/label_sentiment.py`](../datagen/label_sentiment.py)):

- **Labelled from the byte-identical snippet the student will see** — `retrieve`
  returns `blob[:1200]`, lowercased. Showing the teacher the full article while
  the student sees the lede trains the student to guess. This is the most common
  way teacher distillation quietly fails.
- 5-point ordinal: `positive` / `mixed_positive` / `mixed` / `mixed_negative` /
  `negative`. Three classes is too coarse — the reference answers use phrasings
  like "mixed with a negative bias".
- Constrained JSON with a `span` field that must be a literal substring of the
  snippet, which catches confabulated evidence.
- 3-vote self-consistency at temperature 0.7 across three rubric phrasings;
  kept only when the coarse class was unanimous.

**244 of 390 candidates kept (63%).** The 146 discarded are cases where the
teacher was not self-consistent. At ~800 total sequences a noisy label costs more
than a missing one.

| label | n |
|---|---:|
| positive | 120 |
| negative | 100 |
| mixed | 13 |
| mixed_negative | 6 |
| mixed_positive | 5 |

The model has therefore seen only the unambiguous end of the sentiment
distribution — recorded as a limitation in the model card.

## Splits and leakage control

The unit of leakage is (template family × parameter instantiation × entity), not
the row.

| axis | held out from training |
|---|---|
| Entity | `SUN.AX`, `TPG.AX`, `CMW.AX` never a *subject* (may appear inside a basket) |
| Time | year 2017 never a training parameter |
| Article | sentiment articles disjoint between splits |

Paraphrases and perturbations of one fact set share a `gold_key` and are kept on
the same side of the split; otherwise validation measures memorisation. Splitting
is by gold key, never by row.

## Ordering

`train.io.jsonl` is written in **round-robin category order**, and the training
recipe sets `shuffle: false` to preserve it.

Checkpoints land every 20 steps, and step 20 sees only the first 160 rows. Any
prefix must therefore be category-balanced for an early checkpoint to be worth
evaluating. Achieved exactly — the first 160 rows contain **16 of each of the 10
categories**. The build fails if any category appears fewer than 6 times there.

Within a category, examples are ordered shortest-first, which gives a free
curriculum effect.

## Train/serve prompt parity

[`../datagen/render.py`](../datagen/render.py) does:

```python
sys.path.insert(0, str(AGENT_DIR))
from synthesis import SYNTHESIS_SYSTEM_PROMPT, format_evidence
```

The production prompt and evidence renderer are **imported, not copied**. Train/
serve prompt drift is the most common self-inflicted wound in this setup and the
only robust defence is to not have two copies of the string.

## Build gates

[`../datagen/verify.py`](../datagen/verify.py) — all must pass:

```
ok    prompt parity with the served agent
ok    system prompt is imported, not copied
ok    train/val leakage
ok    prefix category balance
ok    token budget
ok    target answer hygiene
train=790 val=144 budget=1024
```

| gate | what it prevents |
|---|---|
| prompt parity | silent train/infer mismatch that would cost the entire LoRA |
| imported prompt | a copied string drifting from the served one |
| leakage | validation measuring memorisation |
| prefix balance | an early checkpoint trained on one category |
| token budget | right-truncation silently deleting the assistant span |
| answer hygiene | tool names, config, or pipeline internals leaking into `answer` |

Token counts come from the **real tokenizer** over HTTP
(`10.0.1.11:8001/tokenize`), not an estimate, and are sqlite-cached.
