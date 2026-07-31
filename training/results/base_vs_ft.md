# Base vs fine-tuned Nemotron — controlled comparison

Reproduce with:

```bash
bash training/scripts/serve_ft.sh --with-adapters
python training/eval/fingerprint.py --arms ck20 ck40 ck60 ck80 ck100
python training/eval/run_eval.py --arms base ck20 ck40 ck60 ck80 ck100 \
    --conditions clean --out training/results/public15.md
python training/eval/run_eval.py --arms base ck40 ck100 \
    --conditions clean noisy insufficient shuffled --prompt-2x2 \
    --questions training/eval_sets/heldout19.jsonl \
    --frozen training/eval_sets/frozen_heldout19 \
    --out training/results/heldout19_full.md
```

Detail tables: [`public15.md`](public15.md), [`heldout19_full.md`](heldout19_full.md).
Raw generations: `*_raw.json` beside each.

## Method

Every arm answers the same questions from **byte-identical frozen tool
evidence**. Qwen routing and tool execution are held fixed — what the handout
asks for — so the only variable between arms is the synthesis model's weights.
All arms are served from one vLLM process as separate model ids, so switching arm
is a different `model` field rather than a restart, giving identical hardware
state between measurements.

- Judge: the calibrated component judge, 15/15 on reference answers and 12/12 on
  adversarial negatives ([`judge_calibration.md`](judge_calibration.md))
- Decoding: greedy, `temperature=0`, `seed=0`
- Deltas: paired bootstrap over **questions** (2,000 resamples), because
  components within a question are correlated; p-values from an exact sign-flip
  permutation test on paired differences

Adapters were verified genuinely applied before any scoring: at temperature 0
every arm differs from base on 5/5 probe questions, and vLLM echoes the adapter
id per request ([`adapter_fingerprint.json`](../../logs/eval/adapter_fingerprint.json)).

## ⚠ Read first: one of our two eval sets is a biased instrument

| set | n | questions & reference answers written by | valid for accuracy? |
|---|---:|---|---|
| `public15` | 15 | **the organizers** | **yes** |
| `heldout19` | 19 | our own generators | **no — see below** |

`heldout19` holds out the right *parameters* (reserved tickers, year 2017) and is
verified to share no fact set with training. But its `expected_fact` strings come
from the **same verbalizers that wrote the training targets**, so the fine-tuned
model reproduces them near-verbatim while base states the same correct value in
its own words and is marked down:

```
EXPECTED: "there are 1,326 AFR records matching whole-word superannuation in 2017"
ck40    : "There are 1,326 AFR records matching whole-word superannuation in 2017."
base    : "The once-per-record whole-word superannuation AFR count for 2017 is 1326."
```

Both are factually correct. The fine-tuned answer scores higher for matching our
phrasing, not for being more accurate.

**`heldout19` therefore measures format adherence to our own templates, and its
large deltas must not be read as an accuracy win.** We report it below for what
it is validly good for — format control and robustness across evidence
conditions — and rest the accuracy claim on `public15` alone.

This is a flaw in our eval design, found by cross-checking the two sets against
each other. A fixed version would author reference answers independently of the
generators.

## Headline: `public15` (the organizers' questions)

| arm | component | strict | halluc. numbers | words p50 | gen time |
|---|---:|---:|---:|---:|---:|
| `base` | **57.1%** | 40.4% | 6.7% | 26 | 24.3s |
| `ck20` | 41.1% | 39.1% | 26.7% | 13 | 11.0s |
| `ck40` | 45.6% | 40.2% | 20.0% | 13 | 12.1s |
| `ck60` | 51.7% | 46.7% | 20.0% | 13 | 11.8s |
| `ck80` | 50.0% | 50.0% | 13.3% | 13 | 11.4s |
| `ck100` | 54.7% | **51.3%** | 13.3% | 13 | 11.3s |

### Component score — no arm beats base

| arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| `ck20` | −16.0% ⚠ | [−35.3%, +0.0%] | 0.1875 | 1/4/10 |
| `ck40` | −11.6% ⚠ | [−27.8%, +0.7%] | 0.1875 | 1/4/10 |
| `ck60` | −5.4% ⚠ | [−28.0%, +15.7%] | 0.6562 | 3/3/9 |
| `ck80` | −7.1% ⚠ | [−23.1%, +5.3%] | 0.5000 | 2/3/10 |
| `ck100` | −2.4% ⚠ | [−20.0%, +13.3%] | 0.8750 | 4/2/9 |

⚠ = 95% CI includes zero.

**On the organizers' questions the fine-tune does not improve component score.**
Every interval crosses zero, so nothing is distinguishable from base in either
direction — but the point estimates are negative and we will not describe that as
a win.

### Strict score — this one does improve

Strict requires the judge **and** the numeric tolerance check to agree, so it
penalises answers that win a verdict while mis-stating or mis-formatting the
figure.

| arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| `ck20` | −1.3% ⚠ | [−20.0%, +20.0%] | 1.0000 | 2/2/11 |
| `ck40` | −0.2% ⚠ | [−14.9%, +18.7%] | 1.0000 | 2/3/10 |
| `ck60` | +6.2% ⚠ | [−7.3%, +23.6%] | 0.5625 | 3/2/10 |
| `ck80` | +9.6% ⚠ | [−0.7%, +25.1%] | 0.2500 | 3/1/11 |
| `ck100` | **+10.9%** | **[+1.3%, +26.4%]** | 0.1250 | **4/0/11** |

`ck100` improves strict score with a bootstrap interval that excludes zero and
**never loses a question** (4 wins, 0 losses, 11 ties). Note the p-value cannot
go below 0.125 here: with only 4 discordant pairs the exact sign-flip test has a
floor of 2/2⁴. The interval and the p-value are not in conflict — the sample is
simply too small for the permutation test to resolve. We report both rather than
quoting whichever looks better.

### Trend: undertrained, not overfit

The gap closes monotonically, ck20 (−16.0%) → ck100 (−2.4%). 100 steps at
effective batch 8 is **1.01 epochs** over 790 rows, at half the reference
learning rate. Nothing here looks like overfitting; it looks like a run that
stopped early. This also contradicts the handout's suggestion that step 20 is
often the best checkpoint — on this data it is reliably the worst.

### Latency roughly halves

24.3s → 11.3s, and median answer length drops 26 → 13 words. Under a 60-second
budget with a −20% penalty beyond it, this is worth real points on the hidden
set, and it is the most robust improvement we measured.

## Where the fine-tune loses on `public15`

Only two questions regress, with specific causes.

**MHQ040** — dimensions and date range:

```
base : ...18 tickers and 1774 records per ticker, ... start '2015-01-02' and end '2021-12-30'.
ck100: The ASX data runs 18 tickers. Covering 2 Feb 2015 through 30 Dec 2021.
```

`2 Feb 2015` is wrong — the evidence says 2 Jan — and the 1,774-rows component is
dropped. A genuine hallucination plus an omission.

**MHQ061** — peak year *and* peak month:

```
base : ...2020 with 1452 cases, and the month ... May 2020 with 218 cases.
ck100: It peaked in 2020 with 1,452 matching records.
```

The month component is missing. This is exactly the failure the
sampled-component-subset design was meant to prevent, and it did not fully work:
training on many 1-component examples appears to have taught brevity as a prior
strong enough to drop a requested component.

## The serious regression: invention under insufficient evidence

Four evidence conditions on `heldout19`. The `insufficient` condition removes an
evidence block, so the requested fact is genuinely unsupported and the **correct**
behaviour is to say so.

| arm | clean | noisy | insufficient | shuffled |
|---|---:|---:|---:|---:|
| `base` | 80.7% | 87.7% | 5.3% | 86.0% |
| `ck40` | 94.7% | 82.5% | 7.9% | 79.8% |
| `ck100` | 89.5% | 86.8% | 14.9% | 89.5% |

Component score is the wrong lens for `insufficient` — a low score there is
correct behaviour. The right lens is whether the arm invents figures:

| arm | hallucinated-number rate under `insufficient` |
|---|---:|
| `base` | **0.0%** |
| `ck40` | **73.7%** |
| `ck100` | **73.7%** |

```
base : "The basket average for the 2017 report is not available as no tool
        results were gathered."
ck40 : "The worst performer in 2017 was ANZ.AX, dropping 50.04%. The non-Tabcorp
        basket averaged 6.78%. ANZ.AX ended at 17.11, compared to 16.40 ... It
        ranked 17 of 20."
```

Every figure in the ck40 answer is fabricated. Base declines correctly.

This is the most damaging finding in the sweep. The rules require stating the
limitation rather than inventing a figure; an invented answer scores zero and is
the failure a judge is most likely to notice.

**Cause — corpus balance, not hyperparameters.** Refusal is 5.1% of 790 rows and
the empty-trace negatives number ~10 examples, against ~95% of rows where
complete evidence is present and a confident terse answer is correct. The model
learned the dominant pattern. The N1/N2 negative design was right in kind and far
too small in quantity.

**Fix, not applied here for time reasons:** raise the insufficient-evidence share
to ~20% of the corpus and keep the discrimination guard explicit — for each
insufficient case, the same question shape with complete evidence, so the model
learns to distinguish rather than to hedge. This is a data change; the pipeline
supports it directly via `TARGET_MIX` in `datagen/build.py`.

## Prompt fairness control (2×2)

The system prompt was shortened 332 → 145 tokens to fit the sequence budget, so
the fine-tuned arms were trained on the short prompt. Without this control the
comparison would silently be "base + long prompt vs FT + short prompt".

| arm | short prompt (shipped) | long prompt (pre-fine-tune) |
|---|---:|---:|
| `base` | 80.7% | **90.3%** |
| `ck40` | **94.7%** | 86.0% |
| `ck100` | 89.5% | **94.7%** |

**This control earns its keep.** Base scores 9.6pp *higher* on the long prompt —
so the prompt we shortened for training genuinely handicaps the base arm, and any
comparison using only the short prompt overstates the fine-tune. Giving each arm
its better prompt: base 90.3%, ck40 94.7%, ck100 94.7% — a much narrower spread
than the short-prompt-only view suggests.

*(On `heldout19`, so also subject to the template-bias caveat above. The
direction of the base-arm effect is unaffected by that bias, since it compares
base against itself.)*

## Robustness to evidence shape

`shuffled` reverses block order and payload key order without changing any value;
`noisy` prepends an irrelevant block. Neither fine-tuned arm collapses —
ck100 is 89.5% clean and 89.5% shuffled, ck40 94.7% → 79.8%, base 80.7% → 86.0%.
The model is reading the evidence rather than copying from fixed positions.

`leaked_reasoning_rate` and `format_violation_rate` are 0% for every arm and
condition, base included.

## Checkpoint selection

The pre-registered rule: disqualify any arm with hallucinated-number rate above
base or error rate above 2%; rank survivors by component score on held-out
questions; tie-break inside the paired CI by the probe set, then p95 latency,
then earlier step.

Applying it honestly is awkward, because the rule assumed `heldout19` would be a
valid accuracy instrument and it is not:

- On the **valid** set (`public15`), **`ck100` is the best fine-tuned arm**:
  highest component score (54.7%), highest strict score (51.3%, CI excluding
  zero, 4W/0L/11T), lowest hallucination rate among FT arms (13.3%), and the
  smallest gap to base (−2.4%).
- On `heldout19`, `ck40` leads — but that set rewards reproducing our own
  phrasing, so its ranking is not trustworthy here.
- **Every arm fails the disqualification clause** on the `insufficient`
  condition (73.7% vs base 0.0%).

**Selected: `ck100`**, on the `public15` evidence, with the abstention regression
documented as a known limitation rather than hidden. `ck40` was the initial pick
before the template bias in `heldout19` was identified; artifacts for both are
retained and the choice is a one-line config change.

## Honest summary

| | verdict |
|---|---|
| Component accuracy vs base | **Not improved.** All CIs cross zero; point estimates negative. |
| Strict score (judge + tolerance) | **Improved**, +10.9pp for ck100, CI [+1.3, +26.4], 4W/0L/11T. |
| Latency | **Roughly halved**, 24.3s → 11.3s. |
| Answer length | 26 → 13 median words, closely matching reference style. |
| Leaked reasoning / format violations | 0% for every arm, base included. |
| Robustness to shuffled/noisy evidence | Holds up; no positional copying. |
| Insufficient evidence | **Regressed badly.** 0% → 73.7% invented figures. |
| Overfitting | Not observed. Undertrained — 1.01 epochs at half the reference LR. |

We are not claiming a headline accuracy win, because the measurement does not
support one. The defensible claims are the latency halving, the strict-score
improvement, the format control, and a precisely diagnosed regression with an
identified cause and a concrete fix.
