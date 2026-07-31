# ClawAndOrder

**Team:** ClawAndOrder — Cognitivo Hackathon submission.

An evidence-grounded market-signal agent that answers financial-market questions over the three
approved local datasets (RBA cash-rate decisions, ASX daily bars, the AFR news corpus). Qwen plans
and emits tool calls, the agent runtime executes them deterministically against the raw data, and a
LoRA fine-tuned `Llama-3.1-Nemotron-Nano-8B-v1` writes the final answer from the verified tool
results. Every number in an answer is computed by application code from the supplied data — the
models decide *what* to look up and *how to say it*, never *what the value is*.

| | |
|---|---|
| Agent source | [src/agent/](src/agent/) — see [src/agent/README.md](src/agent/README.md) for internals |
| Fine-tuning target | `Llama-3.1-Nemotron-Nano-8B-v1`, LoRA rank 32 — selected checkpoint **`ck100`** |
| Training evidence | [training/](training/) — [MODEL_CARD.md](training/MODEL_CARD.md), [data_prep/DATA_CARD.md](training/data_prep/DATA_CARD.md), [logs/](training/logs/), [results/](training/results/) |
| Registered endpoint & pinned commit | [submission.json](submission.json) |
| Data loaded at start-up | RBA 175 decisions (2010-02-03 → 2026-06-17); ASX 18 tickers × 1,774 bars (2015-01-02 → 2021-12-30); AFR 219,538 articles across 85 month files (2015-01-02 → 2021-12-29) |

---

## Run the agent

Exactly this, from the repository root, on the agent/head node:

```bash
source ~/team.env                                   # organizer-supplied endpoints + credentials
python -m venv .venv && source .venv/bin/activate   # first run only
pip install -r src/agent/requirements.txt           # first run only

cd src/agent && uvicorn server:app --host 0.0.0.0 --port 8001
```

- `--host 0.0.0.0` because the harness calls from another machine.
- `--port 8001` on the head node. The port is not mandated by the package — the harness reads
  `agent.endpoint` from `submission.json`, so any port works provided the two agree. If you change
  `SERVER_PORT`, change `submission.json` to match. 8001 is also the port the fine-tuned Nemotron
  vLLM uses on the *fine-tuning* node; those are different machines, so there is no clash.
- **`DOMAIN_PREDICT_MODE` defaults to `llm`**, which routes final synthesis to the fine-tuned
  Nemotron — the only mode valid for official evaluation. The variable now needs setting only to go
  deliberately *back* to `mock`, the bootstrap mode that replaces synthesis with a deterministic
  template. Because the value is read at import, a server already running in `mock` stays there:
  check `diagnostics.synthesis_mode` on any `/query` response, and restart if it is not `llm`.
- `GET /health` returns **503 for the first ~30s** while the corpus loads, then 200. That is
  deliberate: health is a hard gate, and answering 200 early would let the harness start against an
  agent that cannot yet answer.

Serving the fine-tuned model, on the fine-tuning node, before starting the agent:

```bash
bash training/scripts/serve_ft.sh --with-adapters   # base + every LoRA checkpoint as separate model ids
```

All configuration is read from environment variables in [src/agent/config.py](src/agent/config.py);
no endpoint, host, IP or credential is hard-coded. [src/agent/.env.example](src/agent/.env.example)
documents every variable.

Local verification:

```bash
cd src/agent && pytest tests -m "not slow"   # RBA + ASX, ~0.2s
cd src/agent && pytest tests                 # adds the AFR corpus + index (~30s warm-up)
python src/agent/eval/run_public_eval.py --workers 3 --show-answers
```

---

## Endpoints and response shape

The graded contract is two endpoints.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness gate. `200 {"status":"ok"}` once every dataset is queryable; `503 {"status":"loading"\|"error","detail":…}` otherwise. |
| `POST` | `/query` | `{"question": "..."}` → the answer object below. |

Non-graded, for humans and debugging only: `GET /` (operations dashboard), `GET /api/status`
(telemetry), `GET /api/public-questions` (the 15 calibration cases), `POST /api/score` (heuristic
component self-check).

### `POST /query`

```http
POST /query
Content-Type: application/json

{"question": "How many times did the RBA change the cash rate target, and how many were increases versus decreases?"}
```

Response — `answer` is the only graded field; `steps` and `tool_trace` are populated for organizer
diagnostics, and `diagnostics` is an extra key of our own:

```json
{
  "answer": "41 of the 175 decision records changed the cash rate target: 20 increases and 21 decreases.",
  "steps": 2,
  "tool_trace": [
    {
      "tool": "query_data",
      "args": {"dataset": "rba", "metric": "count_changes"},
      "result": "{\"metric\": \"count_changes\", \"total_records\": 175, \"changes\": 41, \"increases\": 20, \"decreases\": 21, …}"
    }
  ],
  "diagnostics": {
    "latency_seconds": 11.4,
    "brain_calls": 2,
    "tool_calls": 1,
    "tool_failures": 0,
    "synthesis_mode": "llm",
    "notes": []
  }
}
```

| field | type | notes |
|---|---|---|
| `answer` | string | Always present, never empty. Contains every component the question asked for, or states plainly which component the data cannot support. |
| `steps` | integer | Reasoning steps taken (brain turns). |
| `tool_trace` | array | Ordered `{tool, args, result}` per executed call. Only successful calls appear — failures are filtered before synthesis so the model never reads an error block as evidence. |
| `diagnostics` | object | Ours, not part of the contract. `synthesis_mode` is `llm`, `mock`, or `mock-fallback`. |

Reliability properties the contract depends on:

- **A valid body is always returned.** Any unhandled exception is caught and converted into a
  well-formed response with a grounded-failure `answer`, because a malformed response scores zero
  while a stated limitation can still earn components. `503` is returned only when the datasets
  themselves are unavailable.
- **Three concurrent `/query` requests are the design point**, matching the documented harness
  default. The datastore is loaded once and read-only thereafter, per-request state lives in an
  `AgentState` object, and AFR search results are LRU-cached per pattern.
- **The whole request is deadline-bounded** at `QUERY_DEADLINE_SECONDS=50`, below the 60s mark where
  20% of earned points is deducted, with `SYNTHESIS_RESERVE_SECONDS=20` held back so planning cannot
  consume the synthesis budget. On deadline the loop stops gathering and synthesises what it has —
  partial credit over a timeout zero.

---

## Architecture

Three roles, mapped one-to-one onto the Challenge Brief's
[Required Model Roles](Participant_Package/Challenge_Brief.md#required-model-roles), kept strictly
separate because the separation is itself scored.

```
POST /query {"question": "..."}
      │
      ▼
  AgentState  (question, messages, tool_trace, steps, deadline)
      │
      ├──►  brain.plan()               ROLE 1 — Qwen3.6-35B-A3B-FP8 via `agent-brain`
      │       plans the approach, selects the tool, emits tool calls + arguments,
      │       reviews returned results, decides whether another call is required
      │                                        │ tool_calls
      │                                        ▼
      │     tools.execute()            ROLE 2 — agent runtime (application code)
      │       validates arguments, executes against the local datasets,
      │       records the trace, returns structured results to Qwen
      │       query_data / retrieve / dataset_coverage
      │                                        │ results
      │     ◄──────── loop until the brain stops asking ─────────────────────────┘
      │       bounded by MAX_AGENT_STEPS=8 and QUERY_DEADLINE_SECONDS=50
      ▼
  synthesis.synthesize()              ROLE 3 — fine-tuned Nemotron via `DOMAIN_FT_MODEL`
      receives the question + accumulated verified tool results, writes the answer
      ▼
{"answer": "...", "steps": N, "tool_trace": [...]}
```

| Role | Component | Owns | Never does |
|---|---|---|---|
| 1. Planning & tool-call generation | Qwen `agent-brain` — [src/agent/brain.py](src/agent/brain.py) | Planning, tool selection, tool-call generation, deciding to iterate | Touch a dataset; write the answer. **Not fine-tuned.** |
| 2. Runtime tool execution | [src/agent/tools/](src/agent/tools/), [orchestrator.py](src/agent/orchestrator.py), [datastore.py](src/agent/datastore.py) | Argument validation, deterministic execution, the trace, error handling, timeouts | Decide *what* to call |
| 3. Answer synthesis | Fine-tuned Nemotron — [src/agent/synthesis.py](src/agent/synthesis.py) | The final grounded `answer` | Select or call a tool. **This is the fine-tuned model.** |

### Retrieval and deterministic computation

All dataset access goes through three tools, whose full metric catalogue is declared in the tool
schema in [src/agent/tools/\_\_init\_\_.py](src/agent/tools/__init__.py):

| tool | covers |
|---|---|
| `query_data(dataset, metric, …)` | Every derived fact: RBA rate lookups, change counts and cycle summaries; ASX returns, rankings, average volume, drawdowns, correlations and baskets; AFR counts by year and month. |
| `retrieve(…)` | AFR article text — exact headline+date lookup, and word-boundary or regex search over the corpus. |
| `dataset_coverage()` | Dataset shape and date ranges, so feasibility can be checked before an answer is attempted. |

Retrieval over AFR is exact rather than semantic, because the benchmark's article questions ask for
counts and term matches rather than similarity. The search surface is one pre-lowercased blob per
article combining `HEADLINE + SUBHEAD + INTRO + TEXT`, so case-insensitivity, four-field scope and
once-per-record counting cannot be got wrong by a caller. Three prefilter strategies (token postings
index → literal substring screen → contiguous-corpus regex) are tried in order, all returning
identical results, taking the worst case from ~22s to ~0.1–5.5s; a brute-force scan pins them in
[tests/test_metrics.py](src/agent/tests/test_metrics.py).

Everything numeric is computed in Python from the raw records — no model arithmetic anywhere. The
deterministic decisions that materially affect scoring (on-or-before rate resolution rather than
nearest, tickers read from the data rather than the filename, `exclude_tickers` for "non-Tabcorp",
baskets as the mean of constituent returns) are documented in
[src/agent/README.md](src/agent/README.md).

### Environment finding: recovering Qwen's tool calls

The supplied vLLM behind the `agent-brain` alias has been observed running without
`--enable-auto-tool-choice`, so the OpenAI `tool_calls` field comes back empty. Qwen still plans
correctly — right metric, right arguments — but returns the call as text
(`<tool_call><function=query_data><parameter=dataset>rba…`). Unparsed, every question becomes a
"no tool use" answer, which the handout scores as 0%. [brain.py](src/agent/brain.py) therefore also
recovers tool calls from the message text, covering both that markup and the Hermes JSON style, and
strips `<think>` blocks so reasoning is never re-read as evidence. This is additive: when the
endpoint exposes native tool calls, the native field takes precedence and the fallback never runs.
Pinned against a verbatim live capture in
[tests/test_brain_parsing.py](src/agent/tests/test_brain_parsing.py).

---

## Training summary

### What was fine-tuned

`Llama-3.1-Nemotron-Nano-8B-v1`, LoRA rank 32 / alpha 32 / dropout 0.1, `match_all_linear: true`
(7 module types × 32 layers = 224 adapted modules, ~168 MB adapter). NeMo AutoModel inside
`nvcr.io/nvidia/nemo:25.09`, one NVIDIA GB10, ~24.6 GiB peak, ~8.5 s/step. Recipe:
[training/configs/lora_nemotron_r32.yaml](training/configs/lora_nemotron_r32.yaml). Launcher:
[training/scripts/train.sh](training/scripts/train.sh).

It is fine-tuned for **one job**: read the question plus the verified tool evidence and write the
`answer` — state every requested component exactly, invent nothing, and where the evidence does not
support a component, say so rather than fill the gap. It does no planning and no tool calling. Qwen
is never fine-tuned. Checkpoints land every 20 steps; **`ck100` is the shipped arm**, selected on the
evidence in the evaluation section below.

Shipped configuration, and the two deviations from the handout's reference baseline, both forced by
measurement rather than preference:

| parameter | reference | used | why |
|---|---|---|---|
| LoRA rank / alpha | 32 / 32 | 32 / 32 | accepted |
| max steps / checkpoint every | 100 / 20 | 100 / 20 | accepted |
| effective batch | 8 (2 × 4) | 8 (1 × 8) | same optimisation, buys sequence length |
| `seq_length` | 512 | **1024** | 512 right-truncates the label. Measured with the model's own tokenizer, only 3 of 13 real question shapes fit at 512 (the 4-call cross-dataset shape is 4,520 tokens); an over-long row keeps the system prompt and silently drops the entire assistant span, i.e. trains on nothing. |
| peak LR | 5e-5 | **2e-5** | 5e-5 diverged reproducibly here — two independent runs, one immediately and one at step ~44 as the ramp crossed ~4.5e-5 (grad_norm 35 → 152 → 824 → 1,384 → 36,352, with no recovery under cosine decay). |
| warmup steps | 50 | **20** | at 100 total steps a 50-step warmup spends half the run below working LR, and the low-LR region was never where the instability lived |

Two changes made 1024 sufficient rather than merely larger: lossless evidence compaction
([src/agent/compact.py](src/agent/compact.py) — line form instead of JSON, no values elided,
`window_return` 896 → 84 tokens) and a shorter system prompt (332 → 145 tokens). After both, 11 of 13
shapes fit; rows that still exceed the budget are dropped, not truncated.

Also active: `answer_only_loss_mask: true`, verified genuinely masking rather than silently masking
nothing — `num_label_tokens` logs ~2,600–4,200 per global batch of 8, consistent with loss on the
answer span only — and `shuffle: false`, to preserve a deliberately category-balanced row order so
that the step-20 checkpoint is worth evaluating.

### Preparation method

**Tool-grounded synthetic supervision with sampled component subsets**, built by
[training/datagen/build.py](training/datagen/build.py) from the supplied datasets only:
**790 train / 144 validation rows**, no hidden evaluation data, no answer keyed to a question id.

```bash
python training/datagen/label_sentiment.py            # Qwen teacher labels (cached, optional)
python training/datagen/build.py --budget 1024 --target 800
python training/datagen/verify.py                     # every gate must pass
```

The five decisions that define the method:

1. **Every gold answer is derived from a real tool payload.** The generator calls the same
   `src/agent/tools/{rba,asx,afr}.py` functions the live agent calls, keeps the returned payload, and
   renders the answer from it with a verbalizer. Targets are grounded by construction and numerically
   exact for free. Generation is parameterised over `(dataset, metric, args)` tuples, never keyed to a
   question id, and the 15 public questions are not used as templates.
2. **Component subsets are sampled per example.** A template that always renders
   `rank_annual_returns → "X best, Y worst"` teaches a metric→sentence map, and the model then drops
   the third component when a hidden question asks for three. Because grading is per-component, the
   skill actually being bought is *component-following*: each payload is decomposed into its available
   components, a subset of 1–4 is sampled, and both question and answer are rendered over exactly that
   subset in the order asked. One payload yields 6–10 distinct examples.
3. **Train/serve prompt parity is structural, not checked.**
   [datagen/render.py](training/datagen/render.py) *imports* `SYNTHESIS_SYSTEM_PROMPT` and
   `format_evidence` from the served agent rather than reimplementing them, so the two cannot drift;
   `verify.py` asserts byte-identity against live agent traces. Sizing follows consumption rather than
   the 48,000-sample reference — 100 steps × batch 8 ≈ 800 sequences, so a 48,000-row file would be
   sampled at under 2% and diversity beats volume.
4. **Realistic evidence, including perturbations.** 35% of examples carry perturbations modelled on
   real Qwen habits: coverage probe, distractor block, shuffled block order, contradictory duplicate
   (same metric re-run without an exclusion), truncated payload. Error blocks are deliberately absent,
   because the orchestrator filters `ok=False` before synthesis and that distribution never occurs at
   inference.
5. **Negative and robustness examples (~12%)** that answer *and then state the gap*, never a bare
   refusal and never an empty response — plus a discrimination guard of ~2 complete-evidence examples
   per hedging example, so the model does not learn "question looks like this → hedge". Target answers
   never mention tools or the evidence format, so pipeline internals cannot leak into a graded field.

Composition across 10 categories: sentiment 24.3%, afr_counts 15.4%, rba_counts 13.0%, asx_returns
9.1%, rba_asx_event 8.6%, coverage 7.6%, composite 6.5%, asx_volume 6.2%, refusal 5.1%, asx_drawdown
4.2%. Sentiment and refusal are oversampled because they are the behaviours least copyable straight
out of the evidence. Token distribution min 223 / p50 592 / p95 958 / max 1,021, counted with the real
tokenizer over HTTP rather than estimated.

Sentiment is the only label needing a teacher: Qwen labels articles offline from the byte-identical
snippet the student will see, on a 5-point ordinal scale, as constrained JSON with a `span` field that
must be a literal substring of the snippet, under 3-vote self-consistency across three rubric
phrasings — **244 of 390 candidates kept (63%)**. The labels go into Nemotron's targets; Qwen's own
weights are never modified.

Leakage control is by `(template family × parameter instantiation × entity)`, not by row: tickers
`SUN.AX` / `TPG.AX` / `CMW.AX` are never a training *subject*, year 2017 is never a training
parameter, sentiment articles are disjoint across splits, and paraphrases sharing a `gold_key` stay on
the same side of the split. Six build gates must pass before a corpus ships: prompt parity, imported
prompt, leakage, prefix category balance, token budget, answer hygiene.

### Where the supporting evidence is stored

| evidence | path |
|---|---|
| Model card — config, deviations with the measured trajectories that forced them, checkpoint-selection rule | [training/MODEL_CARD.md](training/MODEL_CARD.md) |
| Data card — provenance, composition, formatting, perturbations, leakage control, build gates | [training/data_prep/DATA_CARD.md](training/data_prep/DATA_CARD.md) |
| Corpus and build report | [training/data/](training/data/) — `train.io.jsonl`, `val.io.jsonl`, `build_report.json`, `sentiment_labels.jsonl` |
| Data-generation and verification code | [training/datagen/](training/datagen/) |
| Recipe | [training/configs/lora_nemotron_r32.yaml](training/configs/lora_nemotron_r32.yaml) |
| Launch and serve scripts | [training/scripts/](training/scripts/) |
| **All five training runs, including the three failures**, with a written diagnosis of each | [training/logs/](training/logs/), indexed by [training/logs/README.md](training/logs/README.md) |
| Evaluation harness | [training/eval/](training/eval/) |
| Held-out and adversarial question sets, frozen evidence | [training/eval_sets/](training/eval_sets/) |
| Judge calibration and sweep output | [training/results/](training/results/) |
| Agent-side run log (sentiment labelling) | [logs/](logs/) |

The failed runs are kept deliberately. Two located real configuration faults — a `torchrun` rendezvous
hang caused by AutoModel's own `init_method="tcp://localhost:<port>"` under
`TORCHELASTIC_USE_AGENT_STORE=True`, and a dropped `lr_scheduler` block that `build_lr_scheduler()`
silently accepts as "no schedule at all" — and run 4 is the learning-rate evidence that justifies 2e-5.

---

## Base versus fine-tuned evaluation

### Method organizers should use to assess the final model

Reproduce the whole comparison:

```bash
bash training/scripts/serve_ft.sh --with-adapters      # base + ck20..ck100 as separate model ids
python training/eval/freeze_evidence.py                # capture real tool traces once
python training/eval/fingerprint.py --arms ck20 ck40 ck60 ck80 ck100   # prove adapters are applied

python training/eval/run_eval.py --arms base ck20 ck40 ck60 ck80 ck100 \
    --conditions clean --out training/results/public15.md
python training/eval/run_eval.py --arms base ck40 ck100 \
    --conditions clean noisy insufficient shuffled --prompt-2x2 \
    --questions training/eval_sets/heldout19.jsonl \
    --frozen training/eval_sets/frozen_heldout19 \
    --out training/results/heldout19_full.md
```

Written up in [training/results/base_vs_ft.md](training/results/base_vs_ft.md), with detail tables in
[public15.md](training/results/public15.md) and
[heldout19_full.md](training/results/heldout19_full.md) and every generation in the `*_raw.json`
beside each.

To assess the shipped model directly:

1. **Confirm the fine-tuned model is actually in the loop.** `GET /api/status` reports
   `synthesis_live` and `config.domain_predict_mode`; every `/query` response carries
   `diagnostics.synthesis_mode`, which is `llm` only when synthesis went to `DOMAIN_FT_MODEL`. vLLM
   echoes the served model id per request, so a LoRA arm reporting the base id would mean the adapter
   was not applied. `fingerprint.py` checks this before any scoring: at temperature 0 every arm
   differs from base on 5/5 probe questions
   ([logs/eval/adapter_fingerprint.json](logs/eval/adapter_fingerprint.json)).
2. **Compare arms on frozen evidence, not live traces.** `freeze_evidence.py` runs the real brain and
   the real tools once under `DOMAIN_PREDICT_MODE=mock`, then every arm is replayed against
   byte-identical evidence. Qwen at temperature 0 through vLLM is not bit-reproducible, so without
   this the delta silently absorbs planning variance instead of measuring synthesis.
3. **Grade with the calibrated component judge** — one expected fact per call, verdict read from
   YES/NO token logprobs at `temperature=0, seed=0`, thinking disabled. The headline metric is the
   same component score the organizers compute, reported alongside a strict variant (LLM verdict AND
   deterministic numeric tolerance check) and their disagreement rate.
4. **Read the interval, not the point estimate.** Deltas come with a paired bootstrap over questions
   (2,000 resamples — over questions rather than components, which correlate within a question) and an
   exact sign-flip permutation p-value. Any delta whose 95% CI crosses zero is printed with a ⚠ and is
   explicitly not claimed as an improvement.
5. **Four evidence conditions** — `clean`, `noisy`, `insufficient`, `shuffled` — because a clean-only
   result measures only the best case. Plus a `--prompt-2x2` fairness control: the system prompt was
   shortened 332 → 145 tokens to fit the sequence budget, so without running both arms against both
   prompts the comparison would silently be "base + long prompt vs FT + short prompt".
6. **Deterministic secondary metrics** alongside the judge, needing no model calls:
   `hallucinated_number_rate` (numeric literals asserted in the answer that appear in neither the
   evidence nor the question, allowing for rounding — computable only because the evidence is frozen),
   `hedge_rate`, `leaked_reasoning_rate`, `format_violation_rate`, `empty_rate`, `answer_words_p50`.
   These target the failure modes fine-tuning is meant to fix, which a component score alone hides: a
   base model that hedges every figure or narrates its reasoning can still score respectably on
   components while being unusable.

Checkpoint selection is **pre-registered**, and deliberately not on validation loss — val loss is
cross-entropy against our own synthetic phrasing, so minimising it rewards matching our templates,
which is the exact overfitting mode of concern:

1. Disqualify any arm with `hallucinated_number_rate` above base, or `error_rate > 2%`.
2. Rank survivors by component score on the held-out questions.
3. Tie-break inside the paired CI by the adversarial probe set, then p95 latency, then earlier step.
4. If the leader's edge over `ck20` sits inside the CI, take `ck20`.

**Outcome: `ck100`.** Applying the rule honestly was awkward, and the report says so. It assumed our
own `heldout19` set would be a valid accuracy instrument, and it is not (below); *every* arm fails the
disqualification clause under insufficient evidence; and the intent that `public15` be reported but
never used as a tie-break could not be honoured once the alternative set proved biased. `ck100` leads
every fine-tuned arm on the trustworthy set. `ck40` was the initial pick before the bias was found;
artifacts for both are retained and switching is a one-line config change.

### Judge calibration — complete

Before it grades any arm, the judge is gated on two published sets
([training/results/judge_calibration.md](training/results/judge_calibration.md)):

| gate | result |
|---|---|
| Reference answers must score full marks | **100.0%** — 150.0 / 150.0 across all 15 public questions, none below full marks |
| Adversarial and equivalence triples | **12 / 12 = 100.0%** — rejects hedged-count, wrong-context, no-number, off-by-one-date, thinking-out-loud and refusal; accepts comma, ISO-date, reference-date, percent-prose, trailing-zero and word-number equivalences |

### Results

Two evaluation sets, and **only one of them is a valid accuracy instrument**:

| set | n | questions and reference answers written by | valid for accuracy? |
|---|---:|---|---|
| `public15` | 15 | the organizers | **yes** |
| `heldout19` | 19 | our own generators | **no** |

`heldout19` holds out the right parameters (reserved tickers, year 2017) and shares no fact set with
training, but its `expected_fact` strings come from the same verbalizers that wrote the training
targets — so the fine-tuned model reproduces them near-verbatim while base states the same correct
value in its own words and is marked down. It shows +47pp, and that number measures phrasing, not
accuracy. We found this by cross-checking the two sets against each other, and we rest every accuracy
claim on `public15` alone. `heldout19` is reported only for what it is validly good for: format
control and robustness across evidence conditions.

**Headline, on the organizers' 15 questions:**

| arm | component | strict | halluc. numbers | words p50 | gen time |
|---|---:|---:|---:|---:|---:|
| `base` | **57.1%** | 40.4% | 6.7% | 26 | 24.3s |
| `ck20` | 41.1% | 39.1% | 26.7% | 13 | 11.0s |
| `ck40` | 45.6% | 40.2% | 20.0% | 13 | 12.1s |
| `ck60` | 51.7% | 46.7% | 20.0% | 13 | 11.8s |
| `ck80` | 50.0% | 50.0% | 13.3% | 13 | 11.4s |
| **`ck100`** (selected) | 54.7% | **51.3%** | 13.3% | 13 | 11.3s |

| measure | base | `ck100` | paired delta |
|---|---:|---:|---|
| Component score | 57.1% | 54.7% | −2.4%, 95% CI [−20.0%, +13.3%] ⚠ |
| Strict (judge **and** numeric tolerance agree) | 40.4% | **51.3%** | **+10.9%**, 95% CI [+1.3%, +26.4%], 4W/0L/11T |
| Median answer length | 26 words | **13 words** | matches the reference style |
| Generation time | 24.3s | **11.3s** | roughly halved |
| Leaked reasoning / format violations | 0% | 0% | every arm, every condition |

⚠ = interval crosses zero.

**We do not claim an accuracy improvement.** On the organizers' questions no arm beats base on
component score, every interval crosses zero, and the point estimates are negative. What did improve
is real but narrower: strict score, which penalises answers that win a verdict while mis-stating or
mis-formatting the figure, rises +10.9pp with an interval excluding zero and never loses a question;
and latency roughly halves, which is worth points under the 60-second penalty threshold with three
concurrent questions in flight. The p-value floor there is 0.125 — with only 4 discordant pairs the
exact sign-flip test cannot resolve further — and the report quotes both rather than the flattering
one.

The trend across checkpoints is monotone, ck20 (−16.0%) → ck100 (−2.4%). 100 steps at effective batch
8 is **1.01 epochs**, at half the reference learning rate. Nothing looks overfit; it looks
undertrained. This also contradicts the handout's suggestion that step 20 is often the best
checkpoint — here it is reliably the worst.

**The serious regression** is abstention. Under the `insufficient` condition, where an evidence block
is removed so the requested fact is genuinely unsupported and the correct behaviour is to say so:

| arm | hallucinated-number rate under `insufficient` |
|---|---:|
| `base` | **0.0%** |
| `ck40` | 73.7% |
| `ck100` | 73.7% |

Base declines correctly; the fine-tuned arms invent every figure. The cause is corpus balance rather
than hyperparameters — refusal is 5.1% of 790 rows with ~10 empty-trace negatives, against ~95% of
rows where complete evidence is present and a confident terse answer is right, so the model learned
the dominant pattern. The N1/N2 negative design was right in kind and far too small in quantity. The
fix is a data change the pipeline already supports (`TARGET_MIX` in `datagen/build.py`): raise the
insufficient-evidence share to ~20% and keep the discrimination guard explicit. Not applied here, for
time.

**The prompt fairness control earned its keep.** Base scores 9.6pp *higher* on the long pre-fine-tune
prompt than on the shortened one the fine-tune was trained against, so any short-prompt-only
comparison overstates the fine-tune. Giving each arm its better prompt narrows the spread to base
90.3% / ck40 94.7% / ck100 94.7% (on `heldout19`, so subject to the caveat above — though the
base-arm effect compares base against itself and is unaffected by it).

Full report, including the two regressing `public15` questions with their causes:
[training/results/base_vs_ft.md](training/results/base_vs_ft.md).

---

## Known limitations and failure cases

### The one that matters most

- **The fine-tune invents figures when the evidence is insufficient.** 0% → 73.7% hallucinated-number
  rate under the `insufficient` condition, where base declines correctly. The rules require stating
  the limitation rather than producing a number, an invented answer scores zero, and this is the
  failure a judge is most likely to notice. Diagnosed to corpus balance (5.1% refusal rows against
  ~95% complete-evidence rows), with a concrete data-side fix the pipeline already supports. Not
  applied before the deadline. If a hidden question asks for something the datasets cannot support,
  expect a confident wrong answer rather than a correct refusal.
- **Component accuracy did not improve over base**, and we do not claim it did. See the results
  section: the defensible gains are strict score, latency and format adherence.

### Operational — check before evaluation

- **`DOMAIN_PREDICT_MODE` is read at import**, so a server started in `mock` stays in `mock` even once
  the adapter is serving. The code default is now `llm`, but a process launched earlier will not pick
  that up. Verify with `diagnostics.synthesis_mode` on any `/query`, or `synthesis_live` on
  `/api/status`, and restart if it is not `llm` — a `mock` or `mock-fallback` answer is valid, grounded
  and well-formed, but it does not use the fine-tuned model and would not earn the 30% model-quality
  category.
- **The repository must be pushed and the pinned commit must exist publicly.** `submission.json`
  declares the team id, repository URL, commit SHA, and both endpoints — a commit that exists only
  locally makes the architecture review impossible.
- **`agent.endpoint` declares the cluster-link address** `10.0.1.10:8001`. This node also has a venue
  LAN address (`10.3.8.133`); if the harness runs outside the two-node cluster network, that is the
  reachable one and `submission.json` must be changed to match. `GET /health` is a hard gate — an
  unreachable endpoint means zero hidden-question points.

### Data and correctness

- **The highest-cash-rate date disagrees with the handout.** The handout's partial-credit example
  expects `2010-11-02`; the supplied corpus contains exactly one Nov-2010 row
  (`3 Nov 2010,+0.25,4.75`), so `2010-11-03` is the only value derivable from the data —
  `2010-11-02` is the announcement date, not the effective date. The tools report the effective date.
  If hidden questions are graded against announcement dates, RBA date components will be marked
  wrong; resolving that needs an organizer ruling, not a code change.
- **92 AFR articles ship with an empty `PUBLICATIONDATE`.** They are kept so the corpus total still
  matches the organizer's conversion summary (219,538) and they count toward corpus-wide totals, but
  they are excluded from year and month filters because there is no defensible bucket for them.
- **No semantic retrieval.** `QDRANT_URL` is read but unused; AFR retrieval is exact headline+date
  lookup and regex search, which is what the benchmark's article questions actually ask for.
  Embedding retrieval is the first thing to add for open-ended "why did X move" questions.
- **`AFR_MAX_FILES` produces wrong counts** and must be unset for evaluation. It exists only to cut
  the warm-up during local iteration.

### Model behaviour

- **Undertrained.** 100 steps at effective batch 8 over 790 rows is 1.01 epochs, at half the reference
  learning rate, and the checkpoint trend was still improving monotonically at ck100. The corpus is
  sized to the step budget, not to what the task could absorb.
- **Brevity can cost a requested component.** On `MHQ061` the fine-tuned arm gave the peak year and
  dropped the peak month, and on `MHQ040` it dropped the row-count component and mis-stated a date
  (`2 Feb 2015` where the evidence says 2 Jan). Median answer length fell 26 → 13 words, which matches
  the reference style and is mostly good, but the sampled-component-subset design did not fully
  prevent the model learning brevity as a prior strong enough to drop a component that was asked for.
- **One of our two evaluation sets is a biased instrument.** `heldout19`'s expected facts were written
  by the same verbalizers that produced the training targets, so it rewards our phrasing rather than
  correctness and its +47pp is not an accuracy result. It is used only for format control and
  robustness. A corrected version would author reference answers independently of the generators.
- **Sentiment has seen only the unambiguous end of its distribution.** Labels are kept only where
  three rubric phrasings agreed (244/390, 63%), and the discarded 37% are exactly the ambiguous
  cases. The kept distribution is also skewed — 120 positive, 100 negative, but only 13 mixed, 6
  mixed-negative, 5 mixed-positive — so nuanced "mixed with a negative bias" phrasings are the
  weakest behaviour.
- **Entities outside the supplied datasets are untested and out of scope.** Held-out axes are
  `SUN.AX` / `TPG.AX` / `CMW.AX` and year 2017.
- **The adapter is trained for this agent's exact prompt format.** It is not a general-purpose finance
  model and should not be used outside this pipeline.
- **Recovered tool-call parsing is a workaround, not a contract.** If the brain emits a markup that
  neither the native field nor the two pinned text formats cover, the question degrades to a no-tool
  answer. Both known formats are pinned by tests against live captures.

### Performance

- **~30s warm-up and ~1.7 GB resident** for the AFR corpus and index; `/health` is 503 throughout.
  `AFR_BUILD_INDEX=false` trades memory for a much slower worst case.
- **Worst-case AFR pattern is ~5.5s.** Three concurrent copies of the same uncached hard question is
  roughly 17s of CPU — comfortable against the 60s threshold, but not unlimited. At
  `QUERY_DEADLINE_SECONDS=50` the loop stops gathering and synthesises what it has, trading
  completeness for partial credit over a zero.
- **The heuristic self-scorer is not the judge.** `POST /api/score` and
  `src/agent/eval/run_public_eval.py` use numeric and keyword matching, which is blind to synonyms and
  strict about wording, so it under-reports on sentiment components. Treat a miss as "look at this",
  not as a verdict.

---

## Repository layout

```
README.md              this file
submission.json        team identity, pinned commit, agent + model endpoints
src/agent/             the agent: server, orchestrator, brain, tools, synthesis, dashboard, tests
training/              recipe, data generation, corpus, eval harness, logs, results, cards
logs/                  non-sensitive run logs
data set/              organizer-supplied RBA / ASX / AFR corpora
Participant_Package/   challenge materials, public questions, validation schema, handouts
```

No credentials, hidden evaluation material, or machine-specific secrets are committed.
[src/agent/.env.example](src/agent/.env.example) documents every variable without carrying a value
that matters.

## Attribution

The initial scaffold was a clone of the public teaching repository
[masoodfaisal/langchain-basics](https://github.com/masoodfaisal/langchain-basics). None of its code
survives — the Chinook demo agent, LangGraph wiring, memory store and eval harness were all removed —
but the project layout and the FastAPI / env-var conventions started there.
