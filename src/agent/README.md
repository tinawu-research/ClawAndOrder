# ClawAndOrder — market signal agent

Agent service for the Cognitivo hackathon: answers financial-market questions
over the approved RBA, ASX and AFR datasets, and exposes the two endpoints the
evaluation harness calls.

> Repository-level docs (team name, submission metadata, training evidence) live
> in the repository root `README.md`. This file documents the agent itself.

## Architecture

```
POST /query {"question": "..."}
      │
      ▼
  AgentState  (question, messages, tool_trace, steps, deadline)
      │
      ├──►  brain.plan()          Qwen3.6-35B-A3B-FP8 via `agent-brain`
      │       plans, selects tools, emits tool calls + arguments
      │                                      │ tool_calls
      │                                      ▼
      │     tools.execute()        application code validates + runs the call
      │       query_data / retrieve / dataset_coverage
      │       exact structured results from the raw data
      │                                      │ results
      │     ◄────────────────── loop until the brain stops asking ─────────────┘
      │       (bounded by MAX_AGENT_STEPS and QUERY_DEADLINE_SECONDS)
      ▼
  synthesis.synthesize()          fine-tuned Nemotron via `domain-ft`
      receives question + verified tool results, writes the final answer
      ▼
{"answer": "...", "steps": N, "tool_trace": [...]}
```

Three responsibilities, kept strictly separate because the separation is itself
scored:

| Component | Owns | Never does |
|---|---|---|
| Qwen `agent-brain` ([brain.py](brain.py)) | Planning, tool selection, tool-call generation | Touch a dataset; write the answer |
| Agent runtime ([tools/](tools/), [orchestrator.py](orchestrator.py)) | Validating and executing tool calls, recording the trace | Decide *what* to call |
| Fine-tuned Nemotron ([synthesis.py](synthesis.py)) | Final grounded answer synthesis | Select or call a tool |

Qwen is supplied and is not fine-tuned. Nemotron is the team's fine-tuning
target and is not the tool-calling model.

## Run it

```bash
source ~/team.env                 # organizer-supplied endpoints + credentials
pip install -r requirements.txt

# Port 8001 on the head node is MANDATORY — Setup_Instructions states that any
# other port causes the eval to fail. Bound to 0.0.0.0 so the endpoint resolves
# whether the harness reaches it over localhost or the node IP.
uvicorn server:app --host 0.0.0.0 --port 8001
```

> **Port 8001 is overloaded.** It is both the agent HTTP server on the *head*
> node and the fine-tuned Nemotron vLLM on the *fine-tuning/model* node. On a
> two-node cluster those are different machines, so both can use 8001 — but if
> the adapter is ever served on the head node, they collide and one must move.
> Older handout material (`02_execution_guide.md`'s uvicorn command, the
> `submission.json` example) still says port 5000; that is stale.

`GET /health` returns **503 while the corpus loads** (~25s) and 200 once every
dataset is queryable. That is deliberate: health is a hard gate, and answering
200 early would let the harness start against an agent that cannot yet answer.

Before official evaluation:

```bash
export DOMAIN_PREDICT_MODE=llm    # bootstrap default is `mock`
```

In `mock` mode the pipeline runs end to end but final synthesis is a
deterministic template, **not** the fine-tuned model. The dashboard shows a
persistent warning banner while that is the case.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness gate. 200 = ready. |
| `POST` | `/query` | `{"question": "..."}` → `{"answer", "steps", "tool_trace"}` |
| `GET` | `/` | Operations dashboard (humans only) |
| `GET` | `/api/status` | Telemetry: datasets, config, recent latencies |
| `GET` | `/api/public-questions` | The 15 calibration cases |
| `POST` | `/api/score` | Heuristic component self-check |

Only `answer` is graded. `steps` and `tool_trace` are optional diagnostics and
are both populated. A `diagnostics` object is added for our own debugging.

## Dashboard

`GET /` serves a single self-contained page — no build step, no CDN. It shows
health and predict-mode at a glance, dataset coverage, a question box preloaded
with the 15 calibration cases, the answer with latency against the 60s penalty
threshold, the full tool trace, and a per-component self-check against the
reference facts when the question came from the calibration set.

## Tools

All dataset access goes through `query_data(dataset, metric, ...)`, plus
`retrieve` for article text and `dataset_coverage` for feasibility checks. The
full metric catalogue is in the tool schema in
[tools/\_\_init\_\_.py](tools/__init__.py) — it is spelled out there because the
brain cannot discover it at run time, and a wrong metric name is the difference
between full marks and zero.

Notable behaviours, each of which exists because getting it wrong loses points:

- **`rba/lookup_rate` resolves on-or-before**, never nearest. The decision table
  runs to 17 Jun 2026, so a nearest-date lookup returns rates from the future.
- **ASX tickers are read from the data, never the filename.**
  `Aurizon-ASX-2015-2021.jsonl` contains `AZJ.AX`.
- **`exclude_tickers=["TAH.AX"]`** implements "non-Tabcorp". Without it the
  highest-average-volume ticker flips from `AMP.AX` to `TAH.AX`.
- **AFR matching is structurally correct**: the search surface is one
  pre-lowercased blob per article combining HEADLINE + SUBHEAD + INTRO + TEXT, so
  case-insensitivity, four-field scope and once-per-record counting cannot be
  got wrong by a caller. Word boundaries are the caller's job, so `terms=[...]`
  applies them automatically; `pattern=` is used verbatim for the cases where the
  question dictates an exact regex.
- **Baskets are the arithmetic mean of constituent returns**, not a weighted
  index.

### AFR search performance

A naive implementation re-runs the regex over ~800 MB of article text per call.
That is ~22s single-threaded, and the harness sends three questions at once
against a 60s penalty threshold. Three strategies, tried in order, all returning
identical results:

| Strategy | When | Cost |
|---|---|---|
| Token postings index | Every branch has a fully word-delimited literal (`\bunemployment\b`) | ~0.1s |
| Literal substring screen | Every branch has a guaranteed literal (`rate cut`) | ~5.5s |
| Contiguous-corpus regex pass | Neither applies | ~22s |

The token index requires a literal to be delimited on *both* sides before
treating it as mandatory. That rule is load-bearing: the branch `rate cut`
legitimately matches the text "rate cuts", where the indexed token is `cuts`, so
requiring `cut` silently dropped true matches. `test_prefilter_is_exact_versus_full_scan`
pins every strategy against a brute-force scan.

Results are LRU-cached per pattern, so a multi-part question reusing one pattern
across a total, a per-year split and a per-month split pays once.

## Verification

```bash
pytest tests -m "not slow"   # RBA + ASX, ~0.2s
pytest tests                 # adds the AFR corpus + index (~25s warm-up)
```

Every expected value in [tests/test_metrics.py](tests/test_metrics.py) is taken
from `public_questions.jsonl` or a worked example in the organizer handout. The
tool layer reproduces all of them, including to the cent on average daily volume
(`11,635,671.71`) and to the day on all six drawdown endpoint dates.

End-to-end scoring against a running agent:

```bash
python eval/run_public_eval.py --workers 3 --show-answers
```

This applies the same slow-response penalty schedule as the official scorer.
Its component check is a numeric/keyword heuristic, not an LLM judge — treat
failures as items to inspect, not as a verdict.

## Configuration

Everything is read from environment variables in [config.py](config.py); no
endpoint, hostname, IP or credential is hard-coded. See
[.env.example](.env.example) for the full list. On the cluster, `source
~/team.env`.

## Cluster findings that affect scoring

Two things about the supplied environment materially change the result, and
neither is fixable from inside this code alone.

**1. `agent-brain` does not emit native tool calls.** The vLLM server behind the
alias is not started with `--enable-auto-tool-choice`, so the OpenAI `tool_calls`
field is always empty. Qwen still plans correctly — it selects the right metric
with the right arguments — but returns the call as text:

```
<tool_call><function=query_data><parameter=dataset>rba</parameter>…
```

Unparsed, every question becomes a "no tool use" answer, which the handout
documents as scoring 0%. [brain.py](brain.py) therefore recovers tool calls from
the text, covering both this markup and the Hermes JSON style, and strips
`<think>` reasoning so it is never re-read as evidence. This is additive: if the
endpoint is later restarted with a tool-call parser, the native field takes
precedence and the fallback never runs. Tests in
[tests/test_brain_parsing.py](tests/test_brain_parsing.py) pin the behaviour
against a verbatim capture from the live endpoint.

**2. The `domain-ft` alias currently points at Qwen, not Nemotron.** It answers,
but with the same vLLM fingerprint as `agent-brain`, and nothing is listening on
port 8001. So setting `DOMAIN_PREDICT_MODE=llm` today would route synthesis to
Qwen and produce a submission that never uses the fine-tuned model — which the
30% model-quality category explicitly checks for. The LiteLLM route must be
repointed to port 8001 on the fine-tuning node once the adapter is served.

## Known limitations

- **`DOMAIN_PREDICT_MODE=mock` is the default.** Left unset, the submission does
  not use the fine-tuned model at all. The startup log and the dashboard both
  warn about it, but nothing forces it — that is a deliberate choice so the
  bootstrap integration path still works before an adapter exists.
- **Highest-cash-rate date disagrees with the handout.** The handout's
  partial-credit example states the judge expected `2010-11-02`. The supplied
  corpus contains exactly one Nov-2010 row, `3 Nov 2010,+0.25,4.75`, so
  `2010-11-03` is the only value derivable from the data — `2010-11-02` is the
  announcement date, not the effective date. The tools report the effective date.
  If hidden questions are graded against announcement dates, RBA date components
  will be marked wrong; resolving that needs an organizer ruling, not a code
  change.
- **92 AFR articles ship with an empty `PUBLICATIONDATE`.** They are kept so the
  corpus total still matches the organizer's conversion summary (219,538), and
  they count toward corpus-wide totals, but they are excluded from year/month
  filters because there is no defensible bucket for them.
- **`Setup_Instructions.md` places the UTF-8 BOM on the RBA CSV**; it is actually
  on `RBA-rates.jsonl`. Both are opened `utf-8-sig`, which is correct either way.
- **~25s warm-up and ~1.7 GB resident** for the AFR corpus and index. Fine on a
  128 GB node; set `AFR_BUILD_INDEX=false` to trade memory for a much slower
  worst-case query, or `AFR_MAX_FILES=N` for fast local iteration — the latter
  produces **wrong counts** and must be unset for evaluation.
- **Worst-case AFR pattern is ~5.5s.** Three concurrent copies of the same
  uncached hard question is roughly 17s of CPU. Comfortable against the 60s
  threshold, but not unlimited: the loop stops gathering at
  `QUERY_DEADLINE_SECONDS=50` and synthesises what it has, trading completeness
  for partial credit over a zero.
- **No Qdrant / embedding retrieval.** `QDRANT_URL` is read but unused; AFR
  retrieval is exact headline+date lookup and regex search, which is what the
  benchmark's article questions actually ask for. Semantic retrieval would be
  the first thing to add for open-ended article questions.
- **The heuristic self-scorer is not the judge.** It is blind to synonyms and
  strict about wording, so it under-reports on sentiment components.

## Attribution

The initial scaffold was a clone of the public teaching repository
[masoodfaisal/langchain-basics](https://github.com/masoodfaisal/langchain-basics).
None of its code survives — the Chinook demo agent, LangGraph wiring, memory
store and eval harness were all removed — but the project layout and the
FastAPI/env-var conventions started there.
