# Handoff — 2026-07-31

Written on leaving. Covers one open bug (outside this folder, so left unfixed),
what changed inside this folder, and the state of the running services.

---

## Open bug: the brain re-issues the same tool calls every round

**Owner: `brain.py` — not this folder, so untouched.**

### Symptom

On multi-part questions the brain re-sends its entire batch of tool calls
verbatim, several times over, until it hits `MAX_AGENT_STEPS=8`. MHQ076 from the
public set:

```
 1. afr/count{terms:["QBE"]}   2. asx/annual_return{2021}   3. asx/rank_annual_returns{2021}
 4. afr/count{terms:["QBE"]}   5. asx/annual_return{2021}   6. asx/rank_annual_returns{2021}
 7-9.   (identical third round)
10-12.  (identical fourth round)
→ 12 tool calls, notes: ["reached MAX_AGENT_STEPS=8"]
```

Only the first three were needed. Same pattern on MHQ072 (2 rounds) and MHQ084.
This is pre-existing — it predates the tools merge and reproduces on the older
build too.

### Root cause

`agent-brain` is not emitting native `tool_calls`, so every question goes through
the text-recovery path (`brain.py` logs `recovered N tool call(s) from assistant
text` each time). That path leaves the raw markup in the assistant message:

`brain.py:457` — `message["content"] = strip_reasoning(content)`

`strip_reasoning` removes `<think>` blocks only. Verified:

```python
raw = '<think>...</think>\n<tool_call>\n{"name":"query_data",...}\n</tool_call>'
brain.strip_reasoning(raw)
# → '<tool_call>\n{"name": "query_data", ...}\n</tool_call>'   ← markup survives
```

`orchestrator.py:142-148` then appends that content to `state.messages`. On the
next round the model reads its own `<tool_call>` markup as ordinary transcript
text and re-emits it.

The comment directly above `brain.py:457` already states the intent — reasoning
must not survive into the transcript, because it gets re-read as gathered
evidence. Tool-call markup is the same failure mode; it just isn't covered.

### Why it matters

Each wasted round is a real brain call. MHQ076 spends 8 of its 12 calls on
duplicates and hits the step cap, so a genuinely four-part question can exhaust
the budget before the last part is ever gathered.

### Suggested fix

Strip the `<tool_call>` / `<function=>` markup from `content` once the calls have
been recovered. The regexes already exist in `brain.py` (`_TOOL_CALL_BLOCK_RE`,
`_XML_FUNCTION_RE`). Once recovered into `message["tool_calls"]`, the text form is
redundant — the structured field is what the next round should read. Check
whether the endpoint accepts an empty `content` alongside `tool_calls`; if not, a
short placeholder works.

Secondary: recovered call IDs restart at `text_call_1` every round
(`brain.py:134,151`), so round 2's `tool_call_id` collides with round 1's.
Probably harmless once the markup is gone, but worth making unique.

### Repro

`POST /query` with the MHQ076 prompt; read `diagnostics.tool_calls` and
`diagnostics.notes`.

---

## Already fixed in this folder — don't redo

1. **Per-ticker fan-out.** ASX metrics take the whole ticker list at once, but
   the brain was issuing one call per ticker. MHQ072 spent six of eight steps on
   identical calls and hit the cap. The ASX catalogue block in
   `__init__.py` now says to pass them together. MHQ072: 8 calls → 6, no cap,
   and the answer now matches the reference on all six components.

   Note on placement: the first attempt put this as a separate paragraph *after*
   the AFR matching rules, which pushed that guidance further from the end of a
   ~5.5k-char description and regressed MHQ084 from 4 calls to 8 (cap hit).
   Moving it inside the ASX section fixed it. **If you edit this description,
   A/B MHQ084 and MHQ076 before and after** — position matters as much as
   wording.

2. **Invented tool names.** The brain sometimes flattens the call, emitting
   `rba(metric=...)` or `event_window(...)` instead of
   `query_data(dataset=..., metric=...)`. `execute()` now recovers unambiguous
   names and tags the result with `recovered_from_tool_name` so it stays visible
   in the trace. Ambiguous names (`count`, `coverage` — present in more than one
   dataset) and dataset-only names with no metric still fail with the catalogue
   attached.

   If `recovered_from_tool_name` starts appearing often, fix the description
   rather than widening the fallback.

3. **`whole_word` is deliberately absent from the schema.** It only ever weakens
   `terms` by dropping the `\b` anchors, which inflates a count with no sign that
   it did — and the reference derivations all assume anchored matching.
   Deliberate stemming already has a path: `pattern=` is used verbatim and
   unanchored. `query_data` drops unknown keys, so a brain that passes it anyway
   is ignored rather than failed. See the comment above `_QUERY_DATA_DESCRIPTION`.

Verification: 83 tests pass (`cd src/agent && python3 -m pytest tests/ -q`), and
all 12 ground-truth values in `.claude/skills/verify` reproduce exactly — 41 RBA
changes (20 up / 21 down), longest hold 1036 days, 2022-23 +4.25pp over 13 hikes,
2018 best BHP +22.17% / worst AMP -50.04%, worst drawdown AMP -82.45%, basket
5-12 Jun 2019 +2.88%, basket 2019 +20.11%, QBE 2021 +35.57%.

---

## Service state at handoff

- **Port 8001** — pid 151769, `DOMAIN_PREDICT_MODE=llm`. Started *before* the
  tools edits above, so it is running **stale code** and needs a restart to pick
  them up.
- **`domain-ft` is live.** Nemotron is deployed and synthesis is really running
  through it (`diagnostics.synthesis_mode: "llm"`, not `mock-fallback`). Confirm
  with `/api/status` → `synthesis_live: true`.
- Restart, under tmux so it survives the session:

  ```bash
  tmux new-session -d -s agent -c ~/projects/clawandorder/src/agent
  tmux send-keys -t agent 'DOMAIN_PREDICT_MODE=llm ../../.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001' Enter
  ```

  Warm-up is ~25s (the AFR token index); `/health` returns `loading` until ready.

### Still open

- **`~/team.env` does not exist.** Services are running on `config.py` defaults.
  `LITELLM_KEY` defaults to `sk-local-cluster`, which currently works — but if
  the organizers issue a different key, every brain call fails. Source the file
  and restart once it arrives.
- **`submission.json` is still the template**: `mock-team`, `172.20.x.x`,
  placeholder commit SHA. The agent endpoint must be this node's IP on port
  **8001** (the template's `:5000` contradicts the handout — the handout wins).
  The model endpoint is the second node: `http://10.0.1.11:8001/v1`.
- **`.claude/skills/verify` is stale.** Its "Driving the tools" section documents
  the deleted langchain layer (`tools.rba_rates`, `ALL_TOOLS`, `ainvoke`) and
  will fail with `ModuleNotFoundError`. The reference values and dataset
  conventions in it are still correct and worth keeping — replace the driving
  section with `python -m tools` (see `__main__.py` in this folder).
