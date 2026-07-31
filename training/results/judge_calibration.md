# Judge calibration

- judge model: `agent-brain`
- prompt version: `v1`
- protocol: one expected fact per call, verdict read from YES/NO token logprobs,
  `temperature=0`, `seed=0`, thinking disabled.

## Gate 1 - reference answers score full marks

Score on the 15 public reference answers: **100.0%** (150.0/150.0 points). Questions below full marks: **0**.

| question | earned | max | difficulty |
|---|---:|---:|---|
| MHQ001 | 10.00 | 10.00 | easy |
| MHQ035 | 10.00 | 10.00 | medium |
| MHQ040 | 10.00 | 10.00 | easy |
| MHQ045 | 10.00 | 10.00 | medium |
| MHQ049 | 10.00 | 10.00 | medium |
| MHQ055 | 10.00 | 10.00 | hard |
| MHQ058 | 10.00 | 10.00 | easy |
| MHQ061 | 10.00 | 10.00 | medium |
| MHQ067 | 10.00 | 10.00 | hard |
| MHQ072 | 10.00 | 10.00 | medium |
| MHQ074 | 10.00 | 10.00 | hard |
| MHQ076 | 10.00 | 10.00 | easy |
| MHQ080 | 10.00 | 10.00 | medium |
| MHQ084 | 10.00 | 10.00 | medium |
| MHQ090 | 10.00 | 10.00 | hard |

## Gate 2 - adversarial and equivalence triples

Agreement with published labels: **12/12 = 100.0%**.

| case | judge | gold | p(yes) |
|---|---|---|---:|
| hedged-count | NO | NO | 0.000 |
| wrong-context | NO | NO | 0.000 |
| no-number | NO | NO | 0.000 |
| off-by-one-date | NO | NO | 0.007 |
| thinking-out-loud | NO | NO | 0.000 |
| refusal | NO | NO | 0.000 |
| comma-equivalence | YES | YES | 0.881 |
| iso-date-equivalence | YES | YES | 0.986 |
| reference-date-equivalence | YES | YES | 0.958 |
| percent-prose-equivalence | YES | YES | 0.984 |
| trailing-zero-equivalence | YES | YES | 0.986 |
| word-number-equivalence | YES | YES | 0.998 |
