# Base vs fine-tuned Nemotron — controlled comparison

Every arm answers the same questions from byte-identical frozen tool evidence. Qwen routing and tool execution are held fixed, so the only variable between arms is the synthesis model's weights.

- Judge: calibrated component judge (see `judge_calibration.md`)
- Baseline arm: `base`
- Decoding: greedy (`temperature=0`, `seed=0`)


## Component score

| arm | clean |
|---|---|
| `base` | 80.7% |
| `ck20` | 77.2% |
| `ck40` | 94.7% |
| `ck60` | 79.0% |
| `ck80` | 92.1% |
| `ck100` | 89.5% |

## Paired delta vs `base`

Bootstrap over questions (2,000 resamples), not components — components within a question are correlated. `p` is an exact sign-flip permutation test on paired differences.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck20` | clean | 80.7% | 77.2% | **-3.5%** ⚠ | [-26.3%, +18.4%] | 0.8125 | 4/3/12 |
| `ck40` | clean | 80.7% | 94.7% | **+14.0%** | [+3.5%, +27.2%] | 0.0625 | 5/0/14 |
| `ck60` | clean | 80.7% | 78.9% | **-1.8%** ⚠ | [-21.9%, +18.4%] | 0.9062 | 4/3/12 |
| `ck80` | clean | 80.7% | 92.1% | **+11.4%** ⚠ | [-8.8%, +31.6%] | 0.3438 | 6/2/11 |
| `ck100` | clean | 80.7% | 89.5% | **+8.8%** ⚠ | [-7.9%, +26.3%] | 0.4062 | 5/1/13 |

⚠ = 95% CI includes zero; not a demonstrated improvement.


### Strict score (judge **and** numeric tolerance must agree)

The headline metric follows the organizers' LLM judge. This one also requires the stated value to pass tolerance arithmetic, so it catches answers a judge accepts but whose figures are mis-stated or mis-formatted.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck20` | clean | 47.4% | 71.9% | **+24.6%** | [+6.1%, +43.9%] | 0.0391 | 7/1/11 |
| `ck40` | clean | 47.4% | 94.7% | **+47.4%** | [+28.9%, +68.4%] | 0.0010 | 11/0/8 |
| `ck60` | clean | 47.4% | 73.7% | **+26.3%** | [+5.3%, +47.4%] | 0.0469 | 6/1/12 |
| `ck80` | clean | 47.4% | 92.1% | **+44.7%** | [+23.7%, +68.4%] | 0.0029 | 11/1/7 |
| `ck100` | clean | 47.4% | 89.5% | **+42.1%** | [+21.1%, +65.8%] | 0.0049 | 10/1/8 |

## Secondary metrics (deterministic, no judge)

`hallucinated_number_rate` counts numeric literals asserted in the answer that appear in neither the evidence nor the question, allowing for rounding. It is computable only because the evidence is frozen.

| arm | condition | hedge | halluc. num | leaked reasoning | format viol. | empty | words p50 |
|---|---|---|---|---|---|---|---|
| `base` | clean | 0.0% | 10.5% | 0.0% | 0.0% | 0.0% | 26 |
| `ck20` | clean | 0.0% | 5.3% | 0.0% | 0.0% | 0.0% | 14 |
| `ck40` | clean | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 13 |
| `ck60` | clean | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 7 |
| `ck80` | clean | 0.0% | 5.3% | 0.0% | 0.0% | 0.0% | 10 |
| `ck100` | clean | 0.0% | 5.3% | 0.0% | 0.0% | 0.0% | 13 |

## Provenance

| arm | prompt | condition | served model id | errors | gen seconds |
|---|---|---|---|---|---|
| `base` | short | clean | `nemotron-8b-finance` | 0.0% | 14.6 |
| `ck20` | short | clean | `ck20` | 0.0% | 9.9 |
| `ck40` | short | clean | `ck40` | 0.0% | 10.5 |
| `ck60` | short | clean | `ck60` | 0.0% | 8.1 |
| `ck80` | short | clean | `ck80` | 0.0% | 10.0 |
| `ck100` | short | clean | `ck100` | 0.0% | 9.1 |

The served model id is echoed by vLLM per request. A LoRA arm reporting the base id would mean the adapter was not applied.
