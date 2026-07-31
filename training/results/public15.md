# Base vs fine-tuned Nemotron — controlled comparison

Every arm answers the same questions from byte-identical frozen tool evidence. Qwen routing and tool execution are held fixed, so the only variable between arms is the synthesis model's weights.

- Judge: calibrated component judge (see `judge_calibration.md`)
- Baseline arm: `base`
- Decoding: greedy (`temperature=0`, `seed=0`)


## Component score

| arm | clean |
|---|---|
| `base` | 57.1% |
| `ck20` | 41.1% |
| `ck40` | 45.6% |
| `ck60` | 51.7% |
| `ck80` | 50.0% |
| `ck100` | 54.7% |

## Paired delta vs `base`

Bootstrap over questions (2,000 resamples), not components — components within a question are correlated. `p` is an exact sign-flip permutation test on paired differences.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck20` | clean | 57.1% | 41.1% | **-16.0%** ⚠ | [-35.3%, +0.0%] | 0.1875 | 1/4/10 |
| `ck40` | clean | 57.1% | 45.6% | **-11.6%** ⚠ | [-27.8%, +0.7%] | 0.1875 | 1/4/10 |
| `ck60` | clean | 57.1% | 51.7% | **-5.4%** ⚠ | [-28.0%, +15.7%] | 0.6562 | 3/3/9 |
| `ck80` | clean | 57.1% | 50.0% | **-7.1%** ⚠ | [-23.1%, +5.3%] | 0.5000 | 2/3/10 |
| `ck100` | clean | 57.1% | 54.7% | **-2.4%** ⚠ | [-20.0%, +13.3%] | 0.8750 | 4/2/9 |

⚠ = 95% CI includes zero; not a demonstrated improvement.


### Strict score (judge **and** numeric tolerance must agree)

The headline metric follows the organizers' LLM judge. This one also requires the stated value to pass tolerance arithmetic, so it catches answers a judge accepts but whose figures are mis-stated or mis-formatted.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck20` | clean | 40.4% | 39.1% | **-1.3%** ⚠ | [-20.0%, +20.0%] | 1.0000 | 2/2/11 |
| `ck40` | clean | 40.4% | 40.2% | **-0.2%** ⚠ | [-14.9%, +18.7%] | 1.0000 | 2/3/10 |
| `ck60` | clean | 40.4% | 46.7% | **+6.2%** ⚠ | [-7.3%, +23.6%] | 0.5625 | 3/2/10 |
| `ck80` | clean | 40.4% | 50.0% | **+9.6%** ⚠ | [-0.7%, +25.1%] | 0.2500 | 3/1/11 |
| `ck100` | clean | 40.4% | 51.3% | **+10.9%** | [+1.3%, +26.4%] | 0.1250 | 4/0/11 |

## Secondary metrics (deterministic, no judge)

`hallucinated_number_rate` counts numeric literals asserted in the answer that appear in neither the evidence nor the question, allowing for rounding. It is computable only because the evidence is frozen.

| arm | condition | hedge | halluc. num | leaked reasoning | format viol. | empty | words p50 |
|---|---|---|---|---|---|---|---|
| `base` | clean | 0.0% | 6.7% | 0.0% | 6.7% | 0.0% | 30 |
| `ck20` | clean | 0.0% | 26.7% | 0.0% | 0.0% | 0.0% | 17 |
| `ck40` | clean | 0.0% | 20.0% | 0.0% | 0.0% | 0.0% | 22 |
| `ck60` | clean | 0.0% | 20.0% | 0.0% | 0.0% | 0.0% | 22 |
| `ck80` | clean | 0.0% | 13.3% | 0.0% | 0.0% | 0.0% | 18 |
| `ck100` | clean | 0.0% | 13.3% | 0.0% | 0.0% | 0.0% | 20 |

## Provenance

| arm | prompt | condition | served model id | errors | gen seconds |
|---|---|---|---|---|---|
| `base` | short | clean | `nemotron-8b-finance` | 0.0% | 24.3 |
| `ck20` | short | clean | `ck20` | 0.0% | 11.0 |
| `ck40` | short | clean | `ck40` | 0.0% | 12.1 |
| `ck60` | short | clean | `ck60` | 0.0% | 11.8 |
| `ck80` | short | clean | `ck80` | 0.0% | 11.4 |
| `ck100` | short | clean | `ck100` | 0.0% | 11.3 |

The served model id is echoed by vLLM per request. A LoRA arm reporting the base id would mean the adapter was not applied.
