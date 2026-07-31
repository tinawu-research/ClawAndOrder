# Base vs fine-tuned Nemotron — controlled comparison

Every arm answers the same questions from byte-identical frozen tool evidence. Qwen routing and tool execution are held fixed, so the only variable between arms is the synthesis model's weights.

- Judge: calibrated component judge (see `judge_calibration.md`)
- Baseline arm: `base`
- Decoding: greedy (`temperature=0`, `seed=0`)


## Component score

| arm | clean | noisy | insufficient | shuffled |
|---|---|---|---|---|
| `base` | 80.7% | 87.7% | 5.3% | 86.0% |
| `ck40` | 94.7% | 82.5% | 7.9% | 79.8% |
| `ck100` | 89.5% | 86.8% | 14.9% | 89.5% |

## Paired delta vs `base`

Bootstrap over questions (2,000 resamples), not components — components within a question are correlated. `p` is an exact sign-flip permutation test on paired differences.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck40` | clean | 80.7% | 94.7% | **+14.0%** | [+3.5%, +27.2%] | 0.0625 | 5/0/14 |
| `ck100` | clean | 80.7% | 89.5% | **+8.8%** ⚠ | [-7.9%, +26.3%] | 0.4062 | 5/1/13 |
| `ck40` | noisy | 87.7% | 82.5% | **-5.3%** ⚠ | [-26.3%, +15.8%] | 1.0000 | 2/3/14 |
| `ck100` | noisy | 87.7% | 86.8% | **-0.9%** ⚠ | [-14.0%, +14.0%] | 1.0000 | 2/3/14 |
| `ck40` | insufficient | 5.3% | 7.9% | **+2.6%** ⚠ | [-5.3%, +10.5%] | 1.0000 | 2/1/16 |
| `ck100` | insufficient | 5.3% | 14.9% | **+9.6%** | [+1.8%, +18.4%] | 0.1250 | 4/0/15 |
| `ck40` | shuffled | 86.0% | 79.8% | **-6.1%** ⚠ | [-21.9%, +6.1%] | 0.5000 | 2/3/14 |
| `ck100` | shuffled | 86.0% | 89.5% | **+3.5%** ⚠ | [-13.2%, +17.5%] | 0.7656 | 5/2/12 |

⚠ = 95% CI includes zero; not a demonstrated improvement.


### Strict score (judge **and** numeric tolerance must agree)

The headline metric follows the organizers' LLM judge. This one also requires the stated value to pass tolerance arithmetic, so it catches answers a judge accepts but whose figures are mis-stated or mis-formatted.

| arm | condition | base | arm | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|---|---|---|
| `ck40` | clean | 47.4% | 94.7% | **+47.4%** | [+28.9%, +68.4%] | 0.0010 | 11/0/8 |
| `ck100` | clean | 47.4% | 89.5% | **+42.1%** | [+21.1%, +65.8%] | 0.0049 | 10/1/8 |
| `ck40` | noisy | 43.9% | 82.5% | **+38.6%** | [+12.3%, +64.0%] | 0.0149 | 11/2/6 |
| `ck100` | noisy | 43.9% | 86.8% | **+43.0%** | [+22.8%, +65.8%] | 0.0039 | 9/0/10 |
| `ck40` | insufficient | 5.3% | 7.9% | **+2.6%** ⚠ | [-5.3%, +10.5%] | 1.0000 | 2/1/16 |
| `ck100` | insufficient | 5.3% | 14.9% | **+9.6%** | [+1.8%, +18.4%] | 0.1250 | 4/0/15 |
| `ck40` | shuffled | 50.0% | 77.2% | **+27.2%** | [+7.0%, +49.1%] | 0.0410 | 8/2/9 |
| `ck100` | shuffled | 50.0% | 89.5% | **+39.5%** | [+15.8%, +63.2%] | 0.0112 | 10/2/7 |

## Secondary metrics (deterministic, no judge)

`hallucinated_number_rate` counts numeric literals asserted in the answer that appear in neither the evidence nor the question, allowing for rounding. It is computable only because the evidence is frozen.

| arm | condition | hedge | halluc. num | leaked reasoning | format viol. | empty | words p50 |
|---|---|---|---|---|---|---|---|
| `base` | clean | 0.0% | 10.5% | 0.0% | 0.0% | 0.0% | 26 |
| `base` | noisy | 0.0% | 10.5% | 0.0% | 0.0% | 0.0% | 24 |
| `base` | insufficient | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 34 |
| `base` | shuffled | 0.0% | 10.5% | 0.0% | 0.0% | 0.0% | 25 |
| `ck40` | clean | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 13 |
| `ck40` | noisy | 0.0% | 15.8% | 0.0% | 0.0% | 0.0% | 13 |
| `ck40` | insufficient | 0.0% | 73.7% | 0.0% | 0.0% | 0.0% | 14 |
| `ck40` | shuffled | 0.0% | 10.5% | 0.0% | 0.0% | 0.0% | 13 |
| `ck100` | clean | 0.0% | 5.3% | 0.0% | 0.0% | 0.0% | 13 |
| `ck100` | noisy | 0.0% | 5.3% | 0.0% | 0.0% | 0.0% | 10 |
| `ck100` | insufficient | 0.0% | 73.7% | 0.0% | 0.0% | 0.0% | 15 |
| `ck100` | shuffled | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 10 |

## Prompt fairness control (2×2)

The system prompt was shortened (332 → 145 tokens) to fit the sequence budget, so the fine-tuned arm was trained on the short prompt. Without this control the comparison would silently be "base + long prompt vs FT + short prompt". Both arms are run against both prompts.

| arm | prompt | condition | component score |
|---|---|---|---|
| `base` | short (shipped) | clean | 80.7% |
| `base` | long (pre-FT) | clean | 90.3% |
| `ck100` | short (shipped) | clean | 89.5% |
| `ck100` | long (pre-FT) | clean | 94.7% |
| `ck40` | short (shipped) | clean | 94.7% |
| `ck40` | long (pre-FT) | clean | 86.0% |

## Provenance

| arm | prompt | condition | served model id | errors | gen seconds |
|---|---|---|---|---|---|
| `base` | short | clean | `nemotron-8b-finance` | 0.0% | 14.6 |
| `base` | short | noisy | `nemotron-8b-finance` | 0.0% | 17.7 |
| `base` | short | insufficient | `nemotron-8b-finance` | 0.0% | 18.3 |
| `base` | short | shuffled | `nemotron-8b-finance` | 0.0% | 15.0 |
| `base` | long | clean | `nemotron-8b-finance` | 0.0% | 16.0 |
| `ck40` | short | clean | `ck40` | 0.0% | 9.3 |
| `ck40` | short | noisy | `ck40` | 0.0% | 10.7 |
| `ck40` | short | insufficient | `ck40` | 0.0% | 44.0 |
| `ck40` | short | shuffled | `ck40` | 0.0% | 11.0 |
| `ck40` | long | clean | `ck40` | 0.0% | 10.7 |
| `ck100` | short | clean | `ck100` | 0.0% | 9.1 |
| `ck100` | short | noisy | `ck100` | 0.0% | 10.3 |
| `ck100` | short | insufficient | `ck100` | 0.0% | 9.3 |
| `ck100` | short | shuffled | `ck100` | 0.0% | 9.9 |
| `ck100` | long | clean | `ck100` | 0.0% | 10.2 |

The served model id is echoed by vLLM per request. A LoRA arm reporting the base id would mean the adapter was not applied.
