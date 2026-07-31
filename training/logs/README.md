# Training run logs

Every run, including the failures. The failures are kept deliberately: two of
them located real configuration faults, and the learning-rate evidence in run 4
is what justifies the deviation from the reference baseline documented in
[`../MODEL_CARD.md`](../MODEL_CARD.md).

All runs use the same data (790 rows), the same seed (1111), and the same
container (`nvcr.io/nvidia/nemo:25.09`) on one NVIDIA GB10.

| # | log | launcher | LR schedule | outcome |
|---|---|---|---|---|
| 1 | `nemo_lora_20260731-130815.log` | torchrun | — | hang in rendezvous |
| 2 | `nemo_lora_20260731-132916.log` | torchrun | — | hang in rendezvous |
| 3 | `nemo_lora_20260731-133633.log` | python | **none** | diverged by step 6 |
| 4 | `nemo_lora_20260731-133914.log` | python | warmup 50 → 5e-5 | diverged at step ~44 |
| 5 | `nemo_lora_20260731-135718.log` | python | warmup 20 → **2e-5** | **shipped** |

## Runs 1–2 — torchrun rendezvous hang

```
The client socket has timed out after 20min while trying to connect to (localhost, 57117)
```

Not a network problem. AutoModel's single-worker path picks its own free port and
calls `init_process_group` with `init_method="tcp://localhost:<port>"`:

```python
free_port = find_free_port()
init_pg_kwargs["init_method"] = f"tcp://localhost:{free_port}"
```

Under `torchrun`, `TORCHELASTIC_USE_AGENT_STORE=True` puts the TCPStore into
*client* mode, so rank 0 tries to **connect** to that port rather than listen on
it. Nothing is listening, and it blocks for the full distributed timeout.

This is why the usual fixes were inert — `--network=host`, a static rendezvous,
and `MASTER_ADDR` are all overridden by that `init_method` assignment.

Verified directly in the container:

```
under torchrun:  TORCHELASTIC_USE_AGENT_STORE=True  -> FAIL (client socket timed out)
plain python:    TORCHELASTIC_USE_AGENT_STORE=None  -> OK
```

Fix: launch with plain `python`. A single-process, single-GPU run needs nothing
`torchrun` provides. `dist_env.timeout_minutes` was also cut 20 → 5 so a future
misconfiguration reports in five minutes instead of twenty.

## Run 3 — no LR scheduler at all

```
step 0  loss 5.1104    step 3  loss 2.6863    step 5  loss 12.7175
step 1  loss 2.6142    step 4  loss 10.5366   step 6  loss 19.0064   grad_norm 18,176
```

Cause: the `lr_scheduler` block was dropped while porting the recipe between
container formats. `build_lr_scheduler()` returns `None` when the block is
absent — no warning, no error — giving a constant 5e-5 from step 0 with no
warmup.

The tell is in the log itself: run 3 has **no** `Building LR scheduler ...`
line, while runs 4 and 5 do. That absence is the whole diagnosis.

## Run 4 — warmup restored, peak still 5e-5

Warmup fixed the step-0 blowup. Comparing runs 3 and 4 at step 4, same seed and
same data, only the schedule differing:

| | run 3 (no warmup) | run 4 (warmup 50) |
|---|---:|---:|
| loss | 10.5366 | **1.4675** |
| grad_norm | 18,176 | **66.5** |

But the peak itself was still too hot. The run was healthy for 43 steps and then
came apart exactly as the ramp crossed ~4.5e-5:

| step | loss | grad_norm | lr |
|---:|---:|---:|---:|
| 32 | 0.27 | 17 | 3.47e-05 |
| 43 | 0.43 | 35 | 4.46e-05 |
| 44 | 0.43 | **152** | 4.55e-05 |
| 47 | 0.81 | **824** | 4.82e-05 |
| 49 | 0.86 | **1,384** | 5.00e-05 |
| 64 | 9.50 | **36,352** | 3.90e-05 |

It never recovered even as cosine decay brought the LR back below levels it had
been stable at — a damaged optimiser state, not one bad batch.

Checkpoints from this run are preserved on the fine-tuning node as
`checkpoints/run4_lr5e5_diverged/`. Steps 20 and 40 predate the divergence and
are usable; 60 and 80 are not.

## Run 5 — shipped

Peak LR 2e-5, warmup 20. Same seed and data as run 4, so step 0 reproduces
exactly (loss 5.1104) and the schedule is the only variable.

At peak LR:

| | run 4 at its peak (step 49) | run 5 at its peak (step 19) |
|---|---:|---:|
| loss | 0.86 | **0.47** |
| grad_norm | **1,384** | **27.5** |

Warmup shortened 50 → 20 because at 100 total steps a 50-step warmup spent half
the run below working LR, and the low-LR region was never where the instability
lived.

## Reading the logs

- `num_label_tokens` ~2,600–4,200 per global batch of 8 confirms
  `answer_only_loss_mask` is genuinely active — loss is on the answer span only,
  not the whole sequence.
- `mem` holds at ~24.6 GiB of 121 GB available. `local_batch_size: 1` was chosen
  to buy sequence length and, in hindsight, was over-cautious on memory.
- `[val]` lines appear every 20 steps against the 144-row validation split.
- Checkpoint directories are named `epoch_0_step_<N>` with `N` **0-indexed**, so
  `epoch_0_step_19` is the checkpoint written after 20 optimizer steps. The
  serving script maps these to arm names `ck20`, `ck40`, ….
