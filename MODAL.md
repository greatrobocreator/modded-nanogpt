# Running the speedrun on Modal

## One-time setup

```bash
uv tool install modal
modal setup                                   # opens browser, stores token in ~/.modal.toml
modal run modal_train.py --download-only      # caches ~2 GB of FineWeb tokens in a Modal volume
```

No other credentials are needed: the tokenized dataset (`kjj0/fineweb10B-gpt2`) is public on
Hugging Face, and there is no wandb or other tracking service to sign into.

## Running

```bash
modal run --detach modal_train.py             # current record, 8xH100
```

`--detach` keeps the run alive if your laptop disconnects; progress streams to the terminal
either way. The first run pays ~7 minutes of `torch.compile` latency; compile caches persist
in the volume, so later runs on unchanged model code start much faster.

Variations:

```bash
modal run modal_train.py --script train_gpt_medium.py --chunks 30   # GPT-2 medium track
NANOGPT_GPU='H100!:2' modal run modal_train.py                      # debug on fewer GPUs
NANOGPT_TIMEOUT=7200 modal run modal_train.py                       # raise the 1h kill switch
```

## Cheap smoke runs (~$1)

For experiments on training *dynamics*, run the first 20% of training on a single H100:

```bash
NANOGPT_GPU='H100!:1' modal run --detach modal_train.py \
    --stop-frac 0.2 --val-every 50 --run-id baseline-20pct
```

- `--stop-frac 0.2` stops after 20% of the steps. All schedules (LR, attention windows,
  batch size, seq len) are still computed over the full run, so the truncated run's loss
  curve is exactly the first 20% of a full run — dynamics are unchanged, and the final
  window extension for validation is skipped so the last eval stays comparable to a full
  run's intermediate evals at the same step.
- `--val-every 50` gives ~7 val points for plotting (default 250 gives 2). Each eval is a
  full 10.5M-token val pass, noticeable on 1 GPU — don't go much denser than 25.
- `--run-id <name>` names the log file in the volume instead of a random UUID.
- `--run-evals` additionally runs HellaSwag after training.
- These knobs only exist in `train_gpt.py`, not `train_gpt_medium.py`.

Why 1 GPU rather than 8 for smoke runs: training math is identical (`world_size` 1 uses
8-way gradient accumulation, same token stream, same updates) and total training
GPU-seconds are ~the same, but the fixed overhead — ~7 min of torch.compile, kernel
warmup, setup — bills at 1 GPU-rate instead of 8. A 20% run is dominated by that
overhead, so 1xH100 costs ~$1 (~$0.50 with warm compile caches, ~20 min wallclock) vs
~$4.5 on 8xH100 (~10 min wallclock). The tradeoffs: wallclock is longer, timing numbers
are meaningless (fine — smoke runs measure loss, not speed), and systems experiments
(comm patterns, multi-GPU kernels) still need 8 GPUs to test what they actually change.

## Suggested experiment flow

1. Branch off `master`, make a change to `train_gpt.py`.
2. Smoke-run baseline and experiment at `--stop-frac 0.2` on 1xH100, compare val-loss
   curves at equal steps.
3. Only if the curve looks at least as good: full 8xH100 runs (several, for a mean —
   inter-run variance is real).

`world_size` must divide 8 (1/2/4/8 GPUs); gradient accumulation keeps training math identical,
only wallclock changes. Stick to H100s: the code uses FP8 and Flash Attention 3, which need
Hopper. `H100!` pins real H100s — plain `H100` may be silently upgraded to H200, which breaks
timing comparability with the official records (timed on 8xH100).

## Cost (July 2026 Modal prices)

| Setup | Rate | Typical record run |
| - | - | - |
| 8x H100 | ~$31.6/h | ~90 s timed + compile/warmup/eval ≈ $4–8 |
| 2x H100 | ~$7.9/h | same total GPU-seconds, 4x wallclock |

Modal's Starter plan includes $30/month free credits. The timeout defaults to 1 hour
(`NANOGPT_TIMEOUT` to change), so a hung run can't burn more than ~$32.

## Tracking experiments

The repo has its own convention, used for all record submissions — no wandb:

- Every run writes `logs/<run_id>.txt` containing the full `train_gpt.py` source, the
  environment (torch/CUDA/GPU info), and one line per eval:
  `step:N/M val_loss:X train_time:Yms step_avg:Zms`.
- The final `val_loss` and `train_time` of that file are the result. Records require mean
  val loss ≤ 3.28 with p<0.01 across several runs (see README "Rules"), so run experiments
  in batches and compare means, not single runs — inter-run variance is real.
- `modal_train.py` copies each run's log into the volume. Retrieve and compare:

```bash
modal volume ls modded-nanogpt-data logs
modal volume get modded-nanogpt-data logs/<run_id>.txt
grep val_loss <run_id>.txt | tail -3
```

To label an experiment, set a descriptive `run_id` in `Hyperparameters` before launching —
the log file is then named after it. Keep one git branch per experiment; the log embeds the
exact source, so every result stays reproducible.
