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
