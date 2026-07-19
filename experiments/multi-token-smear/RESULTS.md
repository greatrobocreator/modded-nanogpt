# Multi-token smear — results

## Round 1 full runs (2026-07-19)

9 full runs: k ∈ {1, 2, 3} × 3 repeats, **1×H100**, 1390 steps, `--val-every 155`.
1×H100 chosen over 8×H100 for the loss ablation: grad accumulation keeps the token
schedule identical, and the ~7-min compile warmup is paid on 1 GPU instead of 8
(~$1.5/run vs ~$5). k=5 held pending a clean step-time check. Plots:
`plots/loss_curves_full.png`, `plots/final_loss_full.png` (from `plot_results.py`).

### Final val loss @ step 1390

| config | runs | mean ± sd |
|---|---|---|
| k=1 | 3.2760 / 3.2790 / 3.2817 | 3.2789 ± 0.0029 |
| k=2 | 3.2757 / 3.2801 / 3.2795 | 3.2784 ± 0.0024 |
| k=3 | 3.2796 / 3.2780 / 3.2768 | 3.2781 ± 0.0014 |

Monotone trend in the hypothesized direction (k=3 −0.0008 vs k=1) but ~0.4σ of the
difference SE (±0.0018): **statistically a dead heat at n=3**. Per the plan's
criterion (escalate only if ≥ 0.002 better), this does not justify a 10-seed
record-attempt campaign on loss grounds alone.

### Per-offset lambdas at 100% of training (mean ± sd over 3 runs)

| config | λ₁ | λ₂ | λ₃ |
|---|---|---|---|
| k=1 | 0.2223 ± 0.0030 | | |
| k=2 | 0.2338 ± 0.0094 | 0.0457 ± 0.0079 | |
| k=3 | 0.2336 ± 0.0033 | 0.0490 ± 0.0052 | 0.0233 ± 0.0040 |

Trajectory vs the 33% smoke readout: λ₁ decayed (0.32-0.36 → 0.22-0.23), λ₂ held
(≈ 0.046-0.049 at both points), **λ₃ doubled** (0.011 → 0.023) — longer offsets gain
weight in the later stages (larger windows/sequences). The λ₁-grows-with-k effect
persists but compressed (0.222 → 0.234).

### Gate verdict at 100% of training

`gate mean ≈ 0.500-0.5015, std ≈ 0.006 (d=1) / 0.003 (d=2) / 0.002 (d=3)`, weight
norm ≈ 0.09-0.12 for every config. The content gate wakes up but modulates by well
under ±2% — **effectively a constant 0.5 multiplier even at full training**. The
round-2 static-kernel ablation (drop the gate) is strongly motivated.

### Step time

Modal H100 hosts are bimodal (~240 vs ~310 ms/step stage-1 floors across identical
configs). k=2 matched k=1 at every stage (237/444/650 vs 240/438/632 ms). All three
k=3 full runs landed on slow hosts (311/607/892), but the two clean k=3 *smoke* runs
at ~241 ms already bound the k=3 kernel at ≈ neutral — placement luck, not kernel
cost. Any record claim still requires an 8×H100 confirmation run.

### Observation: stage-transition robustness

At the step-465 val (right at the stage-1→2 window/seq jump), k=2 was lower than
every k=1 run (4.254-4.275 vs 4.279-4.375, and k=1's spread there is wide). By step
620 the gap is gone. Weak evidence (n=3) that the smear buffers the transition;
not load-bearing for any decision.

## Round 1 smoke runs (2026-07-19)

12 runs: k ∈ {1, 2, 3, 5} × 3 repeats, 1×H100, `--stop-frac 0.33 --val-every 155`
(459/1390 steps = all of training stage 1). Logs: `smear-k{K}-smoke-r{R}.txt` in the
Modal volume. Baseline: `baseline-33pct.txt` (master, single run).

### Final val loss @ step 459

| config | runs | mean ± σ |
|---|---|---|
| baseline | 4.1946 | (1 run) |
| k=1 | 4.1913 / 4.1814 / 4.1889 | 4.1872 ± 0.0052 |
| k=2 | 4.1926 / 4.1967 / 4.1886 | 4.1926 ± 0.0041 |
| k=3 | 4.1936 / 4.2102 / 4.1987 | 4.2008 ± 0.0085 |
| k=5 | 4.2002 / 4.1930 / 4.1918 | 4.1950 ± 0.0045 |

No separation beyond run noise (σ ≈ 0.005), as expected at smoke scale — and no
damage. k=1 passed the regression gate: loss overlays baseline, so the einsum
rewrite is behaviorally equivalent to master's cat-based smear.

### Per-offset lambdas (zero-init; value after 459 steps, mean of 3 runs)

| config | λ₁ | λ₂ | λ₃ | λ₄ | λ₅ |
|---|---|---|---|---|---|
| k=1 | 0.3177 | | | | |
| k=2 | 0.3378 | 0.0445 | | | |
| k=3 | 0.3557 | 0.0493 | 0.0114 | | |
| k=5 | 0.3632 | 0.0510 | 0.0073 | 0.0129 | 0.0086 |

- **λ₂ is robustly nonzero** (0.042–0.054 in all 9 k≥2 runs, σ ≈ 0.003; ~14% of λ₁).
  The model wants offset 2 — the round-1 hypothesis survives.
- **λ₁ grows with k**: 0.318 → 0.363 monotone, gap ≈ 6σ. Offsets reinforce rather
  than split a fixed budget; total smear mass rises ~40% from k=1 to k=5.
- **d ≥ 3 is marginal**: λ₃ ≈ 0.01 (consistent), λ₄/λ₅ noisy ≈ 0.01. Decay is fast
  (~7× per offset step), so the strided-offsets round-2 variant is not indicated.
- **Content gates never moved**: sigmoid output 0.5000 (max |dev| 0.0009) for every
  offset in every run at this training fraction — the smear is effectively
  0.5·λ_d·x[t−d]. Makes the round-2 static-kernel ablation the lead follow-up.

### Step time

Contaminated by Modal host variance (11 concurrent containers; same-config repeats
differ by up to 60%, e.g. k=3 at 241 vs 387 ms/step). Best per-config 155-step
segment (clean-host proxy): k=1 239.3, k=2 243.4, k=3 240.7, k=5 306.0 ms.

- Two k=3 runs at k=1 speed bound the einsum cost at **≤ +0.6%**; this also bounds
  k=2 (its kernel does strictly less work).
- k=5 never landed on a clean host (all 9 segments ≥ 306 ms): either bad luck ×3 or
  a real kernel cliff — needs one solo staggered rerun before any full-run decision.

### Decision (per plan criteria)

- No NaN anywhere; no config killed by loss.
- Promote **k=2 and k=3** to approval-gated full 8×H100 runs (plus a fresh k=1 full
  baseline).
- **Hold k=5** pending a clean step-time measurement.
- Round-2 lead: static-kernel ablation (drop the dead content gate).
