# Multi-token smear — experiment plan

Extend the 1-position smear (PR #130 by @ClassicLarry) to k previous tokens with
per-offset content gates. Round 1 is deliberately narrow: establish whether the model
wants offsets d > 1 at the embedding level at all, and what weights it assigns them.

## Background

Current master smears each token embedding with its predecessor, gated by the first
12 dims of the current token's embedding (`train_gpt.py:1403-1405`):

```python
smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :12]))
x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
```

`smear_gate = nn.Linear(12, 1)` and `smear_lambda` are zero-init, so training starts
as an exact no-op. Motivation from the PR: many attention heads were consistently
attending to the prior token; attention is an expensive way to do that. The PR's
stated value: "a new design space — directly modeling close-range information passing
between tokens outside of attention." This experiment is the next step in that space.

Prior evidence that width > 1 helps: short causal convolutions are load-bearing in
H3, Mamba (conv width 4), Based, and "Canon layers" — all content-independent
convs over the sequence. Ours is the gated, embedding-level analog.

## Hypothesis

Heads also spend capacity on offsets 2..5 (and PR #130 freed only offset 1).
Pulling those offsets into a cheap gated sum lets attention spend its budget
elsewhere. Expected effect size is small (PR #130 was worth ~5 steps, ~0.001 loss),
so round 1 optimizes for information per dollar, not statistical confirmation:
the learned per-offset weights are themselves the primary readout.

## Round-1 variant: per-offset gates

For offsets d = 1..k, each with its own gate column and scalar:

```python
x[t] += sum_d  lambda_d * sigmoid(W_d @ x_embed[t][:12]) * x_embed[t-d]
```

- All contributions read the *original* embeddings (`x_embed`), and all gates read
  the original current-token embedding — no compositional cascading, matching
  PR #130 semantics at k=1.
- The sum over d is math notation, not the implementation. Everything is matrix
  ops: one `nn.Linear(12, k)` matmul produces all k gates, and the shifted
  weighted sum is one einsum over a zero-copy `as_strided` view of the
  front-padded embeddings:

  ```python
  gates = smear_lambdas.bfloat16() * torch.sigmoid(self.smear_gate(x[:, :12]))  # (T, k)
  xp = F.pad(x, (0, 0, k, 0))                     # k zero rows absorb t-d < 0
  shifts = xp.as_strided((k, T, D), (D, D, 1))    # shifts[e, t] = x[t-(k-e)], no copy
  x = x + torch.einsum('etc,te->tc', shifts, gates)
  ```

  The `.bfloat16()` cast is required, not stylistic: einsum does not type-promote
  ("expected scalar type BFloat16 but found Float"), and the fp32 `(k,)` lambdas
  would otherwise promote the gates. Master got away without a cast only because
  its `smear_lambda` is 0-dim, and scalar promotion keeps bf16.
  Gate column e maps to offset k−e (column semantics are ours to assign;
  zero-init makes the ordering arbitrary). Zero-padding replaces PR #130's
  `torch.cat` prefix trick: out-of-range terms contribute exactly 0. An
  unrolled add chain likely fuses to the same kernel under torch.compile —
  benchmark both, step-time parity is the constraint.
- `lambda_d`: k slots in `self.scalars` instead of 1. This shifts `skip_lambda`'s
  index and the pad size — update `init_misc` (`train_gpt.py:1269-1279`) and the
  forward unpack (`train_gpt.py:1363-1364`) together.
- Param group: existing `smear_gate` adam group (lr_mul 0.01) covers the wider
  Linear unchanged.
- Doc boundaries: like PR #130, no masking across BOS in round 1 (noted as a
  round-2 ablation — matters more at k=5).
- Zero-init ⇒ step-0 equivalence with baseline for every k.

Configs: **k = 1 (control, must reproduce master), 2, 3, 5.**

Instrumentation (the cheap signal): at the final step, rank 0 prints per-offset
`lambda_d` and the mean sigmoid gate activation over the last val batch. Smoke runs
can't resolve 0.001 final-loss effects, but they *can* show whether the model grows
nonzero weight on d > 1 within the first third of training.

## Protocol

Infra: Modal launcher from the `modal-setup` branch (see `MODAL.md`).
Run naming: `smear-k{K}-{smoke|full}-r{repeat}`.

**Rule: no full 8×H100 run without explicit approval.** The default loop is
hypothesis → smoke runs → review the results together; full runs are launched in
batches only after that review signs off on which configs earned one.

1. **k=1 regression check** — one smoke run; loss curve must overlay master's
   baseline smoke within noise, step time neutral, no compile graph breaks.
2. **3 smoke repeats per config** (1×H100, `--stop-frac 0.33 --val-every 155`,
   ~$1 each — covers all of stage 1 with sparse val evals, per `modal_train.py`
   guidance): no NaN, loss-curve sanity vs baseline band, step time neutral,
   record learned lambda_d / gate stats. Repeats capture run-to-run noise
   (runs are nondeterministic; there is no seed knob).
3. **After approval: 1 full 8×H100 run per surviving config** (~$4.5 each):
   final val loss vs baseline. A single run only *ranks* — treat differences
   < 2σ (σ ≈ 0.001 from PR #130's data) as noise. Re-establish the baseline
   number with our own k=1 full run rather than quoting historical records
   (master and hardware moved).
4. **Escalate to 8-10 full seeds** (again approval-gated) only if a config looks
   ≥ 0.002 better or we want an upstreamable claim, per ClassicLarry's 10-run
   t-test methodology.

Estimated round-1 cost: ~10 smoke runs + up to 4 full runs ≈ **$25**.

## Decision criteria

- **Kill a config**: NaN; > 0.5% step-time regression without a loss win;
  lambda_d ≈ 0 for all d > 1 by end of smoke (model doesn't want the offsets).
- **Promote to round 2**: any k ranks ≥ 1σ better than baseline on the full run,
  or the lambda_d profile is informative (e.g. slow decay → try strided).

## Round 2 candidates (chosen after seeing round-1 weights)

- **Softmax-weighted average** over the last k tokens plus self, one outer lambda —
  offsets compete instead of summing independently (the "mini attention head
  without K/Q" version of the idea).
- **Static kernel ablation** (drop the content gate) — is gating even needed for d > 1?
- **Strided offsets** {1, 2, 4, 8} if the learned decay is slow.
- **Gate from the source token** x[t-d] (or both) instead of only the current token.
- **Doc-boundary masking** at BOS.
- **Other injection points**: pre-lm_head, or per-layer (Canon-layer style).

## Follow-up (separate experiment): hand-built circuit primitives

The broader agenda: any attention pattern computable from token *identity* alone can
be replaced by precomputed gather indices + a gated add — O(T·dim) instead of
attention — freeing heads for content-dependent work. Candidates, roughly in order
of implementation cost:

1. **Punctuation / sink primitive.** Precompute, from `input_seq` alone, the index of
   the previous newline / period / EOS for each position; gather the residual there
   and add gated (last one, or an exponentially- or softmax-weighted last few).
   Rationale: well-known heads attend to previous punctuation as segment summaries
   / sink tokens.
2. **Previous-occurrence / induction primitive.** For each t, find the last s < t
   with `input_seq[s] == input_seq[t]` (one scatter over vocab ids, O(T)), retrieve
   the *continuation* x[s+1], add gated — a K/Q-free induction head. Extension:
   approximate matching via bigram signature or embedding similarity. Note master's
   bigram embeddings (PR #299) and value embeds already capture some of this;
   measure overlap before claiming wins.
3. Same recipe for any other identity-defined pattern (BOS sink, previous same-word
   with different case, etc.), injectable at layer 0, mid-layers, or pre-logits.
