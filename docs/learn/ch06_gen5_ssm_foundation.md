# ch06 — Gen5 SSM/Mamba and Foundation Models (2023–)

> Companion notebook: `notebooks/ch06_gen5.ipynb` — MambaTSAD faithful vs fixed (v2)

## State-space models and selective gating

The core of Mamba (S6) is making the state update
\(h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t\) **input-dependent (selective)**:
\(\Delta_t, B_t, C_t = f(x_t)\). It handles long dependencies in O(T) without the
O(T²) attention of a Transformer.

**Δ discretization and the uniform-sampling assumption**: discretizing the continuous
SSM as \(\bar{A} = \exp(\Delta A)\) makes Δ play the role of "time between samples" —
i.e. an implicit **assumption of uniform sampling**. Applying an SSM directly to
irregularly sampled data (the industrial reality of ch09) breaks this assumption.

## MambaTSAD and the faithful/fixed experiment

> **Correction (v2).** An earlier version of this page described four "known issues"
> and a faithful-vs-fixed result table (mean VUS-PR 0.251 → 0.502). That v1
> implementation was written **without reading the official code**, and most of its
> "issues" turn out not to exist upstream: for example, the official model feeds the HP
> *cycle*, not the trend, and there is no CPU branch that drops selectivity. Those
> results are retracted, moved to `benchmarks/results/_superseded/mamba_tsad_v1/`, and
> must not be cited. The lesson stands, aimed at us: **check the source before claiming
> a reproduction.**

The v2 implementation was rebuilt after reading the official repository
(aaaceo890/mamba_tsad @ `08fec05`). It is a behavioral reimplementation (the repository
has no license, so no code was copied):

- **Model.** HP pre-filter, then three RMSNorm→S6 blocks without residuals, with
  FFT-period moving-average (AMA) decomposition after each block.
- **Training.** Sequential, unshuffled batches that carry hidden state across batches.
- **Scoring.** Non-overlapping test windows.

Issues verified in the official code:

| Issue | Official behavior | Flag (faithful → fixed) |
|---|---|---|
| (a) inference hidden-state indexing | the state carried to the next test window is `hs[:, b-1]`, which indexes time by the batch size. With test batch 1 it passes the state of the **first** step, not the last | `state_index: upstream → last` |
| (b) CPU/GPU branch | the CPU branch of the training loop still calls `.cuda()`, so the official code cannot train on CPU (its results are GPU-only) | none (this implementation is device-invariant) |
| (c) HP objective | solves `(I + 2λD₁ᵀD₁ + λD₂ᵀD₂)τ = y`, an extra first-difference term, instead of the HP definition `(I + λD₂ᵀD₂)τ = y` | `hp_objective: upstream → standard` |
| (d) AMA period | the PSD is averaged over the **whole batch** and all channels, giving one period per batch. Training (batch 128) and inference (batch 1) therefore estimate it from different populations | `ama_period: batch → sample` |

Reproduce:

- `configs/mamba_paper_repro.yaml`: paper entities and official hyperparameters.
- `configs/mamba_ablation.yaml`: faithful, then each issue fixed alone, then all fixed, as separate leaderboard rows via `model_alias`.
- `configs/mamba_compare.yaml`: small CPU smoke run.

No v2 results are published yet — the official protocol needs a GPU, about one hour
per epoch per SMD machine on CPU.

## Foundation models and zero-shot

MOMENT, Chronos and TimesFM are general-purpose models pretrained on large time-series
corpora. Two adapter styles for TSAD:

- **zero-shot reconstruction** (`moment`): masked-reconstruction error as the score
- **forecast residual** (`chronos`, `timesfm`, `timesfm3`): the residual of a forecaster as the score

Attractive for cold starts with no training data (ch09), but (1) inference is
expensive and (2) performance can collapse on domains far from the pretraining
distribution (industrial sensors). Install with `pip install tsad-forge[foundation]`.

### The shared forecast-residual protocol

All forecast-based adapters use one fixed protocol so they stay comparable:
`fit()` is zero-shot (load weights, keep the last `context` steps of train as the
initial context) and `score()` slides with stride = `horizon`, predicting **every
test step exactly once**. Three score conventions are available: `residual`
(|x − median|, the default and the one used on the leaderboard), `crps` (mean
pinball loss over the quantiles) and `interval` (deviation normalised by the
prediction-interval width).

### TimesFM 3.0 and the "rank #1" question

TimesFM 3.0 (Aug 2026, 330M params, ~1T training points) adds native multivariate
forecasting with variate attention, and ranks #1 on GIFT-Eval, fev-bench and TIME —
on point *and* probabilistic metrics. Those are **forecasting** benchmarks with no
anomaly labels, so the ranking says nothing directly about TSAD. This repo turns
the claim into a measurable question with three registered variants on identical
weights and settings:

| Name | Isolates |
|---|---|
| `timesfm3` | native multivariate joint forecast, point residual |
| `timesfm3_ci` | channel-independent control (variate attention off) |
| `timesfm3_prob` | probabilistic score (CRPS over the 9 quantiles) |

Every run also records `fc_mase` / `fc_wql` / `fc_crps` next to `vus_pr`, so
forecast skill and detection skill can be read off the same row. The
counterintuitive part matters: **a better forecaster can be a worse detector**,
because a model with enough context will extrapolate a level shift straight
through the anomaly and leave a near-zero residual.

Caveats to keep in mind when reading those rows: TimesFM 3.0's variate attention
covers at most 32 channels per forward pass (SMD has 38, SWaT 51 — the library
chunks them), and the 3.0 weights are under a non-commercial license while 2.5
stays Apache-2.0. Full discussion: [Forecasting benchmarks vs. TSAD
benchmarks](../benchmarks/forecasting-benchmarks.md).
