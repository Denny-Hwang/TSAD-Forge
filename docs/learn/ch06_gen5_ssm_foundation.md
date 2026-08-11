# ch06 — Gen5 SSM/Mamba and Foundation Models (2023–)

> Companion notebook: `notebooks/ch06_gen5.ipynb` — MambaTSAD faithful vs fixed

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

The public implementation of MambaTSAD (Chen et al., IEEE SPL 2024) has four known
issues. This repository ships both variants side by side to **quantify how much
implementation quality moves benchmark numbers**
(`mamba_tsad_faithful` vs `mamba_tsad_fixed`):

| Issue | Description |
|---|---|
| hidden-state indexing | the output reads \(h_{t-1}\) — an off-by-one |
| CPU/GPU branch | the CPU path loses selectivity → a different model per device |
| HP-filter objective | feeds the trend instead of the cycle component |
| global-FFT AMA | one FFT over the whole series → a fixed window that ignores local period changes |

Reproduce: `python benchmarks/run_all.py --profile configs/mamba_compare.yaml`
The two variants appear as separate models on the leaderboard. **The lesson: a
substantial part of reported performance can hinge on implementation details —
which is exactly why reproduction studies matter.**

## Foundation models and zero-shot

MOMENT, Chronos and TimesFM are general-purpose models pretrained on large time-series
corpora. Two adapter styles for TSAD:

- **zero-shot reconstruction** (`moment`): masked-reconstruction error as the score
- **forecast residual** (`chronos`, `timesfm`): the residual of a forecaster as the score

Attractive for cold starts with no training data (ch09), but (1) inference is
expensive and (2) performance can collapse on domains far from the pretraining
distribution (industrial sensors). Install with `pip install tsad-forge[foundation]`.
