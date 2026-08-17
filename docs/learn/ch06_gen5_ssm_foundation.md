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

### Paper-setting reproduction (SPL 2024 entities)

`configs/mamba_paper_repro.yaml` runs both variants on the exact entities the
paper's official code uses (SMD ×5, SMAP A-4/T-1, MSL C-2, SWaT). Differences
from the paper: original-source data (not the repo's redistributed zip), and the
VUS-PR protocol instead of PA-F1 — so absolute numbers are **not** comparable
with the paper's table; the faithful-vs-fixed *relative* comparison is the point.
Results on the SMD portion (3 seeds; NASA/SWaT data must be placed locally):

| SMD machine | faithful (VUS-PR ± seed std) | fixed | Δ |
|---|---|---|---|
| machine-1-1 | 0.340 ± 0.042 | **0.748** ± 0.015 | +0.408 |
| machine-1-6 | 0.384 ± 0.015 | **0.663** ± 0.029 | +0.279 |
| machine-2-1 | 0.324 ± 0.017 | **0.429** ± 0.003 | +0.105 |
| machine-3-2 | 0.099 ± 0.013 | **0.192** ± 0.010 | +0.092 |
| machine-3-7 | 0.109 ± 0.110 | **0.477** ± 0.015 | +0.368 |
| **mean** | 0.251 | **0.502** | +0.251 |

The fixed variant wins on **all five machines**, doubles the mean VUS-PR, and
cuts the mean seed standard deviation from 0.039 to 0.014 (event-F1: 0.092 →
0.337) at identical runtime — the four implementation issues account for half
of the achievable score in the paper's own setting.

## Foundation models and zero-shot

MOMENT, Chronos and TimesFM are general-purpose models pretrained on large time-series
corpora. Two adapter styles for TSAD:

- **zero-shot reconstruction** (`moment`): masked-reconstruction error as the score
- **forecast residual** (`chronos`, `timesfm`): the residual of a forecaster as the score

Attractive for cold starts with no training data (ch09), but (1) inference is
expensive and (2) performance can collapse on domains far from the pretraining
distribution (industrial sensors). Install with `pip install tsad-forge[foundation]`.
