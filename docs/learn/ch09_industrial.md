# ch09 — Industrial Practice: a Semiconductor FDC/ATE Perspective

> Companion notebook: `notebooks/ch09_industrial.ipynb` — the BYOD workflow

## The gap between benchmarks and the fab

| Benchmark assumption | Semiconductor FDC/ATE reality |
|---|---|
| a normal train span exists | "normal" changes with every recipe and PM cycle |
| uniform sampling | per-step sampling rates differ; event-driven logging |
| one long continuous series | many short fragments per wafer/lot |
| point labels | weak labels at the level of "this lot failed" |

## Regime vs fault — the most common failure mode

Recipe switches, post-PM behavior and seasonal shifts are **operating-regime changes**:
statistically anomalous, but not faults. Mitigations:

1. split data by regime metadata (recipe ID, …) and train per-regime models
2. drift-adaptive thresholds like DSPOT (ch08)
3. separate changepoint detection from anomaly detection — SKAB labels both

## Sparse labels, irregular sampling, cold start

- **Sparse labels**: select models with threshold-free metrics (VUS), guarantee the
  false-alarm rate with conformal thresholds
- **Irregular sampling**: breaks the SSM Δ-discretization assumption (ch06) —
  resample, or add the interval as a feature. Matrix Profile also assumes uniform
  sampling.
- **Cold start**: no train data for a new tool or recipe — foundation-model zero-shot
  (ch06) or transfer from similar equipment. The SECOM card (docs/datasets/secom.md)
  illustrates the difference between a time-series problem and static FDC summary
  statistics.

## The BYOD workflow

```bash
# unlabeled CSV: scores + threshold decisions only
tsad-forge run --model sub_pca --data my_sensor.csv

# with a label column, the full metric suite is computed
tsad-forge run --model iforest --data my_labeled.csv --seed 0
```

CSV conventions: numeric columns = channels, optional `timestamp` column = ordering,
optional `label` column = 0/1. Missing values are linearly interpolated. The span
before the first anomaly becomes train (preserving the unsupervised assumption).

**Recommended starting combo**: `sub_pca` + `conformal` — simple, fast, with a
false-alarm guarantee. Only then consider deep models (the leaderboard is the
evidence for this ordering).
