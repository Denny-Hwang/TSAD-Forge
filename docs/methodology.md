# Benchmark methodology

This page states exactly how every number on the leaderboard is produced —
including the policies that could bias results if left implicit.

## Protocol

- **Split**: each dataset's official train/test split; train is assumed
  normal-dominated (unsupervised setting). No test statistics leak into
  preprocessing — z-score normalization uses train mean/std only.
- **Seeds**: deterministic models run once; stochastic (deep) models run 3 seeds
  and all runs are stored individually (seed averages are computed at
  aggregation time, never before).
- **Thresholding** is separated from models: continuous scores are produced by
  the model, decisions by the `thresholding` module (quantile by default;
  SPOT/DSPOT and conformal available). Threshold-free metrics (VUS-PR primary)
  are always reported alongside.
- **Raw scores are stored** (`benchmarks/results/scores/*.npz`) so every metric
  can be recomputed without re-running models.

## Hyperparameter policy

**All models run with paper/library defaults. No per-dataset tuning is
performed, for any model.** This is a deliberate fairness choice: tuning
budget is a hidden variable that has flipped rankings in prior TSAD
comparisons (see TSB-AD's HP-sensitivity discussion). It also means every
number here is a *default-configuration* result — a tuned deployment may do
better, and models whose defaults were set on similar data have an inherent
advantage. A validation-split HP-selection protocol is on the roadmap.

## Measurement

- `runtime_fit_s` / `runtime_score_s`: wall-clock around `fit()` and `score()`
  only, with **no profiler active** inside the timed sections.
  `runtime_s = fit + score`. Threshold calibration is excluded.
- `peak_mem_mb`: peak process-RSS increment over the section baseline, sampled
  by a background thread every 20 ms. Sub-interval spikes can be missed —
  an accepted trade-off for unbiased runtimes.
- `peak_vram_mb`: recorded (in the run's JSON summary) only when the model
  actually ran on CUDA, from `torch.cuda.max_memory_allocated`. CPU-only runs
  have no VRAM number — earlier versions mislabeled host memory under this
  name; that column is gone.

## Aggregation rules

- The unit of aggregation is the **entity** (a dataset channel/machine/series).
  Leaderboard means are entity means of seed means.
- Every mean is shown with an **entity-bootstrap 95% CI** and the number of
  entities it covers. A mean over few entities with a wide CI is weak evidence.
- A **Friedman test** over the complete-coverage entity set gates the ranking
  claim: if it does not reject, adjacent ranks must be read as ties. The
  critical-difference diagram shows which pairs are separable.
- Results are deduplicated by experiment identity
  (model, dataset, channel, seed, data-hash): when a default-config change
  causes a re-run, only the latest result counts — stale runs can never be
  double-counted.
- Missing data (restricted datasets, blocked mirrors) is reported as `no_data`,
  never silently dropped.

## Online track (experimental)

`online_*` models are **prequential**: each point is scored using only the
past, then the state is updated — unlike batch models, which score the test
set after seeing all of it. The two families appear in the same tables, but
this asymmetry favors batch models; read online rows as a harder setting.
`mean_detection_delay` (steps from event start to first alarm; missed events
are charged the full event length) is reported for threshold-based runs.

## Known limitations

- The **lite** profile is a preview subset — a handful of entities per
  dataset, chosen for CI speed, not representativeness. Full-matrix results
  will replace it.
- PA-F1 is computed only behind `--legacy-pa`, with a warning: point
  adjustment inflates scores enough to make random scores look
  state-of-the-art (Kim et al., AAAI 2022).
- Numbers reported in prior papers under PA are **not comparable** with any
  number here.
