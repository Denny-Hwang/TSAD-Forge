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
- **Normalization options** (`normalize_params`): `std_floor` lower-bounds the
  train standard deviation and `clip` truncates z-scores. Channels that are
  *nearly* constant in train (e.g. SMD machine-1-1 channel 33, std 1.5e-5) reach
  |z| ≈ 7×10⁴ in test and make squared reconstruction errors explode to ~10⁸;
  `std_floor: 0.01` is a safe floor for min-max-scaled data. The default (empty)
  keeps the legacy behavior so existing results stay comparable. Every run
  records `diag_test_input_absmax` so the problem is visible after the fact.
- **Score health**: NaN/inf anomaly scores fail the run instead of being
  coerced into a metric — a numerically collapsed model must not get a score.
- **VUS buffer width** (`sliding_window`): legacy fixed `100`, or `"auto"` for a
  per-series period estimated from the autocorrelation (the TSB-AD convention;
  own implementation, numerical parity with TSB-AD's helper is not tested).
  The width actually used is recorded as `diag_vus_window`.
- **Execution device** (`device: auto | cpu | cuda | cuda:N`): resolved before
  hashing and part of the experiment identity for torch models — CPU and GPU
  runs of the same config are different experiments (and some models, e.g.
  `mamba_tsad_faithful`, deliberately behave differently per device). Each run
  JSON also records an environment fingerprint (Python/library versions, CUDA,
  GPU name — no host identity) and `git_dirty`.

## Hyperparameter policy

**All models run with paper/library defaults. No per-dataset tuning is
performed, for any model.** This is a deliberate fairness choice: tuning
budget is a hidden variable that has flipped rankings in prior TSAD
comparisons (see TSB-AD's HP-sensitivity discussion). It also means every
number here is a *default-configuration* result — a tuned deployment may do
better, and models whose defaults were set on similar data have an inherent
advantage. A validation-split HP-selection protocol is on the roadmap.

**Training budget (DL).** `configs/lite_dl.yaml` is a CPU smoke profile
(3–5 epochs, window 32) — its DL numbers understate what these models can do
and must not be read as a generation verdict. `configs/lite_dl_gpu.yaml` gives
every Gen3–5 model the same budget: up to 50 epochs with **early stopping on a
temporal validation split** (the last 20% of *train* — test labels are never
used), learning-rate reduction on plateau, and bf16 autocast on CUDA. It uses
the same entity list as `configs/lite.yaml`, so DL and Gen1–2 results are
comparable entity by entity. The number of epochs actually run is recorded per
run (`diag_epochs_run`).

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
- **Coverage-aware comparison**: models are not all run on the same entities
  (the DL profile covers far fewer). The leaderboard therefore sorts by
  **normalized rank** — each entity's rank rescaled to [0, 1] so entities with
  different numbers of competitors are comparable — and adds a
  **shared-entity** mean over only the entities *every* model ran on. The
  generation chart likewise draws only entities covered by every generation,
  one point per (model, entity).
- Data variants without a channel key (e.g. `synthetic` with different
  `n_dims`) are separate entities, suffixed with their data hash.
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
