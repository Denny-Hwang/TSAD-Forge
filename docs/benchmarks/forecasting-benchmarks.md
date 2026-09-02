# Forecasting benchmarks vs. TSAD benchmarks

> "TimesFM-3 ranks #1 on average across GIFT-Eval, fev-bench and TIME, on both
> point and probabilistic metrics."

That claim is real, and it is **not** a claim about anomaly detection. This page
records what those three benchmarks actually measure, which parts of the claim
transfer to TSAD-Forge, and what we added to this repository so the question can
be answered with numbers instead of assumption.

## The three benchmarks

| Benchmark | Scale | Task | Ranking metrics | License |
|---|---|---|---|---|
| [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) (Aksu et al., ICLR 2025) | 23 datasets, 97 configs, ~144k series, 177M points, 7 domains, 10 frequencies | Point + probabilistic forecasting | MASE, CRPS (rank-aggregated) | Apache-2.0 |
| [fev-bench](https://arxiv.org/abs/2509.26468) (Shchur et al., 2025) | 100 tasks, 7 domains, 46 with covariates | Forecasting incl. covariates | SQL, MASE, WQL, WAPE → win rates + skill scores with bootstrap CIs | Apache-2.0 (`fev` library) |
| [TIME](https://arxiv.org/abs/2602.12147) (2026) | 50 fresh datasets, 98 tasks | Strict zero-shot forecasting, built to avoid pretraining leakage | MASE/CRPS family, pattern-level aggregation | see upstream repo |

Three things they have in common, and all three matter here:

1. **The task is forecasting.** Given a context window, predict the next
   *h* steps. There is no notion of "this segment is anomalous".
2. **There are no anomaly labels.** None of the 250-odd tasks carries an
   event-level ground truth of the kind VUS-PR, affiliation-F1 or event-F1
   need.
3. **The metrics reward being right on average** (MASE, WQL, CRPS). TSAD
   metrics reward *separating* a rare event from a normal background —
   a ranking problem over a heavily imbalanced label set.

## Why we did not add them as datasets

The honest answer to "can these benchmarks be added to TSAD-Forge?" is: **not as
benchmarks**. Ingesting GIFT-Eval's 97 configs would give us 177M unlabeled
points and nothing to score a detector against. Manufacturing labels (e.g.
thresholding residuals) would produce exactly the kind of synthetic, detector-
flattering ground truth that [Wu & Keogh (TKDE 2021)](https://arxiv.org/abs/2009.13807)
warn about, and it would silently redefine the benchmark.

The TSAD analogue of GIFT-Eval already exists and is already wired into this
repo: **TSB-AD** (Liu & Paparrizos, NeurIPS 2024), see
[datasets/tsb_ad.md](../datasets/tsb_ad.md). That is where the "does a
foundation model win?" question gets settled for anomaly detection.

## Why a #1 forecaster is not automatically a #1 detector

Forecast-residual detectors score a point by how badly the model predicted it.
This creates a failure mode that better forecasting makes *worse*:

- **The model can forecast the anomaly.** A long-context model that has seen the
  first half of a level shift will happily extrapolate it. Residual → ~0 → the
  event is invisible. A weaker forecaster keeps predicting the old level and
  flags it.
- **Contaminated context.** In TSAD the context window *is* test data. Anomalous
  points enter the context and the model conditions on them.
- **Sharpness cuts both ways.** Probabilistic skill (WQL/CRPS) rewards
  calibrated intervals. Wide, well-calibrated intervals score well on
  forecasting and desensitise interval-normalised anomaly scores.
- **Rank-1 on average ≠ rank-1 per domain.** Aggregate rank hides
  distribution shift; industrial sensor traces (SMD, SWaT, SKAB) look nothing
  like retail or energy series that dominate forecasting corpora.

None of this says TimesFM-3 is bad at TSAD. It says the claim is untested here,
and the repo's job is to test it rather than restate it.

## What we added instead

Everything below is in this repository and reproducible.

### 1. The model, in three variants

| Registered name | Checkpoint | What it isolates |
|---|---|---|
| `timesfm3` | `google/timesfm-3.0-pytorch` | Native **multivariate** joint forecast (variate attention), point residual score |
| `timesfm3_ci` | same weights | **Channel-independent** control — same weights, variate attention off |
| `timesfm3_prob` | same weights | **Probabilistic** score: mean pinball loss over the 9 quantiles (CRPS approximation) |
| `timesfm` | `google/timesfm-2.5-200m-pytorch` | TimesFM 2.5 baseline (Apache-2.0 weights) |

`timesfm3` vs `timesfm3_ci` isolates exactly the headline feature of the 3.0
release. `timesfm3` vs `timesfm3_prob` splits the point/probabilistic axis the
same way the forecasting leaderboards do.

### 2. Forecast-quality metrics recorded alongside detection metrics

Every forecast-based Gen5 run now also records the metrics those three
benchmarks rank on, computed on the same TSAD test split
(`tsad_forge/evaluation/forecast_metrics.py`):

- `fc_mase` — MAE scaled by the train seasonal-naive MAE
- `fc_wql` — weighted quantile loss
- `fc_crps` — quantile-approximated CRPS
- `fc_mae` — raw mean absolute error

They land in `benchmarks/results/*.parquet` next to `vus_pr`, so
"forecast skill vs. detection skill" is a join, not a new experiment. **A low
`fc_*` with a low `vus_pr` is the interesting cell**: it means the model
predicted the anomalies well, which is precisely what a residual detector cannot
afford.

Read them as diagnostics, not as a leaderboard: they are computed on a split
that contains anomalies, so "lower is better" holds for forecasting and carries
no implication for detection.

### 3. A runnable profile

```bash
pip install "tsad-forge[foundation]" "timesfm[torch]" chronos-forecasting momentfm
python benchmarks/run_all.py --profile configs/foundation.yaml
```

The profile pairs TimesFM 3.0/2.5, Chronos and MOMENT with the same entities the
`lite` profile uses, so the Gen1–Gen4 leaderboard rows are directly comparable.
It is deliberately excluded from CI (CLAUDE.md §9: no heavy training in CI) and
needs HuggingFace access for the weights.

## Status of results

**Not yet run in this repository.** The pretrained weights require HuggingFace
access, which is unavailable in the environment where the adapter was written,
so there are no TimesFM rows in `benchmarks/results/` and no TimesFM entries in
the leaderboard. Claiming numbers without running them would violate CLAUDE.md
§9. The adapters are covered by tests against a stubbed backend
(`tests/test_timesfm.py`) that pin the array contracts, scoring modes and
coverage, so the run is a matter of executing the profile on a machine with
model access.

## Licensing

The `timesfm` package code is Apache-2.0 and is used as a **pip dependency
only** — no code is vendored (CLAUDE.md §10.2).

**TimesFM 3.0 pretrained weights are not Apache-2.0.** They are distributed
under `timesfm-non-commercial-license-v1.0` and are restricted to
non-commercial, non-production use. TimesFM weights up to 2.5 remain Apache-2.0.

The `timesfm3`, `timesfm3_ci` and `timesfm3_prob` adapters emit a `UserWarning`
naming that license on `fit()` when the default 3.0 checkpoint is used. For
commercial or production evaluation, use `timesfm` (2.5). This does not affect
TSAD-Forge's own Apache-2.0 license: no weights and no third-party code are
redistributed here. See [THIRD_PARTY_NOTICES.md](https://github.com/Denny-Hwang/TSAD-Forge/blob/main/THIRD_PARTY_NOTICES.md).

## References

- Das et al., *A decoder-only foundation model for time-series forecasting*, ICML 2024. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- TimesFM 3.0 release and license notice — [google-research/timesfm](https://github.com/google-research/timesfm)
- Aksu et al., *GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation*, ICLR 2025. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393)
- Shchur et al., *fev-bench: A Realistic Benchmark for Time Series Forecasting*, 2025. [arXiv:2509.26468](https://arxiv.org/abs/2509.26468)
- *It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks*, 2026. [arXiv:2602.12147](https://arxiv.org/abs/2602.12147)
- Liu & Paparrizos, *TSB-AD*, NeurIPS 2024 — the TSAD counterpart.
- Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed*, TKDE 2021. [arXiv:2009.13807](https://arxiv.org/abs/2009.13807)
