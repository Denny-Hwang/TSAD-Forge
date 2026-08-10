# TSAD-Forge

**A Reproducible Benchmarking and Learning Ecosystem for Time-Series Anomaly Detection**

TSAD-Forge organizes time-series anomaly detection (TSAD) methods by *generation* — from
1930s statistical control charts to 2020s state-space and foundation models — and evaluates
them under a single, honest protocol:

- **VUS-PR is the primary metric** (Paparrizos et al., VLDB 2022). Point-adjusted F1 (PA-F1)
  is never computed by default because it inflates performance (Kim et al., AAAI 2022) — it is
  available only behind a `--legacy-pa` flag, with a warning attached.
- **Simple baselines are respected**: leaderboards honestly show when Sub-PCA, Matrix Profile,
  or IForest beat deep models (cf. TSB-AD, NeurIPS 2024).
- **Everything is reproducible**: fixed seeds, config-hash-tracked runs, raw scores stored so
  metrics can be recomputed.

> 한국어 학습 자료는 [docs/learn/](docs/learn/)에서 제공됩니다.

## Quickstart

```bash
git clone https://github.com/Denny-Hwang/TSAD-Forge.git && cd TSAD-Forge
pip install -e ".[dev]"
tsad-forge run --model dummy --data synthetic
```

This runs a dummy detector on synthetic data end-to-end (generate → fit → score → threshold →
metrics → results parquet/JSON under `benchmarks/results/`).

### Bring Your Own Data (BYOD)

```bash
tsad-forge run --model dummy --data path/to/your.csv
```

CSV/parquet with an optional `timestamp` column and an optional `label` column. Without labels
you get scores + threshold decisions; with labels you get the full metric suite.
(Real models land in milestones M3–M5; see `CLAUDE.md` §8.)

## Repository layout

See [CLAUDE.md](CLAUDE.md) for the full specification (Korean). In short:

| Path | Purpose |
|---|---|
| `tsad_forge/data/` | Dataset registry, loaders (unified `TSADDataset` schema), download CLI |
| `tsad_forge/models/` | Detectors by generation (Gen1 statistical → Gen5 SSM/foundation), unified `BaseDetector` API |
| `tsad_forge/evaluation/` | Metrics (VUS-PR primary), thresholding (SPOT/conformal/quantile), protocol |
| `tsad_forge/synthetic/` | Synthetic anomaly injectors (spike / level shift / pattern / contextual) |
| `tsad_forge/runner/` | Config-based experiment runner with resume support |
| `benchmarks/` | Benchmark matrix runner + results (parquet + JSON) |
| `docs/` | Learning track (ch01–ch10, Korean) + leaderboard, published to GitHub Pages |

## Datasets

Datasets are **never committed** to this repository. `tsad-forge download <dataset>` fetches
public datasets with SHA256 verification; restricted datasets (Yahoo S5, SWaT/WADI) get
application instructions and local-placement loaders only. See `docs/datasets/`.

## License

Apache-2.0 for code in this repository. Third-party code and datasets are tracked in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — only permissive-licensed (MIT/BSD/Apache-2.0)
code is ever vendored.
