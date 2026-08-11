# SKAB (Skoltech Anomaly Benchmark)

- **Source**: https://github.com/waico/SKAB (Katser & Kozitsin, 2020)
- **Contents**: water-circulation testbed with 8 sensors; 35 experiments
  (valve1/valve2/other) plus one anomaly-free run, 1-second sampling
- **License**: the repository is AGPL-3.0 — **code is never copied**. Only the data
  CSVs are used (the authors state the data is free to use; citation required).
- **Download**: `tsad-forge download skab` or `--subset valve1`
- **Loader**: `load_skab(experiment="valve1/0")` — train comes from `anomaly-free.csv`

## Known flaws

- **Short experiments** (~1k steps each) — too little data for most deep models.
- **Separate changepoint column**: this loader uses only the anomaly labels;
  changepoints are ignored.
- Operating conditions differ between the anomaly-free (train) run and the test
  experiments — a covariate shift exists by construction.
