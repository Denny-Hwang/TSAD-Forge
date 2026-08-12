# SKAB (Skoltech Anomaly Benchmark)

**What is this data?** A physical water-circulation testbed at Skoltech: a pump, valves and 8 sensors (vibration, pressure, current, flow, temperature). Each of the 35 short experiments injects a real physical fault (valve closing, cavitation, ...). A rare benchmark with *physically induced, precisely timed* faults.

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

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/skab.html" width="100%" height="720" frameborder="0"></iframe>
