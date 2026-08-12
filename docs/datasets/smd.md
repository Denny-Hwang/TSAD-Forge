# SMD (Server Machine Dataset)

**What is this data?** Five weeks of server-machine telemetry from a large internet company: 38 channels per machine (CPU load, network, memory, disk I/O, ...) sampled every minute. Anomalies are real operational incidents annotated by domain experts. The canonical benchmark for *multivariate server monitoring*.

- **Source**: OmniAnomaly (Su et al., KDD 2019) — https://github.com/NetManAIOps/OmniAnomaly
- **Contents**: 28 machines (machine-1-1 … 3-11), 38 dimensions, ~25–30k steps each for
  train and test, 1-minute sampling
- **License**: repository is MIT. Only the data files are used.
- **Download**: `tsad-forge download smd` (all machines) or `--subset machine-1-1,machine-2-1`
- **Loader**: `load_smd(machine="machine-1-1")`

## Known flaws

- **Event-length variance**: anomaly events range from a few steps to thousands of steps
  depending on the machine — reporting only machine averages is misleading. This
  repository stores per-machine results separately.
- **Near-constant channels**: some channels are almost constant — zero-variance handling
  is required during normalization (the loader handles this).
- **Train cleanliness**: the claim that the train split is anomaly-free rests on the
  original authors' assertion.

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/smd.html" width="100%" height="720" frameborder="0"></iframe>
