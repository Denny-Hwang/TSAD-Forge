# SMD (Server Machine Dataset)

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
