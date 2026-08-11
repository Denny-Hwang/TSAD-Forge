# UCR Anomaly Archive (2021)

- **Source**: Wu & Keogh (TKDE 2021) — https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- **Contents**: 250 univariate series. The filename encodes the metadata:
  `NNN_UCR_Anomaly_<name>_<trainEnd>_<anomStart>_<anomEnd>.txt`
- **License**: free for research
- **Download**: `tsad-forge download ucr` (zip, ~250 MB)
- **Loader**: `load_ucr(series=1)` or `load_ucr(series="001_UCR_Anomaly_...txt")`

## Design intent and caveats

- The archive was built as a critique of flawed benchmarks (trivial anomalies,
  mislabeled data, excessive prior information) — with a **strict one-anomaly-per-series**
  convention.
- Because of that single-anomaly convention, event-level metrics behave almost
  binarily on this archive. Keep that in mind when reading VUS-PR.
- Files with DISTORTED/NOISE prefixes are variants of the same underlying series —
  the 250 series are not fully independent.
