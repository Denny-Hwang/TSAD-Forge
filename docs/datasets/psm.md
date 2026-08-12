# PSM (Pooled Server Metrics, eBay)

**What is this data?** Pooled Server Metrics from eBay production infrastructure: 25 aggregated CPU/memory/network channels across many server nodes, minute-level, with 13 weeks of train and 8 weeks of labeled test. A *large-scale IT operations* benchmark.

- **Source**: https://github.com/eBay/RANSynCoders (Abdulaal et al., KDD 2021)
- **Contents**: 25-dimensional server metrics; 130k+ train / 87k+ test steps,
  1-minute sampling
- **License**: repository is Apache-2.0
- **Download**: `tsad-forge download psm`
- **Loader**: `load_psm()`

## Known flaws

- **Missing values in train**: train.csv contains many NaNs — the loader linearly
  interpolates (documented default behavior).
- **Coarse label boundaries**: labels are aggregated per minute, so event boundaries
  can be imprecise.
- Most prior reported numbers are PA-F1 based — not comparable.

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/psm.html" width="100%" height="720" frameborder="0"></iframe>
