# PSM (Pooled Server Metrics, eBay)

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
