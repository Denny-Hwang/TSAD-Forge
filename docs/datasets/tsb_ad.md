# TSB-AD-U / TSB-AD-M

- **Source**: TheDatumOrg/TSB-AD (Liu & Paparrizos, NeurIPS 2024)
- **Contents**: 1070 univariate (U) / 200 multivariate (M) series curated from 40
  datasets. Train length is encoded in the filename: `..._tr_<N>_1st_<firstAnomaly>.csv`
- **License**: Apache-2.0
- **Download**: `tsad-forge download tsb_ad_u` / `tsb_ad_m`
  (hosted at https://www.thedatum.org/datasets/ — behind restrictive firewalls,
  download manually and unzip under `data/tsb_ad_u/`)
- **Loader**: `load_tsb_ad(filename="...", variant="u")`

## Reference protocol

This repository adopts the TSB-AD evaluation protocol (official train/test splits,
VUS-PR as primary metric) as its reference standard. The metric implementations in M2
are verified for numerical agreement against the official TSB-AD implementation.
