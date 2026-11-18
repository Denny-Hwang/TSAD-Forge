# NASA SMAP / MSL

**What is this data?** Real spacecraft telemetry from two NASA missions — the SMAP satellite (soil moisture) and the MSL Curiosity rover. Each channel pairs one telemetry value with 24 one-hot command flags; anomalies are incidents from NASA's ISA reports. The canonical *aerospace telemetry* benchmark.

- **Source**: NASA telemanom (Hundman et al., KDD 2018) — https://github.com/khundman/telemanom
- **Contents**: 55 SMAP channels and 27 MSL channels; per-channel train/test npy arrays
  (25 dims: 1 telemetry value + 24 command features)
- **License**: NASA open data. The telemanom code itself is not used (only the label CSV).
- **Download**: `tsad-forge download smap` — S3 (`telemanom/data.zip`) + label CSV.
  If S3 is blocked in your environment, download the HuggingFace mirror
  `appleparan/telemanom` manually and place files under
  `data/smap_msl/train/<chan>.npy` and `data/smap_msl/test/<chan>.npy`.
  As of 2026-11 the S3 URL returns HTTP 403. The mirror (BSD-3-Clause) keeps the arrays
  under `data/data/train/` and `data/data/test/` (82 channels each), for example
  `https://huggingface.co/datasets/appleparan/telemanom/resolve/main/data/data/train/P-1.npy`.
  After placing them, re-run `tsad-forge download smap` to fetch the label CSV and write
  `MANIFEST.json`. No pinned checksums exist for SMAP/MSL yet, so the manifest is a
  record of what you received, not a verification against known-good values.
- **Loader**: `load_smap(channel="P-1")`, `load_msl(channel="M-6")`

## Known flaws (stated openly)

- **Quasi-binary channels**: 24 of the 25 dimensions are one-hot-like command features.
  Almost all information lives in the single telemetry dimension — the benefit of
  multivariate models can be overestimated here.
- **Sparse labels**: 1–3 anomaly events per channel, so event-level metrics have
  high variance.
- **History of PA inflation**: most prior state-of-the-art claims on this dataset were
  PA-F1 based and are not comparable with the VUS-PR results in this repository
  (Kim et al., AAAI 2022).