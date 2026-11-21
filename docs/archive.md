# Research archive

Auto-generated from `archive/papers.yaml` (`tsad-forge archive render`).
Every method the project tracks, from first sighting to benchmark. New arXiv
candidates are proposed weekly as GitHub issues by the archive-scan workflow;
a maintainer decides what enters the ledger.

## Benchmarked (10)

| method | year | models (fidelity) | notes |
|---|---|---|---|
| Hotelling T², PCA T²/SPE, subsequence PCA | 1947 | `hotelling_t2` (reference), `pca_t2spe` (reference), `sub_pca` (reference) |  |
| CUSUM / EWMA control charts | 1954 | `cusum` (reference), `ewma` (reference), `online_cusum` (reference), `online_ewma` (reference) |  |
| STL / polynomial / S-H-ESD residual detectors | 1990 | `stl_residual` (library), `poly` (simplified), `sesd` (paper) |  |
| Isolation Forest, LOF, OC-SVM, k-NN, HBOS | 2000 | `iforest` (library), `lof` (library), `ocsvm` (library), `knn` (library), `sub_knn` (reference), `hbos` (reference) |  |
| AE, LSTM-AD, Telemanom, Donut, DAGMM, OmniAnomaly, USAD | 2015 | `ae` (reference), `lstm_ad` (simplified), `lstm_p` (paper), `vae_donut` (simplified), `dagmm` (simplified), `omni_anomaly` (simplified), `usad` (simplified) |  |
| Matrix Profile discords | 2016 | `matrix_profile` (library) |  |
| Spectral Residual (Azure Anomaly Detector) | 2019 | `spectral_residual` (paper) |  |
| GDN, MTAD-GAT, Anomaly Transformer, TranAD, DCdetector, TimesNet | 2020 | `gdn` (simplified), `mtad_gat` (simplified), `anomaly_transformer` (simplified), `tranad` (simplified), `dcdetector` (simplified), `timesnet` (simplified) |  |
| Uniform random score and per-channel |z| (uninformative floors) | 2025 | `dummy` (baseline), `zscore` (baseline) |  |
| ForgeEnsemble rank-consensus baseline | 2025 | `ensemble_simple` (baseline) |  |

## Implemented (no published results yet) (2)

| method | year | models (fidelity) | notes |
|---|---|---|---|
| MambaTSAD — joint selective SSM and detrending (v2, verified against official code) | 2024 | `mamba_tsad_faithful` (upstream_verified), `mamba_tsad_fixed` (upstream_verified) | v1 results retracted; v2 needs a GPU run (configs/mamba_paper_repro.yaml, mamba_ablation.yaml) |
| MOMENT, Chronos, TimesFM 2.5 / 3.0 adapters | 2024 | `moment` (library), `chronos` (library), `timesfm` (library), `timesfm3` (library), `timesfm3_ci` (library), `timesfm3_prob` (library) | needs `[foundation]` extras and HuggingFace weights; configs/foundation.yaml |

## License audit (2)

| method | venue | year | gen | code (license) | notes |
|---|---|---|---|---|---|
| [AnoMamba — reconstruction aligned with TSAD via selective global dependency modeling](https://www.ijcai.org/proceedings/2026/276) | IJCAI 2026 | 2026 | gen5 | [repo](https://github.com/aaaceo890/AnoMamba_ijcai) | follow-up of MambaTSAD by the same authors (linked from the official mamba_tsad README) |
| DADA — general time-series anomaly detector with adaptive bottlenecks and dual adversarial decoders | ICLR 2025 | 2025 | gen5 | — | zero-shot TSAD foundation model; license unclear at last audit (THIRD_PARTY_NOTICES) |

## Candidates (8)

| method | venue | year | gen | code (license) | notes |
|---|---|---|---|---|---|
| CARLA — contrastive representation learning for TSAD |  | 2024 | gen4 | — |  |
| Patch-tokenized TSAD backbones (PatchTST / PatchAD style) |  | 2024 | gen4 | — |  |
| LLM-based zero-shot anomaly detection (SigLLM line) |  | 2024 | gen5 | — | adapter behind the foundation extra; cost-heavy |
| DAMP — streaming left matrix profile discords at scale | KDD 2022 | 2022 | gen2 | — (none found (reference MATLAB)) | reimplement from paper; fits the online track |
| TranAD vendored port (faithful vs our simplified reimplementation) | VLDB 2022 | 2022 | gen4 | [repo](https://github.com/imperial-qore/TranAD) (BSD-3-Clause) | BSD-3 permits vendoring — quantifies our simplifications, MambaTSAD-style |
| SAND — streaming subsequence anomaly detection | VLDB 2021 | 2021 | gen2 | — | strong on TSB-AD; check the official implementation's license |
| SR-CNN — learned head on spectral-residual saliency | KDD 2019 | 2019 | gen3 | — | extend spectral_residual with a small conv head |
| Robust Random Cut Forest | ICML 2016 | 2016 | gen2 | [repo](https://github.com/kLabUM/rrcf) (MIT) | pip package unmaintained (pkg_resources); reimplement (~150 lines) |

## Rejected (1)

| method | venue | year | gen | code (license) | notes |
|---|---|---|---|---|---|
| MERLIN — parameter-free discord discovery | ICDM 2020 | 2020 | gen2 | — | reference implementation license unclear at audit; revisit if upstream clarifies |
