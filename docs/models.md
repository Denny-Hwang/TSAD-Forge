# Model catalog

Auto-generated from `tsad_forge/models/cards.yaml` by `tsad-forge viz`.
Fidelity says how close each implementation is to the cited method:

- **library** — maintained library / official package
- **reference** — classical algorithm implemented from its definition
- **upstream_verified** — behavior checked against the official code
- **paper** — follows the paper (protocol-level omissions only)
- **simplified** — paper-based reimplementation with documented simplifications
- **baseline** — sanity / proposed baseline of this repository

Rows marked **simplified** measure *this repository's* simplified reimplementation, not the original method — read their leaderboard numbers accordingly.

| model | gen | method | citation | fidelity | upstream (license) | deviations |
|---|---|---|---|---|---|---|
| `dummy` | gen0 | Uniform random score (uninformative floor) | TSAD-Forge | baseline | — | — |
| `ensemble_simple` | gen0 | ForgeEnsemble — rank consensus of cheap diverse detectors | TSAD-Forge (proposed baseline) | baseline | — | — |
| `cusum` | gen1 | Two-sided CUSUM control chart | Page, Biometrika 1954 | reference | — | — |
| `ewma` | gen1 | EWMA control chart | Roberts, Technometrics 1959 | reference | — | — |
| `hotelling_t2` | gen1 | Hotelling T² | Hotelling, 1947 | reference | — | ridge-regularized train covariance |
| `online_cusum` | gen1 | Online CUSUM (prequential, Welford statistics) | Page, Biometrika 1954 | reference | — | streaming protocol (score then update); experimental track |
| `online_ewma` | gen1 | Online EWMA (prequential) | Roberts, Technometrics 1959 | reference | — | streaming protocol (score then update); experimental track |
| `pca_t2spe` | gen1 | PCA T² + SPE (Q statistic) | Jackson & Mudholkar, Technometrics 1979 | reference | — | — |
| `poly` | gen1 | Local polynomial one-step residual | TSB-UAD POLY baseline (Paparrizos et al., VLDB 2022) | simplified | — | simplified sliding-window polynomial fit |
| `sesd` | gen1 | S-H-ESD-style seasonal hybrid ESD | Hochenbaum, Vallis & Kejariwal, 2017 ([arXiv](https://arxiv.org/abs/1704.07706)) | paper | [repo](https://github.com/twitter/AnomalyDetection) (GPL-3.0 — not used) | the continuous robust statistic |x - median| / MAD is the score; the ESD decision is left to the thresholding module |
| `stl_residual` | gen1 | STL decomposition residual | Cleveland et al., J. Official Statistics 1990 | library | [repo](https://github.com/statsmodels/statsmodels) (BSD-3-Clause) | period estimated automatically; channel-wise |z| of residuals, max-aggregated |
| `sub_pca` | gen1 | Subsequence PCA reconstruction | TSB-AD baseline (Liu & Paparrizos, NeurIPS 2024) | reference | — | — |
| `zscore` | gen1 | Per-channel |z| maximum | TSAD-Forge | baseline | — | — |
| `hbos` | gen2 | Histogram-Based Outlier Score | Goldstein & Dengel, KI 2012 | reference | — | — |
| `iforest` | gen2 | Isolation Forest | Liu et al., ICDM 2008 | library | [repo](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause) | — |
| `knn` | gen2 | k-NN distance | Ramaswamy et al., SIGMOD 2000 | library | [repo](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause) | — |
| `lof` | gen2 | Local Outlier Factor (novelty mode) | Breunig et al., SIGMOD 2000 | library | [repo](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause) | — |
| `matrix_profile` | gen2 | Matrix Profile discord | Yeh et al., ICDM 2016 | library | [repo](https://github.com/TDAmeritrade/stumpy) (BSD-3-Clause) | — |
| `ocsvm` | gen2 | One-Class SVM (RBF) | Schölkopf et al., Neural Computation 2001 | library | [repo](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause) | training windows capped at 5000 (O(n²) kernel) |
| `spectral_residual` | gen2 | Spectral Residual (Azure Anomaly Detector) | Ren et al., KDD 2019 ([arXiv](https://arxiv.org/abs/1906.03821)) | paper | — | estimated end points and the SR-CNN learned threshold omitted; channels max-aggregated |
| `sub_knn` | gen2 | Subsequence k-NN | TSB-AD baseline (Liu & Paparrizos, NeurIPS 2024) | reference | — | — |
| `ae` | gen3 | Dense autoencoder (window reconstruction) | Sakurada & Yairi, MLSDA workshop 2014 | reference | — | — |
| `dagmm` | gen3 | Deep Autoencoding Gaussian Mixture Model | Zong et al., ICLR 2018 | simplified | — (no official implementation) | flattened windows instead of point-wise inputs<br>covariance regularization simplified to a diagonal clamp |
| `lstm_ad` | gen3 | Stacked-LSTM multi-step forecaster | Malhotra et al., ESANN 2015 | simplified | — (no official implementation) | squared-error score instead of the multivariate Gaussian likelihood of errors (needs a validation set) |
| `lstm_p` | gen3 | Telemanom-style LSTM predictor + EWMA error smoothing | Hundman et al., KDD 2018 ([arXiv](https://arxiv.org/abs/1802.04431)) | paper | [repo](https://github.com/khundman/telemanom) (custom (NASA) — not used) | nonparametric dynamic thresholding is delegated to the thresholding module |
| `omni_anomaly` | gen3 | OmniAnomaly | Su et al., KDD 2019 | simplified | [repo](https://github.com/NetManAIOps/OmniAnomaly) (MIT) | no planar normalizing flow (diagonal Gaussian posterior)<br>standard-normal prior instead of the linear Gaussian state-space prior |
| `usad` | gen3 | USAD | Audibert et al., KDD 2020 | simplified | — (no official public implementation) | alternating losses within an epoch instead of separate per-decoder optimizers |
| `vae_donut` | gen3 | Donut-style window VAE | Xu et al., WWW 2018 ([arXiv](https://arxiv.org/abs/1802.03903)) | simplified | [repo](https://github.com/NetManAIOps/donut) (none (no LICENSE file) — code not used) | no modified ELBO (missing-data injection) and no MCMC imputation<br>generalized from univariate to flattened multivariate windows |
| `anomaly_transformer` | gen4 | Anomaly Transformer | Xu et al., ICLR 2022 ([arXiv](https://arxiv.org/abs/2110.02642)) | simplified | [repo](https://github.com/thuml/Anomaly-Transformer) (MIT) | single loss (reconstruction + association-discrepancy regularizer) instead of the minimax two-phase training<br>one layer, small hidden size |
| `dcdetector` | gen4 | DCdetector (dual-attention contrastive) | Yang et al., KDD 2023 ([arXiv](https://arxiv.org/abs/2306.10347)) | simplified | [repo](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) (unclear) | single patch size instead of multiple<br>channel-averaged representation instead of channel independence<br>simplified stop-gradient |
| `gdn` | gen4 | Graph Deviation Network | Deng & Hooi, AAAI 2021 ([arXiv](https://arxiv.org/abs/2106.06947)) | simplified | [repo](https://github.com/d-ailin/GDN) (MIT) | graph recomputed from embeddings at every forward pass<br>univariate input falls back to a temporal MLP forecaster |
| `mtad_gat` | gen4 | MTAD-GAT | Zhao et al., ICDM 2020 ([arXiv](https://arxiv.org/abs/2009.02040)) | simplified | — (no official implementation) | deterministic decoder instead of the VAE reconstruction path<br>gamma-weighted score combination (default 0.5)<br>1D-conv preprocessing omitted |
| `timesnet` | gen4 | TimesNet (anomaly detection task) | Wu et al., ICLR 2023 ([arXiv](https://arxiv.org/abs/2210.02186)) | simplified | [repo](https://github.com/thuml/Time-Series-Library) (MIT) | one TimesBlock, top-1 period only<br>two 3×3 conv layers instead of the inception block<br>small embedding |
| `tranad` | gen4 | TranAD | Tuli et al., VLDB 2022 ([arXiv](https://arxiv.org/abs/2201.07284)) | simplified | [repo](https://github.com/imperial-qore/TranAD) (BSD-3-Clause) | the two-decoder min-max game simplified to an epoch-weighted combined loss |
| `chronos` | gen5 | Chronos-Bolt forecast-residual adapter | Ansari et al., TMLR 2024 ([arXiv](https://arxiv.org/abs/2403.07815)) | library | [repo](https://github.com/amazon-science/chronos-forecasting) (Apache-2.0) | anomaly score is the forecast residual — a TSAD use of a forecasting model, not a method from the paper |
| `mamba_tsad_faithful` | gen5 | MambaTSAD — official behavior (issues a, c, d reproduced) | Chen et al., IEEE SPL 2024 ([arXiv](https://arxiv.org/abs/2405.19823)) | upstream_verified | [repo](https://github.com/aaaceo890/mamba_tsad) (none (no LICENSE file) — code not copied) | pure-PyTorch parallel scan instead of the official pscan (same recurrence)<br>device-invariant (the official training loop cannot run on CPU) |
| `mamba_tsad_fixed` | gen5 | MambaTSAD — issues a, c, d fixed | Chen et al., IEEE SPL 2024 ([arXiv](https://arxiv.org/abs/2405.19823)) | upstream_verified | [repo](https://github.com/aaaceo890/mamba_tsad) (none (no LICENSE file) — code not copied) | last-step state carry, standard HP objective, per-window AMA period (see mamba_tsad.py) |
| `moment` | gen5 | MOMENT zero-shot reconstruction adapter | Goswami et al., ICML 2024 ([arXiv](https://arxiv.org/abs/2402.03885)) | library | [repo](https://github.com/moment-timeseries-foundation-model/moment) (MIT) | zero-shot (no fine-tuning); fixed 512-step context with half-overlapping segments |
| `timesfm` | gen5 | TimesFM 2.5 forecast-residual adapter | Das et al., ICML 2024 ([arXiv](https://arxiv.org/abs/2310.10688)) | library | [repo](https://github.com/google-research/timesfm) (Apache-2.0 (code and 2.5 weights)) | forecast residual as anomaly score; channel-independent |
| `timesfm3` | gen5 | TimesFM 3.0 multivariate forecast-residual adapter | Das et al., ICML 2024; TimesFM 3.0 (2026) | library | [repo](https://github.com/google-research/timesfm) (code Apache-2.0; 3.0 weights NON-COMMERCIAL) | forecast residual as anomaly score; joint multivariate forecasting |
| `timesfm3_ci` | gen5 | TimesFM 3.0 channel-independent control | Das et al., ICML 2024; TimesFM 3.0 (2026) | library | [repo](https://github.com/google-research/timesfm) (code Apache-2.0; 3.0 weights NON-COMMERCIAL) | forecast residual as anomaly score; channels forecast independently |
| `timesfm3_prob` | gen5 | TimesFM 3.0 probabilistic (quantile) score | Das et al., ICML 2024; TimesFM 3.0 (2026) | library | [repo](https://github.com/google-research/timesfm) (code Apache-2.0; 3.0 weights NON-COMMERCIAL) | quantile-forecast-based score (crps/interval modes) |
