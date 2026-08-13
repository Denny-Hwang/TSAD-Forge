# Research Roadmap & Proposals

What we researched, what we added, and what we recommend adding next — with the
license verdict for each candidate. Everything here follows the repository's rules:
permissive licenses only for vendored code/data, restricted data gets application
guides only.

## Added in this round (verified and shipped)

| Addition | Kind | Why | License verdict |
|---|---|---|---|
| **MGAB** | dataset | chaotic dynamics stress test used by TSB-AD; anomalies invisible to the eye | **CC0 1.0** — fetched from source repo |
| **MBA (MIT-BIH ECG)** | dataset | the classic near-periodic ECG appendix dataset (TranAD et al.) | TranAD repo **BSD-3**; PhysioNet **ODC-BY** |
| **S-H-ESD** (`sesd`) | Gen1 model | Twitter's production anomaly detector (robust STL + ESD statistic) | paper-based own impl (Twitter's R code is GPL-3 — untouched) |
| **Spectral Residual** (`spectral_residual`) | Gen2 model | the algorithm behind Microsoft Azure Anomaly Detector (KDD 2019) | paper-based own impl |
| **HBOS** (`hbos`) | Gen2 model | the fastest widely-used practical baseline (PyOD staple) | paper-based own impl |
| **ForgeEnsemble** (`ensemble_simple`) | proposed baseline | rank-consensus of 4 cheap diverse detectors (sub_pca, sub_knn, iforest, SR); outlier-ensemble theory says diversity beats any single member on average | own method, Apache-2.0 |

Our proposal is deliberately humble: **ForgeEnsemble is registered under Gen0
(baselines)** — it is the reference any published model should beat, not a novelty
claim. Check the leaderboard to see how many generations actually clear it.

## Recommended next — datasets

| Candidate | What it is | License / access | Blocker & plan |
|---|---|---|---|
| **Exathlon** (VLDB 2021) | Spark-cluster traces with injected + labeled root-caused anomalies; the best-annotated *explainable AD* benchmark | repo Apache-2.0; data via external download | large (~GBs); add downloader + loader, run on full profile only |
| **CATS** (Solenix 2023) | 5M-point simulated spacecraft control system, 200 controlled anomalies, contamination-free train | **CC BY 4.0** (Zenodo) | Zenodo host was unreachable from this environment; loader can ship with a manual-placement guide like Yahoo |
| **GHL** (Kaspersky) | gasoil heating loop ICS simulation with attacks | free but registration-gated | application-guide card, loader only |
| **Genesis** (HU Berlin) | pick-and-place demonstrator, port to industrial PLC signals | CC BY-SA 4.0 | share-alike is fine for data use; needs Kaggle/host check |
| **NAB full set** | we ship 5 of 58 streams | data free | extend `download nab --subset all` via repo clone guidance |
| **UCR / TSB-AD full** | already integrated; blocked only by this dev environment's egress policy | free / Apache-2.0 | run `tsad-forge download ucr / tsb_ad_u` on your machine — loaders and lite-profile entries are already wired |

## Recommended next — models

| Candidate | Generation | Why it matters | Plan / license note |
|---|---|---|---|
| **DAMP** (Lu, Keogh et al., KDD 2022) | Gen2 | streaming left-Matrix-Profile; strong and fast on UCR-style data | reference MATLAB has no clear license → reimplement from paper |
| **SAND** (Boniol & Paparrizos, VLDB 2021) | Gen2 | streaming subsequence clustering; TSB-AD top performer | official impl available; verify license, else reimplement |
| **RRCF** (Guha et al., ICML 2016) | Gen2 | the AWS Kinesis production algorithm; true streaming | `rrcf` pip package (MIT) is unmaintained and breaks on modern setuptools (`pkg_resources`) — reimplement (~150 lines) |
| **SR-CNN** (Ren et al., KDD 2019) | Gen3 | learned threshold on SR saliency; the full Azure pipeline | extend our `spectral_residual` with a small conv head |
| **CARLA** (Darban et al., 2024) | Gen4 | contrastive representation TSAD, strong recent results | paper-based reimplementation |
| **PatchTST/PatchAD-style** | Gen4 | patch tokenization is the current strongest TS backbone | compact reimplementation like our TimesNet |
| **SigLLM / LLM-based zero-shot** | Gen5 | LLMs as anomaly detectors (2024-) — expensive but zero-setup | adapter behind `foundation` extra |
| **TranAD faithful** (BSD-3) | Gen4 | we ship a compact reimplementation; a vendored faithful port would quantify our simplifications, MambaTSAD-style | BSD-3 allows vendoring — a `tranad_faithful`/`tranad` pair |

## Recommended next — methodology (practice-driven)

1. **Streaming / online track**: real deployments score point-by-point. Add a
   `score_online()` protocol (fixed memory, one pass) and re-rank models under it —
   RRCF/DAMP/DSPOT are first-class citizens here, batch transformers are not.
2. **Contamination-robustness curve**: sweep `contamination` in the synthetic
   generator (already supported) and report VUS-PR vs train-pollution level per
   model — directly answers the ch09 industrial question.
3. **Threshold-transfer evaluation**: today F1-style metrics use a per-series
   threshold. Practice needs one threshold across many series/machines — evaluate
   with a *global* SPOT/conformal threshold and report the degradation.
4. **Early-detection latency metric**: NAB rewards early detection; add mean
   detection delay (steps from event start to first alarm) as a secondary column.
5. **Cost-normalized leaderboard**: VUS-PR per log-runtime is already visualized;
   promote it to a sortable leaderboard column.

## Not recommended (and why)

- **Yahoo S5 / SWaT / WADI redistribution** — license forbids; application guides ship instead.
- **NAB scoring code, SKAB code, Twitter AnomalyDetection code** — AGPL/GPL; we use
  data only (NAB/SKAB) or reimplement from papers (S-H-ESD).
- **MERLIN / DADA vendoring** — license unclear at audit time; tracked in
  THIRD_PARTY_NOTICES until upstream clarifies.
