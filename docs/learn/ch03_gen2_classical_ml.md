# ch03 — Gen2 Classical Machine Learning (2000–2016)

> Companion notebook: `notebooks/ch03_gen2_classical_ml.ipynb` — UCR discords with stumpy

## Density and boundary methods: LOF, OC-SVM, KNN

Embed the series as **subsequence window vectors** (`embed_windows`) and any generic
outlier detector applies:

- **KNN / Sub-KNN**: distance to the k-th nearest train window. Simple, and very
  strong on univariate data.
- **LOF**: density relative to the local neighborhood — beats KNN when density varies
  across the space.
- **OC-SVM**: learns a minimal boundary around normal data; \(\nu\) controls the
  fraction allowed outside.

## Isolation Forest — "what isolates easily is anomalous"

In trees built from random axes and random splits, **anomalies isolate at shallow
depth**. No density estimation, O(n log n), robust in high dimensions — the most
common practical default. It is stochastic, so this repository evaluates it with
**3 seeds** (`stochastic: true`).

## Matrix Profile — discovering discords

The profile of minimum z-normalized Euclidean distances between subsequences,
\(MP_i = \min_{j} d(w_i, w_j)\). A **discord** (a subsequence whose nearest neighbor
is still far) matches the definition of a collective anomaly exactly.

- One parameter (window length) — close to parameter-free
- Needs no train split (test self-join) — genuinely unsupervised
- Repeated anomalies can hide each other (the *twin freak* problem)

The UCR Anomaly Archive is essentially a discord-discovery benchmark, which is why MP
is so strong there. We use stumpy (BSD-3) as a pip dependency (`matrix_profile`).

## Where is MERLIN?

MERLIN (discord discovery at all lengths) is deferred: the public reference
implementation has no clear license (CLAUDE.md §10-2 — no license, no copying).
Recorded in THIRD_PARTY_NOTICES.md.
