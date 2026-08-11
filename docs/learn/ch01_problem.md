# ch01 — Defining the TSAD Problem

> Companion notebook: [`notebooks/ch01_problem.ipynb`](https://github.com/Denny-Hwang/TSAD-Forge/tree/main/notebooks)

## What is time-series anomaly detection?

Given a time series \(x_1, \dots, x_T\) (univariate D=1 or multivariate D>1), find the
regions that "deviate from normal behavior". Every model in TSAD-Forge outputs a
**continuous anomaly score** \(s_t \in \mathbb{R}\); the binary decision is the job of a
separate thresholding module (ch08).

## Anomaly taxonomy

| Type | Definition | Example |
|---|---|---|
| **Point** | an individual value is abnormal | a sensor spike |
| **Contextual** | the value is in range, but abnormal for its context (time of day, phase) | daytime-level traffic at 4 a.m. |
| **Collective** | individual values are normal, but the subsequence pattern is not | waveform distortion, period break |

`tsad_forge/synthetic/injectors.py` injects these types (spike / level_shift / pattern /
frequency / contextual) — the notebook builds them by hand.

## Unsupervised vs normality-based

Most of what the literature calls "unsupervised TSAD" is actually **normality-based**
(semi-supervised): assume the train span is "mostly normal", learn normal behavior, and
score deviations at test time. Truly unsupervised methods (no labels *and* no normality
assumption) work by self-join on the test series itself — e.g. Matrix Profile discords.

## Contamination

If anomalies leak into train (e.g. NAB's probationary span), the normality model learns
them too and detection degrades. Experiment with
`generate_synthetic(contamination=...)`. In industrial settings (ch09) a perfectly
clean train span rarely exists, which makes this axis important.

## Why TSAD is actually hard

1. **Sparse, ambiguous labels** — anomaly boundaries differ between annotators (Wu & Keogh, TKDE 2021).
2. **Evaluation pitfalls** — point adjustment makes even random scores look SOTA (ch07).
3. **Strong simple baselines** — Sub-PCA and IForest frequently beat deep models (TSB-AD, NeurIPS 2024).

These three facts are the design philosophy of this repository (CLAUDE.md §0).
