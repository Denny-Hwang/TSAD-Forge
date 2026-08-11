# ch07 — Evaluation Methodology (the most important chapter)

> Companion notebook: `notebooks/ch07_evaluation.ipynb` — **making a random score "SOTA" with PA-F1**

## The mirage built by point adjustment (PA)

The PA protocol (standard practice since Xu et al. 2018): "if any single point inside
an anomaly event is detected, count the **entire** event as detected", then compute F1.

**The problem** (Kim et al., AAAI 2022, arXiv:2109.05257): the longer the event, the
more a single lucky hit flips the whole event to true positive. The consequence —

```python
# reproduce it in this repository (tests/test_metrics.py::test_pa_inflates_random_scores)
scores = rng.random(2000)          # completely random scores
pa_f1  = 0.9+                      # "state of the art"
standard_f1 = 0.05                 # actual performance
vus_pr = 0.3                       # the primary metric is not fooled
```

Many of the "F1 0.95+" claims on SMAP/MSL/SMD leaderboards over the years rest on this
protocol. That is why this repository **excludes PA-F1 from default evaluation** and
provides it only behind `--legacy-pa` with a warning (CLAUDE.md §10-4). The
[metric-divergence chart](../leaderboard/index.md) visualizes it.

## The benchmarks themselves are flawed

Wu & Keogh (TKDE 2021): existing benchmarks suffer from (1) trivial anomalies,
(2) mislabeled data, (3) unrealistic anomaly density, (4) run-to-failure bias.
The UCR Anomaly Archive is a response; TSB-AD (NeurIPS 2024) relabels and curates
40 datasets.

## VUS — the primary metric of this repository

Range-AUC (Paparrizos et al., VLDB 2022): put a sqrt-shaped tolerance buffer at label
boundaries and compute AUC from TPR/FPR that jointly reflect existence (finding the
event) and overlap (point coverage). **VUS (Volume Under the Surface)** averages
Range-AUC over buffer widths 0..L — a threshold-free metric robust to the buffer
choice as well.

- **VUS-PR**: the primary metric. With rare anomalies, PR is more discriminative than ROC.
- Our implementation is written from scratch and verified to match the official
  TSB-AD implementation to 1e-9 (`tests/test_metrics_agreement.py`).

## Companion metrics

- **affiliation-F1** (Huet, KDD 2022): time-distance based — lenient to boundary error
- **range-F1** (Tatbul, NeurIPS 2018): existence+overlap weighted range precision/recall
- **event-F1**: event-level recall × point-level precision
- **standard-F1**: the strictest, point-wise (best-F1 uses an oracle threshold — reference only)

Each metric encodes a different definition of "successful detection".
**Do not trust a single-metric ranking; read the divergence between metrics as
information in itself.**
