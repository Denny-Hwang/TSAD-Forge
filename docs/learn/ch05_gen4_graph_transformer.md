# ch05 — Gen4 Graphs and Attention (2020–2023)

> Companion notebook: `notebooks/ch05_gen4.ipynb`

## Graphs: modeling sensor relations explicitly — GDN

In multivariate series, "which sensors move together" is central information for
anomaly detection. GDN (Deng & Hooi, AAAI 2021):

1. a **learnable embedding** \(v_i\) per sensor
2. a top-k graph from embedding cosine similarity
3. graph attention aggregates neighbors to **forecast the next value**
4. the robustly normalized maximum of per-sensor deviations is the score —
   which also tells you **which sensor caused it**

## Attention: association discrepancy — Anomaly Transformer

The insight of Anomaly Transformer (Xu et al., ICLR 2022): **anomalous points can only
attend to their local neighborhood** (they fail to associate with the global pattern).

- series-association: the attention distribution learned from data
- prior-association: a distance-based Gaussian (the baseline of local concentration)
- the KL distance between them (association discrepancy) is **small** for anomalies —
  multiplied with reconstruction error to form the score

DCdetector (KDD 2023) implements the same intuition **without reconstruction**, purely
by contrasting two attention views (patch-wise vs in-patch).

## TimesNet: folding periods into 2D

Find the dominant period by FFT, fold the 1D series into a 2D (period × phase) tensor,
and apply 2D convolutions — capturing intraperiod patterns and interperiod evolution
at once.

## The honest question

This generation sits at the center of the benchmark-inflation controversy. Anomaly
Transformer's original numbers were PA-F1 based; re-evaluated without PA the rankings
change dramatically (Kim et al., AAAI 2022; TSB-AD, NeurIPS 2024). Check the actual
VUS-PR rankings on our [leaderboard](../leaderboard/index.md) — the comparison against
Gen1–2 baselines is the point.
