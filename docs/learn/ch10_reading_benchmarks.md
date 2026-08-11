# ch10 — How to Read the Benchmark

> Companion notebook: `notebooks/ch10_reading_benchmarks.ipynb`

## Reading order

1. **Start with the [generation-evolution chart](../leaderboard/index.md)**: the answer
   to "does newer mean better?" is mostly **no**. Look at how much the distributions
   overlap.
2. **The critical-difference diagram**: if two models' mean ranks differ by less than
   the CD, they are statistically indistinguishable — "first place" is meaningless
   inside that band.
3. **The performance-vs-cost scatter**: would you pay 100× the runtime for +0.02
   VUS-PR? That is the deployment question.
4. **The metric-divergence chart**: models far above the diagonal are models that
   merely *graze* events.

## The honest limitations of these results

- The lite profile is a subset — full-run rankings may differ.
- DL models use small lite configs (reduced epochs/hidden sizes) — potentially unfair
  to them versus the original papers; conversely, nothing was tuned in their favor
  against Gen1–2 either.
- Synthetic-data results depend on the injector design — read them separately from
  real-data results.
- Numbers from PA-F1-based prior papers are **not comparable** with this leaderboard (ch07).

## Reproduction commands

Each leaderboard row's `config_hash` resolves to the full configuration in the results
JSON (`benchmarks/results/*.json`). At the same commit:

```bash
python benchmarks/run_all.py --profile configs/lite.yaml      # Gen0-2
python benchmarks/run_all.py --profile configs/lite_dl.yaml   # Gen3-4
python benchmarks/run_all.py --profile configs/mamba_compare.yaml  # Gen5 comparison
tsad-forge viz                                                 # regenerate charts + leaderboard
```

## Bibliography (complete)

- Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed...*, TKDE 2021 (arXiv:2009.13807)
- Kim et al., *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI 2022 (arXiv:2109.05257)
- Paparrizos et al., *Volume Under the Surface (VUS)*, VLDB 2022
- Liu & Paparrizos, *The Elephant in the Room (TSB-AD)*, NeurIPS 2024
- Sarfraz et al., *Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?*, ICML 2024
- Tatbul et al., *Precision and Recall for Time Series*, NeurIPS 2018
- Huet et al., *Local Evaluation of Time Series Anomaly Detection Algorithms*, KDD 2022
- Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*, KDD 2017
- Hundman et al. (Telemanom), KDD 2018 · Su et al. (OmniAnomaly), KDD 2019 ·
  Audibert et al. (USAD), KDD 2020 · Deng & Hooi (GDN), AAAI 2021 ·
  Xu et al. (Anomaly Transformer), ICLR 2022 · Tuli et al. (TranAD), VLDB 2022 ·
  Yang et al. (DCdetector), KDD 2023 · Wu et al. (TimesNet), ICLR 2023 ·
  Chen et al. (MambaTSAD), IEEE SPL 2024 · Goswami et al. (MOMENT), ICML 2024
