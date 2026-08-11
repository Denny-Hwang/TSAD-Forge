# ch04 — Gen3 Deep Learning: Reconstruction vs Forecasting (2015–2020)

> Companion notebook: `notebooks/ch04_gen3_dl.ipynb`

## Two paradigms

| | Reconstruction | Forecasting |
|---|---|---|
| Training | compress and restore a window | predict the next value(s) from the past |
| Score | reconstruction error | prediction error |
| Representatives | AE, VAE (Donut), USAD | LSTM-AD, LSTM-P (Telemanom) |
| Strengths | collective anomalies | point anomalies, abrupt changes |
| Weaknesses | over-generalization | insensitive to slow drift |

## VAE and reconstruction "probability"

Donut (Xu et al., WWW 2018) scores the **reconstruction probability**
\(-\log p(x|z)\) rather than the reconstruction value. Because it learns the variance,
it can distinguish "a region that is naturally noisy" from "a genuine anomaly" —
the key advantage over a deterministic AE.

## The over-generalization problem

A high-capacity AE **reconstructs the anomalies too**. The countermeasures define
late-Gen3 research:

- **DAGMM**: a GMM over the latent space scores "low normal-density" points via energy
- **USAD**: two adversarially trained decoders sharpen the reconstruction boundary
- **OmniAnomaly**: a stochastic RNN models time-dependent latent distributions

## Implementation notes for this repository

All Gen3 models are reimplemented from the papers (license audit:
THIRD_PARTY_NOTICES.md); simplifications are documented in each module docstring.
For example, our OmniAnomaly is a GRU-VAE without the normalizing flow and the linear
Gaussian SSM prior. Telemanom's dynamic thresholding lives in the `thresholding`
module (SPOT et al.), not in the model — separation of concerns (CLAUDE.md §3).

**Caution**: most original numbers for this generation were reported under PA-F1.
Do not trust them until you have read ch07.
