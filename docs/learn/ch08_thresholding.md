# ch08 — Thresholding and Decisions

> Companion notebook: `notebooks/ch08_thresholding.ipynb`

## Why thresholding is separated from the model

Turning a continuous score \(s_t\) into a binary decision is an **operational
decision** (false-alarm cost vs miss cost). The same model deserves different
thresholds in different deployments. Hence this repository isolates thresholding in
`evaluation/thresholding` and reports threshold-free metrics (VUS) separately from
threshold-based ones (the F1 family).

## EVT and SPOT/DSPOT (Siffer et al., KDD 2017)

The Pickands–Balkema–de Haan theorem of extreme value theory: exceedances
\(y = x - t\) over a sufficiently high initial threshold \(t\) converge — **regardless
of the underlying distribution** — to a Generalized Pareto Distribution (GPD):

\[ P(y > z) \approx \left(1 + \frac{\gamma z}{\sigma}\right)^{-1/\gamma} \]

Fitting the GPD (Grimshaw MLE) yields the threshold for a target exceedance
probability \(q\) (e.g. 10⁻⁴) **without distributional assumptions**:

\[ z_q = t + \frac{\sigma}{\gamma}\left(\left(\frac{qn}{N_t}\right)^{-\gamma} - 1\right) \]

- **SPOT**: static distribution — `spot_threshold`
- **DSPOT**: SPOT after removing a moving-average drift — for non-stationary score streams

The authors' reference code is GPL-3, so ours is **implemented purely from the paper's
equations** (see the license ledger).

## Split-conformal

Take the \(\lceil (n+1)(1-\alpha) \rceil\)-th order statistic of calibration scores
\(s_1..s_n\) (assumed normal): under exchangeability this guarantees a **finite-sample
false-alarm rate ≤ α**. No distributional assumptions, no asymptotics. This repository
calibrates on train scores (`conformal_threshold`).

## A practical guide

| Situation | Recommendation |
|---|---|
| stable score distribution, conservative ops | conformal (α = target false-alarm rate) |
| extreme-tail anomalies, want theory | SPOT (q = exceedance probability) |
| drifting scores | DSPOT |
| quick prototyping | quantile (q=0.99) |

**Oracle best-F1 is a leaderboard reference, never an operating threshold** —
it peeked at the test labels.
