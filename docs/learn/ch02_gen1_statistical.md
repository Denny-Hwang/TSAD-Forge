# ch02 — Gen1 Statistical Methods (1930s–2000s)

> Companion notebook: `notebooks/ch02_gen1_statistical.ipynb` — PCA-T²/SPE on SMD

## Control charts

The first "anomaly detection", born in industrial quality control.

- **CUSUM** (Page 1954): cumulative sums of standardized deviations,
  \(S^+_t = \max(0, S^+_{t-1} + z_t - k)\). Sensitive to small sustained mean shifts;
  the slack \(k\) is half the shift you want to detect.
- **EWMA** (Roberts 1959): \(z_t = \lambda x_t + (1-\lambda) z_{t-1}\).
  Weights recent observations — reacts faster than CUSUM to abrupt changes.

Implementation: `tsad_forge/models/gen1_statistical/control_charts.py`

## Hotelling T² and PCA-T²/SPE

For multivariate normal data \(x \sim \mathcal{N}(\mu, \Sigma)\), the Mahalanobis
distance \(T^2 = (x-\mu)^\top \Sigma^{-1} (x-\mu)\) follows a \(\chi^2\) distribution
with D degrees of freedom.

When D is large, \(\Sigma^{-1}\) becomes unstable, so PCA splits the space:

- **T² (principal subspace)**: normalized squared scores of the top-k components —
  "excessive movement *within* the normal directions of variation"
- **SPE/Q (residual subspace)**: \(\|x - \hat{x}\|^2\) (reconstruction error) —
  "escaping *outside* the normal directions of variation"

The two statistics catch **different kinds of anomalies**. In semiconductor FDC,
T² often maps to process drift and SPE to sensor faults and novel failure modes (ch09).

## Sub-PCA — a baseline you underestimate at your peril

Slicing the series into sliding windows and scoring PCA reconstruction error is
simple — and sits near the top of the TSB-AD leaderboard. It captures temporal
structure through the window while keeping a linear model's stability. Check our
leaderboard and see for yourself.

## STL residuals

Remove trend and seasonality with STL (Season-Trend decomposition using Loess) and
score |z| of the residual. The period is estimated automatically from the dominant
ACF peak (`estimate_period`); aperiodic channels fall back to moving-average detrending.
