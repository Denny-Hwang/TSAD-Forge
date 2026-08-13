"""S-H-ESD-style detector (Seasonal Hybrid ESD; Hochenbaum, Vallis & Kejariwal 2017).

Twitter's production anomaly-detection method: STL-style seasonal decomposition
followed by the (generalized) ESD test on residuals, made robust ("hybrid") by
using median/MAD instead of mean/std.

Implementation notes (paper-based — Twitter's R package is GPL-3 and was neither
read nor copied):
- We expose the *continuous* robust test statistic |x - median| / MAD of the
  seasonal residual as the anomaly score (BaseDetector contract); the original
  binary ESD decision is exactly a threshold on this statistic, which our
  thresholding module applies separately (CLAUDE.md §3).
- Seasonality is removed per channel with the same STL/moving-average fallback as
  STLResidualDetector; median/MAD are estimated on train residuals.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.gen1_statistical.decomposition import STLResidualDetector
from tsad_forge.models.registry import register_model


@register_model("sesd")
class SESDDetector(BaseDetector):
    generation = "gen1"

    def __init__(self, seed: int = 0, period: int | None = None, **params) -> None:
        super().__init__(seed=seed, period=period, **params)
        self._stl = STLResidualDetector(seed=seed, period=period)

    def fit(self, X: np.ndarray) -> SESDDetector:
        X = self._as_2d(X)
        self._stl.fit(X)  # 주기 추정 재사용
        res = np.column_stack(
            [self._stl._residual(X[:, d], self._stl.periods_[d]) for d in range(X.shape[1])]
        )
        self.med_ = np.median(res, axis=0)
        mad = np.median(np.abs(res - self.med_), axis=0)
        self.mad_ = np.where(mad == 0, 1.0, mad * 1.4826)  # 정규 일치 상수
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        res = np.column_stack(
            [self._stl._residual(X[:, d], self._stl.periods_[d]) for d in range(X.shape[1])]
        )
        return (np.abs(res - self.med_) / self.mad_).max(axis=1)
