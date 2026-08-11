"""관리도 계열 (Page 1954; Roberts 1959) — 자체 구현.

채널별 통계량을 계산하고 채널 최대값을 최종 점수로 사용한다 (fault isolation은
점수 산출 이후의 문제로 보고 여기서는 다루지 않는다).
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("cusum")
class CUSUMDetector(BaseDetector):
    """양방향 CUSUM (Page 1954): S⁺/S⁻ 누적합의 최대.

    k(allowance)는 표준화 편차 기준 슬랙 (기본 0.5σ — 1σ 이동 탐지 최적 관례).
    """

    generation = "gen1"

    def __init__(self, seed: int = 0, k: float = 0.5, **params) -> None:
        super().__init__(seed=seed, k=k, **params)
        self.k = k

    def fit(self, X: np.ndarray) -> CUSUMDetector:
        X = self._as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = (self._as_2d(X) - self.mean_) / self.std_
        T, D = Z.shape
        s_pos = np.zeros(D)
        s_neg = np.zeros(D)
        out = np.empty(T)
        for t in range(T):
            s_pos = np.maximum(0.0, s_pos + Z[t] - self.k)
            s_neg = np.maximum(0.0, s_neg - Z[t] - self.k)
            out[t] = np.maximum(s_pos, s_neg).max()
        return out


@register_model("ewma")
class EWMADetector(BaseDetector):
    """EWMA 관리도 (Roberts 1959): z_t = λx_t + (1-λ)z_{t-1}, 점수 = |z_t - μ| / σ_z."""

    generation = "gen1"

    def __init__(self, seed: int = 0, lam: float = 0.2, **params) -> None:
        super().__init__(seed=seed, lam=lam, **params)
        self.lam = lam

    def fit(self, X: np.ndarray) -> EWMADetector:
        X = self._as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = (self._as_2d(X) - self.mean_) / self.std_
        lam = self.lam
        sigma_z = np.sqrt(lam / (2 - lam))  # 점근 표준편차
        z = np.zeros(Z.shape[1])
        out = np.empty(len(Z))
        for t in range(len(Z)):
            z = lam * Z[t] + (1 - lam) * z
            out[t] = np.abs(z / sigma_z).max()
        return out
