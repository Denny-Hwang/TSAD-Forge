"""M0 더미 탐지기 2종.

- dummy: 시드 고정 난수 점수. 파이프라인 스모크 테스트 + ch07 "random score로
  PA-F1 SOTA 만들기" 실험의 기준 모델로 재사용된다.
- zscore: train 통계 기반 z-score. 가장 단순한 실전 baseline이자
  "더미보다 나은 최소 기준선" 역할.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("dummy")
class DummyDetector(BaseDetector):
    """시드 고정 uniform 난수 점수를 내는 무정보 탐지기."""

    generation = "gen0"

    def fit(self, X: np.ndarray) -> DummyDetector:
        self._as_2d(X)  # 형상 검증만
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        rng = np.random.default_rng(self.seed)
        return rng.random(len(X))


@register_model("zscore")
class ZScoreDetector(BaseDetector):
    """train의 채널별 평균/표준편차 기준 |z| 최대값을 점수로 사용."""

    generation = "gen1"

    def fit(self, X: np.ndarray) -> ZScoreDetector:
        X = self._as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)
