"""스트리밍/온라인 탐지 트랙 (리뷰 P1) — experimental.

실무 TSAD의 상당수는 배치가 아니라 온라인이다: 샘플이 한 점씩 도착하고, 모델은
그 시점까지의 정보만으로 점수를 내야 하며(prequential), 상태를 점진 갱신한다.
기존 벤치마크들이 비워둔 축이라 별도 트랙으로 도입한다.

계약:
- `observe(x_t) -> float`: [D] 샘플 하나를 받아 그 시점의 이상 점수를 반환하고
  내부 상태를 갱신한다 (선평가-후갱신: 점수는 x_t 반영 전 상태 기준).
- BaseDetector 호환: `fit(train)`은 warmup(초기 통계 추정), `score(test)`는
  prequential 루프. score()는 진입 시 post-fit 상태로 자동 복원되므로 반복
  호출해도 결정적이다 (runner의 임계값 보정 score(train) 호출과도 안전).

배치 모델과의 공정 비교 주의: 온라인 모델은 test 구간을 한 번만 통과하며
미래를 보지 않는다. 배치 모델(전체 test에 대한 사후 스코어)과 같은 표에
올릴 때는 이 비대칭을 명시할 것.
"""

from __future__ import annotations

import copy

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


class OnlineDetector(BaseDetector):
    """한 샘플씩 처리하는 탐지기의 공통 루프."""

    generation = "gen1"

    def observe(self, x: np.ndarray) -> float:
        """[D] 샘플 하나 → 이상 점수. 내부 상태를 갱신한다."""
        raise NotImplementedError

    def _warmup(self, X: np.ndarray) -> None:
        """train 구간으로 초기 통계를 추정한다 (점수는 버린다)."""
        for x in X:
            self.observe(x)

    def fit(self, X: np.ndarray) -> OnlineDetector:
        X = self._as_2d(X)
        self._init_state(X.shape[1])
        self._warmup(X)
        self._fitted = True
        # score() 반복 호출이 결정적이도록 post-fit 상태 스냅샷 저장
        self._post_fit_state = {
            k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "_post_fit_state"
        }
        return self

    def reset(self) -> None:
        """상태를 fit 직후로 되돌린다 (스트림 재시작)."""
        snap = self._post_fit_state
        self.__dict__.update({k: copy.deepcopy(v) for k, v in snap.items()})

    def score(self, X: np.ndarray) -> np.ndarray:
        """prequential 스코어링: 각 점수는 해당 샘플 도착 전 상태 기준."""
        self._check_fitted()
        self.reset()
        X = self._as_2d(X)
        return np.array([self.observe(x) for x in X], dtype=np.float64)

    def _init_state(self, n_dims: int) -> None:
        raise NotImplementedError


@register_model("online_ewma")
class OnlineEWMA(OnlineDetector):
    """EWMA 관리도 (온라인) — 지수가중 평균/분산 대비 표준화 잔차.

    score = max_d |x_d - ewma_d| / sqrt(ewvar_d). 갱신은 채점 후(선평가-후갱신).
    """

    def __init__(self, seed: int = 0, alpha: float = 0.05, **params) -> None:
        super().__init__(seed=seed, alpha=alpha, **params)
        self.alpha = alpha

    def _init_state(self, n_dims: int) -> None:
        self._mean = np.zeros(n_dims)
        self._var = np.ones(n_dims)
        self._n = 0

    def observe(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if self._n == 0:
            self._mean = x.copy()
            self._n = 1
            return 0.0
        resid = x - self._mean
        score = float(np.max(np.abs(resid) / np.sqrt(self._var + 1e-12)))
        a = self.alpha
        self._mean = (1 - a) * self._mean + a * x
        self._var = (1 - a) * self._var + a * resid**2
        self._n += 1
        return score


@register_model("online_cusum")
class OnlineCUSUM(OnlineDetector):
    """양측 CUSUM (온라인) — 러닝 Welford 통계로 표준화한 뒤 누적합 관리도.

    s+ = max(0, s+ + z - k), s- = max(0, s- - z - k), score = max_d max(s+, s-).
    k(허용 드리프트)는 관리도 관례상 0.5σ.
    """

    def __init__(self, seed: int = 0, k: float = 0.5, **params) -> None:
        super().__init__(seed=seed, k=k, **params)
        self.k = k

    def _init_state(self, n_dims: int) -> None:
        self._n = 0
        self._mean = np.zeros(n_dims)
        self._m2 = np.zeros(n_dims)
        self._pos = np.zeros(n_dims)
        self._neg = np.zeros(n_dims)

    def observe(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if self._n < 2:
            z = np.zeros_like(x)
        else:
            std = np.sqrt(self._m2 / (self._n - 1) + 1e-12)
            z = (x - self._mean) / std
        self._pos = np.maximum(0.0, self._pos + z - self.k)
        self._neg = np.maximum(0.0, self._neg - z - self.k)
        score = float(np.max(np.maximum(self._pos, self._neg)))
        # Welford 갱신 (채점 후)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)
        return score
