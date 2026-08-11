"""scikit-learn(BSD-3) 기반 고전 ML 탐지기 — pip 의존성, 코드 복사 없음.

공통 설계: window>1이면 슬라이딩 윈도우 임베딩 후 novelty 점수를 계산하고
point 점수로 되돌린다 (끝점 할당). 단변량 시계열에서는 윈도우 임베딩이 필수적이다.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from tsad_forge.models._window import embed_windows, window_scores_to_points
from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


class _WindowedDetector(BaseDetector):
    """윈도우 임베딩 + train 서브샘플링 공통 로직."""

    generation = "gen2"
    max_train_windows = 20000  # 대형 데이터셋에서의 계산량 상한 (시드 고정 서브샘플)

    def __init__(self, seed: int = 0, window: int = 1, **params) -> None:
        super().__init__(seed=seed, window=window, **params)
        self.window = window

    def _embed(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        w = min(self.window, len(X))
        return embed_windows(X, w)

    def _subsample(self, W: np.ndarray) -> np.ndarray:
        if len(W) <= self.max_train_windows:
            return W
        idx = np.random.default_rng(self.seed).choice(len(W), self.max_train_windows, replace=False)
        return W[idx]

    def _to_points(self, window_scores: np.ndarray, T: int) -> np.ndarray:
        w = T - len(window_scores) + 1
        return window_scores_to_points(window_scores, T, w)


@register_model("lof")
class LOFDetector(_WindowedDetector):
    """LOF (Breunig et al., SIGMOD 2000) novelty 모드: train 밀도 대비 국소 이상도."""

    def __init__(self, seed: int = 0, window: int = 10, n_neighbors: int = 20, **params):
        super().__init__(seed=seed, window=window, n_neighbors=n_neighbors, **params)
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray) -> LOFDetector:
        W = self._subsample(self._embed(X))
        self.model_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(W) - 1), novelty=True
        ).fit(W)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        return self._to_points(-self.model_.score_samples(self._embed(X)), len(X))


@register_model("ocsvm")
class OCSVMDetector(_WindowedDetector):
    """One-Class SVM (Schölkopf et al., 2001), RBF 커널."""

    max_train_windows = 5000  # OCSVM은 O(n^2) — 더 강한 상한

    def __init__(self, seed: int = 0, window: int = 10, nu: float = 0.05, **params):
        super().__init__(seed=seed, window=window, nu=nu, **params)
        self.nu = nu

    def fit(self, X: np.ndarray) -> OCSVMDetector:
        W = self._subsample(self._embed(X))
        self.model_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale").fit(W)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        return self._to_points(-self.model_.decision_function(self._embed(X)), len(X))


@register_model("iforest")
class IForestDetector(_WindowedDetector):
    """Isolation Forest (Liu et al., ICDM 2008). 확률적 — 시드별 결과가 다르다."""

    def __init__(self, seed: int = 0, window: int = 10, n_estimators: int = 100, **params):
        super().__init__(seed=seed, window=window, n_estimators=n_estimators, **params)
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray) -> IForestDetector:
        W = self._subsample(self._embed(X))
        self.model_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(W)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        return self._to_points(-self.model_.score_samples(self._embed(X)), len(X))


class _KNNBase(_WindowedDetector):
    def __init__(self, seed: int = 0, window: int = 1, k: int = 5, **params):
        super().__init__(seed=seed, window=window, k=k, **params)
        self.k = k

    def fit(self, X: np.ndarray):
        W = self._subsample(self._embed(X))
        self.model_ = NearestNeighbors(n_neighbors=min(self.k, len(W))).fit(W)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        dist, _ = self.model_.kneighbors(self._embed(X))
        return self._to_points(dist[:, -1], len(X))  # k번째 이웃 거리


@register_model("knn")
class KNNDetector(_KNNBase):
    """KNN (Ramaswamy et al., 2000): train 내 k번째 최근접 이웃 거리. point 기반(window=1)."""


@register_model("sub_knn")
class SubKNNDetector(_KNNBase):
    """Sub-KNN: 부분수열(window) 임베딩 KNN — 단변량에서 특히 강한 baseline."""

    def __init__(self, seed: int = 0, window: int = 50, k: int = 5, **params):
        super().__init__(seed=seed, window=window, k=k, **params)
