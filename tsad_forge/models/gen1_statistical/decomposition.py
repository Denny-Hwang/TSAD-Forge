"""분해 기반: STL 잔차 (Cleveland et al. 1990, statsmodels 사용), POLY 잔차 — 자체 구현."""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


def estimate_period(x: np.ndarray, max_lag: int | None = None, min_period: int = 4) -> int:
    """ACF 최대 피크로 주기 추정. 뚜렷한 주기가 없으면 0 반환."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    max_lag = max_lag or min(n // 3, 1000)
    if n < 3 * min_period or x.std() == 0:
        return 0
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:max_lag]
    acf /= acf[0] + 1e-12
    # 첫 골짜기 이후의 최대 피크
    trough = next((i for i in range(1, len(acf) - 1) if acf[i] < acf[i + 1]), 1)
    if trough + min_period >= len(acf):
        return 0
    peak = trough + int(np.argmax(acf[trough:]))
    return peak if peak >= min_period and acf[peak] > 0.1 else 0


@register_model("stl_residual")
class STLResidualDetector(BaseDetector):
    """STL 분해 잔차의 |z|: 채널별 STL(주기 자동 추정) 후 잔차 표준화, 채널 최대.

    주기가 추정되지 않는 채널은 이동평균 detrend 잔차로 대체한다.
    잔차 표준화 기준(μ, σ)은 train 잔차에서 추정한다.
    """

    generation = "gen1"

    def __init__(self, seed: int = 0, period: int | None = None, **params) -> None:
        super().__init__(seed=seed, period=period, **params)
        self.period = period

    def _residual(self, x: np.ndarray, period: int) -> np.ndarray:
        if period >= 4 and len(x) >= 2 * period + 1:
            from statsmodels.tsa.seasonal import STL

            return STL(x, period=period, robust=False).fit().resid
        # 주기 없음: 중앙 이동평균 detrend
        w = max(min(len(x) // 10, 101), 3)
        kernel = np.ones(w) / w
        trend = np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), kernel, "valid")
        return x - trend

    def fit(self, X: np.ndarray) -> STLResidualDetector:
        X = self._as_2d(X)
        self.periods_ = [
            self.period if self.period is not None else estimate_period(X[:, d])
            for d in range(X.shape[1])
        ]
        res = np.column_stack(
            [self._residual(X[:, d], self.periods_[d]) for d in range(X.shape[1])]
        )
        self.res_mean_ = res.mean(axis=0)
        self.res_std_ = res.std(axis=0)
        self.res_std_[self.res_std_ == 0] = 1.0
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        res = np.column_stack(
            [self._residual(X[:, d], self.periods_[d]) for d in range(X.shape[1])]
        )
        return np.abs((res - self.res_mean_) / self.res_std_).max(axis=1)


@register_model("poly")
class POLYDetector(BaseDetector):
    """POLY: 슬라이딩 윈도우 국소 다항 적합의 one-step 예측 잔차 (Li et al. 계열 단순화).

    윈도우 내 시간축 다항(차수 degree)을 최소제곱 적합해 다음 점을 외삽하고
    |실측 - 외삽|을 점수로 한다. 원 논문 대비: 채널 독립 처리 + 채널 최대 집계.
    """

    generation = "gen1"

    def __init__(self, seed: int = 0, window: int = 20, degree: int = 3, **params) -> None:
        super().__init__(seed=seed, window=window, degree=degree, **params)
        self.window = window
        self.degree = degree

    def fit(self, X: np.ndarray) -> POLYDetector:
        self._as_2d(X)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        T, D = X.shape
        w = min(self.window, max(T - 1, 2))
        deg = min(self.degree, w - 1)
        t_hist = np.arange(w)
        # 설계 행렬과 외삽 벡터를 한 번만 준비 (LSQ의 hat 성분)
        V = np.vander(t_hist, deg + 1, increasing=True)  # [w, deg+1]
        pinv = np.linalg.pinv(V)  # [deg+1, w]
        v_next = np.power(float(w), np.arange(deg + 1))  # t=w에서의 기저
        weights = v_next @ pinv  # [w] — 예측 = weights · 윈도우값
        out = np.zeros(T)
        for d in range(D):
            x = X[:, d]
            hist = np.lib.stride_tricks.sliding_window_view(x[:-1], w)  # [T-w, w]
            pred = hist @ weights
            err = np.abs(x[w:] - pred)
            col = np.empty(T)
            col[w:] = err
            col[:w] = err[0] if len(err) else 0.0
            out = np.maximum(out, col)
        return out
