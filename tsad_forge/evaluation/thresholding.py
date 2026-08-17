"""임계값 모듈 (CLAUDE.md §4) — 임계값은 모델이 아닌 이 모듈의 책임.

- quantile: 점수 분포 q-분위수
- spot / dspot: EVT 기반 (Siffer et al., KDD 2017) — 논문 수식 기반 자체 구현.
  (원 저자 참조 구현은 GPL-3이므로 코드를 일절 참조/복사하지 않았다.)
- conformal: split-conformal — 보정(calibration) 점수 분포 기반 유한표본 보장 임계값.

임계값 기반 지표(F1 계열)와 threshold-free 지표(VUS 등)는 분리 보고한다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def quantile_threshold(scores: np.ndarray, q: float = 0.99, **_) -> float:
    """점수 분포의 q-분위수를 임계값으로 사용 (기본 q=0.99)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return float(np.quantile(np.asarray(scores, dtype=np.float64), q))


def _grimshaw_gpd_fit(excesses: np.ndarray, n_candidates: int = 8) -> tuple[float, float]:
    """GPD(gamma, sigma) 적합 — Grimshaw(1993) 방식의 후보 근 탐색 + MLE 선택.

    Siffer et al. (KDD 2017) §3의 절차를 수식으로부터 구현:
    w(x*) = u(x*)v(x*) - 1 = 0 의 근을 수치 탐색하고, 각 근에서
    gamma = v(x*) - 1, sigma = gamma / x* 를 얻어 로그우도 최대인 해를 고른다.
    """
    y = np.asarray(excesses, dtype=np.float64)
    y = y[y > 0]
    if y.size < 4:
        # 표본 부족 시 지수분포 가정 (gamma -> 0)
        return 0.0, float(y.mean()) if y.size else 1.0

    y_min, y_max, y_mean = y.min(), y.max(), y.mean()

    def log_likelihood(gamma: float, sigma: float) -> float:
        if sigma <= 0:
            return -np.inf
        if abs(gamma) < 1e-9:
            return -y.size * np.log(sigma) - y.sum() / sigma
        z = 1 + gamma * y / sigma
        if (z <= 0).any():
            return -np.inf
        return -y.size * np.log(sigma) - (1 + 1 / gamma) * np.log(z).sum()

    # 후보 x* 구간 (논문의 안정 범위): (-1/y_max, 2*(y_mean-y_min)/(y_min^2)] 근방
    eps = 1e-8
    lo = -1 / y_max + eps
    hi = 2 * (y_mean - y_min) / (y_min**2 + eps)
    candidates = np.concatenate(
        [np.linspace(lo, -eps, n_candidates), np.linspace(eps, max(hi, eps * 2), n_candidates)]
    )

    def w(x: float) -> float:
        u = float(np.mean(1.0 / (1 + x * y)))
        v = float(np.mean(np.log1p(x * y))) + 1.0
        return u * v - 1.0

    roots = [0.0]  # x*=0 (지수분포 해) 항상 포함
    for a, b in zip(candidates[:-1], candidates[1:], strict=True):
        if a < 0 < b:
            continue
        wa, wb = w(a), w(b)
        if np.isfinite(wa) and np.isfinite(wb) and wa * wb < 0:
            for _ in range(40):  # 이분법
                m = (a + b) / 2
                if wa * w(m) <= 0:
                    b = m
                else:
                    a, wa = m, w(m)
            roots.append((a + b) / 2)

    best: tuple[float, float, float] = (0.0, float(y_mean), log_likelihood(0.0, y_mean))
    for x_star in roots:
        if x_star == 0.0:
            continue
        gamma = float(np.mean(np.log1p(x_star * y)))
        sigma = gamma / x_star if x_star != 0 else float(y_mean)
        ll = log_likelihood(gamma, sigma)
        if ll > best[2]:
            best = (gamma, sigma, ll)
    return best[0], best[1]


def spot_threshold(
    scores: np.ndarray,
    q: float = 1e-4,
    level: float = 0.98,
    calibration: np.ndarray | None = None,
    **_,
) -> float:
    """SPOT (Siffer et al., KDD 2017): POT/EVT 기반 초과 확률 q의 임계값 z_q.

    calibration(예: train 점수)이 있으면 그것으로, 없으면 scores 자체로 보정한다.
    z_q = t + (sigma/gamma) * ((q*n/N_t)^-gamma - 1)  (gamma=0이면 지수형 극한식)
    """
    cal = np.asarray(calibration if calibration is not None else scores, dtype=np.float64)
    n = cal.size
    t = float(np.quantile(cal, level))  # 초기 임계값 (peaks 추출 기준)
    peaks = cal[cal > t] - t
    n_t = peaks.size
    if n_t < 2:
        return t if n_t == 0 else float(cal.max())
    gamma, sigma = _grimshaw_gpd_fit(peaks)
    r = q * n / n_t
    if abs(gamma) < 1e-9:
        return t - sigma * np.log(r)
    return float(t + (sigma / gamma) * (r**-gamma - 1))


def dspot_threshold(
    scores: np.ndarray,
    q: float = 1e-4,
    level: float = 0.98,
    depth: int = 50,
    calibration: np.ndarray | None = None,
    **_,
) -> float:
    """DSPOT: 이동평균 drift를 제거한 잔차에 SPOT을 적용, 마지막 drift를 더해 복원.

    스트리밍이 아닌 배치 평가용 단순화: 전체 시퀀스의 local drift 제거 후 단일 z_q 산출.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.size <= depth + 2:
        return spot_threshold(x, q=q, level=level, calibration=calibration)
    kernel = np.ones(depth) / depth
    drift = np.convolve(x, kernel, mode="valid")[:-1]  # x[depth:]의 직전-윈도우 평균
    residuals = x[depth:] - drift
    cal_res = residuals if calibration is None else None
    z = spot_threshold(residuals, q=q, level=level, calibration=cal_res)
    return float(z + drift[-1])


def conformal_threshold(
    scores: np.ndarray,
    alpha: float = 0.01,
    calibration: np.ndarray | None = None,
    **_,
) -> float:
    """Split-conformal: 보정 점수의 ceil((n+1)(1-alpha))/n 분위수.

    보정 표본이 정상(교환가능)일 때 오탐률 <= alpha의 유한표본 보장.
    calibration이 없으면 scores 자체를 사용(보장 약화 — 결과 해석 주의).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    cal = np.asarray(calibration if calibration is not None else scores, dtype=np.float64)
    n = cal.size
    rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(cal)[rank - 1])


THRESHOLDERS: dict[str, Callable[..., float]] = {
    "quantile": quantile_threshold,
    "spot": spot_threshold,
    "dspot": dspot_threshold,
    "conformal": conformal_threshold,
}

# train(정상) 점수를 보정에 쓰는 방법들
NEEDS_CALIBRATION = {"spot", "dspot", "conformal"}


def apply_threshold(
    scores: np.ndarray,
    method: str = "quantile",
    train_scores: np.ndarray | None = None,
    **kwargs,
) -> tuple[float, np.ndarray]:
    """임계값을 계산하고 (threshold, 이진 예측)을 반환한다."""
    if method not in THRESHOLDERS:
        raise KeyError(f"unknown thresholding method '{method}'. Available: {sorted(THRESHOLDERS)}")
    if method in NEEDS_CALIBRATION and train_scores is not None:
        kwargs.setdefault("calibration", np.asarray(train_scores))
    th = THRESHOLDERS[method](np.asarray(scores, dtype=np.float64), **kwargs)
    return th, (np.asarray(scores) >= th).astype(int)
