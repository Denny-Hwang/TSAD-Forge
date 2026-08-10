"""임계값 모듈 (CLAUDE.md §4) — 임계값은 모델이 아닌 이 모듈의 책임.

M0 범위: quantile. M2에서 SPOT/DSPOT(EVT, Siffer KDD 2017)과
split-conformal이 추가된다.

모든 thresholder는 (train_scores 또는 test_scores 기반으로) 스칼라 임계값을
반환한다. 임계값 기반 지표와 threshold-free 지표는 분리 보고한다.
"""

from __future__ import annotations

import numpy as np


def quantile_threshold(scores: np.ndarray, q: float = 0.99) -> float:
    """점수 분포의 q-분위수를 임계값으로 사용 (기본 q=0.99)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return float(np.quantile(np.asarray(scores, dtype=np.float64), q))


THRESHOLDERS = {
    "quantile": quantile_threshold,
    # "spot": M2 (EVT), "dspot": M2, "conformal": M2
}


def apply_threshold(
    scores: np.ndarray, method: str = "quantile", **kwargs
) -> tuple[float, np.ndarray]:
    """임계값을 계산하고 (threshold, 이진 예측)을 반환한다."""
    if method not in THRESHOLDERS:
        raise KeyError(f"unknown thresholding method '{method}'. Available: {sorted(THRESHOLDERS)}")
    th = THRESHOLDERS[method](scores, **kwargs)
    return th, (np.asarray(scores) >= th).astype(int)
