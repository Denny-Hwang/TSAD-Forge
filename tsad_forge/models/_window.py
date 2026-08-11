"""슬라이딩 윈도우 임베딩 유틸 — subsequence 기반 탐지기 공용.

윈도우 점수를 point 점수로 되돌릴 때는 각 윈도우 점수를 윈도우 '끝' 인덱스에
할당(causal)하고, 앞의 (window-1)개 지점은 첫 윈도우 점수로 패딩한다.
"""

from __future__ import annotations

import numpy as np


def embed_windows(X: np.ndarray, window: int) -> np.ndarray:
    """[T, D] -> [T-window+1, window*D] 슬라이딩 윈도우 평탄화."""
    T, D = X.shape
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if T < window:
        raise ValueError(f"series length {T} < window {window}")
    view = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)  # [T-w+1, D, w]
    return view.transpose(0, 2, 1).reshape(T - window + 1, window * D)


def window_scores_to_points(window_scores: np.ndarray, T: int, window: int) -> np.ndarray:
    """[T-window+1] 윈도우 점수 -> [T] point 점수 (끝점 할당 + 앞쪽 패딩)."""
    scores = np.empty(T, dtype=np.float64)
    scores[window - 1 :] = window_scores
    scores[: window - 1] = window_scores[0]
    return scores
