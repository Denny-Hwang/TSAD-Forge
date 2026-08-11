"""Matrix Profile / discord (Yeh et al., ICDM 2016) — stumpy(BSD-3-Clause) pip 의존성.

test 시계열 자기 자신에 대한 self-join matrix profile을 점수로 사용한다
(discord 발견의 표준 사용법 — train은 사용하지 않으며, 이는 MP 계열의 정의상 특성).
다변량은 채널별 MP의 최대값을 취한다 (mstump의 결합 MP 대신 채널 독립 —
채널별 discord 감도를 유지하기 위한 선택; docstring에 원 방법과의 차이 명시).
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("matrix_profile")
class MatrixProfileDetector(BaseDetector):
    generation = "gen2"

    def __init__(self, seed: int = 0, window: int = 100, **params) -> None:
        super().__init__(seed=seed, window=window, **params)
        self.window = window

    def fit(self, X: np.ndarray) -> MatrixProfileDetector:
        self._as_2d(X)  # MP는 test self-join — train 미사용 (모듈 docstring 참조)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import stumpy

        self._check_fitted()
        X = self._as_2d(X)
        T, D = X.shape
        m = min(self.window, max(T // 4, 4))
        out = np.zeros(T)
        for d in range(D):
            x = np.ascontiguousarray(X[:, d], dtype=np.float64)
            if np.ptp(x) == 0:  # 상수 채널은 MP 정의 불가
                continue
            mp = stumpy.stump(x, m=m)[:, 0].astype(np.float64)
            mp = np.nan_to_num(mp, nan=0.0, posinf=0.0)
            # 윈도우 점수 -> point: 각 지점을 포함하는 윈도우 중 최대 (discord 보존)
            point = np.zeros(T)
            for offset in range(m):
                seg = mp[: T - m + 1]
                point[offset : offset + len(seg)] = np.maximum(
                    point[offset : offset + len(seg)], seg
                )
            out = np.maximum(out, point)
        return out
