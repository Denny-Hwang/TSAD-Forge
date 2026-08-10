"""통일 모델 API (CLAUDE.md §3).

모든 탐지기는 BaseDetector를 상속한다:
- fit(train): 정상 위주 train 구간으로 학습 (통계 모델은 파라미터 추정)
- score(test) -> np.ndarray[T]: 연속 이상 점수 (클수록 이상)

임계값 적용은 모델의 책임이 아니다 — tsad_forge.evaluation.thresholding이 담당한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """세대 공통 탐지기 인터페이스.

    Class attributes:
        generation: "gen1" ... "gen5" (리더보드 그룹핑에 사용)
        supports_multivariate: D>1 입력 지원 여부
    """

    generation: str = "gen0"
    supports_multivariate: bool = True

    def __init__(self, seed: int = 0, **params) -> None:
        self.seed = seed
        self.params = params
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseDetector:
        """[T, D] train 배열로 학습하고 self를 반환한다."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """[T, D] test 배열에 대한 [T] 연속 이상 점수 (클수록 이상)."""

    # --- 공용 헬퍼 ---

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X[:, None] if X.ndim == 1 else X

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}: call fit() before score()")

    def get_config(self) -> dict:
        """재현용 설정 스냅샷 (results JSON에 기록)."""
        return {"model": type(self).__name__, "seed": self.seed, **self.params}
