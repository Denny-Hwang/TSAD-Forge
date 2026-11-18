"""통일 모델 API (CLAUDE.md §3).

모든 탐지기는 BaseDetector를 상속한다:
- fit(train): 정상 위주 train 구간으로 학습 (통계 모델은 파라미터 추정)
- score(test) -> np.ndarray[T]: 연속 이상 점수 (클수록 이상)

임계값 적용은 모델의 책임이 아니다 — tsad_forge.evaluation.thresholding이 담당한다.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDetector(ABC):
    """세대 공통 탐지기 인터페이스.

    Class attributes:
        generation: "gen1" ... "gen5" (리더보드 그룹핑에 사용)
        supports_multivariate: D>1 입력 지원 여부
        uses_torch: torch로 연산하는 모델 여부. True면 runner가 실행 장치(cpu/cuda)를
            해석해 결과 정체성(config_hash)에 반영한다 (runner/environment.py).
    """

    generation: str = "gen0"
    supports_multivariate: bool = True
    uses_torch: bool = False
    #: 생성자가 `device` 인자를 받는지 (runner가 해석한 장치를 주입할지 결정)
    accepts_device: bool = False
    #: 구현 버전. 같은 이름의 모델을 의미 있게 재구현하면 올린다 — runner가 config에
    #: 기록하고 1이 아니면 config_hash에 넣어, 구버전 결과가 resume/집계에서 신버전과
    #: 같은 실험으로 취급되지 않게 한다.
    model_version: int = 1

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

    # --- 영속화 (리뷰 P1: 학습 비용이 큰 모델의 "한 번 학습, 여러 번 스코어" 지원) ---

    def save(self, path: str | Path) -> Path:
        """학습된 탐지기를 pickle로 저장한다 (torch 모듈 포함 그대로 직렬화).

        주의: pickle은 코드 버전에 결합된다 — 장기 보관/버전 간 이동용 포맷이
        아니라 동일 환경 내 재사용(캐시) 용도다. 신뢰할 수 없는 파일은 절대
        load()로 열지 말 것 (pickle은 임의 코드 실행 가능).
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaseDetector:
        """save()로 저장한 탐지기를 복원한다. 신뢰할 수 있는 파일만 열 것."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, BaseDetector):
            raise TypeError(f"{path} does not contain a BaseDetector (got {type(model).__name__})")
        return model
