"""파운데이션 모델 어댑터 (선택 의존성, extras: foundation).

- moment: MOMENT (Goswami et al., ICML 2024) zero-shot 재구성 어댑터 (momentfm)
- chronos: Chronos (Ansari et al., 2024) 예측 잔차 어댑터 (chronos-forecasting)
- timesfm: TimesFM (Das et al., ICML 2024) 예측 잔차 어댑터 (timesfm)

세 라이브러리 모두 pip 의존성으로만 연결한다 (코드 복사 없음):
momentfm(MIT), chronos-forecasting(Apache-2.0), timesfm(Apache-2.0).
미설치 시 fit()에서 설치 안내와 함께 RuntimeError를 낸다.
사전학습 가중치 다운로드에는 HuggingFace 접근이 필요하다.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


def _require(module: str, pip_name: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(
            f"'{module}' 미설치 — `pip install {pip_name}` 또는 "
            "`pip install tsad-forge[foundation]` 후 사용하세요. "
            "(사전학습 가중치 다운로드에 HuggingFace 접근 필요)"
        ) from e


@register_model("moment")
class MOMENTDetector(BaseDetector):
    """MOMENT zero-shot 재구성: 윈도우(길이 512 고정) 재구성 오차를 점수로 사용."""

    generation = "gen5"

    def __init__(self, seed: int = 0, model_name: str = "AutonLab/MOMENT-1-large", **params):
        super().__init__(seed=seed, model_name=model_name, **params)
        self.model_name = model_name

    def fit(self, X: np.ndarray) -> MOMENTDetector:
        momentfm = _require("momentfm", "momentfm")
        self.pipe_ = momentfm.MOMENTPipeline.from_pretrained(
            self.model_name, model_kwargs={"task_name": "reconstruction"}
        )
        self.pipe_.init()
        self._fitted = True  # zero-shot: fit은 가중치 로드만
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._check_fitted()
        X = self._as_2d(X)
        T, D = X.shape
        L = 512  # MOMENT 고정 컨텍스트
        scores = np.zeros(T)
        counts = np.zeros(T)
        with torch.no_grad():
            for s in range(0, T, L // 2):
                seg = X[s : s + L]
                pad = L - len(seg)
                x = np.pad(seg, ((0, pad), (0, 0)), mode="edge")
                batch = torch.from_numpy(x.T[None]).float()  # [1, D, L]
                out = self.pipe_(x_enc=batch).reconstruction[0].T.numpy()[: len(seg)]
                err = ((out - seg) ** 2).mean(axis=1)
                scores[s : s + len(seg)] += err
                counts[s : s + len(seg)] += 1
        return scores / np.maximum(counts, 1)


class _ForecastResidualBase(BaseDetector):
    """예측 잔차 어댑터 공통: 컨텍스트로 다음 h스텝 예측 → |실측-예측| 점수."""

    generation = "gen5"
    context = 256
    horizon = 16

    def _forecast(self, context: np.ndarray) -> np.ndarray:  # [ctx, D] -> [h, D]
        raise NotImplementedError

    def fit(self, X: np.ndarray):
        self._load()
        self._fitted = True
        return self

    def _load(self) -> None:
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        T, _ = X.shape
        scores = np.zeros(T)
        counts = np.zeros(T)
        for s in range(self.context, T, self.horizon):
            ctx = X[max(0, s - self.context) : s]
            pred = self._forecast(ctx)[: min(self.horizon, T - s)]
            err = np.abs(X[s : s + len(pred)] - pred).mean(axis=1)
            scores[s : s + len(pred)] += err
            counts[s : s + len(pred)] += 1
        # 앞부분(컨텍스트 미충족)은 첫 유효 점수로 패딩
        first = scores[counts > 0][0] if (counts > 0).any() else 0.0
        scores[counts == 0] = first
        counts[counts == 0] = 1
        return scores / counts


@register_model("chronos")
class ChronosResidualDetector(_ForecastResidualBase):
    """Chronos-Bolt 예측 중앙값 잔차 (채널 독립)."""

    def __init__(self, seed: int = 0, model_name: str = "amazon/chronos-bolt-small", **params):
        super().__init__(seed=seed, model_name=model_name, **params)
        self.model_name = model_name

    def _load(self) -> None:
        chronos = _require("chronos", "chronos-forecasting")
        self.pipe_ = chronos.BaseChronosPipeline.from_pretrained(self.model_name)

    def _forecast(self, context: np.ndarray) -> np.ndarray:
        import torch

        preds = []
        for d in range(context.shape[1]):
            q, _ = self.pipe_.predict_quantiles(
                torch.from_numpy(context[:, d]).float(),
                prediction_length=self.horizon,
                quantile_levels=[0.5],
            )
            preds.append(q[0, :, 0].numpy())
        return np.column_stack(preds)


@register_model("timesfm")
class TimesFMResidualDetector(_ForecastResidualBase):
    """TimesFM 예측 잔차 (채널 독립)."""

    def __init__(
        self, seed: int = 0, model_name: str = "google/timesfm-2.0-500m-pytorch", **params
    ):
        super().__init__(seed=seed, model_name=model_name, **params)
        self.model_name = model_name

    def _load(self) -> None:
        timesfm = _require("timesfm", "timesfm")
        self.model_ = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(context_len=self.context, horizon_len=self.horizon),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_name),
        )

    def _forecast(self, context: np.ndarray) -> np.ndarray:
        fc, _ = self.model_.forecast(list(context.T), freq=[0] * context.shape[1])
        return np.asarray(fc)[:, : self.horizon].T
