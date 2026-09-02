"""파운데이션 모델 어댑터 (선택 의존성, extras: foundation).

- moment: MOMENT (Goswami et al., ICML 2024) zero-shot 재구성 어댑터 (momentfm)
- chronos: Chronos (Ansari et al., 2024) 예측 잔차 어댑터 (chronos-forecasting)
- TimesFM 어댑터는 timesfm.py 참조 (2.5 / 3.0 버전 분기가 있어 분리)

세 라이브러리 모두 pip 의존성으로만 연결한다 (코드 복사 없음):
momentfm(MIT), chronos-forecasting(Apache-2.0), timesfm(Apache-2.0).
미설치 시 fit()에서 설치 안내와 함께 RuntimeError를 낸다.
사전학습 가중치 다운로드에는 HuggingFace 접근이 필요하다.
"""

from __future__ import annotations

import warnings

import numpy as np

from tsad_forge.evaluation.forecast_metrics import (
    DEFAULT_QUANTILE_LEVELS,
    forecast_diagnostics,
)
from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model

#: 잔차 -> 이상 점수 변환 방식 (§4 임계값과 분리된, 순수 점수 산출 규약)
SCORE_MODES = ("residual", "crps", "interval")

_EPS = 1e-8


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
    """예측 잔차 어댑터 공통 골격.

    프로토콜 (모든 예측 기반 Gen5 어댑터 공통 — 어댑터 간 비교가 공정하도록 고정):

    1. `fit(train)`은 zero-shot이다. 가중치를 로드하고, train 꼬리 `context` 스텝을
       초기 컨텍스트로 보관한다 (test 앞머리가 컨텍스트 부족으로 버려지지 않게).
    2. `score(test)`는 stride=horizon으로 슬라이딩하며 test의 **모든 스텝을 정확히
       한 번씩** 예측한다 (겹침 없음 → 스텝별 가중 편향 없음).
    3. 점수는 `score_mode`가 결정한다:
       - `residual`  |x - 중앙값 예측| 채널 평균 (점 예측만 필요, 기본값)
       - `crps`      분위 예측의 pinball 평균 (확률 예측 활용)
       - `interval`  |x - q50| / (q_hi - q_lo)  — 예측 구간 폭으로 정규화한 이탈도
       기본값을 `residual`로 두는 이유: 점 예측만 내는 어댑터와 동일 규약으로
       리더보드에서 비교되게 하기 위해서다. 확률 모드는 별도 변형으로 벤치마크한다.
    4. `diagnostics_`에 예측 품질 지표(`fc_mase`/`fc_wql`/`fc_crps`)를 남긴다 —
       "예측 순위 1위가 탐지 1위인가"를 같은 실행에서 검증하기 위한 축이다.
    """

    generation = "gen5"
    default_context = 256
    default_horizon = 16
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    #: 분위 예측 지원 여부 (crps/interval 모드 사용 가능 여부)
    supports_quantiles = False
    #: 하위 클래스가 바꾸는 기본 점수 규약 (등록명별 변형용)
    default_score_mode = "residual"

    def __init__(
        self,
        seed: int = 0,
        context: int | None = None,
        horizon: int | None = None,
        score_mode: str | None = None,
        batch_size: int = 8,
        **params,
    ):
        score_mode = score_mode or self.default_score_mode
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}, got '{score_mode}'")
        ctx = int(context if context is not None else self.default_context)
        hor = int(horizon if horizon is not None else self.default_horizon)
        if ctx < 1 or hor < 1:
            raise ValueError(f"context/horizon must be >= 1 (got {ctx}/{hor})")
        super().__init__(
            seed=seed,
            context=ctx,
            horizon=hor,
            score_mode=score_mode,
            batch_size=int(batch_size),
            **params,
        )
        self.context = ctx
        self.horizon = hor
        self.score_mode = score_mode
        self.batch_size = max(1, int(batch_size))
        self.diagnostics_: dict[str, float] = {}

    # --- 하위 클래스 구현 지점 ---

    def _load(self) -> None:
        """가중치 로드 (fit에서 1회)."""
        raise NotImplementedError

    def _forecast(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """[ctx, D] -> (점 예측 [horizon, D], 분위 예측 [horizon, D, Q] 또는 None)."""
        raise NotImplementedError

    def _forecast_many(
        self, contexts: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """배치 예측 훅. 기본은 순차 호출 — 배치 API가 있는 백엔드는 오버라이드한다."""
        return [self._forecast(c) for c in contexts]

    # --- 공통 구현 ---

    def fit(self, X: np.ndarray):
        self._load()
        X = self._as_2d(X)
        self.train_ = X
        self.train_tail_ = X[-self.context :]
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        T, D = X.shape
        tail = getattr(self, "train_tail_", np.empty((0, D)))
        if tail.shape[1] != D:  # 채널 수가 다르면 꼬리를 쓰지 않는다
            tail = np.empty((0, D))
        ext = np.vstack([tail, X]) if len(tail) else X
        off = len(ext) - T  # ext에서 test가 시작하는 위치

        # 예측 시작점: train 꼬리가 있으면 test 첫 스텝부터, 없으면 컨텍스트 확보 후
        start = off if off > 0 else min(self.context, max(T // 4, 1))
        starts = list(range(start, len(ext), self.horizon))
        contexts = [ext[max(0, s - self.context) : s] for s in starts]

        point = np.full((T, D), np.nan)
        quant: np.ndarray | None = None
        for i in range(0, len(contexts), self.batch_size):
            chunk = contexts[i : i + self.batch_size]
            for s, (p, q) in zip(
                starts[i : i + self.batch_size], self._forecast_many(chunk), strict=True
            ):
                h = min(self.horizon, len(ext) - s)
                lo, hi = s - off, s - off + h  # test 좌표계
                point[lo:hi] = np.asarray(p, dtype=np.float64)[:h]
                if q is not None:
                    q = np.asarray(q, dtype=np.float64)
                    if quant is None:
                        quant = np.full((T, D, q.shape[-1]), np.nan)
                    quant[lo:hi] = q[:h]

        self.diagnostics_ = forecast_diagnostics(
            X, point, quant, train=getattr(self, "train_", None), levels=self.quantile_levels
        )
        return self._to_scores(X, point, quant)

    def _to_scores(self, X: np.ndarray, point: np.ndarray, quant: np.ndarray | None) -> np.ndarray:
        """예측 -> 이상 점수. 예측되지 않은 앞머리는 첫 유효 점수로 채운다."""
        mode = self.score_mode
        if mode != "residual" and quant is None:
            raise RuntimeError(
                f"{type(self).__name__}: score_mode='{mode}'는 분위 예측이 필요하지만 "
                "이 백엔드는 점 예측만 제공합니다 (score_mode='residual' 사용)."
            )
        if mode == "residual":
            per_channel = np.abs(X - point)
        elif mode == "crps":
            from tsad_forge.evaluation.forecast_metrics import pinball_loss

            assert quant is not None
            per_channel = pinball_loss(X, quant, self.quantile_levels).mean(axis=-1) * 2.0
        else:  # interval
            assert quant is not None
            median = quant[..., len(self.quantile_levels) // 2]
            width = quant[..., -1] - quant[..., 0]
            per_channel = np.abs(X - median) / (np.abs(width) + _EPS)

        with warnings.catch_warnings():  # 예측되지 않은 앞머리 행은 전부 NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = np.nanmean(per_channel, axis=1)
        valid = np.isfinite(scores)
        if not valid.any():
            raise RuntimeError(f"{type(self).__name__}: 유효한 예측이 없습니다 (test가 너무 짧음)")
        scores[~valid] = scores[valid][0]  # 앞머리 패딩 (train 꼬리가 있으면 발생하지 않음)
        return scores


@register_model("chronos")
class ChronosResidualDetector(_ForecastResidualBase):
    """Chronos-Bolt 예측 잔차 (채널 독립 — Chronos는 단변량 모델)."""

    supports_multivariate = True  # 채널별 독립 예측으로 처리
    supports_quantiles = True

    def __init__(self, seed: int = 0, model_name: str = "amazon/chronos-bolt-small", **params):
        super().__init__(seed=seed, model_name=model_name, **params)
        self.model_name = model_name

    def _load(self) -> None:
        chronos = _require("chronos", "chronos-forecasting")
        self.pipe_ = chronos.BaseChronosPipeline.from_pretrained(self.model_name)

    def _forecast(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        import torch

        levels = list(self.quantile_levels)
        mid = len(levels) // 2
        points, quants = [], []
        for d in range(context.shape[1]):
            q, _ = self.pipe_.predict_quantiles(
                torch.from_numpy(context[:, d]).float(),
                prediction_length=self.horizon,
                quantile_levels=levels,
            )
            q = q[0].detach().cpu().numpy()  # [horizon, Q]
            quants.append(q)
            points.append(q[:, mid])
        return np.column_stack(points), np.stack(quants, axis=1)  # [h, D], [h, D, Q]
