"""예측 품질 지표 (forecasting metrics) — 이상탐지 지표가 아니라 *진단용*이다.

왜 TSAD 저장소에 예측 지표가 있나?
    Gen5 파운데이션 어댑터(TimesFM/Chronos)는 "예측 잔차"를 이상 점수로 쓴다. 따라서
    탐지 성능은 예측 성능에 얹혀 있다. 그런데 파운데이션 모델의 순위표(GIFT-Eval,
    fev-bench, TIME)는 전부 *예측* 지표(MASE/WQL/CRPS)로 매겨진다. "예측 1위 = 탐지
    1위"인지 검증하려면 같은 데이터에서 두 축을 함께 재야 한다. 이 모듈이 그 축을
    제공하고, runner가 결과 parquet에 `fc_*` 지표로 함께 기록한다
    (docs/benchmarks/forecasting-benchmarks.md).

주의: 여기서 계산되는 값은 *이상 구간을 포함한 test 분할*에 대한 예측 오차다.
      예측 오차가 작다는 것은 모델이 이상까지 잘 맞혔다는 뜻일 수 있고, 그 경우
      잔차 기반 탐지는 오히려 나빠진다. 낮을수록 좋은 지표이지만 "낮을수록 탐지가
      잘 된다"는 함의는 없다 — 이 괴리를 보는 것이 이 지표들의 용도다.

정의는 GIFT-Eval / fev-bench에서 쓰는 표준형을 따른다:
- MASE (Hyndman & Koehler, IJF 2006): MAE를 train 계절 나이브 MAE로 정규화
- WQL (weighted quantile loss): 분위 손실 합 / |y| 합 (fev-bench의 WQL과 동일 형태)
- CRPS: 분위 근사 CRPS = 2 x 평균 pinball loss (등간격 분위 격자 가정)
"""

from __future__ import annotations

import numpy as np

# TimesFM/Chronos가 공통으로 내보내는 기본 분위 격자
DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_EPS = 1e-8


def seasonal_naive_scale(train: np.ndarray, season: int = 1) -> float:
    """MASE 분모: train 구간의 계절 나이브 MAE (전 채널 평균).

    train이 계절 주기보다 짧거나 상수열이면 0이 되므로 _EPS로 하한을 둔다.
    """
    train = np.asarray(train, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
    season = max(1, int(season))
    if len(train) <= season:
        return _EPS
    diff = np.abs(train[season:] - train[:-season])
    scale = float(np.mean(diff)) if diff.size else 0.0
    return max(scale, _EPS)


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """Mean Absolute Scaled Error — MAE / seasonal_naive_scale(train)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])) / max(scale, _EPS))


def pinball_loss(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
) -> np.ndarray:
    """분위별 pinball loss. y_true[...], q_pred[..., Q] -> [..., Q]."""
    y_true = np.asarray(y_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    if q_pred.shape[-1] != lv.size:
        raise ValueError(f"q_pred last dim {q_pred.shape[-1]} != len(levels) {lv.size}")
    err = y_true[..., None] - q_pred
    return np.maximum(lv * err, (lv - 1.0) * err)


def wql(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Weighted Quantile Loss: 2 x sum(pinball) / (Q x sum|y|).

    fev-bench/GIFT-Eval의 WQL과 같은 정규화(관측값 절대합 기준)를 쓴다.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(q_pred).all(axis=-1)
    if not mask.any():
        return float("nan")
    losses = pinball_loss(y_true[mask], q_pred[mask], levels)
    denom = max(float(np.sum(np.abs(y_true[mask]))), _EPS)
    return float(2.0 * losses.sum() / (len(levels) * denom))


def crps_from_quantiles(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """등간격 분위 격자 기반 CRPS 근사 = 2 x 평균 pinball loss.

    분위가 등간격일 때 pinball 평균은 CRPS의 리만 근사이고, 계수 2를 곱하면
    연속 CRPS와 같은 스케일이 된다 (Gneiting & Raftery, JASA 2007).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(q_pred).all(axis=-1)
    if not mask.any():
        return float("nan")
    return float(2.0 * pinball_loss(y_true[mask], q_pred[mask], levels).mean())


def forecast_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_pred: np.ndarray | None = None,
    *,
    train: np.ndarray | None = None,
    season: int = 1,
    levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
) -> dict[str, float]:
    """예측 어댑터가 결과에 남기는 진단 지표 묶음 (`fc_` 접두어).

    y_true/y_pred: [T, D] (예측되지 않은 스텝은 NaN 허용)
    q_pred: [T, D, Q] 또는 None (점 예측만 가능한 모델)
    train: MASE 분모(계절 나이브 스케일) 계산용 train 배열
    """
    out: dict[str, float] = {}
    scale = seasonal_naive_scale(train, season) if train is not None else 1.0
    out["fc_mase"] = mase(y_true, y_pred, scale)
    out["fc_mae"] = float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    if q_pred is not None:
        out["fc_wql"] = wql(y_true, q_pred, levels)
        out["fc_crps"] = crps_from_quantiles(y_true, q_pred, levels)
    return {k: v for k, v in out.items() if np.isfinite(v)}
