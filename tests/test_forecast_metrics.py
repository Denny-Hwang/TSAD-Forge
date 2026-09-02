"""예측 품질 지표 단위 테스트 (evaluation/forecast_metrics.py)."""

import numpy as np
import pytest

from tsad_forge.evaluation.forecast_metrics import (
    DEFAULT_QUANTILE_LEVELS,
    crps_from_quantiles,
    forecast_diagnostics,
    mase,
    pinball_loss,
    seasonal_naive_scale,
    wql,
)

RNG = np.random.default_rng(0)


def test_seasonal_naive_scale_known_value():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert seasonal_naive_scale(x, season=1) == pytest.approx(1.0)
    assert seasonal_naive_scale(x, season=2) == pytest.approx(2.0)


def test_seasonal_naive_scale_degenerate_is_bounded():
    """상수열/짧은 train에서도 0으로 나누지 않는다."""
    assert seasonal_naive_scale(np.ones((10, 2))) > 0
    assert seasonal_naive_scale(np.ones((1, 2))) > 0


def test_mase_scaling():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([2.0, 3.0, 4.0])  # 상수 오차 1.0
    assert mase(y, yhat, scale=1.0) == pytest.approx(1.0)
    assert mase(y, yhat, scale=2.0) == pytest.approx(0.5)


def test_mase_ignores_unpredicted_steps():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([np.nan, 3.0, 4.0])
    assert mase(y, yhat, scale=1.0) == pytest.approx(1.0)


def test_pinball_is_minimised_at_the_true_quantile():
    """분위 손실은 예측이 실제 분위와 같을 때 최소가 된다."""
    sample = RNG.normal(size=20000)
    levels = DEFAULT_QUANTILE_LEVELS
    truth = np.quantile(sample, levels)
    grid = np.broadcast_to(truth, (len(sample), len(levels)))
    best = pinball_loss(sample, grid).mean()
    worse = pinball_loss(sample, grid + 0.3).mean()
    assert best < worse


def test_crps_equals_twice_mean_pinball():
    y = RNG.normal(size=(7, 3))
    q = np.sort(RNG.normal(size=(7, 3, 9)), axis=-1)
    assert crps_from_quantiles(y, q) == pytest.approx(2.0 * pinball_loss(y, q).mean())


def test_crps_rewards_a_sharper_correct_forecast():
    y = np.zeros((200, 1))
    tight = np.broadcast_to(np.linspace(-0.1, 0.1, 9), (200, 1, 9))
    loose = np.broadcast_to(np.linspace(-3.0, 3.0, 9), (200, 1, 9))
    assert crps_from_quantiles(y, tight) < crps_from_quantiles(y, loose)


def test_wql_is_scale_invariant():
    """WQL은 |y| 합으로 정규화하므로 y와 분위를 같은 배로 키우면 값이 유지된다."""
    y = np.abs(RNG.normal(size=(30, 2))) + 1.0
    q = np.sort(y[..., None] + RNG.normal(scale=0.2, size=(30, 2, 9)), axis=-1)
    assert wql(y, q) == pytest.approx(wql(10 * y, 10 * q), rel=1e-9)


def test_wql_rejects_mismatched_quantile_count():
    with pytest.raises(ValueError, match="len\\(levels\\)"):
        wql(np.zeros((4, 1)), np.zeros((4, 1, 3)))


def test_forecast_diagnostics_keys_and_finiteness():
    train = RNG.normal(size=(100, 2))
    y = RNG.normal(size=(40, 2))
    point = y + RNG.normal(scale=0.1, size=y.shape)
    q = np.sort(point[..., None] + RNG.normal(scale=0.3, size=(40, 2, 9)), axis=-1)
    out = forecast_diagnostics(y, point, q, train=train)
    assert set(out) == {"fc_mase", "fc_mae", "fc_wql", "fc_crps"}
    assert all(np.isfinite(v) for v in out.values())


def test_forecast_diagnostics_point_only():
    train = RNG.normal(size=(50, 1))
    y = RNG.normal(size=(20, 1))
    out = forecast_diagnostics(y, y, None, train=train)
    assert set(out) == {"fc_mase", "fc_mae"}
    assert out["fc_mae"] == pytest.approx(0.0)
