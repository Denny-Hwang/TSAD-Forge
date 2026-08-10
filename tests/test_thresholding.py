"""임계값 모듈 테스트 (M2): quantile / SPOT / DSPOT / conformal."""

import numpy as np
import pytest

from tsad_forge.evaluation.thresholding import (
    THRESHOLDERS,
    apply_threshold,
    conformal_threshold,
    dspot_threshold,
    quantile_threshold,
    spot_threshold,
)

RNG = np.random.default_rng(0)


def test_all_methods_registered():
    assert set(THRESHOLDERS) == {"quantile", "spot", "dspot", "conformal"}


def test_quantile():
    scores = np.arange(100, dtype=float)
    th, preds = apply_threshold(scores, method="quantile", q=0.9)
    assert th == pytest.approx(89.1)
    assert preds.sum() == 10
    with pytest.raises(ValueError):
        quantile_threshold(scores, q=1.5)
    with pytest.raises(KeyError, match="unknown thresholding"):
        apply_threshold(scores, method="nope")


def test_spot_exceeds_initial_level_and_flags_outliers():
    """SPOT z_q는 초기 분위수보다 높아야 하고, 극단 이상은 넘겨야 한다."""
    cal = RNG.normal(size=5000)
    th = spot_threshold(cal, q=1e-3, level=0.98)
    assert th > np.quantile(cal, 0.98)
    assert th < 10.0  # 비상식적으로 크지 않음
    # 명백한 outlier는 임계값 초과
    assert (np.array([8.0, 9.0]) > th).all()


def test_spot_with_separate_calibration():
    cal = RNG.normal(size=3000)
    test_scores = np.concatenate([RNG.normal(size=500), [12.0]])
    th, preds = apply_threshold(test_scores, method="spot", train_scores=cal, q=1e-3)
    assert preds[-1] == 1
    assert preds[:500].mean() < 0.02  # 정상 구간 오탐 거의 없음


def test_spot_exponential_tail():
    """지수 꼬리(감마=0 근방)에서도 안정적으로 동작."""
    cal = RNG.exponential(scale=1.0, size=5000)
    th = spot_threshold(cal, q=1e-4, level=0.98)
    assert np.isfinite(th) and th > np.quantile(cal, 0.98)


def test_dspot_tracks_drift():
    """상승 추세가 있는 점수에서 DSPOT 임계값은 마지막 drift 수준을 반영한다."""
    t = np.arange(3000, dtype=float)
    scores = t / 300.0 + RNG.normal(scale=0.5, size=3000)  # drift 0 -> 10
    th_d = dspot_threshold(scores, q=1e-3, depth=100)
    th_s = spot_threshold(scores, q=1e-3)
    assert th_d > 8.0  # 마지막 drift(~10) 근방 기준
    assert np.isfinite(th_s)


def test_conformal_guarantee_rate():
    """정상 데이터에서 conformal 오탐률이 대략 alpha 이하."""
    cal = RNG.normal(size=2000)
    test_scores = RNG.normal(size=10000)
    th = conformal_threshold(test_scores, alpha=0.01, calibration=cal)
    fp_rate = (test_scores >= th).mean()
    assert fp_rate <= 0.02  # 유한표본 여유 포함
    with pytest.raises(ValueError):
        conformal_threshold(test_scores, alpha=1.5)


def test_calibration_passed_through_apply_threshold():
    cal = np.zeros(100)
    scores = np.ones(50)
    th, preds = apply_threshold(scores, method="conformal", train_scores=cal, alpha=0.05)
    assert preds.all()  # 보정 분포(0)보다 훨씬 큰 점수는 전부 이상
