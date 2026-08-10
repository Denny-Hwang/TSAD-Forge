import numpy as np
import pytest

from tsad_forge.evaluation.metrics import compute_metrics, point_adjust
from tsad_forge.evaluation.thresholding import apply_threshold, quantile_threshold


def _perfect_case():
    labels = np.zeros(100, dtype=int)
    labels[40:50] = 1
    scores = labels.astype(float) + np.linspace(0, 0.01, 100)  # 이상 구간이 항상 높음
    return scores, labels


def test_perfect_scores_give_perfect_auc():
    scores, labels = _perfect_case()
    m = compute_metrics(scores, labels)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["best_f1"] == pytest.approx(1.0)


def test_pa_not_computed_by_default():
    scores, labels = _perfect_case()
    m = compute_metrics(scores, labels, threshold=0.5)
    assert "pa_f1" not in m
    assert "standard_f1" in m


def test_legacy_pa_warns_and_computes():
    scores, labels = _perfect_case()
    with pytest.warns(UserWarning, match="PA-F1"):
        m = compute_metrics(scores, labels, threshold=0.5, legacy_pa=True)
    assert "pa_f1" in m


def test_pa_inflates_random_scores():
    """PA 부풀림 재현 (Kim et al. AAAI 2022): random score의 PA-F1 >> standard-F1."""
    rng = np.random.default_rng(0)
    labels = np.zeros(2000, dtype=int)
    for start in range(100, 2000, 400):
        labels[start : start + 80] = 1
    scores = rng.random(2000)
    th = quantile_threshold(scores, q=0.99)
    with pytest.warns(UserWarning):
        m = compute_metrics(scores, labels, threshold=th, legacy_pa=True)
    assert m["pa_f1"] > m["standard_f1"] + 0.3


def test_point_adjust_marks_whole_event():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])
    np.testing.assert_array_equal(point_adjust(pred, labels), [0, 1, 1, 1, 0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        compute_metrics(np.zeros(5), np.zeros(6, dtype=int))


def test_quantile_threshold_and_apply():
    scores = np.arange(100, dtype=float)
    th, preds = apply_threshold(scores, method="quantile", q=0.9)
    assert th == pytest.approx(89.1)
    assert preds.sum() == 10
    with pytest.raises(KeyError, match="unknown thresholding"):
        apply_threshold(scores, method="nope")
    with pytest.raises(ValueError):
        quantile_threshold(scores, q=1.5)
