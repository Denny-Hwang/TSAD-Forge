import numpy as np
import pytest

from tsad_forge.evaluation.metrics import (
    compute_metrics,
    event_f1,
    point_adjust,
    range_f1,
    vus,
)


def _perfect_case():
    labels = np.zeros(100, dtype=int)
    labels[40:50] = 1
    scores = labels.astype(float) + np.linspace(0, 0.01, 100)  # 이상 구간이 항상 높음
    return scores, labels


def test_perfect_scores_give_perfect_metrics():
    scores, labels = _perfect_case()
    m = compute_metrics(scores, labels)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["best_f1"] == pytest.approx(1.0)
    assert m["vus_pr"] > 0.9  # buffer 평균이라 1.0은 아님
    assert m["vus_roc"] > 0.9


def test_vus_ranks_informed_above_random():
    rng = np.random.default_rng(0)
    labels = np.zeros(1000, dtype=int)
    labels[300:340] = 1
    labels[700:720] = 1
    random_scores = rng.random(1000)
    informed = random_scores * 0.3
    informed[295:345] += 0.7
    informed[695:725] += 0.7
    _, vus_pr_rand = vus(random_scores, labels, window_max=50)
    _, vus_pr_inf = vus(informed, labels, window_max=50)
    assert vus_pr_inf > vus_pr_rand + 0.3


def test_vus_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        vus(np.random.default_rng(0).random(50), np.zeros(50, dtype=int))


def test_full_suite_present_with_threshold():
    scores, labels = _perfect_case()
    m = compute_metrics(scores, labels, threshold=0.5)
    for key in (
        "auc_roc",
        "auc_pr",
        "vus_roc",
        "vus_pr",
        "best_f1",
        "standard_f1",
        "event_f1",
        "range_f1",
        "affiliation_f1",
    ):
        assert key in m, key
    assert "pa_f1" not in m  # PA는 기본 계산 금지 (CLAUDE.md §10-4)


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
    th = float(np.quantile(scores, 0.99))
    with pytest.warns(UserWarning):
        m = compute_metrics(scores, labels, threshold=th, legacy_pa=True)
    assert m["pa_f1"] > m["standard_f1"] + 0.3
    # threshold-free 주지표는 random을 낮게 평가해야 정상
    assert m["vus_pr"] < 0.5


def test_point_adjust_marks_whole_event():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])
    np.testing.assert_array_equal(point_adjust(pred, labels), [0, 1, 1, 1, 0])


def test_point_adjust_canonical_index0_quirk():
    """canonical adjust_predicts는 index 0을 backfill하지 않는다 (문서화된 재현)."""
    labels = np.array([1, 1, 1, 0, 0])
    pred = np.array([0, 0, 1, 0, 0])
    np.testing.assert_array_equal(point_adjust(pred, labels), [0, 1, 1, 0, 0])


def test_event_f1_full_detection():
    labels = np.array([0, 1, 1, 0, 1, 0])
    pred = labels.copy()
    assert event_f1(pred, labels) == pytest.approx(1.0, abs=1e-6)
    assert event_f1(np.zeros_like(labels), labels) == 0.0


def test_range_f1_partial_overlap_between_0_and_1():
    labels = np.zeros(60, dtype=int)
    labels[20:40] = 1
    pred = np.zeros(60, dtype=int)
    pred[30:50] = 1  # 절반 겹침 + 바깥 오탐
    v = range_f1(pred, labels)
    assert 0.0 < v < 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        compute_metrics(np.zeros(5), np.zeros(6, dtype=int))
