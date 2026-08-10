"""자체 지표 구현 vs TSB-AD 공식 구현(vendored, Apache-2.0) 수치 일치 검증 (M2 DoD).

케이스: 합성 라벨(단일/다중/경계 이벤트) × 점수(random/informed).
VUS·R-AUC는 참조와 동일 이산화를 쓰므로 사실상 일치(1e-9),
threshold 기반 지표는 정의 일치를 1e-9 수준에서 확인한다.
"""

import numpy as np
import pytest

from tsad_forge.evaluation import metrics as own
from tsad_forge.evaluation._vendor.tsb_ad.basic_metrics import basic_metricor, generate_curve


def _cases():
    rng = np.random.default_rng(42)
    cases = []
    # 1) 중앙 단일 이벤트 + informed score
    labels = np.zeros(600, dtype=int)
    labels[250:280] = 1
    score = rng.random(600) * 0.3
    score[248:282] += 0.7
    cases.append(("single-informed", score, labels))
    # 2) 다중 이벤트 + random score
    labels2 = np.zeros(800, dtype=int)
    for s in (50, 300, 600):
        labels2[s : s + 20] = 1
    cases.append(("multi-random", rng.random(800), labels2))
    # 3) 시퀀스 경계에 걸친 이벤트
    labels3 = np.zeros(500, dtype=int)
    labels3[:15] = 1
    labels3[490:] = 1
    cases.append(("boundary", rng.random(500), labels3))
    return cases


@pytest.mark.parametrize("name,score,labels", _cases())
def test_vus_matches_reference(name, score, labels):
    window = 30
    _, _, _, _, _, _, ref_vus_roc, ref_vus_pr = generate_curve(labels, score, window)
    own_roc, own_pr = own.vus(score, labels, window_max=window)
    assert own_roc == pytest.approx(ref_vus_roc, abs=1e-9), name
    assert own_pr == pytest.approx(ref_vus_pr, abs=1e-9), name


@pytest.mark.parametrize("name,score,labels", _cases())
def test_auc_matches_reference(name, score, labels):
    grader = basic_metricor()
    assert own.compute_metrics(score, labels)["auc_roc"] == pytest.approx(
        grader.metric_ROC(labels, score)
    )
    assert own.compute_metrics(score, labels)["auc_pr"] == pytest.approx(
        grader.metric_PR(labels, score)
    )


@pytest.mark.parametrize("name,score,labels", _cases())
def test_threshold_metrics_match_reference(name, score, labels):
    grader = basic_metricor()
    preds = (score >= np.quantile(score, 0.95)).astype(int)

    own_event = own.event_f1(preds, labels)
    ref_event = grader.metric_EventF1PA(labels, score, preds=preds)
    assert own_event == pytest.approx(ref_event, abs=1e-9), name

    own_aff = own.affiliation_f1(preds, labels)
    ref_aff = grader.metric_Affiliation(labels, score, preds=preds)
    assert own_aff == pytest.approx(ref_aff, abs=1e-9), name

    own_rf1 = own.range_f1(preds, labels)
    ref_rf1 = grader.metric_RF1(labels, score, preds=preds)
    assert own_rf1 == pytest.approx(ref_rf1, abs=1e-9), name

    # PA-F1 (legacy) 도 참조와 일치해야 함
    with pytest.warns(UserWarning):
        own_pa = own.compute_metrics(
            score, labels, threshold=float(np.quantile(score, 0.95)), legacy_pa=True
        )["pa_f1"]
    ref_pa = grader.metric_PointF1PA(labels, score, preds=preds)
    assert own_pa == pytest.approx(ref_pa, abs=1e-9), name
