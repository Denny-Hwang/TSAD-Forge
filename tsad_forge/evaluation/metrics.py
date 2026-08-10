"""평가 지표 (CLAUDE.md §4).

M0 범위: threshold-free 지표(AUC-ROC, AUC-PR)와 best-F1(참고용), 그리고
legacy 플래그 전용 PA-F1(경고 포함)을 제공한다.

M2에서 추가 예정: VUS-PR/VUS-ROC(주지표), affiliation-F1, range-F1, event-F1
+ TSB-AD 공식 구현과의 수치 일치 검증 테스트. M2 전까지 결과에는 VUS-PR이
포함되지 않으므로 M0-M1 결과로 성능 주장을 하지 않는다.

PA(point adjustment)는 기본값으로 절대 사용하지 않는다 — Kim et al. (AAAI 2022,
arXiv:2109.05257)이 보였듯 random score조차 SOTA로 만드는 부풀림이 있다.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tsad_forge.data.schema import label_events

PA_WARNING = (
    "[legacy-pa] PA-F1은 성능을 부풀립니다 (point adjustment는 random score도 "
    "SOTA로 만듭니다; Kim et al., AAAI 2022). 논문 비교 재현 외 용도로 사용하지 "
    "마세요. 주지표는 VUS-PR입니다."
)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    legacy_pa: bool = False,
) -> dict[str, float]:
    """연속 이상 점수와 이진 라벨로 지표 딕셔너리를 계산한다.

    threshold가 주어지면 임계값 기반 지표(standard-F1 등)도 계산한다.
    legacy_pa=True일 때만 PA-F1을 계산하며, UserWarning으로 경고한다.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")

    out: dict[str, float] = {}
    has_both_classes = 0 < labels.sum() < len(labels)

    if has_both_classes:
        out["auc_roc"] = float(roc_auc_score(labels, scores))
        out["auc_pr"] = float(average_precision_score(labels, scores))
        out["best_f1"] = _best_f1(scores, labels)  # 참고용 (oracle threshold)

    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        if has_both_classes:
            out["standard_f1"] = float(f1_score(labels, pred, zero_division=0))
        if legacy_pa:
            warnings.warn(PA_WARNING, UserWarning, stacklevel=2)
            adjusted = point_adjust(pred, labels)
            out["pa_f1"] = float(f1_score(labels, adjusted, zero_division=0))

    return out


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Point adjustment (Xu et al. 2018 프로토콜): 이상 이벤트 내 한 지점이라도
    탐지되면 이벤트 전체를 탐지한 것으로 간주.

    legacy 비교 재현 전용. 기본 평가에 사용 금지 (CLAUDE.md §4).
    """
    adjusted = np.asarray(pred).astype(int).copy()
    for start, end in label_events(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def _best_f1(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 200) -> float:
    """threshold sweep으로 얻는 최대 F1. oracle 임계값이므로 '참고용'으로만 보고."""
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
    best = 0.0
    for th in np.unique(candidates):
        f1 = f1_score(labels, (scores >= th).astype(int), zero_division=0)
        best = max(best, float(f1))
    return best
