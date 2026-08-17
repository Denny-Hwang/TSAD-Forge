"""평가 지표 (CLAUDE.md §4) — 전부 자체 구현, TSB-AD 공식 구현과 수치 일치 테스트로 검증.

주지표: VUS-PR (Paparrizos et al., VLDB 2022).
보조: VUS-ROC, R-AUC-PR/ROC(단일 윈도우), AUC-PR/ROC, affiliation-F1, range-F1,
event-F1, standard-F1. PA-F1은 `legacy_pa=True`일 때만 계산되며 경고가 붙는다
(Kim et al., AAAI 2022: PA는 random score도 SOTA로 만든다).

구현 노트:
- VUS/R-AUC의 이산화(250개 임계값, sqrt 경사 라벨 확장, existence 가중 recall)는
  비교 가능성을 위해 TSB-AD/VUS 공식 구현의 수식을 의도적으로 그대로 따른다.
  수치 일치는 tests/test_metrics_agreement.py에서 vendored 참조로 검증한다.
- affiliation-F1은 vendored TSB-AD affiliation 패키지(Apache-2.0)를 사용한다.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score

from tsad_forge.data.schema import label_events

PA_WARNING = (
    "[legacy-pa] PA-F1은 성능을 부풀립니다 (point adjustment는 random score도 "
    "SOTA로 만듭니다; Kim et al., AAAI 2022). 논문 비교 재현 외 용도로 사용하지 "
    "마세요. 주지표는 VUS-PR입니다."
)

DEFAULT_SLIDING_WINDOW = 100  # TSB-AD get_metrics 기본값과 동일
_EPS = 1e-15


# --- 라벨 확장 (VUS의 sqrt 경사 버퍼) ---


def _extend_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """이상 이벤트 양끝에 폭 window//2의 sqrt 경사 버퍼를 붙인 연속 라벨."""
    ext = labels.astype(float).copy()
    T = len(ext)
    if window <= 0:
        return ext
    for s, e_excl in label_events(labels):
        e = e_excl - 1  # 참조 구현은 양끝 포함 인덱스 사용
        x1 = np.arange(e + 1, min(e + window // 2 + 1, T))
        ext[x1] += np.sqrt(1 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        ext[x2] += np.sqrt(1 - (s - x2) / window)
    return np.minimum(ext, 1.0)


def _merged_events(labels: np.ndarray, window: int) -> list[tuple[int, int]]:
    """이벤트를 window//2만큼 확장 후 겹침 병합한 [start, end(포함)] 목록 (참조 new_sequence)."""
    seq = [(s, e - 1) for s, e in label_events(labels)]
    if not seq:
        return []
    T = len(labels)
    merged = []
    a = max(seq[0][0] - window // 2, 0)
    for i in range(len(seq) - 1):
        if seq[i][1] + window // 2 < seq[i + 1][0] - window // 2:
            merged.append((a, seq[i][1] + window // 2))
            a = seq[i + 1][0] - window // 2
    merged.append((a, min(seq[-1][1] + window // 2, T - 1)))
    return merged


def range_auc(
    scores: np.ndarray, labels: np.ndarray, window: int, n_thresholds: int = 250
) -> tuple[float, float]:
    """단일 버퍼 폭의 (R-AUC-ROC, R-AUC-PR). VUS는 이를 window 축으로 평균한 것."""
    vus_roc, vus_pr = vus(
        scores, labels, window_max=window, n_thresholds=n_thresholds, _single=True
    )
    return vus_roc, vus_pr


def vus(
    scores: np.ndarray,
    labels: np.ndarray,
    window_max: int = DEFAULT_SLIDING_WINDOW,
    n_thresholds: int = 250,
    _single: bool = False,
) -> tuple[float, float]:
    """(VUS-ROC, VUS-PR): buffer width 0..window_max에 대한 R-AUC의 평균.

    _single=True면 window_max 하나의 R-AUC만 계산 (range_auc용).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    T = len(scores)
    P = labels.sum()
    if P == 0 or P == T:
        raise ValueError("labels must contain both classes for VUS")

    seq = [(s, e - 1) for s, e in label_events(labels)]
    big_events = _merged_events(labels, window_max)  # TP 집계 구간 (참조와 동일: 최대 폭 기준)
    score_sorted = -np.sort(-scores)
    thr_idx = np.linspace(0, T - 1, n_thresholds).astype(int)
    preds = scores[None, :] >= score_sorted[thr_idx][:, None]  # [n_thr, T]
    n_pred = preds.sum(axis=1)

    # 성능 최적화 (수치 불변): TP/existence 집계는 big_events 합집합 안에서만 일어나므로
    # 해당 구간만 이어붙인 부분 배열로 계산한다. 전역 [s,e]를 부분 좌표로 매핑.
    idx = (
        np.concatenate([np.arange(s, e + 1) for s, e in big_events]) if big_events else np.arange(0)
    )
    offsets: list[tuple[int, int, int]] = []  # (global_start, global_end, concat_offset)
    off = 0
    for s, e in big_events:
        offsets.append((s, e, off))
        off += e - s + 1

    def _to_sub(s: int, e: int) -> tuple[int, int]:
        for gs, ge, o in offsets:
            if gs <= s <= ge:
                return o + s - gs, o + e - gs
        raise RuntimeError("interval outside merged events")  # pragma: no cover

    preds_sub = preds[:, idx]  # [n_thr, m]
    seq_sub = [_to_sub(s, e) for s, e in seq]

    windows = [window_max] if _single else list(range(window_max + 1))
    aucs, aps = [], []
    for w in windows:
        ext_sub = _extend_labels(labels, w)[idx]
        events_w = _merged_events(labels, w)
        events_w_sub = [_to_sub(s, e) for s, e in events_w]
        tpr_list = np.zeros(n_thresholds + 2)
        fpr_list = np.zeros(n_thresholds + 2)
        prec_list = np.ones(n_thresholds + 1)

        for j in range(n_thresholds):
            pred = preds_sub[j]
            lab = ext_sub.copy()
            existence = 0
            for s, e in events_w_sub:
                lab[s : e + 1] = ext_sub[s : e + 1] * pred[s : e + 1]
                if pred[s : e + 1].any():
                    existence += 1
            for s, e in seq_sub:
                lab[s : e + 1] = 1

            tp = float(np.dot(lab, pred))
            n_labels = float(lab.sum())

            fp = n_pred[j] - tp
            p_new = (P + n_labels) / 2
            recall = min(tp / p_new, 1.0)
            tpr_list[j + 1] = recall * (existence / len(events_w))
            fpr_list[j + 1] = fp / (T - p_new)
            prec_list[j + 1] = tp / n_pred[j] if n_pred[j] > 0 else 0.0

        tpr_list[-1] = 1.0
        fpr_list[-1] = 1.0

        width = fpr_list[1:] - fpr_list[:-1]
        height = (tpr_list[1:] + tpr_list[:-1]) / 2
        aucs.append(float(np.dot(width, height)))

        width_pr = tpr_list[1:-1] - tpr_list[:-2]
        aps.append(float(np.dot(width_pr, prec_list[1:])))

    return float(np.mean(aucs)), float(np.mean(aps))


# --- 임계값 기반 지표 ---


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Point adjustment (Xu et al. 2018): 이벤트 내 1점이라도 탐지되면 이벤트 전체 탐지 간주.

    legacy 비교 재현 전용. 기본 평가에 사용 금지 (CLAUDE.md §4).

    주의: 문헌 표준 구현(donut/OmniAnomaly 계열 `adjust_predicts`)은 backfill 루프가
    `range(i, 0, -1)`이라 시계열 첫 인덱스(0)를 절대 채우지 않는 off-by-one 특성이
    있다. 문헌 수치와의 비교 가능성이 이 지표의 유일한 존재 이유이므로, 여기서는
    그 canonical 동작을 의도적으로 그대로 재현한다 (TSB-AD 참조 구현과 일치 검증됨).
    """
    adjusted = np.asarray(pred).astype(int).copy()
    for start, end in label_events(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
            if start == 0 and not pred[0]:
                adjusted[0] = 0  # canonical 구현의 index-0 미채움 재현
    return adjusted


def event_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    """event-level recall × point-level precision의 F1 (TSB-AD EventF1PA 정의와 동일)."""
    events = label_events(labels)
    if not events:
        return 0.0
    tp = sum(1 for s, e in events if pred[s:e].any())
    rec_e = tp / len(events)
    prec_t = precision_score(labels, pred, zero_division=0)
    return float(2 * rec_e * prec_t / (rec_e + prec_t + _EPS))


def _overlap_reward(event: tuple[int, int], pred_pos: np.ndarray, n_pred_events: int) -> float:
    """range-recall의 overlap 항 (flat bias, cardinality factor 포함; Tatbul NeurIPS 2018)."""
    s, e = event  # e 포함
    length = e - s + 1
    hits = pred_pos[(pred_pos >= s) & (pred_pos <= e)]
    if hits.size == 0:
        return 0.0
    omega = hits.size / length  # flat positional bias
    return omega  # cardinality factor는 호출부에서 곱함


def range_f1(pred: np.ndarray, labels: np.ndarray, alpha: float = 0.2) -> float:
    """Range-based F1 (Tatbul et al., NeurIPS 2018; TSB-AD RF1과 동일 파라미터).

    recall은 existence(alpha) + overlap(1-alpha), precision은 역방향 overlap(alpha=0).
    """
    r = _range_recall(labels, pred, alpha)
    p = _range_recall(pred, labels, 0.0)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def _range_recall(labels: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    real = [(s, e - 1) for s, e in label_events(labels)]
    pred_events = [(s, e - 1) for s, e in label_events(pred)]
    if not real:
        return 0.0
    pred_pos = np.flatnonzero(pred)
    existence = sum(1 for s, e in real if pred[s : e + 1].any())
    overlap = 0.0
    for ev in real:
        card = sum(1 for ps, pe in pred_events if ps <= ev[1] and pe >= ev[0])
        gamma = 1.0 / card if card >= 1 else 0.0
        overlap += _overlap_reward(ev, pred_pos, card) * (gamma if card > 1 else 1.0)
    return (alpha * existence + (1 - alpha) * overlap) / len(real)


def affiliation_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    """Affiliation-F1 (Huet et al., KDD 2022) — vendored TSB-AD 패키지 사용 (Apache-2.0)."""
    from tsad_forge.evaluation._vendor.tsb_ad.affiliation.generics import (
        convert_vector_to_events,
    )
    from tsad_forge.evaluation._vendor.tsb_ad.affiliation.metrics import pr_from_events

    events_pred = convert_vector_to_events(np.asarray(pred).astype(int))
    events_gt = convert_vector_to_events(np.asarray(labels).astype(int))
    if not events_pred or not events_gt:
        return 0.0
    res = pr_from_events(events_pred, events_gt, (0, len(pred)))
    p, r = res["Affiliation_Precision"], res["Affiliation_Recall"]
    return float(2 * p * r / (p + r + _EPS))


def _best_f1(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 200) -> float:
    """threshold sweep 최대 F1 (oracle 임계값 — '참고용'으로만 보고)."""
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
    best = 0.0
    for th in np.unique(candidates):
        best = max(best, float(f1_score(labels, (scores >= th).astype(int), zero_division=0)))
    return best


# --- 통합 엔트리 ---


def mean_detection_delay(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float | None:
    """이상 이벤트 시작 → 첫 경보까지의 평균 지연 (steps). 조기탐지 관점 지표 (리뷰 P1).

    이벤트 [s, e) 안에서 scores >= threshold 인 첫 시점 t의 (t - s).
    이벤트 내 경보가 없으면(미탐) 이벤트 길이 (e - s)를 지연 상한으로 부과한다 —
    미탐을 무시하면 짧은 이벤트만 잡는 모델이 유리해지는 왜곡이 생긴다.
    이벤트가 없으면 None (지표 미산출).
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    padded = np.diff(np.concatenate([[0], labels, [0]]))
    starts, ends = np.flatnonzero(padded == 1), np.flatnonzero(padded == -1)
    if len(starts) == 0:
        return None
    delays = []
    for s, e in zip(starts, ends, strict=True):
        hits = np.flatnonzero(scores[s:e] >= threshold)
        delays.append(int(hits[0]) if hits.size else int(e - s))
    return float(np.mean(delays))


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    legacy_pa: bool = False,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
) -> dict[str, float]:
    """연속 이상 점수와 이진 라벨로 전체 지표를 계산한다.

    threshold-free: vus_pr(주지표), vus_roc, auc_pr, auc_roc, best_f1(참고용).
    threshold 지정 시: standard_f1, event_f1, range_f1, affiliation_f1.
    legacy_pa=True일 때만 pa_f1 (UserWarning 경고 필수).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")

    out: dict[str, float] = {}
    has_both = 0 < labels.sum() < len(labels)

    if has_both:
        out["auc_roc"] = float(roc_auc_score(labels, scores))
        out["auc_pr"] = float(average_precision_score(labels, scores))
        window = min(sliding_window, max(len(scores) // 4, 1))
        vus_roc_v, vus_pr_v = vus(scores, labels, window_max=window)
        out["vus_roc"] = vus_roc_v
        out["vus_pr"] = vus_pr_v  # 주지표
        out["best_f1"] = _best_f1(scores, labels)

    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        if has_both:
            out["standard_f1"] = float(f1_score(labels, pred, zero_division=0))
            out["event_f1"] = event_f1(pred, labels)
            out["range_f1"] = range_f1(pred, labels)
            out["affiliation_f1"] = affiliation_f1(pred, labels)
            delay = mean_detection_delay(scores, labels, threshold)
            if delay is not None:
                out["mean_detection_delay"] = delay
        if legacy_pa:
            warnings.warn(PA_WARNING, UserWarning, stacklevel=2)
            adjusted = point_adjust(pred, labels)
            out["pa_f1"] = float(f1_score(labels, adjusted, zero_division=0))

    return out
