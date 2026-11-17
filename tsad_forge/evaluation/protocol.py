"""실험 프로토콜 유틸 (CLAUDE.md §4, §9).

- 시드 고정 (numpy / random / 가능하면 torch)
- z-score 정규화 (train 통계 기준 — test 누수 방지)
- VUS 버퍼 폭 자동 추정 (TSB-AD 관례: 자기상관 기반 주기)
- 스코어 원본 저장 규약 (지표 재계산 가능하게)
- 스코어 건전성 검사 (비유한 값은 조용히 지표로 흘려보내지 않는다)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """모든 무작위성 시드 고정. torch는 설치된 경우에만."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def zscore_normalize(
    train: np.ndarray,
    test: np.ndarray,
    std_floor: float = 0.0,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """train 통계 기준 z-score 정규화 (기본 전처리). test 통계는 사용하지 않는다.

    Args:
        std_floor: 표준편차 하한. 기본 0.0은 레거시 동작(정확히 0인 채널만 1로 대체).
            train에서 "거의 상수"인 채널(예: SMD machine-1-1의 33번 채널, std=1.5e-5)은
            test에서 |z|가 수만까지 치솟아 제곱 재구성 오차가 1e8대로 발산한다.
            입력이 [0,1]로 min-max된 데이터(SMD 등)에는 1e-2 정도가 안전한 하한이다.
        clip: 지정 시 정규화 후 값을 [-clip, clip]으로 자른다 (train/test 동일 적용).
            극단값이 점수를 지배하는 것을 막지만, 이상 신호 자체를 깎을 수 있으므로
            비교 실험에서는 같은 clip을 모든 모델에 적용해야 한다.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    if std_floor > 0:
        std = np.maximum(std, std_floor)
    tr, te = (train - mean) / std, (test - mean) / std
    if clip is not None:
        tr, te = np.clip(tr, -clip, clip), np.clip(te, -clip, clip)
    return tr, te


#: TSB-AD 관례: 주기 추정 실패/범위 밖이면 이 값을 쓴다.
DEFAULT_PERIOD_FALLBACK = 125


def estimate_sliding_window(
    x: np.ndarray,
    max_lag: int = 400,
    min_period: int = 3,
    max_period: int = 300,
    fallback: int = DEFAULT_PERIOD_FALLBACK,
    max_points: int = 20000,
) -> int:
    """자기상관(ACF)의 최대 국소 극대로 지배 주기를 추정해 VUS 버퍼 폭으로 쓴다.

    TSB-AD(Liu & Paparrizos, NeurIPS 2024)의 평가 관례(시계열별 주기 기반 sliding
    window)를 따른 자체 구현이다 — 코드 복사 없음. 다변량은 첫 채널을 쓴다 (TSB-AD와
    동일). 추정 주기가 [min_period, max_period] 밖이면 fallback을 반환한다.

    주의: TSB-AD 공식 함수와의 수치 일치는 테스트로 보장하지 않는다 (해당 함수는
    vendored 대상이 아니다). 결과 JSON에 실제 사용 폭(diag_vus_window)이 기록되므로
    사후 대조가 가능하다.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    x = x[:max_points]
    x = x - x.mean()
    n = len(x)
    var = float(np.dot(x, x))
    if n < 2 * min_period or var == 0.0:
        return fallback
    nlags = min(max_lag, n - 1)
    # FFT 기반 ACF (statsmodels.acf(fft=True)와 같은 편향 추정량)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conj(f), size)[: nlags + 1] / var
    base = min_period
    seg = acf[base:]
    if len(seg) < 3:
        return fallback
    is_peak = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])
    peaks = np.flatnonzero(is_peak) + 1
    if peaks.size == 0:
        return fallback
    period = int(peaks[np.argmax(seg[peaks])]) + base
    if period < min_period or period > max_period:
        return fallback
    return period


class ScoreHealthError(ValueError):
    """모델이 비유한(NaN/inf) 점수를 냈다 — 지표 계산 전에 실패로 처리한다."""


def check_scores(scores: np.ndarray, expected_len: int) -> np.ndarray:
    """점수 배열의 모양/유한성을 검사해 float64 1-D로 반환한다.

    비유한 점수를 nan_to_num으로 덮으면 "수치 붕괴한 모델"이 그럴듯한 지표를 얻는다.
    여기서는 실패로 올려 run_all의 실패 기록에 남긴다.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != expected_len:
        raise ScoreHealthError(f"score length {len(scores)} != test length {expected_len}")
    bad = ~np.isfinite(scores)
    if bad.any():
        raise ScoreHealthError(
            f"{int(bad.sum())}/{len(scores)} non-finite anomaly scores "
            f"(first at index {int(np.flatnonzero(bad)[0])}). Consider "
            "normalize_params={std_floor: 0.01} or clip for near-constant train channels."
        )
    return scores


def save_scores(scores: np.ndarray, labels: np.ndarray, path: str | Path) -> Path:
    """스코어 원본 + 라벨을 npz로 저장 (지표 재계산 가능하게, CLAUDE.md §4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores, labels=labels)
    return path


def load_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["scores"], data["labels"]
