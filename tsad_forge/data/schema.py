"""통일 데이터셋 스키마 (CLAUDE.md §2).

모든 로더는 TSADDataset을 반환한다. 단변량은 D=1로 통일한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TSADDataset:
    """단일 시계열(또는 단일 엔터티)의 train/test 분할과 test 라벨.

    Attributes:
        train: [T_train, D] float 배열. 학습(정상 위주) 구간.
        test: [T_test, D] float 배열. 평가 구간.
        labels: [T_test] {0,1} 배열. test 구간의 point-level 이상 라벨.
        meta: name, source_url, license, citation, sampling 정보,
            이상 이벤트 수/길이 통계 등.
    """

    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.train = np.asarray(self.train, dtype=np.float64)
        self.test = np.asarray(self.test, dtype=np.float64)
        self.labels = np.asarray(self.labels)
        if self.train.ndim == 1:
            self.train = self.train[:, None]
        if self.test.ndim == 1:
            self.test = self.test[:, None]
        self.validate()

    def validate(self) -> None:
        if self.train.ndim != 2 or self.test.ndim != 2:
            raise ValueError("train/test must be 2-D arrays of shape [T, D]")
        if self.train.shape[1] != self.test.shape[1]:
            raise ValueError(
                f"train/test dimensionality mismatch: {self.train.shape[1]} vs {self.test.shape[1]}"
            )
        if self.labels.ndim != 1:
            raise ValueError("labels must be a 1-D array of shape [T_test]")
        if len(self.labels) != len(self.test):
            raise ValueError(
                f"labels length {len(self.labels)} != test length {len(self.test)}"
            )
        unique = set(np.unique(self.labels).tolist())
        if not unique <= {0, 1}:
            raise ValueError(f"labels must be binary 0/1, got values {sorted(unique)}")

    @property
    def n_dims(self) -> int:
        return self.train.shape[1]

    @property
    def anomaly_rate(self) -> float:
        return float(np.mean(self.labels)) if len(self.labels) else 0.0

    def event_stats(self) -> dict:
        """test 라벨의 이상 이벤트(연속 1 구간) 수와 길이 통계."""
        events = label_events(self.labels)
        lengths = [e - s for s, e in events]
        return {
            "n_events": len(events),
            "min_len": int(min(lengths)) if lengths else 0,
            "max_len": int(max(lengths)) if lengths else 0,
            "mean_len": float(np.mean(lengths)) if lengths else 0.0,
        }


def label_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """이진 라벨에서 [start, end) 이상 이벤트 구간 목록을 추출한다."""
    labels = np.asarray(labels).astype(bool)
    if labels.size == 0:
        return []
    padded = np.concatenate([[False], labels, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))
