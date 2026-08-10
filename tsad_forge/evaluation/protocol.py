"""실험 프로토콜 유틸 (CLAUDE.md §4, §9).

- 시드 고정 (numpy / random / 가능하면 torch)
- z-score 정규화 (train 통계 기준 — test 누수 방지)
- 스코어 원본 저장 규약 (지표 재계산 가능하게)
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


def zscore_normalize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train 통계 기준 z-score 정규화 (기본 전처리). test 통계는 사용하지 않는다."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def save_scores(scores: np.ndarray, labels: np.ndarray, path: str | Path) -> Path:
    """스코어 원본 + 라벨을 npz로 저장 (지표 재계산 가능하게, CLAUDE.md §4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores, labels=labels)
    return path


def load_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["scores"], data["labels"]
