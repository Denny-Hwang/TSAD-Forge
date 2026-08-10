"""합성 이상 주입기 (CLAUDE.md 구조 §1: spike / level shift / pattern / frequency / contextual).

각 injector는 (series[T, D], rng, start, length) -> 수정된 구간을 in-place 주입하고
해당 구간 라벨을 1로 만든다. contamination 실험(train 오염)에도 재사용한다.
"""

from __future__ import annotations

import numpy as np


def inject_spike(x: np.ndarray, rng: np.random.Generator, start: int, length: int) -> None:
    """단발성 스파이크: 구간 내 각 지점에 큰 임펄스."""
    d = rng.integers(x.shape[1])
    magnitude = rng.uniform(4.0, 8.0) * x[:, d].std()
    sign = rng.choice([-1.0, 1.0])
    x[start : start + length, d] += sign * magnitude


def inject_level_shift(x: np.ndarray, rng: np.random.Generator, start: int, length: int) -> None:
    """구간 전체의 평균 수준 이동."""
    d = rng.integers(x.shape[1])
    shift = rng.uniform(2.5, 5.0) * x[:, d].std() * rng.choice([-1.0, 1.0])
    x[start : start + length, d] += shift


def inject_pattern(x: np.ndarray, rng: np.random.Generator, start: int, length: int) -> None:
    """파형 왜곡: 구간을 노이즈 파형으로 대체 (collective anomaly)."""
    d = rng.integers(x.shape[1])
    local_std = x[:, d].std()
    x[start : start + length, d] = rng.normal(
        loc=x[start : start + length, d].mean(), scale=2.0 * local_std, size=length
    )


def inject_frequency(x: np.ndarray, rng: np.random.Generator, start: int, length: int) -> None:
    """주파수 변화: 구간에 고주파 성분 중첩."""
    d = rng.integers(x.shape[1])
    t = np.arange(length)
    freq = rng.uniform(0.3, 0.5)  # 나이퀴스트 근처 고주파
    x[start : start + length, d] += 1.5 * x[:, d].std() * np.sin(2 * np.pi * freq * t)


def inject_contextual(x: np.ndarray, rng: np.random.Generator, start: int, length: int) -> None:
    """맥락적 이상: 값 자체는 정상 범위지만 위상이 뒤집힘 (구간 반전)."""
    d = rng.integers(x.shape[1])
    seg = x[start : start + length, d]
    x[start : start + length, d] = seg.mean() - (seg - seg.mean())


INJECTORS = {
    "spike": inject_spike,
    "level_shift": inject_level_shift,
    "pattern": inject_pattern,
    "frequency": inject_frequency,
    "contextual": inject_contextual,
}


def inject_anomalies(
    x: np.ndarray,
    rng: np.random.Generator,
    n_events: int,
    kinds: list[str] | None = None,
    min_len: int = 1,
    max_len: int = 40,
) -> np.ndarray:
    """x[T, D]에 n_events개 이상을 주입하고 [T] 라벨을 반환한다. x는 in-place 수정."""
    kinds = kinds or list(INJECTORS)
    unknown = set(kinds) - set(INJECTORS)
    if unknown:
        raise ValueError(f"unknown anomaly kinds: {sorted(unknown)}")
    T = len(x)
    labels = np.zeros(T, dtype=int)
    for _ in range(n_events):
        kind = kinds[rng.integers(len(kinds))]
        length = int(rng.integers(min_len, max_len + 1))
        if T - length <= 1:
            raise ValueError(f"series too short (T={T}) for anomaly length {length}")
        start = int(rng.integers(1, T - length))
        INJECTORS[kind](x, rng, start, length)
        labels[start : start + length] = 1
    return labels
