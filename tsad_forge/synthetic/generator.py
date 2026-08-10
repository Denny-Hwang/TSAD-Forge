"""합성 데이터셋 생성기: 스모크 테스트·교육·contamination 실험용."""

from __future__ import annotations

import numpy as np

from tsad_forge.data.schema import TSADDataset
from tsad_forge.synthetic.injectors import inject_anomalies


def _base_signal(T: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """채널별로 주기·위상이 다른 sine + 완만한 추세 + 백색잡음."""
    t = np.arange(T, dtype=np.float64)
    x = np.empty((T, D))
    for d in range(D):
        period = rng.uniform(40, 120)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.8, 1.5)
        trend = rng.uniform(-0.5, 0.5) * (t / T)
        noise = rng.normal(0, 0.1, size=T)
        x[:, d] = amp * np.sin(2 * np.pi * t / period + phase) + trend + noise
    return x


def generate_synthetic(
    n_train: int = 2000,
    n_test: int = 2000,
    n_dims: int = 1,
    n_events: int = 5,
    anomaly_kinds: list[str] | None = None,
    contamination: float = 0.0,
    seed: int = 0,
) -> TSADDataset:
    """합성 TSADDataset 생성.

    Args:
        contamination: train 구간에 주입할 이상 이벤트 비율 실험용.
            0이면 train은 순수 정상 (기본).
        seed: 전체 생성 과정의 시드 (재현성, CLAUDE.md §9).
    """
    rng = np.random.default_rng(seed)
    train = _base_signal(n_train, n_dims, rng)
    test = _base_signal(n_test, n_dims, rng)

    if contamination > 0:
        n_train_events = max(1, int(round(contamination * n_events)))
        inject_anomalies(train, rng, n_train_events, anomaly_kinds)

    labels = inject_anomalies(test, rng, n_events, anomaly_kinds)

    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": "synthetic",
            "source_url": "(generated in-process)",
            "license": "Apache-2.0",
            "citation": "TSAD-Forge synthetic generator",
            "sampling": "uniform (unitless)",
            "seed": seed,
            "n_events_requested": n_events,
            "contamination": contamination,
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
