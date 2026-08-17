"""SKAB (Skoltech Anomaly Benchmark) 로더.

세미콜론 구분 csv: datetime;8개 센서;anomaly;changepoint.
train = anomaly-free/anomaly-free.csv (정상 실험), test = 각 실험 파일.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset

SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


def load_skab(experiment: str = "valve1/0", data_dir: str | Path | None = None) -> TSADDataset:
    """experiment 예: 'valve1/0', 'valve2/3', 'other/11'"""
    root = resolve_data_dir(data_dir) / "skab"
    test_f = root / f"{experiment}.csv"
    if not test_f.exists():
        raise FileNotFoundError(f"{test_f} not found. Run: tsad-forge download skab")

    train_df = pd.read_csv(root / "anomaly-free" / "anomaly-free.csv", sep=";")
    test_df = pd.read_csv(test_f, sep=";")

    train = train_df[SENSOR_COLS].to_numpy(float)
    test = test_df[SENSOR_COLS].to_numpy(float)
    labels = test_df["anomaly"].to_numpy().astype(float).astype(int)

    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": f"skab/{experiment}",
            "source_url": "https://github.com/waico/SKAB",
            "license": "AGPL-3.0 repo — 데이터 파일만 사용 (코드 미사용), 상세는 데이터셋 카드 참조",
            "citation": "Katser & Kozitsin, Skoltech Anomaly Benchmark (SKAB), 2020",
            "sampling": "1 s",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
