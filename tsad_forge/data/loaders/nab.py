"""NAB 로더 (Numenta Anomaly Benchmark).

data/<group>/<file>.csv: timestamp,value. labels/combined_windows.json이
이상 윈도우 [start, end] 타임스탬프 쌍을 제공한다.
NAB 코드는 AGPL이므로 코드는 사용하지 않고 데이터 파일만 사용한다 (데이터는 자유 이용).
train 분할: NAB 규약상 앞 15%가 probationary 구간 — 이를 train으로 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.schema import TSADDataset

PROBATION_RATIO = 0.15


def load_nab(rel_path: str, data_dir: str | Path = "data") -> TSADDataset:
    """rel_path 예: 'realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv'"""
    root = Path(data_dir) / "nab"
    path = root / "data" / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: tsad-forge download nab")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    windows = json.loads((root / "labels" / "combined_windows.json").read_text())
    labels_full = np.zeros(len(df), dtype=int)
    for start, end in windows.get(rel_path, []):
        mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
        labels_full[mask.to_numpy()] = 1

    split = int(len(df) * PROBATION_RATIO)
    values = df[["value"]].to_numpy(np.float64)
    ds = TSADDataset(
        train=values[:split],
        test=values[split:],
        labels=labels_full[split:],
        meta={
            "name": f"nab/{Path(rel_path).stem}",
            "source_url": "https://github.com/numenta/NAB",
            "license": "data: free to use (NAB code is AGPL — code not used)",
            "citation": "Lavin & Ahmad, Evaluating Real-Time Anomaly Detection Algorithms — "
            "the Numenta Anomaly Benchmark, ICMLA 2015",
            "sampling": "5 min (typical)",
            "note": "train은 NAB probationary 앞 15% 구간; train에 이상이 포함될 수 있음",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
