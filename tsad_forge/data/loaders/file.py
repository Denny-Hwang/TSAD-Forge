"""BYOD 로더: CSV/parquet 파일을 TSADDataset으로 변환 (CLAUDE.md §2).

- `timestamp` 열(옵션): 정렬에만 사용하고 값 행렬에서 제외.
- `label` 열(옵션): test 라벨. 없으면 전부 0으로 두고 meta["has_labels"]=False.
- train/test 분할: 라벨이 없으면 앞 train_ratio 구간을 train으로 사용.
  라벨이 있으면 첫 이상 이전 구간까지를 train 후보로 쓰되 최소 비율을 보장.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.schema import TSADDataset

TIMESTAMP_COLS = ("timestamp", "time", "datetime", "date")
LABEL_COLS = ("label", "labels", "is_anomaly", "anomaly")


def load_file(path: str | Path, train_ratio: float = 0.3) -> TSADDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix} (use .csv or .parquet)")

    ts_col = next((c for c in df.columns if c.lower() in TIMESTAMP_COLS), None)
    if ts_col is not None:
        df = df.sort_values(ts_col).reset_index(drop=True)
        df = df.drop(columns=[ts_col])

    label_col = next((c for c in df.columns if c.lower() in LABEL_COLS), None)
    if label_col is not None:
        labels_full = df[label_col].to_numpy().astype(int)
        df = df.drop(columns=[label_col])
        has_labels = True
    else:
        labels_full = np.zeros(len(df), dtype=int)
        has_labels = False

    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if values.shape[1] == 0:
        raise ValueError("no numeric columns found in file")
    if np.isnan(values).any():
        # 결측은 선형 보간 (문서화된 기본 동작)
        values = (
            pd.DataFrame(values).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
        )

    split = int(len(values) * train_ratio)
    if has_labels:
        anomalous = np.flatnonzero(labels_full)
        if anomalous.size and anomalous[0] < split:
            split = int(anomalous[0])  # train은 정상 구간만 (unsupervised 가정)
    split = max(split, min(10, len(values) // 2))

    return TSADDataset(
        train=values[:split],
        test=values[split:],
        labels=labels_full[split:],
        meta={
            "name": path.stem,
            "source_url": str(path),
            "license": "user-provided",
            "citation": "",
            "has_labels": has_labels,
            "train_ratio": split / max(len(values), 1),
        },
    )
