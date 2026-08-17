"""Yahoo S5 로더 — 재배포 금지, 로컬 배치 전용 (CLAUDE.md §2).

Yahoo Webscope에서 신청·승인 후 (docs/datasets/yahoo_s5.md) 아래처럼 배치:
    data/yahoo_s5/A1Benchmark/real_1.csv ... A4Benchmark/...
형식: timestamp,value,is_anomaly (A1/A2) 또는 timestamps,value,anomaly (A3/A4 변형).
라벨이 전 구간에 있으므로 앞 50%를 train으로 사용하되 train 라벨은 버린다(unsupervised).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset


def load_yahoo(
    benchmark: str = "A1Benchmark",
    series: str = "real_1.csv",
    data_dir: str | Path | None = None,
    train_ratio: float = 0.5,
) -> TSADDataset:
    path = resolve_data_dir(data_dir) / "yahoo_s5" / benchmark / series
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Yahoo S5는 재배포 금지입니다 — 신청 및 배치 방법: "
            "docs/datasets/yahoo_s5.md"
        )
    df = pd.read_csv(path)
    label_col = next(c for c in df.columns if "anomaly" in c.lower())
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    values = df[["value"]].to_numpy(float)
    labels_full = df[label_col].to_numpy().astype(int)
    df.sort_values(ts_col)  # 정렬 확인용 (원본은 이미 정렬됨)

    split = int(len(values) * train_ratio)
    ds = TSADDataset(
        train=values[:split],
        test=values[split:],
        labels=labels_full[split:],
        meta={
            "name": f"yahoo_s5/{benchmark}/{Path(series).stem}",
            "source_url": "https://webscope.sandbox.yahoo.com/catalog.php?datatype=s",
            "license": "Yahoo Webscope — 재배포 금지, 연구 목적 신청 필요",
            "citation": "Laptev et al., Yahoo S5 - A Labeled Anomaly Detection Dataset, 2015",
            "note": "train 구간 라벨은 unsupervised 가정에 따라 미사용",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
