"""NASA SMAP/MSL 로더 — telemanom (Hundman et al., KDD 2018).

채널별 npy(train/test) + labeled_anomalies.csv(테스트 구간 이상 시퀀스).
알려진 결함: 다수 채널이 준이진(command 채널 포함 55차원 중 텔레메트리는 1차원)이고
라벨이 희소하다. PA 기반 선행 보고와 직접 비교 불가 (CLAUDE.md §2).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.schema import TSADDataset


def _load_channel(spacecraft: str, channel: str, data_dir: str | Path = "data") -> TSADDataset:
    root = Path(data_dir) / "smap_msl"
    train_f = root / "train" / f"{channel}.npy"
    if not train_f.exists():
        raise FileNotFoundError(
            f"{train_f} not found. Run: tsad-forge download smap "
            "(S3 차단 환경에서는 docs/datasets/smap_msl.md의 수동 배치 안내 참고)"
        )
    train = np.load(train_f)
    test = np.load(root / "test" / f"{channel}.npy")

    labels_df = pd.read_csv(root / "labeled_anomalies.csv")
    row = labels_df[(labels_df["chan_id"] == channel) & (labels_df["spacecraft"] == spacecraft)]
    if row.empty:
        raise KeyError(f"channel '{channel}' not found for spacecraft '{spacecraft}'")
    labels = np.zeros(len(test), dtype=int)
    for start, end in ast.literal_eval(row.iloc[0]["anomaly_sequences"]):
        labels[start : end + 1] = 1  # 원 라벨은 양끝 포함

    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": f"{spacecraft.lower()}/{channel}",
            "source_url": "https://github.com/khundman/telemanom",
            "license": "NASA open data (telemanom repo: custom permissive, 코드 미사용)",
            "citation": "Hundman et al., Detecting Spacecraft Anomalies Using LSTMs and "
            "Nonparametric Dynamic Thresholding, KDD 2018",
            "anomaly_classes": row.iloc[0].get("class", ""),
        },
    )
    ds.meta.update(ds.event_stats())
    return ds


def load_smap(channel: str = "P-1", data_dir: str | Path = "data") -> TSADDataset:
    return _load_channel("SMAP", channel, data_dir)


def load_msl(channel: str = "M-6", data_dir: str | Path = "data") -> TSADDataset:
    return _load_channel("MSL", channel, data_dir)
