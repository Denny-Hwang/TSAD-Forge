"""SWaT / WADI 로더 — iTrust 신청 필요, 재배포 금지 (CLAUDE.md §2).

신청·승인 후 (docs/datasets/swat_wadi.md) 원본 xlsx를 CSV로 내보내 배치:
    data/swat/train.csv  (Normal 구간, 'Normal/Attack' 열 포함 가능)
    data/swat/test.csv   (Attack 구간, 'Normal/Attack' 열 필수)
    data/wadi/train.csv, data/wadi/test.csv (라벨 열 'Attack' 0/1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset

_LABEL_CANDIDATES = ("normal/attack", "attack", "label")


def _split_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    label_col = next((c for c in df.columns if c.strip().lower() in _LABEL_CANDIDATES), None)
    if label_col is None:
        labels = np.zeros(len(df), dtype=int)
    else:
        raw = df[label_col]
        if pd.api.types.is_string_dtype(raw) or raw.dtype == object:
            labels = (raw.str.strip().str.lower() != "normal").astype(int).to_numpy()
        else:
            labels = raw.to_numpy().astype(int)
        df = df.drop(columns=[label_col])
    ts_cols = [c for c in df.columns if "timestamp" in c.lower() or c.strip().lower() == "date"]
    values = (
        df.drop(columns=ts_cols)
        .select_dtypes(include=[np.number])
        .interpolate(limit_direction="both")
        .to_numpy(np.float64)
    )
    return values, labels


def _load_scada(name: str, data_dir: str | Path, sampling: str) -> TSADDataset:
    root = resolve_data_dir(data_dir) / name
    train_f, test_f = root / "train.csv", root / "test.csv"
    if not train_f.exists() or not test_f.exists():
        raise FileNotFoundError(
            f"{root}/train.csv, test.csv가 필요합니다. {name.upper()}는 iTrust 신청 필요 — "
            "docs/datasets/swat_wadi.md 참고"
        )
    train, _ = _split_label(pd.read_csv(train_f))
    test, labels = _split_label(pd.read_csv(test_f))
    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": name,
            "source_url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/",
            "license": "iTrust — 신청 필요, 재배포 금지",
            "citation": "Goh et al., A Dataset to Support Research in the Design of Secure "
            "Water Treatment Systems, CRITIS 2016",
            "sampling": sampling,
        },
    )
    ds.meta.update(ds.event_stats())
    return ds


def load_swat(data_dir: str | Path | None = None) -> TSADDataset:
    return _load_scada("swat", data_dir, "1 s")


def load_wadi(data_dir: str | Path | None = None) -> TSADDataset:
    return _load_scada("wadi", data_dir, "1 s")
