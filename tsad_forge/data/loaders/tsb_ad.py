"""TSB-AD 로더 (Liu & Paparrizos, NeurIPS 2024; Apache-2.0).

CSV 형식: 값 열들 + 'Label' 열. 파일명에 train 길이가 인코딩:
    <id>_<source>_id_<k>_<domain>_tr_<trainLen>_1st_<firstAnomaly>.csv
프로토콜 호환: train = x[:trainLen], test = x[trainLen:] (TSB-AD 규약).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset

_TR_RE = re.compile(r"_tr_(\d+)_")


def parse_train_length(filename: str) -> int:
    m = _TR_RE.search(Path(filename).name)
    if not m:
        raise ValueError(f"cannot parse train length (_tr_N_) from: {filename}")
    return int(m.group(1))


def load_tsb_ad(
    filename: str, variant: str = "u", data_dir: str | Path | None = None
) -> TSADDataset:
    root = resolve_data_dir(data_dir) / f"tsb_ad_{variant.lower()}"
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found under {root}. Run: tsad-forge download tsb_ad_{variant}"
        )
    path = matches[0]
    df = pd.read_csv(path)
    label_col = next(c for c in df.columns if c.lower() == "label")
    labels_full = df[label_col].to_numpy().astype(int)
    values = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy(np.float64)

    t_end = parse_train_length(path.name)
    ds = TSADDataset(
        train=values[:t_end],
        test=values[t_end:],
        labels=labels_full[t_end:],
        meta={
            "name": f"tsb_ad_{variant}/{path.stem}",
            "source_url": "https://github.com/TheDatumOrg/TSB-AD",
            "license": "Apache-2.0",
            "citation": "Liu & Paparrizos, The Elephant in the Room: Towards A Reliable "
            "Time-Series Anomaly Detection Benchmark, NeurIPS 2024",
            "train_end": t_end,
        },
    )
    ds.meta.update(ds.event_stats())
    return ds


def list_tsb_ad_files(variant: str = "u", data_dir: str | Path | None = None) -> list[str]:
    root = resolve_data_dir(data_dir) / f"tsb_ad_{variant.lower()}"
    return sorted(p.name for p in root.rglob("*.csv"))
