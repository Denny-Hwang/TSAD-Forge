"""UCR Anomaly Archive 로더 (Wu & Keogh, TKDE 2021).

파일명이 메타데이터를 담는다:
    NNN_UCR_Anomaly_<name>_<trainEnd>_<anomStart>_<anomEnd>.txt
값은 한 줄당 하나(또는 공백 구분)의 단변량 시계열.
train = x[:trainEnd], test = x[trainEnd:], 라벨은 전체 인덱스 기준
[anomStart, anomEnd] (1-indexed, 양끝 포함)를 test 좌표로 변환.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset

_NAME_RE = re.compile(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


def parse_ucr_filename(filename: str) -> dict:
    m = _NAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"not a UCR anomaly archive filename: {filename}")
    idx, name, train_end, a_start, a_end = m.groups()
    return {
        "index": int(idx),
        "series_name": name,
        "train_end": int(train_end),
        "anomaly_start": int(a_start),
        "anomaly_end": int(a_end),
    }


def load_ucr(series: str | int, data_dir: str | Path | None = None) -> TSADDataset:
    """series: 3자리 인덱스(예: 1, '001') 또는 전체 파일명."""
    root = resolve_data_dir(data_dir) / "ucr"
    if isinstance(series, int) or series.isdigit():
        pattern = f"{int(series):03d}_UCR_Anomaly_*.txt"
        matches = sorted(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"no UCR file matching {pattern} in {root}. Run: tsad-forge download ucr"
            )
        path = matches[0]
    else:
        path = root / series
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: tsad-forge download ucr")

    info = parse_ucr_filename(path.name)
    x = np.loadtxt(path).ravel()
    t_end = info["train_end"]
    labels = np.zeros(len(x) - t_end, dtype=int)
    # 파일명 인덱스는 1-based, 양끝 포함
    s = max(info["anomaly_start"] - 1 - t_end, 0)
    e = info["anomaly_end"] - t_end
    labels[s:e] = 1

    ds = TSADDataset(
        train=x[:t_end],
        test=x[t_end:],
        labels=labels,
        meta={
            "name": f"ucr/{info['index']:03d}_{info['series_name']}",
            "source_url": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/",
            "license": "free for research (UCR archive)",
            "citation": "Wu & Keogh, Current Time Series Anomaly Detection Benchmarks are "
            "Flawed and are Creating the Illusion of Progress, TKDE 2021",
            **info,
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
