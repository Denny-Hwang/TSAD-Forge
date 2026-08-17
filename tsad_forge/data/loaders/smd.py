"""SMD (Server Machine Dataset) 로더 — OmniAnomaly (Su et al., KDD 2019).

28개 머신, 38차원. train/test/test_label의 comma-separated txt.
알려진 결함: 머신 간 이상 이벤트 길이 분산이 크고, 일부 채널은 상수에 가깝다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset


def load_smd(machine: str = "machine-1-1", data_dir: str | Path | None = None) -> TSADDataset:
    root = resolve_data_dir(data_dir) / "smd"
    train_f = root / "train" / f"{machine}.txt"
    if not train_f.exists():
        raise FileNotFoundError(
            f"{train_f} not found. Run: tsad-forge download smd --subset {machine}"
        )
    train = np.loadtxt(train_f, delimiter=",")
    test = np.loadtxt(root / "test" / f"{machine}.txt", delimiter=",")
    labels = np.loadtxt(root / "test_label" / f"{machine}.txt", delimiter=",").astype(int)
    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": f"smd/{machine}",
            "source_url": "https://github.com/NetManAIOps/OmniAnomaly",
            "license": "MIT (OmniAnomaly repository)",
            "citation": "Su et al., Robust Anomaly Detection for Multivariate Time Series "
            "through Stochastic Recurrent Neural Network, KDD 2019",
            "sampling": "1 min",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
