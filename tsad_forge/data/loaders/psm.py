"""PSM (Pooled Server Metrics, eBay) 로더 — RANSynCoders (Abdulaal et al., KDD 2021).

train.csv / test.csv: timestamp_(min) + 25개 feature. test_label.csv: label.
알려진 결함: train에 결측이 있고(보간 필요), 라벨 경계가 분 단위로 거칠다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.schema import TSADDataset


def load_psm(data_dir: str | Path = "data") -> TSADDataset:
    root = Path(data_dir) / "psm"
    train_f = root / "train.csv"
    if not train_f.exists():
        raise FileNotFoundError(f"{train_f} not found. Run: tsad-forge download psm")

    def _values(df: pd.DataFrame) -> np.ndarray:
        df = df.drop(columns=[c for c in df.columns if "timestamp" in c.lower()])
        return df.interpolate(limit_direction="both").to_numpy(np.float64)

    train = _values(pd.read_csv(train_f))
    test = _values(pd.read_csv(root / "test.csv"))
    labels = pd.read_csv(root / "test_label.csv")["label"].to_numpy().astype(int)

    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": "psm",
            "source_url": "https://github.com/eBay/RANSynCoders",
            "license": "Apache-2.0 (RANSynCoders repository)",
            "citation": "Abdulaal et al., Practical Approach to Asynchronous Multivariate "
            "Time Series Anomaly Detection and Localization, KDD 2021",
            "sampling": "1 min",
            "note": "train 결측은 선형 보간",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
