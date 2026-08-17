"""MGAB (Mackey-Glass Anomaly Benchmark) loader — Thill et al., CC0 1.0.

10 univariate series generated from the chaotic Mackey-Glass equations with
synthetically inserted anomalies. Columns: index, value, is_anomaly, is_ignored
(is_ignored marks warm-up/transition segments the original benchmark excludes;
we keep the points but record the mask in meta).

No official train/test split exists: we use the longest anomaly-free prefix
(capped at 30% of the series) as train — documented deviation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset


def load_mgab(series: int = 1, data_dir: str | Path | None = None) -> TSADDataset:
    path = resolve_data_dir(data_dir) / "mgab" / f"{int(series)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: tsad-forge download mgab")
    df = pd.read_csv(path, index_col=0)
    values = df["value"].to_numpy(np.float64)
    labels_full = df["is_anomaly"].to_numpy().astype(int)

    first_anom = int(np.argmax(labels_full)) if labels_full.any() else len(values)
    split = min(first_anom, int(len(values) * 0.3))
    split = max(split, 100)

    ds = TSADDataset(
        train=values[:split],
        test=values[split:],
        labels=labels_full[split:],
        meta={
            "name": f"mgab/{int(series)}",
            "source_url": "https://github.com/MarkusThill/MGAB",
            "license": "CC0 1.0 (public domain)",
            "citation": "Thill, Konen & Baeck, MGAB: The Mackey-Glass Anomaly Benchmark, 2020",
            "sampling": "uniform (unitless, chaotic simulation)",
            "n_ignored": int(df["is_ignored"].sum()) if "is_ignored" in df else 0,
            "note": "no official split — longest anomaly-free prefix (<=30%) used as train",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
