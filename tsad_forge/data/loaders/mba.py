"""MBA (MIT-BIH Supraventricular Arrhythmia, 2-lead ECG) loader.

Processed copy fetched from the TranAD repository (BSD-3-Clause); the underlying
recordings come from PhysioNet's MIT-BIH database (ODC-BY, open). Train/test are
the splits shipped by TranAD; labels.xlsx marks anomalous test regions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsad_forge.data.paths import resolve_data_dir
from tsad_forge.data.schema import TSADDataset


def load_mba(data_dir: str | Path | None = None) -> TSADDataset:
    root = resolve_data_dir(data_dir) / "mba"
    train_f = root / "train.xlsx"
    if not train_f.exists():
        raise FileNotFoundError(f"{train_f} not found. Run: tsad-forge download mba")

    def _values(path: Path) -> np.ndarray:
        df = pd.read_excel(path, engine="openpyxl", index_col=0)
        # first row carries units ("(mV)") — coerce and drop non-numeric rows
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return df.to_numpy(np.float64)

    train = _values(train_f)
    test = _values(root / "test.xlsx")

    # labels.xlsx is a PhysioNet beat-annotation table (Sample index + beat Type).
    # Non-'N' beats are anomalous; following the TranAD preprocessing convention we
    # mark +/-20 samples around each such beat.
    ann = pd.read_excel(root / "labels.xlsx", engine="openpyxl")
    labels = np.zeros(len(test), dtype=int)
    for sample in ann.loc[ann["Type"].astype(str).str.strip() != "N", "Sample"]:
        s = int(sample)
        labels[max(0, s - 20) : min(len(test), s + 20)] = 1

    ds = TSADDataset(
        train=train,
        test=test,
        labels=labels,
        meta={
            "name": "mba",
            "source_url": "https://github.com/imperial-qore/TranAD (data/MBA)",
            "license": "repo BSD-3-Clause; underlying PhysioNet MIT-BIH: ODC-BY",
            "citation": "Moody & Mark, MIT-BIH Arrhythmia Database; processed split from "
            "Tuli et al. (TranAD), VLDB 2022",
            "sampling": "360 Hz ECG (resampled by TranAD preprocessing)",
        },
    )
    ds.meta.update(ds.event_stats())
    return ds
