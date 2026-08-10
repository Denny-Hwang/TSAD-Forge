"""로더 포맷 테스트 — 실데이터 포맷을 흉내낸 fixture로 CI에서 항상 실행."""

import json

import numpy as np
import pandas as pd
import pytest

from tsad_forge.data.loaders.nab import load_nab
from tsad_forge.data.loaders.psm import load_psm
from tsad_forge.data.loaders.skab import SENSOR_COLS, load_skab
from tsad_forge.data.loaders.smap_msl import load_smap
from tsad_forge.data.loaders.smd import load_smd
from tsad_forge.data.loaders.swat_wadi import load_swat
from tsad_forge.data.loaders.tsb_ad import load_tsb_ad, parse_train_length
from tsad_forge.data.loaders.ucr import load_ucr, parse_ucr_filename
from tsad_forge.data.loaders.yahoo import load_yahoo
from tsad_forge.data.registry import get_dataset_entry, list_datasets

RNG = np.random.default_rng(0)


def test_registry_has_all_planned_datasets():
    names = list_datasets()
    for expected in [
        "synthetic",
        "smap",
        "msl",
        "smd",
        "ucr",
        "tsb_ad_u",
        "tsb_ad_m",
        "nab",
        "psm",
        "skab",
        "yahoo_s5",
        "swat",
        "wadi",
    ]:
        assert expected in names
    assert not get_dataset_entry("yahoo_s5").redistributable
    assert not get_dataset_entry("swat").redistributable


def test_smd_loader(tmp_path):
    root = tmp_path / "smd"
    for part, n in [("train", 50), ("test", 40)]:
        (root / part).mkdir(parents=True)
        np.savetxt(root / part / "machine-1-1.txt", RNG.normal(size=(n, 38)), delimiter=",")
    (root / "test_label").mkdir()
    labels = np.zeros(40)
    labels[10:15] = 1
    np.savetxt(root / "test_label" / "machine-1-1.txt", labels, delimiter=",")

    ds = load_smd("machine-1-1", data_dir=tmp_path)
    assert ds.train.shape == (50, 38) and ds.test.shape == (40, 38)
    assert ds.labels.sum() == 5
    assert ds.meta["n_events"] == 1


def test_smap_loader(tmp_path):
    root = tmp_path / "smap_msl"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    np.save(root / "train" / "P-1.npy", RNG.normal(size=(60, 25)))
    np.save(root / "test" / "P-1.npy", RNG.normal(size=(50, 25)))
    pd.DataFrame(
        {
            "chan_id": ["P-1"],
            "spacecraft": ["SMAP"],
            "anomaly_sequences": ["[[10, 19]]"],
            "class": ["[point]"],
            "num_values": [50],
        }
    ).to_csv(root / "labeled_anomalies.csv", index=False)

    ds = load_smap("P-1", data_dir=tmp_path)
    assert ds.test.shape == (50, 25)
    assert ds.labels.sum() == 10  # [10,19] 양끝 포함
    with pytest.raises(FileNotFoundError):
        load_smap("NOPE-1", data_dir=tmp_path)


def test_ucr_loader(tmp_path):
    root = tmp_path / "ucr"
    root.mkdir()
    x = RNG.normal(size=300)
    fname = "007_UCR_Anomaly_testseries_200_251_260.txt"
    np.savetxt(root / fname, x)

    info = parse_ucr_filename(fname)
    assert info == {
        "index": 7,
        "series_name": "testseries",
        "train_end": 200,
        "anomaly_start": 251,
        "anomaly_end": 260,
    }
    ds = load_ucr(7, data_dir=tmp_path)
    assert len(ds.train) == 200 and len(ds.test) == 100
    np.testing.assert_array_equal(np.flatnonzero(ds.labels), np.arange(50, 60))
    with pytest.raises(ValueError, match="not a UCR"):
        parse_ucr_filename("random.txt")


def test_tsb_ad_loader(tmp_path):
    root = tmp_path / "tsb_ad_u"
    root.mkdir()
    fname = "001_NAB_id_1_Facility_tr_30_1st_40.csv"
    labels = np.zeros(80, dtype=int)
    labels[40:45] = 1
    pd.DataFrame({"value": RNG.normal(size=80), "Label": labels}).to_csv(root / fname, index=False)

    assert parse_train_length(fname) == 30
    ds = load_tsb_ad(fname, variant="u", data_dir=tmp_path)
    assert len(ds.train) == 30 and len(ds.test) == 50
    assert ds.labels.sum() == 5


def test_nab_loader(tmp_path):
    root = tmp_path / "nab"
    (root / "data" / "realX").mkdir(parents=True)
    (root / "labels").mkdir()
    ts = pd.date_range("2024-01-01", periods=100, freq="5min")
    pd.DataFrame({"timestamp": ts, "value": RNG.normal(size=100)}).to_csv(
        root / "data" / "realX" / "s1.csv", index=False
    )
    windows = {"realX/s1.csv": [[str(ts[50]), str(ts[59])]]}
    (root / "labels" / "combined_windows.json").write_text(json.dumps(windows))

    ds = load_nab("realX/s1.csv", data_dir=tmp_path)
    assert len(ds.train) == 15  # probationary 15%
    assert ds.labels.sum() == 10


def test_psm_loader(tmp_path):
    root = tmp_path / "psm"
    root.mkdir()
    train = pd.DataFrame(RNG.normal(size=(60, 3)), columns=["feature_0", "feature_1", "feature_2"])
    train.iloc[5, 1] = np.nan  # 결측 보간 확인
    train.insert(0, "timestamp_(min)", range(60))
    train.to_csv(root / "train.csv", index=False)
    test = pd.DataFrame(RNG.normal(size=(40, 3)), columns=["feature_0", "feature_1", "feature_2"])
    test.insert(0, "timestamp_(min)", range(40))
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"timestamp_(min)": range(40), "label": [0] * 30 + [1] * 10}).to_csv(
        root / "test_label.csv", index=False
    )

    ds = load_psm(data_dir=tmp_path)
    assert ds.train.shape == (60, 3)
    assert not np.isnan(ds.train).any()
    assert ds.labels.sum() == 10


def test_skab_loader(tmp_path):
    root = tmp_path / "skab"
    (root / "anomaly-free").mkdir(parents=True)
    (root / "valve1").mkdir()

    def _frame(n, with_labels):
        df = pd.DataFrame(RNG.normal(size=(n, 8)), columns=SENSOR_COLS)
        df.insert(0, "datetime", pd.date_range("2020-01-01", periods=n, freq="s"))
        if with_labels:
            df["anomaly"] = [0.0] * (n - 5) + [1.0] * 5
            df["changepoint"] = 0.0
        return df

    _frame(50, False).to_csv(root / "anomaly-free" / "anomaly-free.csv", sep=";", index=False)
    _frame(30, True).to_csv(root / "valve1" / "0.csv", sep=";", index=False)

    ds = load_skab("valve1/0", data_dir=tmp_path)
    assert ds.train.shape == (50, 8) and ds.test.shape == (30, 8)
    assert ds.labels.sum() == 5


def test_yahoo_loader(tmp_path):
    root = tmp_path / "yahoo_s5" / "A1Benchmark"
    root.mkdir(parents=True)
    labels = np.zeros(100, dtype=int)
    labels[80:85] = 1
    pd.DataFrame(
        {"timestamp": range(100), "value": RNG.normal(size=100), "is_anomaly": labels}
    ).to_csv(root / "real_1.csv", index=False)

    ds = load_yahoo(data_dir=tmp_path)
    assert len(ds.train) == 50 and len(ds.test) == 50
    assert ds.labels.sum() == 5


def test_swat_loader(tmp_path):
    root = tmp_path / "swat"
    root.mkdir()
    pd.DataFrame(
        {"Timestamp": range(50), "s1": RNG.normal(size=50), "s2": RNG.normal(size=50)}
    ).to_csv(root / "train.csv", index=False)
    pd.DataFrame(
        {
            "Timestamp": range(40),
            "s1": RNG.normal(size=40),
            "s2": RNG.normal(size=40),
            "Normal/Attack": ["Normal"] * 30 + ["Attack"] * 10,
        }
    ).to_csv(root / "test.csv", index=False)

    ds = load_swat(data_dir=tmp_path)
    assert ds.train.shape == (50, 2) and ds.test.shape == (40, 2)
    assert ds.labels.sum() == 10


def test_missing_data_message_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="tsad-forge download smd"):
        load_smd("machine-1-1", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="yahoo_s5.md"):
        load_yahoo(data_dir=tmp_path)
