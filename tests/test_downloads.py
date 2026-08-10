"""실제 다운로드 → 로드 → 스키마 검증 (M1 DoD).

네트워크가 필요하므로 로컬 data/가 이미 있으면 그것으로 검증하고,
없으면 TSAD_FORGE_NETWORK_TESTS=1일 때만 실제 다운로드한다 (CI 기본 skip).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tsad_forge.data.download import download_dataset, sha256sum, verify_checksum
from tsad_forge.data.registry import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NETWORK = os.environ.get("TSAD_FORGE_NETWORK_TESTS") == "1"


def _ensure(name: str, marker: Path, **kwargs) -> None:
    if marker.exists():
        return
    if not NETWORK:
        pytest.skip(f"{name} data not present and TSAD_FORGE_NETWORK_TESTS != 1")
    download_dataset(name, data_dir=DATA_DIR, **kwargs)


def _check_schema(ds):
    assert ds.train.ndim == 2 and ds.test.ndim == 2
    assert ds.train.shape[1] == ds.test.shape[1]
    assert len(ds.labels) == len(ds.test)
    assert set(np.unique(ds.labels)) <= {0, 1}
    assert ds.labels.sum() > 0, "test 구간에 이상이 있어야 함"
    assert np.isfinite(ds.train).all() and np.isfinite(ds.test).all()
    for key in ("name", "source_url", "license", "citation"):
        assert ds.meta.get(key), f"meta['{key}'] missing"
    assert ds.meta["n_events"] >= 1


def test_smd_download_load_schema():
    _ensure("smd", DATA_DIR / "smd" / "train" / "machine-1-1.txt", subset=["machine-1-1"])
    ds = load_dataset("smd", machine="machine-1-1", data_dir=DATA_DIR)
    _check_schema(ds)
    assert ds.train.shape[1] == 38


def test_nab_download_load_schema():
    rel = "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    _ensure("nab", DATA_DIR / "nab" / "data" / rel)
    ds = load_dataset("nab", rel_path=rel, data_dir=DATA_DIR)
    _check_schema(ds)
    assert ds.n_dims == 1


def test_psm_download_load_schema():
    _ensure("psm", DATA_DIR / "psm" / "train.csv")
    ds = load_dataset("psm", data_dir=DATA_DIR)
    _check_schema(ds)
    assert ds.train.shape[1] == 25


def test_skab_download_load_schema():
    _ensure("skab", DATA_DIR / "skab" / "valve1" / "0.csv", subset=["valve1"])
    ds = load_dataset("skab", experiment="valve1/0", data_dir=DATA_DIR)
    _check_schema(ds)
    assert ds.train.shape[1] == 8


def test_manifest_checksums_verify():
    """다운로드된 파일이 MANIFEST.json의 sha256과 일치하는지 검증."""
    import json

    manifest_f = DATA_DIR / "psm" / "MANIFEST.json"
    if not manifest_f.exists():
        pytest.skip("psm not downloaded")
    manifest = json.loads(manifest_f.read_text())
    assert manifest, "manifest is empty"
    name, expected = next(iter(manifest.items()))
    verify_checksum(DATA_DIR / "psm" / name, expected)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum(DATA_DIR / "psm" / name, "0" * 64)


def test_sha256sum_stable(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"tsad-forge")
    assert sha256sum(f) == sha256sum(f)


def test_restricted_datasets_refuse_download(tmp_path):
    for name in ("yahoo_s5", "swat", "wadi"):
        with pytest.raises(SystemExit, match="재배포 금지"):
            download_dataset(name, data_dir=tmp_path)
