"""계측 모듈 테스트 (리뷰 P0-1/P0-2) — RSS 샘플러, 결과 dedup, 체크섬 고정."""

import time
import warnings

import numpy as np
import pytest

from tsad_forge.runner.measure import PeakRssSampler
from tsad_forge.runner.results import load_all_results, write_result


def test_peak_rss_sampler_catches_allocation():
    with PeakRssSampler(interval_s=0.01) as mem:
        big = np.ones(25_000_000)  # ~200MB, 샘플 여러 번 돌 동안 유지
        time.sleep(0.1)
        del big
    assert mem.peak_mb > 100  # 200MB 할당의 절반 이상은 관측되어야 함


def test_peak_rss_sampler_short_block_is_nonnegative():
    with PeakRssSampler() as mem:
        pass
    assert mem.peak_mb >= 0.0


def _write(results_dir, cfg_hash, ts_offset=0, value=0.5, data_hash="samedata0000", device=None):
    """동일 (model, dataset, channel, seed) 결과를 다른 config_hash로 기록."""
    import time as _t
    from unittest import mock

    from tsad_forge.runner import results as res

    fake_now = f"2026-01-01T00:00:{ts_offset:02d}+00:00"
    with mock.patch.object(res, "utc_now", return_value=fake_now):
        write_result(
            results_dir,
            {"vus_pr": value},
            model="m",
            generation="gen1",
            dataset="d",
            channel="all",
            seed=0,
            runtime_s=1.0,
            runtime_fit_s=0.6,
            runtime_score_s=0.4,
            peak_mem_mb=10.0,
            data_hash=data_hash,
            cfg_hash=cfg_hash,
            config={} if device is None else {"device": device},
        )
    _t.sleep(0)  # noop — mock 컨텍스트 종료 명시


def test_load_all_results_dedups_superseded_config_hash(tmp_path):
    _write(tmp_path, "oldhash000000", ts_offset=0, value=0.1)
    _write(tmp_path, "newhash000000", ts_offset=1, value=0.9)

    with pytest.warns(UserWarning, match="superseded"):
        df = load_all_results(tmp_path)
    assert len(df) == 1
    assert df["value"].iloc[0] == 0.9  # 최신 timestamp가 남는다
    assert df["config_hash"].iloc[0] == "newhash000000"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw = load_all_results(tmp_path, dedup=False)
    assert len(raw) == 2  # dedup=False면 원본 그대로


def test_dedup_keeps_distinct_data_variants(tmp_path):
    """data_params가 다른 정당한 별개 실험(channel 동일)은 dedup 대상이 아니다."""
    _write(tmp_path, "hashA00000000", ts_offset=0, value=0.1, data_hash="variantA0000")
    _write(tmp_path, "hashB00000000", ts_offset=1, value=0.9, data_hash="variantB0000")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 경고 없이
        df = load_all_results(tmp_path)
    assert len(df) == 2


def test_mps_rows_are_labeled_and_do_not_supersede_cpu(tmp_path):
    """MPS 결과는 `<model>@mps`로 분리되어, 더 최신이어도 CPU 행을 대체하지 않는다."""
    from tsad_forge.runner.results import frame_run_ids

    _write(tmp_path, "cpuhash00000", ts_offset=0, value=0.1, device="cpu")
    _write(tmp_path, "mpshash00000", ts_offset=1, value=0.9, device="mps")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # superseded 경고 없이
        df = load_all_results(tmp_path)
    by_model = df.set_index("model")
    assert set(df["model"]) == {"m", "m@mps"}
    assert by_model.at["m", "value"] == 0.1 and by_model.at["m@mps", "value"] == 0.9
    assert by_model.at["m@mps", "device"] == "mps" and by_model.at["m", "device"] == "cpu"
    assert set(df["run_model"]) == {"m"}  # 파일명의 원래 이름은 그대로
    # run_id 역참조는 접미사 없는 실제 파일명이어야 한다
    for rid in frame_run_ids(df):
        assert (tmp_path / f"{rid}.parquet").exists()


def test_cuda_still_supersedes_cpu(tmp_path):
    """cpu/cuda는 접미사가 없다 — 같은 키의 최신 결과가 대체하는 기존 규칙 유지."""
    _write(tmp_path, "cpuhash00000", ts_offset=0, value=0.1, device="cpu")
    _write(tmp_path, "cudahash0000", ts_offset=1, value=0.9, device="cuda")
    with pytest.warns(UserWarning, match="superseded"):
        df = load_all_results(tmp_path)
    assert list(df["model"]) == ["m"] and df["value"].iloc[0] == 0.9


def test_results_without_json_or_device_are_legacy_cpu(tmp_path):
    _write(tmp_path, "legacyhash00", device=None)  # config에 device 키 없음
    _write(tmp_path, "nojsonhash00", data_hash="otherdata000", device="mps")
    (tmp_path / "m__d__all__seed0__nojsonhash00.json").unlink()  # JSON 없는 parquet
    df = load_all_results(tmp_path)
    assert set(df["device"]) == {"cpu"} and set(df["model"]) == {"m"}


def test_device_label_is_idempotent_on_reload(tmp_path):
    """스냅샷처럼 이미 device/접미사가 붙은 parquet을 다시 읽어도 두 번 붙지 않는다."""
    _write(tmp_path, "mpshash00000", device="mps")
    first = load_all_results(tmp_path)
    out = tmp_path / "again"
    out.mkdir()
    first.to_parquet(out / "snapshot.parquet", index=False)
    again = load_all_results(out)
    assert list(again["model"]) == ["m@mps"] and list(again["run_model"]) == ["m"]


def test_write_manifest_verifies_pinned_checksums(tmp_path, monkeypatch):
    from tsad_forge.data import download as dl

    ds_dir = tmp_path / "mydata"
    ds_dir.mkdir()
    (ds_dir / "a.csv").write_text("hello")

    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    monkeypatch.setattr(dl, "PINNED_CHECKSUMS_DIR", pins_dir)

    # 고정 체크섬 없음 → 기록만 하고 통과
    dl._write_manifest(ds_dir)

    # 일치하는 고정 체크섬 → 통과
    good = dl.sha256sum(ds_dir / "a.csv")
    (pins_dir / "mydata.json").write_text(f'{{"a.csv": "{good}"}}')
    dl._write_manifest(ds_dir)

    # 불일치 → 명확한 에러
    (pins_dir / "mydata.json").write_text('{"a.csv": "deadbeef"}')
    with pytest.raises(RuntimeError, match="checksum mismatch against pinned"):
        dl._write_manifest(ds_dir)

    # 고정 목록에 없는 파일(subset/신규)은 대조 대상 아님
    (pins_dir / "mydata.json").write_text('{"other.csv": "deadbeef"}')
    dl._write_manifest(ds_dir)
