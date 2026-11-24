"""결과 스냅샷: pack → verify → 변조 감지 → unpack → 집계/리더보드가 그대로 읽는다."""

import json
import zipfile

import pytest

from tsad_forge.cli import main
from tsad_forge.runner.experiment import run_experiment
from tsad_forge.runner.results import load_all_results
from tsad_forge.runner.snapshot import (
    _sha256,
    fetch_snapshot,
    pack_snapshot,
    unpack_snapshot,
    verify_snapshot,
)
from tsad_forge.viz.leaderboard import build_leaderboard


@pytest.fixture
def results(tmp_path):
    rd = tmp_path / "results"
    for model in ("zscore", "sub_pca", "dummy"):
        run_experiment({"model": model, "results_dir": str(rd)}, verbose=False)
    return rd


def test_pack_verify_unpack_roundtrip(results, tmp_path):
    snap = pack_snapshot(results, tmp_path / "snaps", "lite-test", with_scores=True)
    manifest = json.loads((snap / "MANIFEST.json").read_text())
    assert manifest["counts"]["runs"] == 3 and manifest["counts"]["models"] == 3
    assert set(manifest["files"]) == {"results.parquet", "runs.jsonl", "scores-000.zip"}
    assert manifest["score_parts"] == ["scores-000.zip"] and manifest["counts"]["scores"] == 3
    assert verify_snapshot(snap) == []

    dest = unpack_snapshot(snap, tmp_path / "restored")
    orig, restored = load_all_results(results), load_all_results(dest)
    cols = ["model", "metric", "value", "config_hash"]
    key = ["model", "metric"]
    assert (
        orig[cols]
        .sort_values(key)
        .reset_index(drop=True)
        .equals(restored[cols].sort_values(key).reset_index(drop=True))
    )
    assert len(list((dest / "scores").glob("*.npz"))) == 3
    _, summary = build_leaderboard(dest)  # 리더보드가 스냅샷을 그대로 읽는다
    assert len(summary) == 3


def test_tampering_is_detected(results, tmp_path):
    snap = pack_snapshot(results, tmp_path / "snaps", "s1")
    with open(snap / "runs.jsonl", "a") as f:
        f.write("{}\n")
    assert verify_snapshot(snap) == ["checksum mismatch: runs.jsonl"]
    with pytest.raises(ValueError, match="verification"):
        unpack_snapshot(snap, tmp_path / "x")


def test_snapshots_are_immutable(results, tmp_path):
    pack_snapshot(results, tmp_path / "snaps", "s1")
    with pytest.raises(FileExistsError):
        pack_snapshot(results, tmp_path / "snaps", "s1")


def _rehash(snap, fname):
    """파일을 고친 뒤 MANIFEST 해시를 맞춘다 (checksum 단계 뒤의 검사를 시험하려고)."""
    m = json.loads((snap / "MANIFEST.json").read_text())
    p = snap / fname
    m["files"][fname] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    (snap / "MANIFEST.json").write_text(json.dumps(m))


def test_zip_slip_rejected(results, tmp_path):
    snap = pack_snapshot(results, tmp_path / "snaps", "s1", with_scores=True)
    with zipfile.ZipFile(snap / "scores-000.zip", "a") as z:
        z.writestr("../evil.npz", b"x")
    # 매니페스트 해시를 맞춰 checksum 단계를 통과시킨 뒤에도 경로 검사로 막혀야 한다
    _rehash(snap, "scores-000.zip")
    with pytest.raises(ValueError, match="unexpected entry"):
        unpack_snapshot(snap, tmp_path / "x")
    assert not (tmp_path / "x").exists()  # 검사가 끝나기 전에는 아무것도 풀지 않는다


def _one_file_per_part(results):
    """npz 하나만 들어가고 둘은 못 들어가는 part 크기."""
    from tsad_forge.runner.snapshot import _ZIP_ENTRY, _ZIP_EOCD

    npz = sorted((results / "scores").glob("*.npz"))
    return max(p.stat().st_size + _ZIP_ENTRY + 2 * len(p.name) for p in npz) + _ZIP_EOCD


def test_scores_split_into_parts_under_limit(results, tmp_path):
    """Release 자산 한도 대응: 스코어가 part_bytes 이하 zip 여러 개로 나뉘고 그대로 복원된다."""
    part_bytes = _one_file_per_part(results)
    snap = pack_snapshot(
        results, tmp_path / "snaps", "split", with_scores=True, part_bytes=part_bytes
    )
    manifest = json.loads((snap / "MANIFEST.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["score_parts"] == ["scores-000.zip", "scores-001.zip", "scores-002.zip"]
    for name in manifest["score_parts"]:
        assert (snap / name).stat().st_size <= part_bytes
        assert name in manifest["files"]
    assert verify_snapshot(snap) == []

    dest = unpack_snapshot(snap, tmp_path / "restored")
    for orig in (results / "scores").glob("*.npz"):
        assert (dest / "scores" / orig.name).read_bytes() == orig.read_bytes()


def test_part_size_too_small_for_one_file(results, tmp_path):
    with pytest.raises(ValueError, match="alone exceeds"):
        pack_snapshot(results, tmp_path / "snaps", "tiny", with_scores=True, part_bytes=100)


def test_missing_part_is_detected(results, tmp_path):
    part_bytes = _one_file_per_part(results)
    snap = pack_snapshot(results, tmp_path / "snaps", "s1", with_scores=True, part_bytes=part_bytes)
    (snap / "scores-001.zip").unlink()
    assert verify_snapshot(snap) == ["missing: scores-001.zip"]
    with pytest.raises(ValueError, match="verification"):
        unpack_snapshot(snap, tmp_path / "x")


def test_duplicate_entry_across_parts_rejected(results, tmp_path):
    part_bytes = _one_file_per_part(results)
    snap = pack_snapshot(results, tmp_path / "snaps", "s1", with_scores=True, part_bytes=part_bytes)
    with zipfile.ZipFile(snap / "scores-000.zip") as z:
        name = z.namelist()[0]
        data = z.read(name)
    with zipfile.ZipFile(snap / "scores-001.zip", "a") as z:
        z.writestr(name, data)
    _rehash(snap, "scores-001.zip")
    with pytest.raises(ValueError, match="duplicate"):
        unpack_snapshot(snap, tmp_path / "x")


def test_format1_single_scores_zip_still_unpacks(results, tmp_path):
    """format 1 스냅샷(단일 scores.zip, counts.scores 없음)도 그대로 읽는다."""
    snap = pack_snapshot(results, tmp_path / "snaps", "v1", with_scores=True)
    (snap / "scores-000.zip").rename(snap / "scores.zip")
    m = json.loads((snap / "MANIFEST.json").read_text())
    m["format_version"] = 1
    m["files"]["scores.zip"] = m["files"].pop("scores-000.zip")
    del m["score_parts"], m["counts"]["scores"]
    (snap / "MANIFEST.json").write_text(json.dumps(m))
    assert verify_snapshot(snap) == []
    dest = unpack_snapshot(snap, tmp_path / "restored")
    assert len(list((dest / "scores").glob("*.npz"))) == 3


def test_fetch_requires_https(tmp_path):
    with pytest.raises(ValueError, match="https"):
        fetch_snapshot("http://example.invalid/releases/download/v1", tmp_path)


def test_cli_pack_and_verify(results, tmp_path, capsys):
    assert (
        main(
            [
                "results",
                "pack",
                "--results-dir",
                str(results),
                "--name",
                "c1",
                "--out",
                str(tmp_path / "snaps"),
                "--with-scores",
                "--part-size-mb",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "gh release upload" in out and "scores:   1 part(s)" in out
    assert main(["results", "verify", str(tmp_path / "snaps" / "c1")]) == 0


def test_snapshot_keeps_mps_runs_and_labels(tmp_path):
    """`@mps` 행의 실행 JSON도 runs.jsonl에 들어가고, 복원 후 이름이 유지된다."""
    from tsad_forge.runner.results import write_result

    rd = tmp_path / "results"
    for device, h in (("cpu", "cpuhash00000"), ("mps", "mpshash00000")):
        write_result(
            rd,
            {"vus_pr": 0.5},
            model="m",
            generation="gen5",
            dataset="d",
            channel="all",
            seed=0,
            runtime_s=1.0,
            runtime_fit_s=0.5,
            runtime_score_s=0.5,
            peak_mem_mb=1.0,
            data_hash="data00000000",
            cfg_hash=h,
            config={"device": device},
        )
    snap = pack_snapshot(rd, tmp_path / "snaps", "mps-test")
    runs = [json.loads(line) for line in (snap / "runs.jsonl").read_text().splitlines()]
    assert {r["config_hash"] for r in runs} == {"cpuhash00000", "mpshash00000"}
    restored = load_all_results(unpack_snapshot(snap, tmp_path / "restored"))
    assert set(restored["model"]) == {"m", "m@mps"}
    # 리더보드의 재현 링크는 `@mps` 행도 실제 파일명(run_id)을 가리킨다
    _, summary = build_leaderboard(rd)
    links = dict(zip(summary.index.get_level_values("model"), summary["example_run"], strict=True))
    assert links["m@mps"] == "m__d__all__seed0__mpshash00000"
    assert (rd / f"{links['m@mps']}.json").exists()
