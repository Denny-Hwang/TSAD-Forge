"""매트릭스 스케줄러 테스트: entity 확장, 필터, 격리/병렬 실행, 타임아웃·크래시·오류
기록, 플러그인, dry-run.
"""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from tsad_forge.data.entities import list_entities
from tsad_forge.runner.matrix import (
    build_jobs,
    expand_combos,
    load_failures,
    record_failure,
    run_matrix,
)
from tsad_forge.runner.results import load_all_results

ROOT = Path(__file__).resolve().parents[1]

PLUGIN_SRC = textwrap.dedent("""
    import os
    import time

    import numpy as np

    from tsad_forge.models.base import BaseDetector
    from tsad_forge.models.registry import register_model


    class _Base(BaseDetector):
        generation = "gen0"

        def fit(self, X):
            self._fitted = True
            return self

        def score(self, X):
            return np.abs(np.asarray(X)).sum(axis=1)


    @register_model("_plug_slow")
    class Slow(_Base):
        def fit(self, X):
            time.sleep(60)
            return super().fit(X)


    @register_model("_plug_crash")
    class Crash(_Base):
        def fit(self, X):
            os._exit(3)


    @register_model("_plug_error")
    class Error(_Base):
        def fit(self, X):
            raise RuntimeError("boom")


    @register_model("_plug_ok")
    class Ok(_Base):
        pass
    """)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """tmp에 플러그인 모듈을 만들고 sys.path에 올린다 (spawn 워커도 sys.path를 물려받는다)."""
    pkg_dir = tmp_path / "plugins"
    pkg_dir.mkdir()
    (pkg_dir / "tsadf_test_plugin.py").write_text(PLUGIN_SRC)
    monkeypatch.syspath_prepend(str(pkg_dir))
    return "tsadf_test_plugin"


# --- entity 확장 -------------------------------------------------------------


def _fake_data(root: Path) -> None:
    (root / "smd" / "train").mkdir(parents=True)
    for m in ("machine-1-2", "machine-1-1", "machine-2-1"):
        (root / "smd" / "train" / f"{m}.txt").write_text("0\n")
    (root / "ucr").mkdir()
    for n in ("002_UCR_Anomaly_B_10_20_30.txt", "001_UCR_Anomaly_A_10_20_30.txt", "notes.txt"):
        (root / "ucr" / n).write_text("0\n")
    (root / "mgab").mkdir()
    for n in ("10.csv", "2.csv", "readme.csv"):
        (root / "mgab" / n).write_text("0\n")


def test_list_entities_is_sorted_and_local(tmp_path):
    _fake_data(tmp_path)
    assert list_entities("smd", tmp_path) == [
        {"machine": "machine-1-1"},
        {"machine": "machine-1-2"},
        {"machine": "machine-2-1"},
    ]
    assert list_entities("ucr", tmp_path) == [{"series": 1}, {"series": 2}]
    assert list_entities("mgab", tmp_path) == [{"series": 2}, {"series": 10}]
    assert list_entities("smap", tmp_path) == []  # 다운로드 안 됨 → 빈 목록
    with pytest.raises(KeyError):
        list_entities("psm", tmp_path)


def test_profile_entities_all_with_limit(tmp_path, monkeypatch):
    _fake_data(tmp_path)
    monkeypatch.setenv("TSAD_FORGE_DATA", str(tmp_path))
    profile = {
        "seeds": [0, 1],
        "models": ["zscore", {"model": "dummy", "stochastic": True}],
        "datasets": [
            "synthetic",
            {"data": "smd", "entities": "all", "limit": 2},
            {"data": "smap", "entities": "all"},
        ],
    }
    combos = expand_combos(profile)
    smd = [c for c in combos if c["data"] == "smd"]
    # zscore 1시드 × 2머신 + dummy 2시드 × 2머신
    assert len(smd) == 6
    assert {c["data_params"]["machine"] for c in smd} == {"machine-1-1", "machine-1-2"}
    jobs, empty = build_jobs(profile, str(tmp_path / "res"))
    assert empty == ["smap"]
    with pytest.raises(ValueError, match="does not support"):
        expand_combos({"models": ["zscore"], "datasets": [{"data": "psm", "entities": "all"}]})


def test_build_jobs_filters_and_lanes(tmp_path):
    profile = {
        "seeds": [0, 1],
        "models": ["zscore", {"model": "dummy", "stochastic": True}],
        "datasets": ["synthetic", {"data": "synthetic", "data_params": {"n_dims": 3}}],
    }
    jobs, _ = build_jobs(profile, str(tmp_path), models=["dummy"], seeds=[1])
    assert len(jobs) == 2 and all(j.tag[0] == "dummy" and j.tag[3] == 1 for j in jobs)
    assert {j.lane for j in jobs} == {"cpu"}
    assert len({j.run_id for j in jobs}) == 2


# --- 격리/병렬 실행 ----------------------------------------------------------


def test_isolated_parallel_matches_in_process(tmp_path):
    profile = {
        "models": ["zscore", "sub_pca"],
        "datasets": ["synthetic", {"data": "synthetic", "data_params": {"n_dims": 3}}],
    }
    a, b = tmp_path / "inproc", tmp_path / "isolated"
    s1 = run_matrix(build_jobs(profile, str(a))[0], results_dir=a)
    s2 = run_matrix(build_jobs(profile, str(b))[0], results_dir=b, n_jobs=2)
    assert s1.counts["done"] == s2.counts["done"] == 4 and not s2.failed

    def _vals(d):
        df = load_all_results(d)
        df = df[df["metric"] == "vus_pr"]
        return df.set_index(["model", "data_hash"])["value"].sort_index()

    np.testing.assert_allclose(_vals(a).to_numpy(), _vals(b).to_numpy())

    # 재실행: 전부 resume
    s3 = run_matrix(build_jobs(profile, str(b))[0], results_dir=b, n_jobs=2)
    assert s3.counts["skipped"] == 4 and s3.counts["done"] == 0


def test_timeout_crash_error_are_recorded(tmp_path, plugin):
    profile = {
        "plugins": [plugin],
        "models": ["_plug_slow", "_plug_crash", "_plug_error", "_plug_ok"],
        "datasets": ["synthetic"],
    }
    rd = tmp_path / "res"
    jobs, _ = build_jobs(profile, str(rd))
    summary = run_matrix(jobs, results_dir=rd, n_jobs=4, timeout=5, plugins=[plugin])
    kinds = {tag[0]: kind for tag, kind in summary.failed}
    assert kinds == {"_plug_slow": "timeout", "_plug_crash": "crashed", "_plug_error": "error"}
    assert summary.counts["done"] == 1
    records = {r["config"]["model"]: r for r in load_failures(rd)}
    assert set(records) == set(kinds)
    assert "boom" in records["_plug_error"]["detail"]
    assert not list(rd.glob("_plug_slow*.parquet"))


def test_in_process_errors_are_recorded_and_cleared_on_success(tmp_path, plugin):
    rd = tmp_path / "res"
    profile = {"plugins": [plugin], "models": ["_plug_error"], "datasets": ["synthetic"]}
    summary = run_matrix(build_jobs(profile, str(rd))[0], results_dir=rd)
    assert summary.failed and load_failures(rd)[0]["kind"] == "error"

    # 같은 run_id가 이후 성공(여기선 resume skip으로 모사)하면 실패 기록이 지워진다
    ok_profile = {"models": ["zscore"], "datasets": ["synthetic"]}
    (job,) = build_jobs(ok_profile, str(rd))[0]
    record_failure(rd, job, "error", "stale")
    run_matrix([job], results_dir=rd)
    assert not any(f["run_id"] == job.run_id for f in load_failures(rd))


# --- CLI ---------------------------------------------------------------------


def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "benchmarks" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_all_dry_run_and_filters(tmp_path, capsys):
    profile = {
        "results_dir": str(tmp_path / "res"),
        "models": ["zscore", "dummy"],
        "datasets": ["synthetic"],
    }
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(profile))
    run_all = _load_run_all()
    assert run_all.main(["--profile", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "combos=2" in out and "pending" in out
    assert not (tmp_path / "res").exists()  # dry-run은 아무것도 실행하지 않는다

    assert run_all.main(["--profile", str(path), "--models", "zscore", "--jobs", "2"]) == 0
    out = capsys.readouterr().out
    assert "done=1" in out
    summary = json.loads(next((tmp_path / "res").glob("zscore*.json")).read_text())
    assert summary["model"] == "zscore"


def test_shipped_profiles_expand(tmp_path, monkeypatch):
    monkeypatch.setenv("TSAD_FORGE_DATA", str(tmp_path))  # 로컬 데이터 없음
    for name in ("lite", "lite_dl", "lite_dl_gpu", "mamba_paper_repro", "full"):
        profile = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        assert isinstance(expand_combos(profile), list), name
