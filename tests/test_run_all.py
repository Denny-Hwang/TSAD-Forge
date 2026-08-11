"""benchmarks/run_all.py 매트릭스 실행 + resume 스모크."""

import importlib.util
import sys
from pathlib import Path

import yaml

from tsad_forge.runner.results import load_all_results

ROOT = Path(__file__).resolve().parents[1]


def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "benchmarks" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lite_matrix_runs_and_resumes(tmp_path):
    profile = {
        "results_dir": str(tmp_path / "results"),
        "seeds": [0, 1],
        "defaults": {"normalize": "zscore"},
        # dummy는 stochastic 표시 → 시드 2개, zscore는 결정적 → 시드 1개
        "models": [{"model": "dummy", "stochastic": True}, "zscore"],
        "datasets": ["synthetic", {"data": "synthetic", "data_params": {"n_dims": 3}}],
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))

    run_all = _load_run_all()
    assert run_all.main(["--profile", str(profile_path)]) == 0

    df = load_all_results(tmp_path / "results")
    # dummy: 2 datasets x 2 seeds = 4, zscore: 2 datasets x 1 seed = 2
    assert df.groupby(["model", "dataset", "seed", "config_hash"]).ngroups == 6

    # 재실행 → 전부 resume skip, 결과 파일 수 불변
    n_files = len(list((tmp_path / "results").glob("*.parquet")))
    assert run_all.main(["--profile", str(profile_path)]) == 0
    assert len(list((tmp_path / "results").glob("*.parquet"))) == n_files


def test_missing_data_is_skipped_not_failed(tmp_path, capsys):
    profile = {
        "results_dir": str(tmp_path / "results"),
        "models": ["zscore"],
        "datasets": [
            "synthetic",
            {"data": "smd", "data_params": {"machine": "machine-1-1", "data_dir": str(tmp_path)}},
        ],
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    run_all = _load_run_all()
    assert run_all.main(["--profile", str(profile_path)]) == 0  # 실패 아님
    out = capsys.readouterr().out
    assert "[no-data]" in out and "done=1" in out
