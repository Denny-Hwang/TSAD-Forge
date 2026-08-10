"""End-to-end: CLI → runner → results parquet/JSON/scores + resume (M0 DoD)."""

import json

import numpy as np
import pandas as pd
import pytest

from tsad_forge.cli import main
from tsad_forge.runner.results import load_all_results


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


def _run(results_dir, *extra):
    return main(
        [
            "run",
            "--model",
            "dummy",
            "--data",
            "synthetic",
            "--results-dir",
            str(results_dir),
            *extra,
        ]
    )


def test_cli_run_end_to_end(results_dir):
    assert _run(results_dir) == 0

    parquets = list(results_dir.glob("*.parquet"))
    jsons = list(results_dir.glob("*.json"))
    scores = list((results_dir / "scores").glob("*.npz"))
    assert len(parquets) == 1 and len(jsons) == 1 and len(scores) == 1

    df = pd.read_parquet(parquets[0])
    expected_cols = {
        "model",
        "generation",
        "dataset",
        "channel",
        "seed",
        "metric",
        "value",
        "runtime_s",
        "peak_vram_mb",
        "commit_hash",
        "config_hash",
        "timestamp",
    }
    assert expected_cols == set(df.columns)
    assert (df["model"] == "dummy").all()
    assert "auc_roc" in set(df["metric"])

    summary = json.loads(jsons[0].read_text())
    assert summary["config"]["model"] == "dummy"
    assert summary["metrics"]["auc_roc"] == pytest.approx(
        df.loc[df["metric"] == "auc_roc", "value"].iloc[0]
    )

    # 스코어 원본에서 지표 재계산 가능 (CLAUDE.md §4)
    data = np.load(scores[0])
    assert data["scores"].shape == data["labels"].shape


def test_resume_skips_and_force_reruns(results_dir, capsys):
    _run(results_dir)
    n_files = len(list(results_dir.glob("*.parquet")))

    _run(results_dir)  # 동일 조합 → resume skip
    assert "[resume] skip" in capsys.readouterr().out
    assert len(list(results_dir.glob("*.parquet"))) == n_files

    _run(results_dir, "--force")
    assert "[resume]" not in capsys.readouterr().out


def test_different_seed_is_new_run(results_dir):
    _run(results_dir)
    _run(results_dir, "--seed", "1")
    df = load_all_results(results_dir)
    assert set(df["seed"]) == {0, 1}


def test_legacy_pa_flag_warns(results_dir):
    with pytest.warns(UserWarning, match="PA-F1"):
        _run(results_dir, "--legacy-pa")
    df = load_all_results(results_dir)
    assert "pa_f1" in set(df["metric"])


def test_byod_csv_with_labels(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    values = rng.normal(size=n)
    labels = np.zeros(n, dtype=int)
    values[300:310] += 8.0
    labels[300:310] = 1
    csv = tmp_path / "mydata.csv"
    pd.DataFrame({"timestamp": np.arange(n), "value": values, "label": labels}).to_csv(
        csv, index=False
    )

    out = tmp_path / "results"
    assert main(["run", "--model", "zscore", "--data", str(csv), "--results-dir", str(out)]) == 0
    df = load_all_results(out)
    auc = df.loc[df["metric"] == "auc_roc", "value"].iloc[0]
    assert auc > 0.95  # 명백한 스파이크는 zscore가 잡아야 함


def test_byod_csv_without_labels(tmp_path):
    csv = tmp_path / "nolabel.csv"
    pd.DataFrame({"value": np.random.default_rng(0).normal(size=200)}).to_csv(csv, index=False)
    out = tmp_path / "results"
    assert main(["run", "--model", "zscore", "--data", str(csv), "--results-dir", str(out)]) == 0
    df = load_all_results(out)
    assert "auc_roc" not in set(df["metric"])  # 라벨 없으면 스코어/임계값만
    assert "threshold" in set(df["metric"])


def test_list_commands(capsys):
    assert main(["list", "models"]) == 0
    assert "dummy" in capsys.readouterr().out
    assert main(["list", "datasets"]) == 0
    assert "synthetic" in capsys.readouterr().out
