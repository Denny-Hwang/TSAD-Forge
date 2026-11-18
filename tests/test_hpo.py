"""HPO 프로토콜 테스트: 탐색 공간, 겹침 거부, 스터디 end-to-end, tuned 프로파일 분리."""

import pytest
import yaml

from tsad_forge.models.cards import get_card
from tsad_forge.runner.hpo import (
    accepted_params,
    build_trial_jobs,
    check_disjoint,
    load_selected,
    load_study,
    sample_trials,
    select_best,
    validate_space,
    write_selected,
)
from tsad_forge.runner.matrix import build_jobs, load_profile, run_matrix
from tsad_forge.runner.results import load_all_results


def test_sample_trials_deterministic_with_default_first():
    space = {"window": [10, 20, 50], "k": {"int": [1, 9]}, "lr": {"log_uniform": [1e-4, 1e-2]}}
    a, b = sample_trials(space, 8, seed=3), sample_trials(space, 8, seed=3)
    assert a == b and len(a) == 8
    assert a[0] == {}  # trial 0 = 기본값
    for t in a[1:]:
        assert t["window"] in (10, 20, 50) and 1 <= t["k"] <= 9 and 1e-4 <= t["lr"] <= 1e-2
    assert sample_trials(space, 8, seed=4) != a
    with pytest.raises(ValueError, match="bad search-space"):
        sample_trials({"x": {"beta": [1, 2]}}, 3, 0)


def test_validate_space_catches_typos():
    assert {"window", "var_ratio"} <= accepted_params("sub_pca")
    if _has_torch():  # TorchDetector 계층의 학습 인자(lr)까지 따라 올라간다
        assert {"window", "hidden", "lr"} <= accepted_params("lstm_p")
    validate_space("sub_pca", {"window": [10]})
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        validate_space("sub_pca", {"windw": [10]})


def _has_torch():
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def test_tuning_must_not_overlap_evaluation():
    ev = [{"data": "smd", "data_params": {"machine": "machine-1-1"}}]
    check_disjoint([{"data": "smd", "data_params": {"machine": "machine-1-2"}}], ev)
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint([{"data": "smd", "data_params": {"machine": "machine-1-1"}}], ev)
    with pytest.raises(ValueError, match="seed explicitly"):
        check_disjoint([{"data": "synthetic"}], ev)


def _write_study(tmp_path):
    eval_profile = {
        "results_dir": str(tmp_path / "eval"),
        "defaults": {"normalize": "zscore"},
        "models": ["sub_pca", "zscore"],
        "datasets": ["synthetic"],
    }
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump(eval_profile))
    study = {
        "name": "t",
        "eval_profile": str(tmp_path / "eval.yaml"),
        "spaces": {"sub_pca": {"window": [10, 25, 80], "var_ratio": [0.5, 0.9]}},
        "models": ["sub_pca"],
        "n_trials": 4,
        "seed": 0,
        "metric": "vus_pr",
        "results_dir": str(tmp_path / "hpo"),
        "output": str(tmp_path / "selected.yaml"),
        "tuning_datasets": [
            {"data": "synthetic", "data_params": {"seed": 11}},
            {"data": "synthetic", "data_params": {"seed": 12, "n_dims": 3}},
        ],
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    return path


def test_study_end_to_end_and_tuned_profile(tmp_path):
    study = load_study(_write_study(tmp_path))
    jobs, trials = build_trial_jobs(study)
    assert len(jobs) == 4 * 2  # trials × tuning entities
    assert {j.tag[0] for j in jobs} == {f"sub_pca#t{k:02d}" for k in range(4)}
    summary = run_matrix(jobs, results_dir=study["results_dir"])
    assert summary.counts["done"] == 8 and not summary.failed

    selected = select_best(study, trials)
    s = selected["sub_pca"]
    assert s["n_trials_completed"] == 4
    assert s["tuning_vus_pr"] >= s["default_vus_pr"]  # trial 0이 기본값이므로 항상 성립
    out = write_selected(study, selected)
    doc = yaml.safe_load(out.read_text())
    assert doc["_provenance"]["n_trials_budget"] == 4
    assert load_selected(out) == {"sub_pca": s["params"]}

    # 평가 프로파일 상속 + tuned_params → sub_pca@tuned는 별도 행, zscore는 기본값 그대로
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(yaml.safe_dump({"extends": "eval.yaml", "tuned_params": str(out)}))
    profile = load_profile(tuned)
    assert profile["defaults"] == {"normalize": "zscore"}
    rd = tmp_path / "eval"
    run_matrix(build_jobs(load_profile(tmp_path / "eval.yaml"), str(rd))[0], results_dir=rd)
    run_matrix(build_jobs(profile, str(rd))[0], results_dir=rd)
    models = set(load_all_results(rd)["model"])
    expected = {"sub_pca", "zscore"} | ({"sub_pca@tuned"} if s["params"] else set())
    assert models == expected
    assert get_card("sub_pca@tuned")["fidelity"] == get_card("sub_pca")["fidelity"]


def test_shipped_studies_validate():
    for f in ("configs/hpo/lite.yaml", "configs/hpo/dl_gpu.yaml"):
        if "dl_gpu" in f and not _has_torch():
            continue
        study = load_study(f)
        assert study["models"] and study["_tuning"]
