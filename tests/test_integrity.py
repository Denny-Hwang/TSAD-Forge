"""결과 무결성 테스트 (리뷰 P0): 해시 하위호환, 실행 장치, 정규화, 점수 건전성,
VUS 윈도우 자동 추정, 공정 집계(공통 entity / 정규화 순위), CLI config 우선순위.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tsad_forge.cli import main
from tsad_forge.evaluation.protocol import (
    DEFAULT_PERIOD_FALLBACK,
    ScoreHealthError,
    check_scores,
    estimate_sliding_window,
    zscore_normalize,
)
from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model
from tsad_forge.runner.environment import environment_fingerprint, resolve_device
from tsad_forge.runner.experiment import resolve_config, run_experiment
from tsad_forge.runner.results import (
    add_entity_column,
    experiment_hash,
    normalized_ranks,
    shared_entities,
)
from tsad_forge.viz.charts import generation_shared_entities

# --- 해시 하위호환 -----------------------------------------------------------

#: benchmarks/results/에 커밋된 실제 결과의 config_hash (dummy, synthetic, seed 0).
#: 신규 config 키 도입 후에도 이 값이 유지되어야 기존 결과가 resume/dedup에서
#: 같은 실험으로 인식된다. 이 테스트가 깨지면 HASH_EXEMPT_DEFAULTS에 새 키의
#: 레거시 값을 등록하라.
GOLDEN_HASHES = {
    "dummy/synthetic": "ca2c02bf2522",
    "dummy/synthetic(n_dims=5,n_events=8)": "4551a546ceb8",
}


def _cpu_cfg(**overrides):
    cfg = resolve_config({"results_dir": "x", **overrides})
    cfg["device"] = resolve_device(cfg["device"], uses_torch=False)
    return cfg


def test_default_config_hash_matches_committed_results():
    assert experiment_hash(_cpu_cfg(model="dummy")) == GOLDEN_HASHES["dummy/synthetic"]
    cfg = _cpu_cfg(model="dummy", data_params={"n_dims": 5, "n_events": 8})
    assert experiment_hash(cfg) == GOLDEN_HASHES["dummy/synthetic(n_dims=5,n_events=8)"]


def test_new_keys_change_hash_only_when_non_legacy():
    base = experiment_hash(_cpu_cfg(model="dummy"))
    assert experiment_hash({**_cpu_cfg(model="dummy"), "device": "cuda"}) != base
    assert experiment_hash(_cpu_cfg(model="dummy", normalize_params={"std_floor": 0.01})) != base
    assert experiment_hash(_cpu_cfg(model="dummy", sliding_window="auto")) != base
    # 실행 부수 키는 정체성에 들어가지 않는다
    assert experiment_hash({**_cpu_cfg(model="dummy"), "results_dir": "elsewhere"}) == base


# --- 실행 장치 ---------------------------------------------------------------


def test_non_torch_models_always_resolve_to_cpu():
    assert resolve_device("cuda", uses_torch=False) == "cpu"
    assert resolve_device("auto", uses_torch=False) == "cpu"


def test_torch_device_resolution():
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu", uses_torch=True) == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto", uses_torch=True) == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda", uses_torch=True)
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu", uses_torch=True)


def test_environment_fingerprint_has_no_host_identity():
    env = environment_fingerprint("cpu")
    assert env["device"] == "cpu" and "numpy" in env["packages"]
    flat = json.dumps(env).lower()
    import getpass
    import socket

    assert socket.gethostname().lower() not in flat
    assert getpass.getuser().lower() not in flat


def test_result_json_records_environment_and_git_state(tmp_path):
    out = run_experiment({"model": "zscore", "results_dir": str(tmp_path)}, verbose=False)
    assert out is not None
    (js,) = tmp_path.glob("*.json")
    summary = json.loads(js.read_text())
    assert summary["config"]["device"] == "cpu"
    assert summary["environment"]["device"] == "cpu"
    assert "git_dirty" in summary
    assert summary["metrics"]["diag_vus_window"] == 100.0
    assert summary["metrics"]["diag_test_input_absmax"] > 0


# --- 정규화 / 점수 건전성 ---------------------------------------------------


def test_std_floor_tames_near_constant_channels():
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.normal(size=500), 0.5 + 1e-5 * rng.normal(size=500)])
    test = train.copy()
    test[100, 1] = 0.9  # near-constant 채널의 작은 절대 변화
    _, te_legacy = zscore_normalize(train, test)
    _, te_floor = zscore_normalize(train, test, std_floor=0.01)
    assert np.abs(te_legacy).max() > 1e4  # 레거시: 발산 (SMD machine-1-1 재현)
    assert np.abs(te_floor).max() < 100
    _, te_clip = zscore_normalize(train, test, clip=10.0)
    assert np.abs(te_clip).max() == pytest.approx(10.0)


def test_legacy_zscore_unchanged():
    rng = np.random.default_rng(1)
    train, test = rng.normal(size=(50, 3)), rng.normal(size=(40, 3))
    tr, te = zscore_normalize(train, test)
    np.testing.assert_allclose(te, (test - train.mean(0)) / train.std(0))
    assert np.allclose(tr.mean(0), 0)


def test_check_scores_rejects_non_finite():
    assert check_scores(np.ones(5), 5).dtype == np.float64
    with pytest.raises(ScoreHealthError, match="non-finite"):
        check_scores(np.array([0.0, np.nan, 1.0]), 3)
    with pytest.raises(ScoreHealthError, match="length"):
        check_scores(np.ones(4), 5)


@register_model("_test_nan_scorer")
class _NaNScorer(BaseDetector):
    generation = "gen0"

    def fit(self, X):
        self._fitted = True
        return self

    def score(self, X):
        out = np.zeros(len(X))
        out[3] = np.inf
        return out


def test_runner_fails_loudly_on_non_finite_scores(tmp_path):
    with pytest.raises(ScoreHealthError):
        run_experiment({"model": "_test_nan_scorer", "results_dir": str(tmp_path)}, verbose=False)
    assert not list(tmp_path.glob("*.parquet"))  # 붕괴한 실행은 결과를 남기지 않는다


# --- VUS 윈도우 자동 추정 ----------------------------------------------------


def test_sliding_window_recovers_period():
    t = np.arange(4000)
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * rng.normal(size=len(t))
    assert abs(estimate_sliding_window(x) - 50) <= 1
    # 다변량은 첫 채널 기준 (TSB-AD 관례)
    X = np.column_stack([x, rng.normal(size=len(t))])
    assert abs(estimate_sliding_window(X) - 50) <= 1


def test_sliding_window_fallbacks():
    assert estimate_sliding_window(np.ones(1000)) == DEFAULT_PERIOD_FALLBACK
    assert estimate_sliding_window(np.arange(4)) == DEFAULT_PERIOD_FALLBACK
    t = np.arange(5000)
    long_period = np.sin(2 * np.pi * t / 1000)  # max_period(300) 초과
    assert estimate_sliding_window(long_period) == DEFAULT_PERIOD_FALLBACK


def test_runner_auto_window_is_recorded(tmp_path):
    run_experiment(
        {"model": "zscore", "sliding_window": "auto", "results_dir": str(tmp_path)},
        verbose=False,
    )
    (js,) = tmp_path.glob("*.json")
    summary = json.loads(js.read_text())
    assert summary["config"]["sliding_window"] == "auto"
    assert summary["metrics"]["diag_vus_window"] >= 3


# --- 공정 집계 ---------------------------------------------------------------


def _frame(rows):
    return pd.DataFrame(rows, columns=["model", "generation", "dataset", "channel", "data_hash"])


def test_entity_disambiguates_data_variants():
    df = _frame(
        [
            ("a", "gen1", "synthetic", "all", "h1aaaaaa"),
            ("a", "gen1", "synthetic", "all", "h2bbbbbb"),
            ("a", "gen1", "smd", "m-1", "h3cccccc"),
        ]
    )
    ent = add_entity_column(df)["entity"].tolist()
    assert ent == ["synthetic#h1aaaa", "synthetic#h2bbbb", "smd/m-1"]


def test_shared_entities_and_normalized_ranks():
    pivot = pd.DataFrame(
        {"e1": [0.9, 0.5, 0.1], "e2": [0.8, np.nan, 0.2]}, index=["m1", "m2", "m3"]
    )
    assert shared_entities(pivot) == ["e1"]
    nr = normalized_ranks(pivot)
    # e1: 3 모델 -> {0, .5, 1}; e2: 2 모델 -> {0, 1}
    assert nr.loc["m1"].tolist() == [0.0, 0.0]
    assert nr.loc["m2", "e1"] == 0.5 and np.isnan(nr.loc["m2", "e2"])
    assert nr.loc["m3"].tolist() == [1.0, 1.0]


def test_generation_shared_entities_ignores_baseline():
    m = pd.DataFrame(
        {
            "generation": ["gen0", "gen1", "gen1", "gen3"],
            "entity": ["only_baseline", "e1", "e2", "e1"],
        }
    )
    assert generation_shared_entities(m) == ["e1"]


# --- CLI ---------------------------------------------------------------------


def test_cli_flags_do_not_clobber_config_file(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: zscore\nseed: 2\n")
    rd = tmp_path / "res"
    assert main(["run", "--config", str(cfg), "--results-dir", str(rd)]) == 0
    (js,) = rd.glob("*.json")
    summary = json.loads(js.read_text())
    assert summary["model"] == "zscore" and summary["seed"] == 2


def test_model_alias_separates_variants(tmp_path):
    from tsad_forge.runner.results import load_all_results

    base = {"model": "sub_pca", "results_dir": str(tmp_path)}
    run_experiment(base, verbose=False)
    run_experiment(
        {**base, "model_alias": "sub_pca_w50", "model_params": {"window": 50}}, verbose=False
    )
    df = load_all_results(tmp_path)
    assert set(df["model"]) == {"sub_pca", "sub_pca_w50"}  # 중복 제거로 사라지지 않는다
    # alias 미지정은 해시에 들어가지 않는다 (레거시 해시 불변)
    assert experiment_hash(_cpu_cfg(model="dummy")) == GOLDEN_HASHES["dummy/synthetic"]
