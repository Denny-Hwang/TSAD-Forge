"""Gen5 테스트 (M5): MambaTSAD v2 (공식 코드 동작 재구현) + 파운데이션 어댑터 가드."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Gen5 모델은 torch 필요 (extras: dl)")

from tsad_forge.models.gen5_ssm_foundation.mamba_tsad import (  # noqa: E402
    ama_decompose,
    hp_filter,
    hp_matrix,
)
from tsad_forge.models.registry import get_model, list_models  # noqa: E402

RNG = np.random.default_rng(5)
T_TR, T_TE = 400, 200
t = np.arange(T_TR + T_TE)
X = np.sin(2 * np.pi * t / 40)[:, None] + RNG.normal(scale=0.1, size=(T_TR + T_TE, 1))
TRAIN, TEST = X[:T_TR], X[T_TR:].copy()
TEST[80:100] += 3.0

SMALL = {"epochs": 2, "window": 20, "batch_size": 32, "d_model": 8, "d_state": 4, "n_blocks": 2}


def test_gen5_registered():
    names = list_models()
    for m in [
        "mamba_tsad_faithful",
        "mamba_tsad_fixed",
        "moment",
        "chronos",
        "timesfm",
        "timesfm3",
        "timesfm3_ci",
        "timesfm3_prob",
    ]:
        assert m in names, m


@pytest.mark.parametrize("name", ["mamba_tsad_faithful", "mamba_tsad_fixed"])
def test_mamba_contract(name):
    model = get_model(name, seed=0, device="cpu", **SMALL)
    scores = model.fit(TRAIN).score(TEST)
    assert scores.shape == (T_TE,)
    assert np.isfinite(scores).all()
    assert model.generation == "gen5" and model.model_version == 2
    # 이상 구간 점수가 정상 구간보다 커야 한다 (최소한의 탐지 능력)
    assert scores[80:100].mean() > np.median(scores)


def test_mamba_is_seed_deterministic():
    a = get_model("mamba_tsad_fixed", seed=1, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    b = get_model("mamba_tsad_fixed", seed=1, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    np.testing.assert_array_equal(a, b)


def test_variants_differ():
    s_f = get_model("mamba_tsad_faithful", seed=0, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    s_x = get_model("mamba_tsad_fixed", seed=0, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    assert not np.allclose(s_f, s_x)


def test_state_index_affects_inference_only():
    """이슈 (a)는 추론 경로에만 있다: 같은 시드면 학습된 가중치는 같고 점수만 다르다."""
    up = get_model("mamba_tsad_faithful", seed=0, device="cpu", **SMALL).fit(TRAIN)
    last = get_model("mamba_tsad_faithful", seed=0, device="cpu", state_index="last", **SMALL).fit(
        TRAIN
    )
    for (k, p1), p2 in zip(
        up.net_.state_dict().items(), last.net_.state_dict().values(), strict=True
    ):
        assert torch.equal(p1, p2), k
    assert not np.allclose(up.score(TEST), last.score(TEST))


def test_ablation_flags_override_variant_and_are_validated():
    m = get_model("mamba_tsad_faithful", hp_objective="standard")
    assert m.flags["hp_objective"] == "standard" and m.flags["ama_period"] == "batch"
    assert m.get_config()["hp_objective"] == "standard"
    with pytest.raises(ValueError, match="ama_period"):
        get_model("mamba_tsad_fixed", ama_period="global")


def test_hp_matrix_objectives():
    L, lam = 50, 1e4
    std, up = hp_matrix(L, lam, "standard"), hp_matrix(L, lam, "upstream")
    y = X[:L, 0]
    # standard 해는 희소 HP 필터와 일치
    np.testing.assert_allclose(np.linalg.solve(std, y), hp_filter(y[:, None], lam)[0][:, 0])
    # upstream은 1차 차분 항 2λD₁ᵀD₁이 추가된 다른 시스템
    d1 = np.eye(L)[:-1] - np.eye(L)[1:]
    np.testing.assert_allclose(up - std, 2 * lam * d1.T @ d1)


def test_hp_filter_decomposition():
    trend, cycle = hp_filter(X)
    np.testing.assert_allclose(trend + cycle, X, atol=1e-8)
    assert np.abs(np.diff(trend[:, 0], 2)).sum() < np.abs(np.diff(X[:, 0], 2)).sum()


def test_ama_batch_vs_sample_period():
    L = 64
    tt = torch.arange(L, dtype=torch.float32)
    fast = torch.sin(2 * np.pi * tt / 8)  # 주기 8
    slow = 0.2 * torch.sin(2 * np.pi * tt / 32)  # 주기 32, 약한 진폭
    x = torch.stack([fast, slow])[:, :, None]  # [B=2, L, C=1]
    _, tr_batch = ama_decompose(x, "batch")
    _, tr_sample = ama_decompose(x, "sample")
    # batch 모드: 강한 샘플(주기 8)의 주기가 배치 전체에 적용 → 느린 샘플도 8-이동평균
    # sample 모드: 느린 샘플은 자기 주기(32)로 평활 → 두 결과가 달라야 한다
    assert torch.allclose(tr_batch[0], tr_sample[0], atol=1e-6)
    assert not torch.allclose(tr_batch[1], tr_sample[1], atol=1e-3)
    resid, trend = ama_decompose(x, "sample")
    assert torch.allclose(resid + trend, x)


def test_mamba_rejects_unknown_options():
    with pytest.raises(TypeError):
        get_model("mamba_tsad_faithful", amp="bf16")


def test_model_version_enters_config_hash():
    from tsad_forge.runner.experiment import plan_experiment

    plan = plan_experiment({"model": "mamba_tsad_fixed", "device": "cpu", "results_dir": "x"})
    assert plan.cfg["model_version"] == 2
    zs = plan_experiment({"model": "zscore", "results_dir": "x"})
    assert zs.cfg["model_version"] == 1  # 레거시 값 → 해시 불변 (test_integrity 골든 해시)
