"""Gen1-2 모델 계약(contract) + 정탐 스모크 테스트 (M3)."""

import numpy as np
import pytest

from tsad_forge.models._window import embed_windows, window_scores_to_points
from tsad_forge.models.registry import get_model, list_models

GEN1 = ["cusum", "ewma", "hotelling_t2", "pca_t2spe", "sub_pca", "stl_residual", "poly"]
GEN2 = ["lof", "ocsvm", "iforest", "knn", "sub_knn", "matrix_profile"]

RNG = np.random.default_rng(7)
TRAIN = RNG.normal(size=(600, 3))
TEST = RNG.normal(size=(400, 3))
TEST[200:210] += 6.0  # 명백한 스파이크


def test_all_gen12_registered():
    names = list_models()
    for m in GEN1 + GEN2:
        assert m in names, m


@pytest.mark.parametrize("name", GEN1 + GEN2)
def test_contract_shape_finite(name):
    model = get_model(name, seed=0)
    scores = model.fit(TRAIN).score(TEST)
    assert scores.shape == (400,)
    assert np.isfinite(scores).all()
    assert model.generation in ("gen1", "gen2")


@pytest.mark.parametrize("name", GEN1 + GEN2)
def test_score_before_fit_raises(name):
    with pytest.raises(RuntimeError, match="call fit"):
        get_model(name).score(TEST)


# matrix_profile 제외: 백색잡음 스모크 데이터에서 discord 정의가 무의미
@pytest.mark.parametrize("name", [m for m in GEN1 + GEN2 if m != "matrix_profile"])
def test_detects_obvious_spike(name):
    scores = get_model(name, seed=0).fit(TRAIN).score(TEST)
    peak = int(np.argmax(scores))
    # 윈도우 끝점 할당 모델은 window 만큼 지연될 수 있음
    window = getattr(get_model(name), "window", 0) or 0
    assert 195 <= peak <= 215 + window, f"{name}: peak at {peak}"


def test_deterministic_models_reproducible():
    for name in ["cusum", "hotelling_t2", "sub_pca", "knn"]:
        s1 = get_model(name, seed=0).fit(TRAIN).score(TEST)
        s2 = get_model(name, seed=1).fit(TRAIN).score(TEST)  # 시드 무관해야 함
        np.testing.assert_allclose(s1, s2, err_msg=name)


def test_iforest_seed_sensitivity():
    s1 = get_model("iforest", seed=0).fit(TRAIN).score(TEST)
    s2 = get_model("iforest", seed=0).fit(TRAIN).score(TEST)
    s3 = get_model("iforest", seed=1).fit(TRAIN).score(TEST)
    np.testing.assert_allclose(s1, s2)
    assert not np.allclose(s1, s3)


def test_univariate_input_works():
    train = RNG.normal(size=500)
    test = RNG.normal(size=300)
    test[150] += 8
    for name in ["cusum", "sub_pca", "sub_knn", "poly"]:
        scores = get_model(name, seed=0).fit(train).score(test)
        assert scores.shape == (300,)


def test_matrix_profile_finds_discord_in_periodic_data():
    """MP의 본령: 주기 신호 속 파형 왜곡(discord) 탐지."""
    t = np.arange(2000)
    x = np.sin(2 * np.pi * t / 50)
    x[1000:1050] = np.sin(2 * np.pi * t[1000:1050] / 12)  # 주기 붕괴 구간
    scores = get_model("matrix_profile", window=50).fit(x[:500]).score(x)
    peak = int(np.argmax(scores))
    assert 950 <= peak <= 1100, f"discord peak at {peak}"


def test_constant_channel_handled():
    train = np.column_stack([RNG.normal(size=300), np.ones(300)])
    test = np.column_stack([RNG.normal(size=200), np.ones(200)])
    for name in ["hotelling_t2", "pca_t2spe", "stl_residual", "matrix_profile"]:
        scores = (
            get_model(name, seed=0, **({"window": 20} if name == "matrix_profile" else {}))
            .fit(train)
            .score(test)
        )
        assert np.isfinite(scores).all(), name


def test_window_utils():
    X = np.arange(20, dtype=float).reshape(10, 2)
    W = embed_windows(X, 3)
    assert W.shape == (8, 6)
    np.testing.assert_array_equal(W[0], [0, 1, 2, 3, 4, 5])
    pts = window_scores_to_points(np.arange(8, dtype=float), 10, 3)
    assert pts.shape == (10,)
    assert pts[0] == pts[1] == pts[2] == 0.0  # 앞쪽 패딩
    with pytest.raises(ValueError):
        embed_windows(X, 100)


NEW_PRACTICAL = ["sesd", "spectral_residual", "hbos", "ensemble_simple"]


@pytest.mark.parametrize("name", NEW_PRACTICAL)
def test_practical_models_contract(name):
    scores = get_model(name, seed=0).fit(TRAIN).score(TEST)
    assert scores.shape == (400,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", ["sesd", "hbos", "ensemble_simple"])
def test_practical_models_detect_spike(name):
    scores = get_model(name, seed=0).fit(TRAIN).score(TEST)
    window = getattr(get_model(name), "window", 0) or 0
    assert 195 <= int(np.argmax(scores)) <= 215 + window, name


def test_spectral_residual_on_spike():
    """SR의 본령(Azure KPI 시나리오): 주기+노이즈 신호 속 스파이크 saliency."""
    t = np.arange(2000)
    x = np.sin(2 * np.pi * t / 50) + np.random.default_rng(0).normal(scale=0.05, size=2000)
    x[1200] += 4.0
    scores = get_model("spectral_residual").fit(x[:500]).score(x)
    assert 1195 <= int(np.argmax(scores)) <= 1205


def test_ensemble_beats_worst_member_on_synthetic():
    from tsad_forge.evaluation.metrics import compute_metrics
    from tsad_forge.synthetic.generator import generate_synthetic

    ds = generate_synthetic(seed=11)
    vals = {}
    for name in ["ensemble_simple", "sub_pca", "sub_knn", "iforest", "spectral_residual"]:
        s = get_model(name, seed=0).fit(ds.train).score(ds.test)
        vals[name] = compute_metrics(s, ds.labels)["vus_pr"]
    members = [v for k, v in vals.items() if k != "ensemble_simple"]
    assert vals["ensemble_simple"] >= min(members)  # 최소한 최악 멤버보다는 나아야
