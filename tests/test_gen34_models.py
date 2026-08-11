"""Gen3-4 DL 모델 스모크 테스트 (M4 DoD) — 소형 config로 CPU에서 빠르게."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="Gen3-4 모델은 torch 필요 (extras: dl)")

from tsad_forge.models.registry import get_model, list_models  # noqa: E402

GEN3 = ["ae", "lstm_ad", "lstm_p", "vae_donut", "dagmm", "omni_anomaly", "usad"]
GEN4 = ["gdn", "mtad_gat", "anomaly_transformer", "tranad", "dcdetector", "timesnet"]

RNG = np.random.default_rng(3)
TRAIN = RNG.normal(size=(300, 3)).astype(np.float64)
TEST = RNG.normal(size=(150, 3)).astype(np.float64)
TEST[70:80] += 6.0

SMALL = {"epochs": 1, "window": 16, "batch_size": 64}


def test_all_gen34_registered():
    names = list_models()
    for m in GEN3 + GEN4:
        assert m in names, m


@pytest.mark.parametrize("name", GEN3 + GEN4)
def test_contract_shape_finite(name):
    model = get_model(name, seed=0, **SMALL)
    scores = model.fit(TRAIN).score(TEST)
    assert scores.shape == (150,)
    assert np.isfinite(scores).all()
    assert model.generation in ("gen3", "gen4")


@pytest.mark.parametrize("name", ["ae", "usad", "tranad"])
def test_seed_reproducibility(name):
    s1 = get_model(name, seed=0, **SMALL).fit(TRAIN).score(TEST)
    s2 = get_model(name, seed=0, **SMALL).fit(TRAIN).score(TEST)
    np.testing.assert_allclose(s1, s2, rtol=1e-5, err_msg=name)


def test_univariate_works():
    train = RNG.normal(size=400)
    test = RNG.normal(size=200)
    for name in ["ae", "lstm_p", "gdn", "timesnet"]:
        scores = get_model(name, seed=0, **SMALL).fit(train).score(test)
        assert scores.shape == (200,), name


def test_reconstruction_models_learn_sine():
    """정상 패턴(사인파) 학습 후 파형 붕괴 구간 점수가 정상 구간보다 높아야 한다."""
    t = np.arange(1200)
    train = np.sin(2 * np.pi * t / 40)[:800]
    test = np.sin(2 * np.pi * t / 40)[800:]
    test = test.copy()
    test[200:250] = np.random.default_rng(0).normal(size=50)  # 패턴 파괴
    for name in ["ae", "usad"]:
        scores = get_model(name, seed=0, window=32, epochs=8).fit(train).score(test)
        anom = scores[200:250].mean()
        normal = np.concatenate([scores[50:150], scores[300:390]]).mean()
        assert anom > normal, f"{name}: anomaly {anom:.4f} <= normal {normal:.4f}"
