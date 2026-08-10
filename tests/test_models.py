import numpy as np
import pytest

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import get_model, list_models


def test_builtin_models_registered():
    models = list_models()
    assert "dummy" in models
    assert "zscore" in models


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("does-not-exist")


@pytest.mark.parametrize("name", ["dummy", "zscore"])
def test_fit_score_contract(name):
    model = get_model(name, seed=0)
    assert isinstance(model, BaseDetector)
    train = np.random.default_rng(0).normal(size=(200, 3))
    test = np.random.default_rng(1).normal(size=(150, 3))
    scores = model.fit(train).score(test)
    assert scores.shape == (150,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="call fit"):
        get_model("dummy").score(np.zeros((10, 1)))


def test_dummy_deterministic_per_seed():
    X = np.zeros((100, 1))
    s1 = get_model("dummy", seed=5).fit(X).score(X)
    s2 = get_model("dummy", seed=5).fit(X).score(X)
    s3 = get_model("dummy", seed=6).fit(X).score(X)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_zscore_detects_obvious_spike():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 1))
    test = rng.normal(size=(200, 1))
    test[100] += 10.0
    scores = get_model("zscore").fit(train).score(test)
    assert scores.argmax() == 100
