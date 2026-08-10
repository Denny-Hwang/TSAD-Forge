import numpy as np
import pytest

from tsad_forge.synthetic.generator import generate_synthetic
from tsad_forge.synthetic.injectors import INJECTORS, inject_anomalies


def test_generator_shapes_and_labels():
    ds = generate_synthetic(n_train=500, n_test=400, n_dims=3, n_events=4, seed=7)
    assert ds.train.shape == (500, 3)
    assert ds.test.shape == (400, 3)
    assert ds.labels.shape == (400,)
    assert ds.labels.sum() > 0
    assert ds.meta["n_events"] >= 1


def test_generator_deterministic_with_seed():
    a = generate_synthetic(n_train=300, n_test=300, seed=42)
    b = generate_synthetic(n_train=300, n_test=300, seed=42)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.test, b.test)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_generator_differs_across_seeds():
    a = generate_synthetic(n_train=300, n_test=300, seed=0)
    b = generate_synthetic(n_train=300, n_test=300, seed=1)
    assert not np.array_equal(a.test, b.test)


def test_contamination_changes_train():
    clean = generate_synthetic(n_train=500, n_test=200, seed=3, contamination=0.0)
    dirty = generate_synthetic(n_train=500, n_test=200, seed=3, contamination=1.0)
    assert not np.array_equal(clean.train, dirty.train)


@pytest.mark.parametrize("kind", sorted(INJECTORS))
def test_each_injector_marks_labels(kind):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    labels = inject_anomalies(x, rng, n_events=2, kinds=[kind], min_len=3, max_len=10)
    assert labels.sum() > 0


def test_unknown_kind_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown anomaly kinds"):
        inject_anomalies(np.zeros((100, 1)), rng, n_events=1, kinds=["nope"])
