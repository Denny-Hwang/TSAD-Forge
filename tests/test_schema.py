import numpy as np
import pytest

from tsad_forge.data.schema import TSADDataset, label_events


def test_univariate_promoted_to_2d():
    ds = TSADDataset(train=np.zeros(10), test=np.ones(8), labels=np.zeros(8, dtype=int))
    assert ds.train.shape == (10, 1)
    assert ds.test.shape == (8, 1)
    assert ds.n_dims == 1


def test_dim_mismatch_raises():
    with pytest.raises(ValueError, match="dimensionality mismatch"):
        TSADDataset(train=np.zeros((10, 2)), test=np.zeros((8, 3)), labels=np.zeros(8, dtype=int))


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="labels length"):
        TSADDataset(train=np.zeros((10, 1)), test=np.zeros((8, 1)), labels=np.zeros(5, dtype=int))


def test_nonbinary_labels_raise():
    with pytest.raises(ValueError, match="binary"):
        TSADDataset(train=np.zeros((10, 1)), test=np.zeros((8, 1)), labels=np.full(8, 2, dtype=int))


def test_label_events():
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert label_events(labels) == [(1, 3), (5, 6), (7, 10)]
    assert label_events(np.zeros(5, dtype=int)) == []


def test_event_stats():
    ds = TSADDataset(
        train=np.zeros((4, 1)),
        test=np.zeros((10, 1)),
        labels=np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0]),
    )
    stats = ds.event_stats()
    assert stats["n_events"] == 2
    assert stats["min_len"] == 1
    assert stats["max_len"] == 2
