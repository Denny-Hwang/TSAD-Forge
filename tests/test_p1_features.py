"""리뷰 P1 기능 테스트: 데이터 경로, timestamps, save/load, 온라인 트랙, 지연 지표, 리더보드 CI."""

import numpy as np
import pandas as pd
import pytest

from tsad_forge.data.paths import default_data_dir, resolve_data_dir
from tsad_forge.data.schema import TSADDataset
from tsad_forge.evaluation.metrics import compute_metrics, mean_detection_delay
from tsad_forge.models.online import OnlineCUSUM, OnlineEWMA
from tsad_forge.models.registry import get_model, list_models

RNG = np.random.default_rng(0)


# --- P1-C: 데이터 경로 ---


def test_data_dir_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv("TSAD_FORGE_DATA", raising=False)
    assert default_data_dir() == resolve_data_dir(None)
    assert str(default_data_dir()) == "data"

    monkeypatch.setenv("TSAD_FORGE_DATA", str(tmp_path))
    assert resolve_data_dir(None) == tmp_path
    # 명시 인자가 환경변수보다 우선
    assert resolve_data_dir("explicit") == resolve_data_dir("explicit")
    assert str(resolve_data_dir("explicit")) == "explicit"


def test_loader_honors_env_var(monkeypatch, tmp_path):
    root = tmp_path / "mgab"
    root.mkdir()
    n = 1000  # 로더의 최소 train 길이(100)보다 충분히 크게
    labels = np.zeros(n, dtype=int)
    labels[500:505] = 1
    pd.DataFrame(
        {"value": RNG.normal(size=n), "is_anomaly": labels, "is_ignored": np.zeros(n, dtype=int)}
    ).to_csv(root / "1.csv")

    from tsad_forge.data.loaders.mgab import load_mgab

    monkeypatch.setenv("TSAD_FORGE_DATA", str(tmp_path))
    ds = load_mgab(series=1)  # data_dir 미지정 → 환경변수로 해석
    assert ds.labels.sum() == 5


# --- P1-D: timestamps ---


def test_schema_timestamps_roundtrip():
    ts = np.arange(10)
    ds = TSADDataset(
        train=RNG.normal(size=(5, 1)),
        test=RNG.normal(size=(10, 1)),
        labels=np.zeros(10, dtype=int),
        test_timestamps=ts,
    )
    np.testing.assert_array_equal(ds.test_timestamps, ts)
    assert ds.train_timestamps is None

    with pytest.raises(ValueError, match="test_timestamps length"):
        TSADDataset(
            train=RNG.normal(size=(5, 1)),
            test=RNG.normal(size=(10, 1)),
            labels=np.zeros(10, dtype=int),
            test_timestamps=np.arange(7),
        )


def test_file_loader_preserves_timestamps(tmp_path):
    from tsad_forge.data.loaders.file import load_file

    n = 60
    df = pd.DataFrame(
        {"timestamp": np.arange(n) * 2, "v": RNG.normal(size=n), "label": [0] * 50 + [1] * 10}
    )
    p = tmp_path / "byod.csv"
    df.to_csv(p, index=False)

    ds = load_file(p)
    assert ds.train_timestamps is not None and ds.test_timestamps is not None
    assert len(ds.train_timestamps) == len(ds.train)
    assert len(ds.test_timestamps) == len(ds.test)
    # 정렬 뒤 보존 (간격 2의 등차)
    assert ds.test_timestamps[1] - ds.test_timestamps[0] == 2


# --- P1-E: save/load ---


def test_save_load_roundtrip(tmp_path):
    train = RNG.normal(size=(200, 3))
    test = RNG.normal(size=(100, 3))
    model = get_model("sub_pca", seed=0).fit(train)
    expected = model.score(test)

    path = model.save(tmp_path / "m.pkl")
    from tsad_forge.models.base import BaseDetector

    loaded = BaseDetector.load(path)
    np.testing.assert_allclose(loaded.score(test), expected)


def test_save_requires_fitted(tmp_path):
    model = get_model("sub_pca", seed=0)
    with pytest.raises(RuntimeError, match="fit"):
        model.save(tmp_path / "m.pkl")


def test_load_rejects_non_detector(tmp_path):
    import pickle

    p = tmp_path / "junk.pkl"
    p.write_bytes(pickle.dumps({"not": "a detector"}))
    from tsad_forge.models.base import BaseDetector

    with pytest.raises(TypeError, match="BaseDetector"):
        BaseDetector.load(p)


# --- P1-F: 온라인 트랙 ---


def _spiky_series(n=500, spike_at=400):
    x = RNG.normal(size=(n, 2))
    x[spike_at : spike_at + 5] += 8.0
    return x


@pytest.mark.parametrize("cls", [OnlineEWMA, OnlineCUSUM])
def test_online_detectors_registered_and_flag_spike(cls):
    assert "online_ewma" in list_models() and "online_cusum" in list_models()
    train = RNG.normal(size=(300, 2))
    test = _spiky_series()
    model = cls(seed=0).fit(train)
    scores = model.score(test)
    assert scores.shape == (len(test),)
    spike_region = scores[400:405].max()
    assert spike_region > np.percentile(scores[:400], 99)  # 스파이크가 최상위 점수


def test_online_score_is_deterministic_across_calls():
    """score()가 상태를 오염시키지 않는다 (post-fit 스냅샷 복원)."""
    train = RNG.normal(size=(300, 2))
    test = _spiky_series()
    model = OnlineEWMA(seed=0).fit(train)
    s1 = model.score(test)
    s2 = model.score(test)
    np.testing.assert_array_equal(s1, s2)


def test_online_is_causal():
    """미래 값 변경이 과거 점수에 영향을 주지 않는다 (prequential 검증)."""
    train = RNG.normal(size=(300, 1))
    test_a = RNG.normal(size=(200, 1))
    test_b = test_a.copy()
    test_b[150:] += 100.0  # 뒤쪽만 변경
    m = OnlineCUSUM(seed=0).fit(train)
    sa = m.score(test_a)
    sb = m.score(test_b)
    np.testing.assert_array_equal(sa[:150], sb[:150])


# --- P1: 지연 지표 ---


def test_mean_detection_delay():
    labels = np.zeros(100, dtype=int)
    labels[20:30] = 1  # 이벤트 1: 즉시 탐지
    labels[60:70] = 1  # 이벤트 2: 3 스텝 지연
    scores = np.zeros(100)
    scores[20] = 1.0
    scores[63] = 1.0
    assert mean_detection_delay(scores, labels, threshold=0.5) == pytest.approx(1.5)

    # 미탐 이벤트는 이벤트 길이를 부과
    scores2 = np.zeros(100)
    scores2[20] = 1.0
    assert mean_detection_delay(scores2, labels, threshold=0.5) == pytest.approx((0 + 10) / 2)

    # 이벤트 없음 → None, compute_metrics에서 키 미포함
    out = compute_metrics(np.zeros(50), np.zeros(50, dtype=int), threshold=0.5)
    assert "mean_detection_delay" not in out


def test_compute_metrics_includes_delay():
    labels = np.zeros(200, dtype=int)
    labels[100:110] = 1
    scores = RNG.normal(size=200) * 0.01
    scores[102] = 5.0
    out = compute_metrics(scores, labels, threshold=1.0)
    assert out["mean_detection_delay"] == pytest.approx(2.0)


# --- P1-A: 리더보드 CI/Friedman ---


def test_bootstrap_ci_and_friedman():
    from tsad_forge.viz.leaderboard import _bootstrap_ci, friedman_pvalue

    row = pd.Series([0.5, 0.6, 0.4, 0.55, 0.45])
    lo, hi = _bootstrap_ci(row)
    assert lo < row.mean() < hi
    assert (hi - lo) < 0.4  # 터무니없이 넓지 않다

    lo2, hi2 = _bootstrap_ci(pd.Series([0.5, 0.6]))  # 엔티티 < 3 → CI 없음
    assert np.isnan(lo2) and np.isnan(hi2)

    rng = np.random.default_rng(1)
    pivot = pd.DataFrame(
        rng.random((5, 8)),
        index=pd.MultiIndex.from_tuples([("gen1", f"m{i}") for i in range(5)]),
    )
    pivot.iloc[0] += 1.0  # 한 모델이 명백히 우세 → 유의해야 함
    fr = friedman_pvalue(pivot)
    assert fr is not None
    p, n_models, n_entities = fr
    assert n_models == 5 and n_entities == 8
    assert p < 0.05

    assert friedman_pvalue(pivot.iloc[:2]) is None  # 모델 3개 미만 → 검정 불가
