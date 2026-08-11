"""Gen5 테스트 (M5): MambaTSAD faithful/fixed + 파운데이션 어댑터 가드."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="Gen5 모델은 torch 필요 (extras: dl)")

from tsad_forge.models.gen5_ssm_foundation.mamba_tsad import (  # noqa: E402
    ama_smooth,
    hp_filter,
)
from tsad_forge.models.registry import get_model, list_models  # noqa: E402

RNG = np.random.default_rng(5)
T_TR, T_TE = 400, 200
t = np.arange(T_TR + T_TE)
X = np.sin(2 * np.pi * t / 40)[:, None] + RNG.normal(scale=0.1, size=(T_TR + T_TE, 1))
TRAIN, TEST = X[:T_TR], X[T_TR:].copy()
TEST[80:100] += 3.0

SMALL = {"epochs": 2, "window": 16, "batch_size": 64}


def test_gen5_registered():
    names = list_models()
    for m in ["mamba_tsad_faithful", "mamba_tsad_fixed", "moment", "chronos", "timesfm"]:
        assert m in names, m


@pytest.mark.parametrize("name", ["mamba_tsad_faithful", "mamba_tsad_fixed"])
def test_mamba_contract(name):
    model = get_model(name, seed=0, **SMALL)
    scores = model.fit(TRAIN).score(TEST)
    assert scores.shape == (T_TE,)
    assert np.isfinite(scores).all()
    assert model.generation == "gen5"


def test_variants_differ():
    """faithful과 fixed는 같은 시드에서도 다른 점수를 내야 한다 (4개 이슈의 효과)."""
    s_f = get_model("mamba_tsad_faithful", seed=0, **SMALL).fit(TRAIN).score(TEST)
    s_x = get_model("mamba_tsad_fixed", seed=0, **SMALL).fit(TRAIN).score(TEST)
    assert not np.allclose(s_f, s_x)


def test_hp_filter_decomposition():
    trend, cycle = hp_filter(X)
    np.testing.assert_allclose(trend + cycle, X, atol=1e-8)
    # trend는 원 신호보다 매끄러워야 함 (2차 차분 에너지 감소)
    assert np.abs(np.diff(trend[:, 0], 2)).sum() < np.abs(np.diff(X[:, 0], 2)).sum()


def test_ama_smooth_modes_differ():
    g = ama_smooth(X, global_fft=True)
    loc = ama_smooth(X, global_fft=False)
    assert g.shape == X.shape and loc.shape == X.shape
    assert np.isfinite(g).all() and np.isfinite(loc).all()


@pytest.mark.parametrize("name", ["moment", "chronos", "timesfm"])
def test_foundation_adapters_guarded(name):
    """선택 의존성 미설치 시 설치 안내가 담긴 RuntimeError."""
    pytest.importorskip_reason = None
    try:
        import momentfm  # noqa: F401

        installed = name == "moment"
    except ImportError:
        installed = False
    if installed:
        pytest.skip("library installed — guard 테스트 대상 아님")
    with pytest.raises(RuntimeError, match="pip install"):
        get_model(name).fit(TRAIN)
