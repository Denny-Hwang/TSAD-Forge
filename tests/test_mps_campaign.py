"""맥미니 MPS 캠페인 드라이버(scripts/mps_campaign.py)의 순수 로직 테스트 (실행·데이터 불필요)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def camp():
    spec = importlib.util.spec_from_file_location("mps_campaign", ROOT / "scripts/mps_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mps_campaign"] = mod  # dataclass가 모듈을 찾을 수 있게
    spec.loader.exec_module(mod)
    return mod


def _job(data, **params):
    return SimpleNamespace(overrides={"data": data, "data_params": params})


def test_phases_are_unique_ordered_and_use_mps_profiles(camp):
    from tsad_forge.runner.matrix import load_profile

    ids = [p.id for p in [*camp.PHASES, camp.RETRY]]
    assert len(ids) == len(set(ids)) and ids == sorted(ids)  # p0 < p1 < ... < p5
    for phase in [*camp.PHASES, camp.RETRY]:
        prof = load_profile(ROOT / phase.profile)
        assert prof["defaults"]["device"] == "mps"
        assert prof["results_dir"] == "benchmarks/results_mps"  # gitignored, never public


def test_phase_selectors(camp):
    assert camp._default_synthetic(_job("synthetic"))
    assert not camp._default_synthetic(_job("synthetic", n_dims=5, n_events=8))
    assert not camp._default_synthetic(_job("smd", machine="machine-1-1"))
    assert camp._machine_1_6(_job("smd", machine="machine-1-6"))
    assert not camp._machine_1_6(_job("smd", machine="machine-1-1"))


def test_state_roundtrip_is_atomic(camp, tmp_path, monkeypatch):
    monkeypatch.setattr(camp, "CAMPAIGN_DIR", tmp_path)
    monkeypatch.setattr(camp, "STATE_PATH", tmp_path / "state.json")
    assert camp.load_state() == {"phases": {}}
    camp.save_state({"phases": {"p0_smoke": {"status": "done"}}})
    assert camp.load_state()["phases"]["p0_smoke"]["status"] == "done"
    assert not (tmp_path / "state.tmp").exists()


def test_bootstrap_ci_needs_three_values(camp):
    assert all(np.isnan(camp._boot_ci(np.array([0.1, 0.2]))))
    lo, hi = camp._boot_ci(np.array([0.1, 0.2, 0.3, 0.4]))
    assert 0.1 <= lo <= 0.25 <= hi <= 0.4
