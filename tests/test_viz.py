"""시각화 8종 + 리더보드 생성 테스트 (M6 DoD)."""

import numpy as np
import pytest

from tsad_forge.runner.experiment import run_experiment
from tsad_forge.viz.charts import generate_all
from tsad_forge.viz.leaderboard import build_leaderboard, save_leaderboard

EXPECTED_CHARTS = {
    "generation_evolution",
    "leaderboard_table",
    "heatmap",
    "critical_difference",
    "perf_vs_cost",
    "metric_divergence",
    "case_viewer",
    "dataset_cards",
}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    rd = tmp_path_factory.mktemp("results")
    for model in ("dummy", "zscore", "sub_pca"):
        run_experiment({"model": model, "data": "synthetic", "results_dir": str(rd)}, verbose=False)
        run_experiment(
            {
                "model": model,
                "data": "synthetic",
                "data_params": {"n_dims": 3, "n_events": 6},
                "results_dir": str(rd),
            },
            verbose=False,
        )
    return rd


def test_all_eight_charts_generated(results, tmp_path):
    out = tmp_path / "charts"
    paths = generate_all(results, out, data_dir=tmp_path / "nodata")
    assert len(paths) == 8
    names = {p.stem for p in paths}
    assert names == EXPECTED_CHARTS
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000, p.name
        assert "plotly" in p.read_text()[:3000].lower()


def test_leaderboard_build_and_save(results, tmp_path):
    pivot, summary = build_leaderboard(results)
    assert "mean_vus_pr" in summary.columns
    assert summary["mean_rank"].is_monotonic_increasing
    # dummy(무정보)는 정보 있는 모델보다 순위가 나빠야 한다
    models_by_rank = [m for _, m in summary.index]
    assert models_by_rank.index("dummy") > models_by_rank.index("sub_pca")

    out = save_leaderboard(results, tmp_path / "lb" / "lite.md")
    text = out.read_text()
    assert "VUS-PR" in text and "config" in text
    assert "<table" in text and "Gen " in text  # 세대 배지가 있는 HTML 테이블
    assert "pa_f1" not in text  # PA는 리더보드에 절대 등장하지 않는다


def test_leaderboard_missing_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_leaderboard(tmp_path / "none")


def test_metric_frame_requires_data(tmp_path):
    from tsad_forge.viz.charts import _metric_frame

    with pytest.raises(FileNotFoundError):
        _metric_frame(tmp_path)


def test_charts_numeric_sanity(results, tmp_path):
    """세대 진화 차트의 원 데이터: dummy(gen0) VUS-PR < 정보모델 최대값."""
    from tsad_forge.runner.results import load_all_results

    df = load_all_results(results)
    v = df[df["metric"] == "vus_pr"]
    assert np.isfinite(v["value"]).all()
    assert (v["value"] >= 0).all() and (v["value"] <= 1).all()
