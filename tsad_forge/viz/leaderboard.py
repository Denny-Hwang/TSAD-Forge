"""리더보드 생성 (M3: 마크다운 테이블, M6: 인터랙티브 확장).

results parquet(long-format)에서 (모델 × 데이터셋) 지표 테이블을 만든다.
- 시드가 여러 개면 평균 (시드 수는 각 셀의 n으로 기록)
- 주지표 VUS-PR 기준 정렬, 평균 순위(mean rank) 포함
- 각 모델 행에 config_hash를 병기해 재현 커맨드를 추적 가능하게 (CLAUDE.md §9)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsad_forge.runner.results import load_all_results

PRIMARY_METRIC = "vus_pr"


def build_leaderboard(
    results_dir: str | Path, metric: str = PRIMARY_METRIC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pivot 테이블, 요약 테이블) 반환.

    pivot: index=(generation, model), columns=dataset/channel, values=metric(시드 평균)
    summary: 모델별 평균 지표·평균 순위·총 실행시간
    """
    df = load_all_results(results_dir)
    if df.empty:
        raise FileNotFoundError(f"no results under {results_dir}")
    m = df[df["metric"] == metric].copy()
    if m.empty:
        raise ValueError(f"metric '{metric}' not found in results")
    m["entity"] = m["dataset"] + "/" + m["channel"]
    m.loc[m["channel"] == "all", "entity"] = m.loc[m["channel"] == "all", "dataset"]

    pivot = m.pivot_table(
        index=["generation", "model"], columns="entity", values="value", aggfunc="mean"
    )
    ranks = pivot.rank(ascending=False, axis=0)
    summary = pd.DataFrame(
        {
            f"mean_{metric}": pivot.mean(axis=1),
            "mean_rank": ranks.mean(axis=1),
            "n_entities": pivot.notna().sum(axis=1),
        }
    ).sort_values("mean_rank")
    runtime = (
        df[df["metric"] == metric]
        .groupby(["generation", "model"])["runtime_s"]
        .mean()
        .rename("mean_runtime_s")
    )
    summary = summary.join(runtime)
    cfg = (
        df.groupby(["generation", "model"])["config_hash"]
        .agg(lambda s: ",".join(sorted(set(s))[:3]))
        .rename("config_hash")
    )
    summary = summary.join(cfg)
    return pivot.round(4), summary.round(4)


def leaderboard_markdown(results_dir: str | Path, metric: str = PRIMARY_METRIC) -> str:
    pivot, summary = build_leaderboard(results_dir, metric)
    lines = [
        f"# Leaderboard — {metric.upper().replace('_', '-')}",
        "",
        "Primary metric: **VUS-PR** (Paparrizos et al., VLDB 2022). Values are seed"
        " averages. PA-F1 is intentionally not shown (CLAUDE.md §4).",
        "",
        "## Model summary (sorted by mean rank)",
        "",
        summary.reset_index().to_markdown(index=False),
        "",
        "## Model × dataset",
        "",
        pivot.reset_index().to_markdown(index=False),
        "",
        "Reproduce with `python benchmarks/run_all.py --profile configs/lite.yaml`"
        " — each row's config_hash resolves to its full configuration in the results JSON.",
    ]
    return "\n".join(lines)


def save_leaderboard(
    results_dir: str | Path, out_path: str | Path, metric: str = PRIMARY_METRIC
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(leaderboard_markdown(results_dir, metric))
    return out_path
