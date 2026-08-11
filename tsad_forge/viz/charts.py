"""시각화 8종 (CLAUDE.md §6) — results parquet + 스코어 npz에서 Plotly HTML 생성.

1. generation_evolution: 세대별 VUS-PR 분포 (box) — "세대가 오르면 성능이 오르는가?"
2. leaderboard_table: 지표 선택 드롭다운 정렬 테이블
3. heatmap: 모델×데이터셋 VUS-PR
4. critical_difference: 평균 순위 + Nemenyi CD (자체 구현)
5. perf_vs_cost: VUS-PR vs 실행시간 산점도
6. metric_divergence: PA-F1 vs VUS-PR — 저장된 원본 스코어에서 PA-F1을 재계산해
   평가 부풀림을 시각적으로 증명 (기본 results에는 PA-F1이 없음 — 의도된 설계)
7. case_viewer: 대표 시계열 + 세대별 스코어 오버레이 + 정답 음영
8. dataset_cards: 이상 길이 분포·채널 통계
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tsad_forge.runner.results import load_all_results

GEN_ORDER = ["gen0", "gen1", "gen2", "gen3", "gen4", "gen5"]
GEN_LABEL = {
    "gen0": "Gen0 baseline",
    "gen1": "Gen1 Statistical",
    "gen2": "Gen2 Classical ML",
    "gen3": "Gen3 DL Recon",
    "gen4": "Gen4 Graph/Transformer",
    "gen5": "Gen5 SSM/Foundation",
}
PRIMARY = "vus_pr"


def _metric_frame(results_dir: str | Path) -> pd.DataFrame:
    df = load_all_results(results_dir)
    if df.empty:
        raise FileNotFoundError(f"no results under {results_dir}")
    df = df[~df["dataset"].str.startswith("synthetic") | True]  # 전체 유지 (필터 훅)
    df["entity"] = np.where(
        df["channel"] == "all", df["dataset"], df["dataset"] + "/" + df["channel"]
    )
    return df


def _write(fig: go.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


# 1 ---------------------------------------------------------------------------


def generation_evolution(df: pd.DataFrame, out_dir: Path) -> Path:
    m = df[df["metric"] == PRIMARY]
    fig = go.Figure()
    for gen in GEN_ORDER:
        vals = m[m["generation"] == gen]["value"]
        if len(vals):
            fig.add_trace(go.Box(y=vals, name=GEN_LABEL[gen], boxmean=True))
    fig.update_layout(
        title="세대별 VUS-PR 분포 — 세대가 올라간다고 성능이 오르는가?",
        yaxis_title="VUS-PR (entity×seed 분포)",
        showlegend=False,
    )
    return _write(fig, out_dir, "generation_evolution")


# 2 ---------------------------------------------------------------------------


def leaderboard_table(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = sorted(df["metric"].unique())
    metrics = [m for m in metrics if m not in ("threshold", "n_predicted_anomalies")]
    pivots = {
        met: df[df["metric"] == met]
        .pivot_table(index=["generation", "model"], columns="entity", values="value")
        .round(4)
        for met in metrics
    }

    def _cells(p: pd.DataFrame):
        p = p.reset_index()
        return [p[c] for c in p.columns], list(p.columns)

    default = PRIMARY if PRIMARY in pivots else metrics[0]
    cells, header = _cells(pivots[default])
    fig = go.Figure(go.Table(header={"values": header}, cells={"values": cells}))
    buttons = []
    for met in metrics:
        c, h = _cells(pivots[met])
        buttons.append(
            {
                "label": met,
                "method": "restyle",
                "args": [{"header": [{"values": h}], "cells": [{"values": c}]}],
            }
        )
    fig.update_layout(
        title="리더보드 (지표 선택 드롭다운) — 주지표 VUS-PR",
        updatemenus=[{"buttons": buttons, "x": 1.0, "y": 1.15}],
    )
    return _write(fig, out_dir, "leaderboard_table")


# 3 ---------------------------------------------------------------------------


def heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    m = df[df["metric"] == PRIMARY]
    pivot = m.pivot_table(index="model", columns="entity", values="value")
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar_title="VUS-PR",
            zmin=0,
        )
    )
    fig.update_layout(title="모델 × 데이터셋 VUS-PR 히트맵", xaxis_tickangle=45)
    return _write(fig, out_dir, "heatmap")


# 4 ---------------------------------------------------------------------------


def critical_difference(df: pd.DataFrame, out_dir: Path, alpha: float = 0.05) -> Path:
    """평균 순위 기반 CD 다이어그램 (Demšar 2006, Nemenyi) — 자체 구현."""
    m = df[df["metric"] == PRIMARY]
    pivot = m.pivot_table(index="model", columns="entity", values="value")
    pivot = pivot.dropna(axis=1, how="any")  # 공통 entity만 (순위 비교 조건)
    if pivot.shape[1] < 2:
        warnings.warn("공통 entity가 부족해 CD 다이어그램 생략", stacklevel=2)
        pivot = m.pivot_table(index="model", columns="entity", values="value").fillna(0)
    ranks = pivot.rank(ascending=False, axis=0).mean(axis=1).sort_values()
    k, n = len(ranks), pivot.shape[1]
    # Nemenyi 임계 차이: q_alpha(k) * sqrt(k(k+1)/(6N)) — q값 근사(정규 기반 Tukey 근사)
    q_alpha = 2.343 + 0.407 * math.log(max(k - 1, 1))  # 근사식 (k<=20 오차 ~2%)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks.values,
            y=list(ranks.index),
            mode="markers+text",
            text=[f"{v:.2f}" for v in ranks.values],
            textposition="middle right",
        )
    )
    best = ranks.iloc[0]
    fig.add_vrect(
        x0=best,
        x1=best + cd,
        fillcolor="rgba(0,120,0,0.1)",
        line_width=0,
        annotation_text=f"CD={cd:.2f} (α={alpha})",
    )
    fig.update_layout(
        title=f"Critical Difference — 평균 순위 (공통 entity {n}개 기준)",
        xaxis_title="평균 순위 (낮을수록 좋음)",
        yaxis=dict(autorange="reversed"),
    )
    return _write(fig, out_dir, "critical_difference")


# 5 ---------------------------------------------------------------------------


def perf_vs_cost(df: pd.DataFrame, out_dir: Path) -> Path:
    m = df[df["metric"] == PRIMARY]
    agg = (
        m.groupby(["generation", "model"])
        .agg(vus_pr=("value", "mean"), runtime=("runtime_s", "mean"), mem=("peak_vram_mb", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    for gen in GEN_ORDER:
        g = agg[agg["generation"] == gen]
        if len(g):
            fig.add_trace(
                go.Scatter(
                    x=g["runtime"],
                    y=g["vus_pr"],
                    mode="markers+text",
                    text=g["model"],
                    textposition="top center",
                    name=GEN_LABEL[gen],
                    marker={"size": 8 + 4 * np.log1p(g["mem"].fillna(0))},
                )
            )
    fig.update_layout(
        title="성능 vs 비용 — VUS-PR vs 평균 실행시간 (마커 크기 ∝ 메모리)",
        xaxis={"title": "실행시간 (s, log)", "type": "log"},
        yaxis_title="평균 VUS-PR",
    )
    return _write(fig, out_dir, "perf_vs_cost")


# 6 ---------------------------------------------------------------------------


def metric_divergence(results_dir: str | Path, df: pd.DataFrame, out_dir: Path) -> Path:
    """PA-F1 vs VUS-PR: 저장된 스코어 npz에서 PA-F1(oracle 임계값)만 재계산.

    VUS-PR은 results parquet의 기존 값을 재사용한다 (재계산 없음 — run_id로 조인).
    """
    from sklearn.metrics import f1_score

    from tsad_forge.evaluation.metrics import point_adjust
    from tsad_forge.evaluation.protocol import load_scores

    m = df[df["metric"] == PRIMARY].copy()
    m["run_id"] = (
        m["model"]
        + "__"
        + m["dataset"]
        + "__"
        + m["channel"]
        + "__seed"
        + m["seed"].astype(str)
        + "__"
        + m["config_hash"]
    )
    vus_by_run = m.set_index("run_id")["value"].to_dict()

    rows = []
    for npz in sorted(Path(results_dir).glob("scores/*.npz")):
        model = npz.stem.split("__")[0]
        vus_pr = vus_by_run.get(npz.stem)
        if vus_pr is None:
            continue
        try:
            scores, labels = load_scores(npz)
        except Exception:
            continue
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        # PA-F1: oracle threshold sweep (문헌 최상 관행 재현 — 그래서 더 부풀려짐)
        best_pa = 0.0
        for q in np.linspace(0.80, 0.999, 30):
            pred = (scores >= np.quantile(scores, q)).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_pa = max(best_pa, f1_score(labels, point_adjust(pred, labels)))
        rows.append({"model": model, "vus_pr": vus_pr, "pa_f1": best_pa, "run": npz.stem})
    d = pd.DataFrame(rows)
    fig = go.Figure()
    if len(d):
        gen_map = df.drop_duplicates("model").set_index("model")["generation"].to_dict()
        d["generation"] = d["model"].map(gen_map).fillna("gen0")
        for gen in GEN_ORDER:
            g = d[d["generation"] == gen]
            if len(g):
                fig.add_trace(
                    go.Scatter(
                        x=g["vus_pr"],
                        y=g["pa_f1"],
                        mode="markers",
                        name=GEN_LABEL[gen],
                        hovertext=g["run"],
                    )
                )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": "gray"})
    fig.update_layout(
        title="평가 부풀림의 증거: PA-F1 (oracle) vs VUS-PR — 대각선 위쪽일수록 부풀림",
        xaxis={"title": "VUS-PR (주지표)", "range": [0, 1]},
        yaxis={"title": "PA-F1 (legacy, oracle threshold)", "range": [0, 1]},
    )
    return _write(fig, out_dir, "metric_divergence")


# 7 ---------------------------------------------------------------------------


def case_viewer(results_dir: str | Path, df: pd.DataFrame, out_dir: Path) -> Path:
    """대표 케이스: synthetic(seed 0) 시계열 + 세대별 대표 모델 스코어 오버레이."""
    from tsad_forge.data.schema import label_events
    from tsad_forge.evaluation.protocol import load_scores
    from tsad_forge.synthetic.generator import generate_synthetic

    ds = generate_synthetic(seed=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ds.test[:, 0], name="series", line={"color": "black", "width": 1}))
    for s, e in label_events(ds.labels):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.15)", line_width=0)

    gen_map = df.drop_duplicates("model").set_index("model")["generation"].to_dict()
    seen_gens: set[str] = set()
    for npz in sorted(Path(results_dir).glob("scores/*__synthetic__all__seed0__*.npz")):
        model = npz.stem.split("__")[0]
        gen = gen_map.get(model)
        if gen is None or gen in seen_gens:
            continue
        seen_gens.add(gen)
        scores, _ = load_scores(npz)
        if len(scores) != len(ds.test):
            continue
        norm = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
        fig.add_trace(
            go.Scatter(y=norm, name=f"{GEN_LABEL.get(gen, gen)}: {model}", opacity=0.7, yaxis="y2")
        )
    fig.update_layout(
        title="사례 뷰어 — synthetic(seed 0), 붉은 음영=정답 이상 구간",
        yaxis={"title": "value"},
        yaxis2={"title": "score (min-max)", "overlaying": "y", "side": "right"},
    )
    return _write(fig, out_dir, "case_viewer")


# 8 ---------------------------------------------------------------------------


def dataset_cards(out_dir: Path, data_dir: str | Path = "data") -> Path:
    """이상 이벤트 길이 분포 + 기본 통계 (로컬에 있는 데이터셋만)."""
    from tsad_forge.data.registry import load_dataset

    candidates = [
        ("synthetic", {}),
        ("smd", {"machine": "machine-1-1", "data_dir": data_dir}),
        ("psm", {"data_dir": data_dir}),
        ("skab", {"experiment": "valve1/0", "data_dir": data_dir}),
        (
            "nab",
            {"rel_path": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", "data_dir": data_dir},
        ),
    ]
    fig = go.Figure()
    stats_rows = []
    for name, kw in candidates:
        try:
            ds = load_dataset(name, **kw)
        except (FileNotFoundError, KeyError):
            continue
        from tsad_forge.data.schema import label_events

        lengths = [e - s for s, e in label_events(ds.labels)]
        if lengths:
            fig.add_trace(go.Box(y=lengths, name=ds.meta.get("name", name), boxpoints="all"))
        stats_rows.append(
            {
                "dataset": ds.meta.get("name", name),
                "D": ds.n_dims,
                "T_test": len(ds.test),
                "anomaly_rate": round(ds.anomaly_rate, 4),
                "n_events": len(lengths),
            }
        )
    fig.update_layout(
        title="데이터셋 카드 — 이상 이벤트 길이 분포 (로그 스케일)",
        yaxis={"title": "이벤트 길이 (스텝)", "type": "log"},
        annotations=[
            {
                "text": "<br>".join(
                    f"{r['dataset']}: D={r['D']}, T={r['T_test']}, "
                    f"rate={r['anomaly_rate']}, events={r['n_events']}"
                    for r in stats_rows
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 1.0,
                "showarrow": False,
                "align": "left",
                "font": {"size": 10},
            }
        ],
    )
    return _write(fig, out_dir, "dataset_cards")


# 통합 ------------------------------------------------------------------------


def generate_all(
    results_dir: str | Path = "benchmarks/results",
    out_dir: str | Path = "docs/assets/charts",
    data_dir: str | Path = "data",
) -> list[Path]:
    out_dir = Path(out_dir)
    df = _metric_frame(results_dir)
    paths = [
        generation_evolution(df, out_dir),
        leaderboard_table(df, out_dir),
        heatmap(df, out_dir),
        critical_difference(df, out_dir),
        perf_vs_cost(df, out_dir),
        metric_divergence(results_dir, df, out_dir),
        case_viewer(results_dir, df, out_dir),
        dataset_cards(out_dir, data_dir),
    ]
    print(f"generated {len(paths)} charts under {out_dir}")
    return paths
