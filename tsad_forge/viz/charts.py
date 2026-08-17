"""Visualization suite (CLAUDE.md §6) — 8 Plotly HTML charts from results parquet + score npz.

1. generation_evolution: VUS-PR distribution per generation (box)
2. leaderboard_table: sortable table with a metric dropdown
3. heatmap: model x dataset VUS-PR
4. critical_difference: mean ranks + Nemenyi CD (own implementation)
5. perf_vs_cost: VUS-PR vs runtime scatter
6. metric_divergence: PA-F1 (recomputed offline from saved raw scores) vs VUS-PR —
   visual proof of PA inflation without polluting the default results
7. case_viewer: representative series + per-generation score overlays (stacked
   subplots — never a dual axis) with ground-truth shading
8. dataset_cards: anomaly event length distributions + per-dataset stats table

Layout rules applied everywhere: axis ranges padded so marks never touch the frame,
automargin on tick labels, recessive grid, legend outside the plot area, explicit
heights sized to content so labels cannot collide.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Two-line variants for category tick labels (6 long names collide on one line)
GEN_LABEL_2L = {
    "gen0": "Gen0<br>baseline",
    "gen1": "Gen1<br>Statistical",
    "gen2": "Gen2<br>Classical ML",
    "gen3": "Gen3<br>DL Recon",
    "gen4": "Gen4<br>Graph/Transformer",
    "gen5": "Gen5<br>SSM/Foundation",
}
# Validated categorical palette (fixed slot order; gen0 baseline wears neutral ink)
GEN_COLOR = {
    "gen0": "#52514e",
    "gen1": "#2a78d6",
    "gen2": "#eb6834",
    "gen3": "#1baf7a",
    "gen4": "#eda100",
    "gen5": "#e87ba4",
}
PRIMARY = "vus_pr"

_GRID = "#e8e8e6"
_INK = "#0b0b0b"
_INK2 = "#52514e"


def _base_layout(fig: go.Figure, title: str, height: int = 520) -> None:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 16, "color": _INK}, "x": 0.02, "xanchor": "left"},
        font={"size": 12, "color": _INK},
        height=height,
        margin={"l": 70, "r": 40, "t": 64, "b": 64},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, automargin=True)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, automargin=True)


def _padded_range(lo: float, hi: float, frac: float = 0.06) -> list[float]:
    span = (hi - lo) or 1.0
    return [lo - frac * span, hi + frac * span]


def _short(label: str, limit: int = 26) -> str:
    """Compact entity label for tick text (keep the dataset prefix, trim the rest)."""
    tail = label.rsplit("/", 1)
    name = tail[-1].removesuffix(".csv")
    # NAB-style directory prefixes carry no identity once the dataset is named
    name = re.sub(r"^(real|artificial)[A-Za-z]+_", "", name)
    prefix = tail[0] + "/" if len(tail) > 1 else ""
    out = prefix + name
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _metric_frame(results_dir: str | Path) -> pd.DataFrame:
    df = load_all_results(results_dir)
    if df.empty:
        raise FileNotFoundError(f"no results under {results_dir}")
    df = df.copy()
    df["entity"] = np.where(
        df["channel"] == "all", df["dataset"], df["dataset"] + "/" + df["channel"]
    )
    df["entity"] = df["entity"].map(_short)
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
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=GEN_LABEL_2L[gen],
                    boxmean=True,
                    marker_color=GEN_COLOR[gen],
                    line={"width": 2},
                )
            )
    _base_layout(fig, "VUS-PR by generation — does newer mean better?", height=540)
    fig.update_yaxes(title_text="VUS-PR (entities × seeds)", range=[-0.05, 1.05])
    fig.update_xaxes(tickangle=0, tickfont={"size": 11})
    fig.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 64, "b": 84})
    return _write(fig, out_dir, "generation_evolution")


# 2 ---------------------------------------------------------------------------


def leaderboard_table(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = sorted(df["metric"].unique())
    metrics = [m for m in metrics if m not in ("threshold", "n_predicted_anomalies")]
    if PRIMARY in metrics:  # primary metric first in the dropdown
        metrics = [PRIMARY] + [m for m in metrics if m != PRIMARY]
    pivots = {
        met: df[df["metric"] == met]
        .pivot_table(index=["generation", "model"], columns="entity", values="value")
        .round(3)
        for met in metrics
    }

    def _cells(p: pd.DataFrame):
        p = p.reset_index()
        return [p[c] for c in p.columns], list(p.columns)

    def _wrap_header(h: str) -> str:
        # long entity names break onto two lines at the dataset separator
        return h.replace("/", "/<br>") if len(h) > 14 and "/" in h else h

    cells, header = _cells(pivots[metrics[0]])
    n_cols = len(header)
    n_rows = len(cells[0])
    # Explicit width per column: the iframe scrolls horizontally instead of
    # squeezing 15+ columns into unreadable slivers.
    col_w = [110, 170] + [134] * (n_cols - 2)
    fig = go.Figure(
        go.Table(
            columnwidth=col_w,
            header={
                "values": [f"<b>{_wrap_header(h)}</b>" for h in header],
                "fill_color": "#eef2f7",
                "align": "left",
                "font": {"size": 11, "color": _INK},
                "height": 48,
            },
            cells={
                "values": cells,
                "align": "left",
                "font": {"size": 11, "color": _INK2},
                "height": 26,
                "fill_color": [["#fcfcfb", "#f4f4f2"] * (n_rows // 2 + 1)],
            },
        )
    )
    buttons = []
    for met in metrics:
        c, h = _cells(pivots[met])
        buttons.append(
            {
                "label": met,
                "method": "restyle",
                "args": [
                    {
                        "header.values": [[f"<b>{_wrap_header(x)}</b>" for x in h]],
                        "cells.values": [c],
                    }
                ],
            }
        )
    fig.update_layout(
        template="plotly_white",
        title={
            "text": "Leaderboard — pick a metric (primary: VUS-PR)",
            "font": {"size": 16},
            "x": 0.02,
            "xanchor": "left",
        },
        width=max(980, sum(col_w) + 60),
        height=140 + 26 * n_rows,
        margin={"l": 20, "r": 20, "t": 96, "b": 20},
        updatemenus=[
            {  # top-right so it never covers the title
                "buttons": buttons,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.04,
                "yanchor": "bottom",
                "direction": "down",
            }
        ],
        paper_bgcolor="#fcfcfb",
    )
    return _write(fig, out_dir, "leaderboard_table")


# 3 ---------------------------------------------------------------------------


def heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    m = df[df["metric"] == PRIMARY]
    pivot = m.pivot_table(index="model", columns="entity", values="value")
    order = pivot.mean(axis=1).sort_values(ascending=True).index  # best on top
    pivot = pivot.loc[order]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",  # sequential = one hue, light -> dark
            colorbar={"title": {"text": "VUS-PR", "side": "right"}, "thickness": 14},
            zmin=0,
            zmax=1,
            xgap=2,
            ygap=2,
            hovertemplate="model=%{y}<br>entity=%{x}<br>VUS-PR=%{z:.3f}<extra></extra>",
        )
    )
    _base_layout(fig, "Model × dataset VUS-PR heatmap", height=180 + 26 * len(pivot))
    fig.update_xaxes(tickangle=40, tickfont={"size": 11})
    fig.update_yaxes(tickfont={"size": 11})
    fig.update_layout(margin={"l": 130, "r": 60, "t": 64, "b": 130})
    return _write(fig, out_dir, "heatmap")


# 4 ---------------------------------------------------------------------------


def critical_difference(df: pd.DataFrame, out_dir: Path, alpha: float = 0.05) -> Path:
    """Mean-rank critical-difference view (Demšar 2006, Nemenyi) — own implementation."""
    m = df[df["metric"] == PRIMARY]
    pivot = m.pivot_table(index="model", columns="entity", values="value")
    # Shared-coverage selection: keep the entities covered by (almost) every model,
    # then the models that cover all of them — maximizes N for the rank test instead
    # of letting one sparse model shrink the shared set to nothing.
    coverage = pivot.notna().sum(axis=0)
    wide = coverage[coverage >= 0.9 * len(pivot)].index
    pivot = pivot[wide].dropna(axis=0, how="any") if len(wide) >= 2 else pivot
    pivot = pivot.dropna(axis=1, how="any")  # rank comparability
    if pivot.shape[1] < 2:
        warnings.warn("not enough shared entities for a CD diagram", stacklevel=2)
        pivot = m.pivot_table(index="model", columns="entity", values="value").fillna(0)
    ranks = pivot.rank(ascending=False, axis=0).mean(axis=1).sort_values()
    k, n = len(ranks), pivot.shape[1]
    # Nemenyi critical difference: q_alpha(k) * sqrt(k(k+1)/(6N)) (Tukey-based approx)
    q_alpha = 2.343 + 0.407 * math.log(max(k - 1, 1))  # ~2% error for k<=20
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    gen_map = df.drop_duplicates("model").set_index("model")["generation"].to_dict()

    fig = go.Figure()
    best = float(ranks.iloc[0])
    fig.add_vrect(
        x0=best,
        x1=best + cd,
        fillcolor="rgba(42,120,214,0.08)",
        line_width=0,
    )
    for gen in GEN_ORDER:  # one trace per generation -> legend carries identity
        sel = [mname for mname in ranks.index if gen_map.get(mname, "gen0") == gen]
        if not sel:
            continue
        fig.add_trace(
            go.Scatter(
                x=[ranks[mname] for mname in sel],
                y=sel,
                mode="markers",
                name=GEN_LABEL[gen],
                marker={"size": 10, "color": GEN_COLOR[gen]},
                hovertemplate="%{y}: mean rank %{x:.2f}<extra></extra>",
            )
        )
    _base_layout(
        fig,
        f"Critical difference — mean ranks over {n} shared entities "
        f"(shaded band = statistically tied with best, CD={cd:.2f}, α={alpha})",
        height=170 + 26 * k,
    )
    fig.update_xaxes(
        title_text="mean rank (lower is better)",
        range=_padded_range(float(ranks.min()), max(float(ranks.max()), best + cd), 0.08),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(ranks.index[::-1]),  # best on top
        tickfont={"size": 11},
    )
    fig.update_layout(
        margin={"l": 150, "r": 40, "t": 84, "b": 64},
        legend={"orientation": "h", "y": -0.10, "yanchor": "top", "x": 0},
    )
    return _write(fig, out_dir, "critical_difference")


# 5 ---------------------------------------------------------------------------


def perf_vs_cost(df: pd.DataFrame, out_dir: Path) -> Path:
    m = df[df["metric"] == PRIMARY]
    agg = (
        m.groupby(["generation", "model"])
        .agg(vus_pr=("value", "mean"), runtime=("runtime_s", "mean"), mem=("peak_mem_mb", "mean"))
        .reset_index()
    )
    # Direct-label only the best model per generation; everything else is hover-only
    # (labelling all ~30 points is exactly the collision mess we're avoiding).
    top = set(agg.loc[agg.groupby("generation")["vus_pr"].idxmax(), "model"])
    fig = go.Figure()
    for gen in GEN_ORDER:
        g = agg[agg["generation"] == gen]
        if not len(g):
            continue
        fig.add_trace(
            go.Scatter(
                x=g["runtime"],
                y=g["vus_pr"],
                mode="markers+text",
                text=[name if name in top else "" for name in g["model"]],
                textposition="top center",
                textfont={"size": 11, "color": _INK2},
                name=GEN_LABEL[gen],
                marker={
                    "size": 9 + 3 * np.log1p(g["mem"].fillna(0)),
                    "color": GEN_COLOR[gen],
                    "line": {"width": 2, "color": "#fcfcfb"},
                },
                customdata=g["model"],
                hovertemplate="%{customdata}<br>runtime=%{x:.2f}s<br>VUS-PR=%{y:.3f}<extra></extra>",
            )
        )
    _base_layout(
        fig,
        "Performance vs cost — VUS-PR vs mean fit+score runtime (marker size ~ peak host memory)",
        height=560,
    )
    lo, hi = float(agg["runtime"].min()), float(agg["runtime"].max())
    log_lo, log_hi = math.log10(max(lo, 1e-3)), math.log10(max(hi, 1e-3))
    pad = 0.08 * (log_hi - log_lo or 1.0)
    fig.update_xaxes(
        title_text="mean runtime (s, log scale)",
        type="log",
        range=[log_lo - pad, log_hi + pad],
        dtick=1,  # decade ticks only — minor log labels collide at this density
    )
    fig.update_yaxes(
        title_text="mean VUS-PR",
        range=_padded_range(float(agg["vus_pr"].min()), float(agg["vus_pr"].max()), 0.12),
    )
    fig.update_layout(legend={"orientation": "h", "y": -0.15, "yanchor": "top", "x": 0})
    return _write(fig, out_dir, "perf_vs_cost")


# 6 ---------------------------------------------------------------------------


def metric_divergence(results_dir: str | Path, df: pd.DataFrame, out_dir: Path) -> Path:
    """PA-F1 vs VUS-PR: recompute only PA-F1 (oracle threshold) from saved score npz.

    VUS-PR is reused from the results parquet (joined by run_id, no recompute).
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
        best_pa = 0.0  # oracle threshold sweep — the literature's best practice, hence inflated
        for q in np.linspace(0.80, 0.999, 30):
            pred = (scores >= np.quantile(scores, q)).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_pa = max(best_pa, f1_score(labels, point_adjust(pred, labels)))
        rows.append({"model": model, "vus_pr": vus_pr, "pa_f1": best_pa, "run": npz.stem})
    d = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": _INK2, "width": 1}
    )
    fig.add_annotation(  # placed in the (empty) below-diagonal half
        x=0.78,
        y=0.10,
        text="above the diagonal = PA inflation",
        showarrow=False,
        font={"size": 11, "color": _INK2},
    )
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
                        marker={
                            "size": 8,
                            "color": GEN_COLOR[gen],
                            "line": {"width": 1.5, "color": "#fcfcfb"},
                        },
                        customdata=g["run"],
                        hovertemplate="%{customdata}<br>VUS-PR=%{x:.3f}<br>PA-F1=%{y:.3f}<extra></extra>",
                    )
                )
    _base_layout(fig, "Evidence of evaluation inflation: PA-F1 (oracle) vs VUS-PR", height=560)
    fig.update_xaxes(title_text="VUS-PR (primary metric)", range=[-0.04, 1.04])
    fig.update_yaxes(title_text="PA-F1 (legacy, oracle threshold)", range=[-0.04, 1.04])
    fig.update_layout(legend={"orientation": "h", "y": -0.15, "yanchor": "top", "x": 0})
    return _write(fig, out_dir, "metric_divergence")


# 7 ---------------------------------------------------------------------------


def case_viewer(results_dir: str | Path, df: pd.DataFrame, out_dir: Path) -> Path:
    """Representative case: series on top, per-generation scores below (no dual axis)."""
    from tsad_forge.data.schema import label_events
    from tsad_forge.evaluation.protocol import load_scores
    from tsad_forge.synthetic.generator import generate_synthetic

    ds = generate_synthetic(seed=0)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.65],
        vertical_spacing=0.06,
        subplot_titles=(
            "series (red shading = ground-truth anomalies)",
            "anomaly scores (min-max scaled per model)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            y=ds.test[:, 0], name="series", line={"color": _INK, "width": 1}, showlegend=False
        ),
        row=1,
        col=1,
    )
    for s, e in label_events(ds.labels):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(208,59,59,0.15)", line_width=0, row=1, col=1)
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(208,59,59,0.08)", line_width=0, row=2, col=1)

    # Representative = the BEST model per generation on this entity (by VUS-PR),
    # not the first file found — a random baseline as "representative" is noise.
    m = df[(df["metric"] == PRIMARY) & (df["dataset"] == "synthetic") & (df["seed"] == 0)]
    best = (
        m.sort_values("value", ascending=False)
        .drop_duplicates("generation")
        .set_index("generation")["model"]
        .to_dict()
    )
    npz_by_model = {
        p.stem.split("__")[0]: p
        for p in sorted(Path(results_dir).glob("scores/*__synthetic__all__seed0__*.npz"))
    }
    for gen in GEN_ORDER:
        if gen == "gen0":
            continue  # the uninformed baseline is pure noise here — skip for legibility
        model = best.get(gen)
        npz = npz_by_model.get(model)
        if npz is None:
            continue
        scores, _ = load_scores(npz)
        if len(scores) != len(ds.test):
            continue
        norm = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
        fig.add_trace(
            go.Scatter(
                y=norm,
                name=f"{GEN_LABEL[gen]}: {model}",
                line={"color": GEN_COLOR[gen], "width": 1.8},
                opacity=0.9,
            ),
            row=2,
            col=1,
        )
    _base_layout(fig, "Case viewer — synthetic (seed 0)", height=700)
    fig.update_yaxes(title_text="value", automargin=True, row=1, col=1)
    fig.update_yaxes(title_text="score (0–1)", range=[-0.06, 1.06], row=2, col=1)
    fig.update_xaxes(title_text="time step", title_standoff=8, row=2, col=1)
    fig.update_layout(  # legend well below the x title — no overlap at any width
        legend={"orientation": "h", "y": -0.16, "yanchor": "top", "x": 0},
        margin={"l": 80, "r": 40, "t": 64, "b": 150},
    )
    return _write(fig, out_dir, "case_viewer")


# 8 ---------------------------------------------------------------------------


def dataset_cards(out_dir: Path, data_dir: str | Path = "data") -> Path:
    """Event-length distributions (box) + a separate stats table — no overlapping text."""
    from tsad_forge.data.registry import load_dataset
    from tsad_forge.data.schema import label_events

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
    boxes = []
    stats_rows = []
    for name, kw in candidates:
        try:
            ds = load_dataset(name, **kw)
        except (FileNotFoundError, KeyError):
            continue
        lengths = [e - s for s, e in label_events(ds.labels)]
        if lengths:
            boxes.append((ds.meta.get("name", name), lengths))
        stats_rows.append(
            {
                "dataset": ds.meta.get("name", name),
                "dims": ds.n_dims,
                "test length": len(ds.test),
                "anomaly rate": round(ds.anomaly_rate, 4),
                "events": len(lengths),
            }
        )
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.12,
        specs=[[{"type": "box"}], [{"type": "table"}]],
        subplot_titles=("anomaly event length (steps, log scale)", ""),
    )
    for i, (name, lengths) in enumerate(boxes):
        fig.add_trace(
            go.Box(
                y=lengths,
                name=name,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"size": 5, "color": list(GEN_COLOR.values())[1:][i % 5]},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        fig.add_trace(
            go.Table(
                columnwidth=[220, 70, 110, 120, 80],
                header={
                    "values": [f"<b>{c}</b>" for c in sdf.columns],
                    "fill_color": "#eef2f7",
                    "align": "left",
                    "font": {"size": 12},
                    "height": 40,
                },
                cells={
                    "values": [sdf[c] for c in sdf.columns],
                    "align": "left",
                    "font": {"size": 12, "color": _INK2},
                    "height": 26,
                },
            ),
            row=2,
            col=1,
        )
    _base_layout(fig, "Dataset cards — event lengths and basic statistics", height=680)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(tickangle=15, row=1, col=1)
    return _write(fig, out_dir, "dataset_cards")


# aggregate -------------------------------------------------------------------


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
