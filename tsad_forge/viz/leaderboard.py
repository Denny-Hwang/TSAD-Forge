"""Leaderboard generation — styled HTML tables inside the markdown page.

The plain markdown table with 30 rows was hard to scan, so the generated page uses
HTML tables with (a) a colored generation badge per row, (b) a subtle row tint in the
generation's hue, and (c) a light sequential shading on the primary-metric column.
Values are seed averages; each row's config_hash resolves to the full configuration
in the results JSON (CLAUDE.md §9).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsad_forge.runner.results import load_all_results
from tsad_forge.viz.charts import GEN_COLOR, GEN_LABEL

PRIMARY_METRIC = "vus_pr"

# 10% tint of each generation hue for row backgrounds (readable zebra substitute)
_GEN_TINT = {
    "gen0": "rgba(82,81,78,0.07)",
    "gen1": "rgba(42,120,214,0.07)",
    "gen2": "rgba(235,104,52,0.07)",
    "gen3": "rgba(27,175,122,0.07)",
    "gen4": "rgba(237,161,0,0.09)",
    "gen5": "rgba(232,123,164,0.08)",
}

_TABLE_CSS = "border-collapse:collapse;width:100%;font-size:0.82rem;line-height:1.35;"
_TH = (
    "text-align:left;padding:6px 10px;border-bottom:2px solid #d5d5d2;"
    "background:#eef2f7;white-space:nowrap;"
)
_TD = "padding:5px 10px;border-bottom:1px solid #e8e8e6;"


def _badge(gen: str) -> str:
    color = GEN_COLOR.get(gen, "#52514e")
    label = gen.replace("gen", "Gen ")
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:10px;'
        f'background:{color};color:#ffffff;font-size:0.72rem;font-weight:600;"'
        f' title="{GEN_LABEL.get(gen, gen)}">{label}</span>'
    )


def _shade(value: float, vmax: float = 1.0) -> str:
    """Light sequential blue shading for the primary-metric cell."""
    if pd.isna(value):
        return ""
    alpha = 0.05 + 0.30 * max(0.0, min(float(value) / vmax, 1.0))
    return f"background:rgba(42,120,214,{alpha:.2f});"


def _wrap_head(h: str) -> str:
    return h.replace("/", "/<br>") if len(h) > 14 and "/" in h else h


def build_leaderboard(
    results_dir: str | Path, metric: str = PRIMARY_METRIC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pivot, summary) frames.

    pivot: index=(generation, model), columns=entity, values=metric (seed mean)
    summary: per-model mean metric, mean rank, entity coverage, runtime, config refs
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


def _summary_html(summary: pd.DataFrame, metric: str) -> str:
    head_cells = "".join(
        f'<th style="{_TH}">{h}</th>'
        for h in [
            "generation",
            "model",
            f"mean {metric}",
            "mean rank",
            "entities",
            "runtime (s)",
            "config",
        ]
    )
    rows = []
    for (gen, model), r in summary.iterrows():
        tint = _GEN_TINT.get(gen, "")
        cfg = str(r["config_hash"]).split(",")[0]
        rows.append(
            f'<tr style="background:{tint}">'
            f'<td style="{_TD}white-space:nowrap;">{_badge(gen)}</td>'
            f'<td style="{_TD}font-weight:600;">{model}</td>'
            f'<td style="{_TD}{_shade(r[f"mean_{metric}"])}text-align:right;">'
            f"{r[f'mean_{metric}']:.3f}</td>"
            f'<td style="{_TD}text-align:right;">{r["mean_rank"]:.2f}</td>'
            f'<td style="{_TD}text-align:right;">{int(r["n_entities"])}</td>'
            f'<td style="{_TD}text-align:right;">{r["mean_runtime_s"]:.2f}</td>'
            f'<td style="{_TD}"><code style="font-size:0.72rem;">{cfg}</code></td>'
            "</tr>"
        )
    return (
        f'<table style="{_TABLE_CSS}"><thead><tr>{head_cells}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _pivot_html(pivot: pd.DataFrame) -> str:
    entities = list(pivot.columns)
    head_cells = "".join(f'<th style="{_TH}">{h}</th>' for h in ["generation", "model"]) + "".join(
        f'<th style="{_TH}white-space:normal;min-width:86px;">{_wrap_head(e)}</th>'
        for e in entities
    )
    rows = []
    for (gen, model), r in pivot.iterrows():
        tint = _GEN_TINT.get(gen, "")
        cells = "".join(
            f'<td style="{_TD}{_shade(r[e])}text-align:right;">'
            + ("–" if pd.isna(r[e]) else f"{r[e]:.3f}")
            + "</td>"
            for e in entities
        )
        rows.append(
            f'<tr style="background:{tint}">'
            f'<td style="{_TD}white-space:nowrap;">{_badge(gen)}</td>'
            f'<td style="{_TD}font-weight:600;">{model}</td>{cells}</tr>'
        )
    return (
        f'<div style="overflow-x:auto;"><table style="{_TABLE_CSS}">'
        f"<thead><tr>{head_cells}</tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def leaderboard_markdown(results_dir: str | Path, metric: str = PRIMARY_METRIC) -> str:
    pivot, summary = build_leaderboard(results_dir, metric)
    # keep pivot rows in summary (rank) order
    pivot = pivot.loc[summary.index]
    lines = [
        f"# Leaderboard — {metric.upper().replace('_', '-')}",
        "",
        "Primary metric: **VUS-PR** (Paparrizos et al., VLDB 2022). Values are seed"
        " averages; cell shading follows the metric value, row tint and badge follow"
        " the generation. PA-F1 is intentionally not shown (CLAUDE.md §4).",
        "",
        "## Model summary (sorted by mean rank)",
        "",
        _summary_html(summary, metric),
        "",
        "## Model × dataset",
        "",
        _pivot_html(pivot),
        "",
        "Reproduce with `python benchmarks/run_all.py --profile configs/lite.yaml`"
        " — each row's config hash resolves to its full configuration in the results JSON.",
    ]
    return "\n".join(lines)


def save_leaderboard(
    results_dir: str | Path, out_path: str | Path, metric: str = PRIMARY_METRIC
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(leaderboard_markdown(results_dir, metric))
    return out_path
