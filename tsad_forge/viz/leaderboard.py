"""Leaderboard generation — styled HTML tables inside the markdown page.

The plain markdown table with 30 rows was hard to scan, so the generated page uses
HTML tables with (a) a colored generation badge per row, (b) a subtle row tint in the
generation's hue, and (c) a light sequential shading on the primary-metric column.
Values are seed averages; each row's config_hash resolves to the full configuration
in the results JSON (CLAUDE.md §9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tsad_forge.models.cards import get_card
from tsad_forge.runner.results import (
    add_entity_column,
    load_all_results,
    normalized_ranks,
    shared_entities,
)
from tsad_forge.viz.charts import GEN_COLOR, GEN_LABEL

PRIMARY_METRIC = "vus_pr"
_BOOTSTRAP_N = 2000
_BOOTSTRAP_SEED = 0
# 리더보드 행 → 재현 설정(run JSON) 링크의 베이스 (CLAUDE.md §9)
_RESULTS_BLOB_URL = "https://github.com/Denny-Hwang/TSAD-Forge/blob/main/benchmarks/results"

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


_FIDELITY_STYLE = {
    "simplified": "background:#fdecc8;color:#7a4b00;",
    "upstream_verified": "background:#d7f0e3;color:#0d5c3a;",
    "variant": "background:#ececea;color:#52514e;",
}


def _fidelity_badge(level: str) -> str:
    style = _FIDELITY_STYLE.get(level, "background:#eef2f7;color:#26425f;")
    return (
        f'<a href="../../models/" style="text-decoration:none;"><span style="padding:1px 7px;'
        f'border-radius:9px;font-size:0.72rem;{style}">{level}</span></a>'
    )


def _shade(value: float, vmax: float = 1.0) -> str:
    """Light sequential blue shading for the primary-metric cell."""
    if pd.isna(value):
        return ""
    alpha = 0.05 + 0.30 * max(0.0, min(float(value) / vmax, 1.0))
    return f"background:rgba(42,120,214,{alpha:.2f});"


def _wrap_head(h: str) -> str:
    return h.replace("/", "/<br>") if len(h) > 14 and "/" in h else h


def _bootstrap_ci(row: pd.Series, n_boot: int = _BOOTSTRAP_N) -> tuple[float, float]:
    """엔티티 축 bootstrap 95% CI (리뷰 P1 — 불확실성 없는 mean 단독 표기 금지).

    엔티티 3개 미만이면 CI가 무의미하므로 (nan, nan)을 반환한다.
    """
    vals = row.dropna().to_numpy()
    if len(vals) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def friedman_pvalue(pivot: pd.DataFrame) -> tuple[float, int, int] | None:
    """공통 엔티티에서의 Friedman 검정. (p, n_models, n_entities) 또는 None.

    "모델 간 순위 차이가 우연 수준인가"에 대한 전역 검정 — CD 다이어그램의 수치 근거.
    커버리지가 이질적일 때(예: DL 프로파일은 소수 엔티티만 실행) 전 모델 교집합이
    비어버리므로, CD 다이어그램과 같은 규칙으로 최대 커버리지의 90% 이상인 모델만
    남긴 뒤 그 안에서 무결측 열을 취한다.
    """
    coverage = pivot.notna().sum(axis=1)
    core = pivot.loc[coverage >= 0.9 * coverage.max()]
    complete = core.loc[:, core.notna().all(axis=0)]
    if complete.shape[0] < 3 or complete.shape[1] < 3:
        return None
    _, p = stats.friedmanchisquare(*[complete.loc[idx].to_numpy() for idx in complete.index])
    return float(p), complete.shape[0], complete.shape[1]


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
    m = df[df["metric"] == metric]
    if m.empty:
        raise ValueError(f"metric '{metric}' not found in results")
    m = add_entity_column(m)

    pivot = m.pivot_table(
        index=["generation", "model"], columns="entity", values="value", aggfunc="mean"
    )
    ci = pivot.apply(_bootstrap_ci, axis=1)
    shared = shared_entities(pivot)
    summary = pd.DataFrame(
        {
            f"mean_{metric}": pivot.mean(axis=1),
            "ci_lo": ci.str[0],
            "ci_hi": ci.str[1],
            # 원시 평균 순위는 entity별 참가 모델 수가 달라 척도가 섞인다 — 참고용
            "mean_rank": pivot.rank(ascending=False, axis=0).mean(axis=1),
            # 정렬 기준: entity별 [0,1] 정규화 순위의 평균 (0=최고)
            "norm_rank": normalized_ranks(pivot).mean(axis=1),
            # 전 모델 공통 entity에서의 평균 — 커버리지 편향 없는 유일한 직접 비교
            f"shared_{metric}": pivot[shared].mean(axis=1) if shared else np.nan,
            "n_entities": pivot.notna().sum(axis=1),
        }
    ).sort_values(["norm_rank", "mean_rank"])
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
    # 대표 run_id — config 셀이 실제 재현 설정(run JSON)으로 링크되게 (CLAUDE.md §9)
    first = df.sort_values(["dataset", "channel", "seed"]).groupby(["generation", "model"]).first()
    example = (
        first["dataset"]
        + "__"
        + first["channel"]
        + "__seed"
        + first["seed"].astype(str)
        + "__"
        + first["config_hash"]
    )
    example_runs: list[str] = [
        f"{idx[1]}__{rest}"  # type: ignore[index]  # MultiIndex(gen, model)
        for idx, rest in example.items()
    ]
    example = pd.Series(example_runs, index=example.index, name="example_run")
    summary = summary.join(example).round(4)
    summary["fidelity"] = [
        (get_card(str(model)) or {}).get("fidelity", "variant") for _, model in summary.index
    ]
    # attrs는 join/round를 거치며 보존이 보장되지 않으므로 마지막에 다시 붙인다
    summary.attrs["n_shared_entities"] = len(shared)
    return pivot.round(4), summary


def _summary_html(summary: pd.DataFrame, metric: str) -> str:
    head_cells = "".join(
        f'<th style="{_TH}">{h}</th>'
        for h in [
            "generation",
            "model",
            "fidelity",
            f"mean {metric} [95% CI]",
            f"shared-entity {metric}",
            "norm. rank",
            "entities",
            "runtime (s)",
            "config",
        ]
    )
    rows = []
    for idx, r in summary.iterrows():
        gen, model = str(idx[0]), str(idx[1])  # type: ignore[index]  # MultiIndex(gen, model)
        tint = _GEN_TINT.get(gen, "")
        cfg = str(r["config_hash"]).split(",")[0]
        ci = (
            f' <span style="font-size:0.72rem;color:#5f5e5b;">'
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]</span>"
            if pd.notna(r.get("ci_lo"))
            else ""
        )
        rows.append(
            f'<tr style="background:{tint}">'
            f'<td style="{_TD}white-space:nowrap;">{_badge(gen)}</td>'
            f'<td style="{_TD}font-weight:600;">{model}</td>'
            f'<td style="{_TD}">{_fidelity_badge(str(r["fidelity"]))}</td>'
            f'<td style="{_TD}{_shade(r[f"mean_{metric}"])}text-align:right;white-space:nowrap;">'
            f"{r[f'mean_{metric}']:.3f}{ci}</td>"
            f'<td style="{_TD}text-align:right;">'
            + ("–" if pd.isna(r[f"shared_{metric}"]) else f'{r[f"shared_{metric}"]:.3f}')
            + "</td>"
            f'<td style="{_TD}text-align:right;">{r["norm_rank"]:.2f}</td>'
            f'<td style="{_TD}text-align:right;">{int(r["n_entities"])}</td>'
            f'<td style="{_TD}text-align:right;">{r["mean_runtime_s"]:.2f}</td>'
            f'<td style="{_TD}"><a href="{_RESULTS_BLOB_URL}/{r["example_run"]}.json" '
            f'title="example run JSON (full config + metrics)">'
            f'<code style="font-size:0.72rem;">{cfg}</code></a></td>'
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
    for idx, r in pivot.iterrows():
        gen, model = str(idx[0]), str(idx[1])  # type: ignore[index]  # MultiIndex(gen, model)
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


def _coverage_line(summary: pd.DataFrame, metric: str) -> str:
    n_shared = summary.attrs.get("n_shared_entities", 0)
    cov = summary["n_entities"]
    if cov.min() == cov.max():
        return f"Every model was run on the same {int(cov.max())} entities."
    head = (
        f"**Coverage differs across models** ({int(cov.min())}–{int(cov.max())} entities)."
        f" `mean {metric}` averages over *each model's own* entities and is not a fair"
        " cross-model comparison; `shared-entity` averages only over the"
        f" **{n_shared}** entities every model was run on"
    )
    tail = (
        ", and `norm. rank` is the per-entity rank rescaled to [0, 1] (0 = best) so that"
        " entities with different numbers of competitors are comparable."
    )
    if n_shared == 0:
        head += " (none here — no model set is fully comparable on this page)"
    return head + tail


def leaderboard_markdown(results_dir: str | Path, metric: str = PRIMARY_METRIC) -> str:
    pivot, summary = build_leaderboard(results_dir, metric)
    # keep pivot rows in summary (rank) order
    pivot = pivot.loc[summary.index]
    fr = friedman_pvalue(pivot)
    friedman_line = (
        f"Friedman test across the {fr[1]} models with complete coverage of the"
        f" {fr[2]} shared entities: **p = {fr[0]:.2e}** — "
        + (
            "model ranking differences are statistically significant overall; see the"
            " critical-difference diagram for which pairs are separable."
            if fr[0] < 0.05
            else "ranking differences are **not** statistically significant at this"
            " benchmark size; treat adjacent ranks as ties."
        )
        if fr
        else "Friedman test skipped: not enough models with complete shared-entity coverage."
    )
    lines = [
        f"# Leaderboard — {metric.upper().replace('_', '-')}",
        "",
        '!!! warning "Lite profile — preview"',
        "    These numbers come from the **lite** subset (CI-sized: a handful of"
        " entities per dataset), not the full benchmark matrix. Means are shown with"
        " entity-bootstrap 95% CIs and an `entities` count — a mean over few entities"
        " with a wide CI is weak evidence. Do not quote ranks from this page without"
        " those qualifiers.",
        "",
        "Primary metric: **VUS-PR** (Paparrizos et al., VLDB 2022). Values are seed"
        " averages; cell shading follows the metric value, row tint and badge follow"
        " the generation. PA-F1 is intentionally not shown (CLAUDE.md §4).",
        "",
        friedman_line,
        "",
        _coverage_line(summary, metric),
        "",
        "`fidelity` (see the [model catalog](../models.md)): **simplified** rows measure"
        " this repository's simplified reimplementation, not the original method.",
        "",
        "## Model summary (sorted by normalized rank)",
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
