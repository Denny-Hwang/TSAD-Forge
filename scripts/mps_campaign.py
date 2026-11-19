#!/usr/bin/env python
"""맥미니(Apple MPS) 장기 실험 캠페인 드라이버 — 계획: benchmarks/CAMPAIGN_MPS.md.

여러 날 걸리는 실행을 단계(phase)로 나눠 순서대로 돌리고, 언제 멈춰도 같은 명령으로
이어서 돌 수 있게 한다.

- 재개: 결과 parquet이 이미 있는 실행은 건너뛴다 (run_all과 같은 resume 규칙).
  단계 상태는 _campaign/state.json에 남는다. "done"인 단계는 다시 보지 않고,
  "running"이었던 단계는 남은 실행만 다시 돈다.
- 실패/timeout은 단계를 막지 않는다. 마지막 retry 단계가 남은 실행을 긴 timeout,
  동시 실행 1개로 한 번 더 돈다.
- 기록: 단계 시작/종료는 state.json과 events.jsonl, 실행별 진행은 campaign.log
  (래퍼가 stdout을 붙인다), 15분마다 STATUS.md, 단계가 끝날 때마다 reports/*.md.

사용 (보통은 scripts/mps_campaign.sh를 쓴다):
    python scripts/mps_campaign.py run       # 남은 단계 실행 (재개)
    python scripts/mps_campaign.py status    # 진행률·ETA·실패 (언제든)
    python scripts/mps_campaign.py report    # 비교 보고서 다시 쓰기 (언제든)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results_mps"
CAMPAIGN_DIR = RESULTS_DIR / "_campaign"
STATE_PATH = CAMPAIGN_DIR / "state.json"
EVENTS_PATH = CAMPAIGN_DIR / "events.jsonl"
STATUS_PATH = CAMPAIGN_DIR / "STATUS.md"
REPORTS_DIR = CAMPAIGN_DIR / "reports"

LITE_DL_GPU = "configs/lite_dl_gpu_mps.yaml"
ABLATION = "configs/mamba_ablation_mps.yaml"
PAPER_REPRO = "configs/mamba_paper_repro_mps.yaml"
CPU_RESULTS = ROOT / "benchmarks" / "results"
STATUS_EVERY_S = 900


def _always(job) -> bool:
    return True


def _default_synthetic(job) -> bool:
    o = job.overrides
    return o.get("data") == "synthetic" and not o.get("data_params")


def _machine_1_6(job) -> bool:
    return (job.overrides.get("data_params") or {}).get("machine") == "machine-1-6"


@dataclass(frozen=True)
class Phase:
    id: str
    title: str
    profile: str
    seeds: tuple[int, ...] | None = None
    select: Callable = _always
    torch_jobs: int = 1
    timeout_s: float = 14400
    report: str | None = None  # 단계 종료 후 쓸 보고서 ("ablation" | "lite_dl_gpu")


PHASES: list[Phase] = [
    Phase(
        "p0_smoke",
        "Smoke: every lite_dl_gpu model once on MPS (default synthetic, seed 0)",
        LITE_DL_GPU,
        seeds=(0,),
        select=_default_synthetic,
        torch_jobs=1,
        timeout_s=3600,
    ),
    Phase(
        "p1_ablation_m16",
        "MambaTSAD ablation (fix a/c/d) on SMD machine-1-6, seeds 0-2",
        ABLATION,
        select=_machine_1_6,
        torch_jobs=1,
        timeout_s=14400,
        report="ablation",
    ),
    Phase(
        "p2_ablation_rest",
        "MambaTSAD ablation (fix a/c/d) on the other 7 entities, seeds 0-2",
        ABLATION,
        torch_jobs=1,
        timeout_s=14400,
        report="ablation",
    ),
    Phase(
        "p3_lite_dl_gpu_seed0",
        "lite_dl_gpu on MPS, all 21 entities, seed 0",
        LITE_DL_GPU,
        seeds=(0,),
        torch_jobs=2,
        timeout_s=14400,
        report="lite_dl_gpu",
    ),
    Phase(
        "p4_lite_dl_gpu_seed12",
        "lite_dl_gpu on MPS, all 21 entities, seeds 1-2",
        LITE_DL_GPU,
        seeds=(1, 2),
        torch_jobs=2,
        timeout_s=14400,
        report="lite_dl_gpu",
    ),
]
RETRY = Phase(
    "p5_retry",
    "Retry every unfinished run of p0-p4 once (timeout 12 h, one at a time)",
    LITE_DL_GPU,  # 실제 대상은 retry_jobs()가 모은다
    torch_jobs=1,
    timeout_s=43200,
    report="all",
)


# --- 상태 -------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"phases": {}}


def save_state(state: dict) -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)  # 원자적 교체 — 중간에 죽어도 state.json이 깨지지 않는다


def log_event(kind: str, **fields) -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps({"time": _now(), "event": kind, **fields}, default=str) + "\n")
    print(f"[campaign {_now()}] {kind} {json.dumps(fields, default=str)}", flush=True)


# --- 계획 -------------------------------------------------------------------


def phase_jobs(phase: Phase) -> list:
    from tsad_forge.runner.matrix import build_jobs, load_profile

    profile = load_profile(ROOT / phase.profile)
    seeds = list(phase.seeds) if phase.seeds is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jobs, _ = build_jobs(profile, str(RESULTS_DIR), seeds=seeds)
    return [j for j in jobs if phase.select(j)]


def retry_jobs(state: dict) -> list:
    no_data = set(state.get("no_data", []))
    seen: dict[str, object] = {}
    for phase in PHASES:
        for j in phase_jobs(phase):
            if not j.exists and j.overrides.get("data") not in no_data:
                seen.setdefault(j.run_id, j)
    return list(seen.values())


# --- 실행 -------------------------------------------------------------------


def _run_phase(phase: Phase, jobs: list, state: dict) -> None:
    from tsad_forge.runner.matrix import run_matrix

    entry = state["phases"].setdefault(phase.id, {})
    todo = [j for j in jobs if not j.exists]
    entry.update(
        status="running",
        title=phase.title,
        started=entry.get("started") or _now(),
        resumed=_now(),
        n_total=len(jobs),
        n_todo_at_start=len(todo),
    )
    save_state(state)
    log_event("phase_start", phase=phase.id, total=len(jobs), todo=len(todo))
    summary = run_matrix(
        todo,
        results_dir=str(RESULTS_DIR),
        n_jobs=1,
        torch_jobs=phase.torch_jobs,
        timeout=phase.timeout_s,
    )
    state["no_data"] = sorted(set(state.get("no_data", [])) | {str(d) for d in summary.no_data})
    remaining = [j for j in jobs if not _exists(j)]
    entry.update(
        status="done" if not remaining else "done_with_failures",
        finished=_now(),
        summary=summary.line(),
        failed=[{"tag": list(t), "kind": k} for t, k in summary.failed],
        n_remaining=len(remaining),
    )
    save_state(state)
    log_event("phase_end", phase=phase.id, summary=summary.line(), remaining=len(remaining))


def _exists(job) -> bool:
    return (RESULTS_DIR / f"{job.run_id}.parquet").exists()


def _status_loop(stop: threading.Event) -> None:
    while not stop.wait(STATUS_EVERY_S):
        try:
            write_status()
        except Exception as e:  # noqa: BLE001 — 상태 기록 실패가 실행을 멈추면 안 된다
            print(f"[campaign] status write failed: {e}", flush=True)


def run_campaign() -> int:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN_DIR / "driver.pid").write_text(str(os.getpid()))
    state = load_state()
    log_event("driver_start", pid=os.getpid())
    stop = threading.Event()
    threading.Thread(target=_status_loop, args=(stop,), daemon=True).start()
    try:
        for phase in [*PHASES, RETRY]:
            if state["phases"].get(phase.id, {}).get("status") in ("done", "done_with_failures"):
                continue
            jobs = retry_jobs(state) if phase is RETRY else phase_jobs(phase)
            _run_phase(phase, jobs, state)
            if phase.report:
                _safe_reports()
            write_status()
        state["finished"] = _now()
        save_state(state)
        log_event("campaign_finished")
        _safe_reports()
        write_status()
    finally:
        stop.set()
        (CAMPAIGN_DIR / "driver.pid").unlink(missing_ok=True)
    return 0


# --- 상태 보고 ----------------------------------------------------------------


def _runtime_table() -> pd.DataFrame:
    rows = []
    for js in RESULTS_DIR.glob("*.json"):
        try:
            d = json.loads(js.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(
            {"model": d.get("model"), "dataset": d.get("dataset"), "rt": d.get("runtime_s")}
        )
    return pd.DataFrame(rows, columns=["model", "dataset", "rt"])


def _estimate_s(job, rt: pd.DataFrame) -> float:
    """남은 실행 1개의 예상 시간: 같은 (모델, 데이터셋) → 같은 데이터셋 → 전체 중앙값."""
    model, dataset = job.tag[0], job.tag[1]
    for sel in (
        (rt["model"] == model) & (rt["dataset"] == dataset),
        rt["dataset"] == dataset,
        rt["model"].notna(),
    ):
        vals = rt.loc[sel, "rt"].dropna()
        if len(vals):
            return float(vals.median())
    return 1800.0


def status_text() -> str:
    from tsad_forge.runner.matrix import load_failures

    state = load_state()
    rt = _runtime_table()
    lines = [f"# MPS campaign status — {_now()}", ""]
    pid_file = CAMPAIGN_DIR / "driver.pid"
    alive = False
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            alive = True
        except (OSError, ValueError):
            pass
    lines.append(f"Driver running: **{'yes' if alive else 'no'}**")
    lines += [
        "",
        "| phase | status | done / total | pending | est. remaining |",
        "|---|---|---|---|---|",
    ]
    total_eta = 0.0
    for phase in [*PHASES, RETRY]:
        entry = state["phases"].get(phase.id, {})
        if phase is RETRY:
            jobs = retry_jobs(state) if entry else []
            n_total = entry.get("n_total", len(jobs))
            pending = [j for j in jobs if not _exists(j)]
            n_done = n_total - len(pending)
        else:
            jobs = phase_jobs(phase)
            pending = [j for j in jobs if not _exists(j)]
            n_total, n_done = len(jobs), len(jobs) - len(pending)
        eta = sum(_estimate_s(j, rt) for j in pending) / max(phase.torch_jobs, 1)
        if entry.get("status") not in ("done", "done_with_failures"):
            total_eta += eta
        lines.append(
            f"| {phase.id} | {entry.get('status', 'not started')} | {n_done} / {n_total}"
            f" | {len(pending)} | {eta / 3600:.1f} h |"
        )
    lines += ["", f"Estimated time left for unfinished phases: **{total_eta / 3600:.1f} h**"]
    lines.append("(medians of finished runs with the same model and dataset; rough)")
    failures = load_failures(RESULTS_DIR)
    if failures:
        lines += ["", f"## Recorded failures ({len(failures)})", ""]
        lines += [f"- `{f['run_id']}` — {f['kind']} ({f.get('timestamp', '')})" for f in failures]
    orphans = [p.name for p in RESULTS_DIR.glob("*.parquet") if not p.with_suffix(".json").exists()]
    if orphans:
        lines += ["", "## Parquet without JSON (run killed mid-write — delete and rerun)", ""]
        lines += [f"- `{o}`" for o in orphans]
    return "\n".join(lines) + "\n"


def write_status() -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(status_text())


# --- 비교 보고서 --------------------------------------------------------------


def _load_runs(jobs: list) -> pd.DataFrame:
    """완료된 실행의 JSON 요약 → (model, generation, dataset, channel, seed, vus_pr, runtime_s)."""
    rows = []
    for j in jobs:
        js = RESULTS_DIR / f"{j.run_id}.json"
        if not js.exists():
            continue
        d = json.loads(js.read_text())
        rows.append(
            {
                "model": d["model"],
                "generation": d["generation"],
                "dataset": d["dataset"],
                "channel": d["channel"],
                "seed": d["seed"],
                "value": d["metrics"].get("vus_pr"),
                "runtime_s": d.get("runtime_s"),
            }
        )
    cols = ["model", "generation", "dataset", "channel", "seed", "value", "runtime_s"]
    return pd.DataFrame(rows, columns=cols)


def _entity(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].where(df["channel"] == "all", df["dataset"] + "/" + df["channel"])


def _boot_ci(vals: np.ndarray, n: int = 10000) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.3f}"


def report_ablation() -> str:
    from scipy import stats

    jobs = phase_jobs(Phase("x", "", ABLATION))
    df = _load_runs(jobs)
    lines = [
        "# MambaTSAD ablation on MPS — relative comparison",
        "",
        f"Generated {_now()}. Profile `{ABLATION}`, VUS-PR, seed means.",
        "",
        "- Variants apply one fix to `faithful`: fix_a = hidden-state index (last step),"
        " fix_c = standard HP-filter objective, fix_d = per-sample AMA period."
        " `fixed` applies all of them. Fix (b) (CPU/GPU branch) does not change the computation"
        " here, so it has no variant.",
        "- MPS numbers are **not** comparable to CUDA or to the paper (which reports PA-F1 /"
        " affiliation-F1). Only differences between variants on the same device mean anything.",
        "",
    ]
    if df.empty:
        return "\n".join([*lines, "No finished runs yet."]) + "\n"
    df["entity"] = _entity(df)
    seeds = df.groupby(["model", "entity"])["seed"].nunique().unstack()
    mean = df.groupby(["model", "entity"])["value"].mean().unstack()
    order = [
        m
        for m in (
            "mamba_tsad_faithful",
            "mamba_tsad_fix_a",
            "mamba_tsad_fix_c",
            "mamba_tsad_fix_d",
            "mamba_tsad_fixed",
        )
        if m in mean.index
    ]
    mean, seeds = mean.loc[order], seeds.loc[order]
    ents = list(mean.columns)
    lines += ["## VUS-PR by entity (seed mean; seeds finished in brackets)", ""]
    lines.append("| variant | " + " | ".join(ents) + " | mean |")
    lines.append("|---" * (len(ents) + 2) + "|")
    for m in order:
        cells = [
            f"{_fmt(mean.at[m, e])} [{int(seeds.at[m, e]) if pd.notna(seeds.at[m, e]) else 0}]"
            for e in ents
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {_fmt(mean.loc[m].mean())} |")
    if "mamba_tsad_faithful" in mean.index:
        base = mean.loc["mamba_tsad_faithful"]
        lines += [
            "",
            "## Difference from faithful (variant − faithful, entity-bootstrap 95% CI)",
            "",
            "| variant | entities | mean Δ | 95% CI | median Δ | wins / losses | Wilcoxon p |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in order[1:]:
            d = (mean.loc[m] - base).dropna().to_numpy()
            if not len(d):
                continue
            lo, hi = _boot_ci(d)
            p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
            lines.append(
                f"| {m} | {len(d)} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
                f" {np.median(d):+.3f} | {int((d > 0).sum())} / {int((d < 0).sum())} | {_fmt(p)} |"
            )
        m16 = "smd/machine-1-6"
        if m16 in mean.columns:
            lines += ["", f"## {m16} by seed", ""]
            per_seed = df[df["entity"] == m16].pivot_table(
                index="model", columns="seed", values="value"
            )
            per_seed = per_seed.loc[[m for m in order if m in per_seed.index]]
            lines.append("| variant | " + " | ".join(f"seed {s}" for s in per_seed.columns) + " |")
            lines.append("|---" * (len(per_seed.columns) + 1) + "|")
            for m, row in per_seed.iterrows():
                lines.append(f"| {m} | " + " | ".join(_fmt(v) for v in row) + " |")
    rt = df.groupby("model")["runtime_s"].agg(["count", "median", "sum"])
    lines += ["", "## Runtime", "", "| variant | runs | median s | total h |", "|---|---|---|---|"]
    for m, r in rt.loc[[m for m in order if m in rt.index]].iterrows():
        lines.append(f"| {m} | {int(r['count'])} | {r['median']:.0f} | {r['sum'] / 3600:.2f} |")
    return "\n".join(lines) + "\n"


def report_lite_dl_gpu() -> str:
    from tsad_forge.runner.results import add_entity_column, load_all_results, normalized_ranks
    from tsad_forge.viz.leaderboard import friedman_pvalue

    jobs = phase_jobs(Phase("x", "", LITE_DL_GPU))
    mps = _load_runs(jobs)
    lines = [
        "# lite_dl_gpu on MPS vs the CPU lite leaderboard — relative comparison",
        "",
        f"Generated {_now()}. VUS-PR, seed means.",
        "",
        "- `<model>@mps` rows: Gen3-5 with the real training budget (`configs/lite_dl_gpu.yaml`:"
        " up to 50 epochs, early stopping on a temporal validation split, window 100) on Apple MPS.",
        "- Plain Gen3-4 rows: the CPU smoke budget (`configs/lite_dl.yaml`, 3-5 epochs) — under-trained DL.",
        "- Gen0-2 rows are CPU results from `benchmarks/results` (no torch; device does not matter).",
        "- MPS numbers are not comparable to CUDA or published numbers. Entities with unfinished"
        " runs make the shared-entity set smaller; check the counts.",
        "",
    ]
    if mps.empty:
        return "\n".join([*lines, "No finished lite_dl_gpu runs yet."]) + "\n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = load_all_results(CPU_RESULTS)
    cpu = cpu[cpu["metric"] == "vus_pr"]
    mps = mps.assign(model=mps["model"] + "@mps")
    # data_hash는 parquet에만 있다 — synthetic 변형 구분을 위해 CPU 쪽과 같은 규칙으로 맞춘다
    from tsad_forge.runner.results import config_hash

    def _dh(job) -> str:
        o = job.overrides
        return config_hash({"data": o["data"], "data_params": o.get("data_params", {})})

    dh = {(j.tag[0] + "@mps", j.tag[1], j.tag[2], j.tag[3]): _dh(j) for j in jobs}
    mps["data_hash"] = [
        dh.get((r.model, r.dataset, r.channel, r.seed), "") for r in mps.itertuples()
    ]
    cols = ["model", "generation", "dataset", "channel", "seed", "data_hash", "value"]
    both = add_entity_column(pd.concat([cpu[cols], mps[cols]], ignore_index=True))
    pivot = both.pivot_table(index=["generation", "model"], columns="entity", values="value")
    mps_rows = [i for i in pivot.index if str(i[1]).endswith("@mps")]
    full_mps = [i for i in mps_rows if pivot.loc[i].notna().all()]
    keep = [i for i in pivot.index if not str(i[1]).endswith("@mps")] + full_mps
    lines += [
        f"MPS models with all {pivot.shape[1]} entities finished: **{len(full_mps)} / {len(mps_rows)}**"
        " (only these enter the combined ranking below).",
        "",
    ]
    # 1) 학습 예산 효과: 같은 모델의 CPU 스모크 vs MPS 실제 예산 (공통 엔티티에서)
    lines += [
        "## Training-budget effect (same model, entities finished on MPS)",
        "",
        "| model | entities | CPU smoke | MPS budget | Δ | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for gen, model in mps_rows:
        base = model.removesuffix("@mps")
        if (gen, base) not in pivot.index:
            lines.append(
                f"| {model} | {int(pivot.loc[(gen, model)].notna().sum())} | — |"
                f" {_fmt(pivot.loc[(gen, model)].mean())} | — | — |"
            )
            continue
        pair = pivot.loc[[(gen, base), (gen, model)]].dropna(axis=1)
        d = (pair.iloc[1] - pair.iloc[0]).to_numpy()
        lo, hi = _boot_ci(d)
        lines.append(
            f"| {model} | {pair.shape[1]} | {_fmt(pair.iloc[0].mean())} | {_fmt(pair.iloc[1].mean())}"
            f" | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
        )
    # 2) 통합 순위: Gen0-2 CPU + Gen3-4 CPU 스모크 + 전 엔티티 완료된 @mps
    sub = pivot.loc[keep]
    ranks = normalized_ranks(sub).mean(axis=1)
    summary = pd.DataFrame(
        {"mean": sub.mean(axis=1), "norm_rank": ranks, "n": sub.notna().sum(axis=1)}
    ).sort_values("norm_rank")
    fr = friedman_pvalue(sub)
    lines += [
        "",
        "## Combined ranking (normalized rank, 0 = best)",
        "",
        (
            f"Friedman test, {fr[1]} models × {fr[2]} entities: p = {fr[0]:.2e}"
            if fr
            else "Friedman test skipped (not enough complete rows)."
        ),
        "",
        "| # | generation | model | mean VUS-PR | 95% CI | norm. rank | entities |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, ((gen, model), r) in enumerate(summary.iterrows(), 1):
        lo, hi = _boot_ci(sub.loc[(gen, model)].to_numpy(dtype=float))
        lines.append(
            f"| {k} | {gen} | {model} | {_fmt(r['mean'])} | [{_fmt(lo)}, {_fmt(hi)}] |"
            f" {r['norm_rank']:.3f} | {int(r['n'])} |"
        )
    # 3) 세대별 분포: 기본 설정 행만 (@tuned 제외), DL은 @mps(실제 예산)만
    gen_rows = [
        i
        for i in keep
        if "@tuned" not in str(i[1])
        and (i[0] in ("gen0", "gen1", "gen2") or str(i[1]).endswith("@mps"))
    ]
    if gen_rows:
        long = sub.loc[gen_rows].stack().rename("v").reset_index()
        g = long.groupby("generation")["v"].describe()[["count", "mean", "25%", "50%", "75%"]]
        lines += [
            "",
            "## By generation (default-config rows; Gen3-5 = @mps real budget only)",
            "",
            "| generation | points | mean | q1 | median | q3 |",
            "|---|---|---|---|---|---|",
        ]
        for gen, r in g.iterrows():
            lines.append(
                f"| {gen} | {int(r['count'])} | {r['mean']:.3f} | {r['25%']:.3f} |"
                f" {r['50%']:.3f} | {r['75%']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(which: str = "all") -> list[Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    if which in ("all", "ablation"):
        p = REPORTS_DIR / "ablation.md"
        p.write_text(report_ablation())
        out.append(p)
    if which in ("all", "lite_dl_gpu"):
        p = REPORTS_DIR / "lite_dl_gpu.md"
        p.write_text(report_lite_dl_gpu())
        out.append(p)
    return out


def _safe_reports() -> None:
    try:
        for p in write_reports("all"):
            log_event("report_written", path=str(p.relative_to(ROOT)))
    except Exception as e:  # noqa: BLE001 — 보고서 실패가 캠페인을 멈추면 안 된다
        log_event("report_failed", error=repr(e))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["run", "status", "report", "plan"])
    args = p.parse_args(argv)
    os.chdir(ROOT)
    if args.command == "run":
        return run_campaign()
    if args.command == "status":
        text = status_text()
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(text)
        print(text)
        return 0
    if args.command == "report":
        for path in write_reports("all"):
            print(path)
        return 0
    for phase in PHASES:  # plan
        jobs = phase_jobs(phase)
        todo = [j for j in jobs if not j.exists]
        print(f"{phase.id:<22} total={len(jobs):<4} todo={len(todo):<4} {phase.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
