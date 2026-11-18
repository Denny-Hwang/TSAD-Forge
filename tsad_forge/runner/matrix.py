"""벤치마크 매트릭스 스케줄러 (CLAUDE.md §5) — 프로파일 확장 + 병렬/격리 실행 + 실패 기록.

두 실행 모드:
- **in-process (기본, jobs=1·timeout 없음)**: 레거시와 같은 순차 실행. 테스트·디버깅용.
- **isolated (jobs>1 또는 timeout 지정 또는 isolate=True)**: 실행마다 별도 프로세스
  (spawn). 한 실행의 segfault/OOM-kill/무한 루프가 매트릭스 전체를 멈추지 않는다.
  두 레인으로 나눠 동시에 돈다:
    * cpu 레인 — torch를 쓰지 않는 모델(Gen1-2 등). 최대 `jobs`개 병렬, 각 프로세스의
      BLAS/OpenMP 스레드 수를 코어/jobs로 제한해 과구독을 막는다.
    * torch 레인 — torch 모델. 기본 동시 1개 (GPU 1장 / CPU torch는 자체 병렬).

실패는 `<results_dir>/failures/<run_id>.json`에 유형별로 남는다
(`timeout` | `oom` | `score_health` | `crashed` | `error`). 같은 실험이 이후 성공하면
실패 기록은 지워진다. 데이터 부재(FileNotFoundError)는 실패가 아니라 `no_data`다.

프로파일 `plugins: [my_pkg.my_models]` — 나열된 모듈을 부모와 각 워커에서 import한다.
코어를 수정하지 않고 실험용 모델(@register_model)이나 데이터셋을 붙이는 경로다.
플러그인 목록은 실험 정체성(config_hash)에 들어가지 않는다 — 같은 이름의 모델은
같은 실험이다.

프로파일 데이터셋 항목의 확장 문법:
    - { data: smd, entities: all }            # 로컬에 있는 SMD 머신 전부
    - { data: ucr, entities: all, limit: 50 } # 정렬 순서 앞 50개 (결정적)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAILURE_DIR = "failures"
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# --- 플러그인 -----------------------------------------------------------------


def import_plugins(plugins: list[str] | None) -> None:
    """프로파일의 plugins 모듈을 import한다 (등록 부수효과). 중복 import는 무해."""
    import importlib

    for mod in plugins or []:
        importlib.import_module(mod)


# --- 프로파일 확장 -----------------------------------------------------------


def _expand_datasets(profile: dict) -> tuple[list[dict], list[str]]:
    """datasets 항목을 entity 단위 data spec 목록으로 펼친다. (specs, 빈 열거 데이터셋)"""
    from tsad_forge.data.entities import list_entities, supports_enumeration

    specs: list[dict] = []
    empty: list[str] = []
    for raw in profile["datasets"]:
        spec: dict = {"data": raw} if isinstance(raw, str) else dict(raw)
        mode = spec.pop("entities", None)
        limit = spec.pop("limit", None)
        if mode is None:
            specs.append(spec)
            continue
        if mode != "all":
            raise ValueError(f"unknown entities mode '{mode}' (expected 'all')")
        name = spec["data"]
        if not supports_enumeration(name):
            raise ValueError(f"dataset '{name}' does not support `entities: all`")
        base_params = dict(spec.get("data_params", {}))
        # 데이터 루트: data_params.data_dir > $TSAD_FORGE_DATA > ./data (로더와 같은 규칙)
        found = list_entities(name, data_dir=base_params.get("data_dir"))
        if limit is not None:
            found = found[: int(limit)]
        if not found:
            empty.append(name)
        for ent in found:
            specs.append({**spec, "data_params": {**base_params, **ent}})
    return specs, empty


def expand_combos(profile: dict) -> list[dict]:
    """(model × dataset × seed) 조합. 결정적 모델은 seeds[0]만, stochastic은 전체 시드."""
    seeds = profile.get("seeds", [0])
    datasets, _ = _expand_datasets(profile)
    combos = []
    for model_spec in profile["models"]:
        model_spec = {"model": model_spec} if isinstance(model_spec, str) else dict(model_spec)
        stochastic = model_spec.pop("stochastic", False)
        model_seeds = seeds if stochastic else seeds[:1]
        for data_spec in datasets:
            for seed in model_seeds:
                combos.append({**model_spec, **data_spec, "seed": seed})
    return combos


def _matches(value: str, allowed: list[str] | None) -> bool:
    return allowed is None or value in allowed


@dataclass
class Job:
    overrides: dict
    run_id: str
    lane: str  # "cpu" | "torch"
    tag: tuple
    exists: bool


def build_jobs(
    profile: dict,
    results_dir: str | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
) -> tuple[list[Job], list[str]]:
    """프로파일 → 실행 계획 목록 (필터 적용). (jobs, entity가 하나도 없는 데이터셋)."""
    from tsad_forge.runner.experiment import plan_experiment

    import_plugins(profile.get("plugins"))
    results_dir = results_dir or profile.get("results_dir", "benchmarks/results")
    defaults = profile.get("defaults", {})
    _, empty = _expand_datasets(profile)
    jobs: list[Job] = []
    for combo in expand_combos(profile):
        if not (
            (_matches(combo["model"], models) or _matches(str(combo.get("model_alias")), models))
            and _matches(str(combo["data"]), datasets)
            and (seeds is None or combo["seed"] in seeds)
        ):
            continue
        overrides = {**defaults, **combo, "results_dir": results_dir}
        plan = plan_experiment(overrides)
        jobs.append(
            Job(
                overrides=overrides,
                run_id=plan.run_id,
                lane="torch" if plan.uses_torch else "cpu",
                tag=(plan.name, plan.dataset, plan.channel, combo["seed"]),
                exists=plan.exists(),
            )
        )
    return jobs, empty


# --- 실패 기록 ---------------------------------------------------------------


def _failure_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / FAILURE_DIR / f"{run_id}.json"


def record_failure(results_dir: str | Path, job: Job, kind: str, detail: str) -> Path:
    from tsad_forge.runner.results import utc_now

    path = _failure_path(results_dir, job.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": job.run_id,
                "kind": kind,
                "detail": detail[-4000:],
                "config": job.overrides,
                "timestamp": utc_now(),
            },
            indent=2,
            default=str,
        )
    )
    return path


def clear_failure(results_dir: str | Path, run_id: str) -> None:
    _failure_path(results_dir, run_id).unlink(missing_ok=True)


def load_failures(results_dir: str | Path) -> list[dict]:
    d = Path(results_dir) / FAILURE_DIR
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def classify_exception(exc: BaseException) -> str:
    from tsad_forge.evaluation.protocol import ScoreHealthError

    if isinstance(exc, ScoreHealthError):
        return "score_health"
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "oom"
    return "error"


# --- 실행 --------------------------------------------------------------------


@dataclass
class MatrixSummary:
    counts: Counter = field(default_factory=Counter)
    no_data: set = field(default_factory=set)
    failed: list = field(default_factory=list)  # (tag, kind)

    def line(self) -> str:
        c = self.counts
        return (
            f"done={c['done']}  skipped(resume)={c['skipped']}  "
            f"no_data={len(self.no_data)}  failed={len(self.failed)}"
        )


def _run_in_process(job: Job, force: bool) -> tuple[str, str]:
    from tsad_forge.runner.experiment import run_experiment

    try:
        result = run_experiment(job.overrides, force=force)
    except FileNotFoundError as e:
        return "no_data", str(e)
    except Exception as e:  # noqa: BLE001 — 실패 유형을 분류해 기록한다
        return classify_exception(e), traceback.format_exc()
    return ("skipped" if result is None else "done"), ""


def _child(
    overrides: dict, force: bool, threads: int | None, plugins: list[str] | None, conn
) -> None:
    """격리 실행 진입점 (spawn). 무거운 import 전에 스레드 수를 고정한다."""
    if threads:
        for var in _THREAD_ENV:
            os.environ[var] = str(threads)
    try:
        import_plugins(plugins)
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return
    job = Job(overrides=overrides, run_id="", lane="", tag=(), exists=False)
    status, detail = _run_in_process(job, force)
    conn.send((status, detail))
    conn.close()


@dataclass
class _Active:
    job: Job
    proc: Any  # multiprocessing SpawnProcess
    conn: Any  # multiprocessing Connection (recv end)
    started: float


def run_matrix(
    jobs: list[Job],
    *,
    results_dir: str | Path,
    force: bool = False,
    n_jobs: int = 1,
    torch_jobs: int = 1,
    timeout: float | None = None,
    isolate: bool = False,
    plugins: list[str] | None = None,
    poll_s: float = 0.2,
) -> MatrixSummary:
    summary = MatrixSummary()
    todo = []
    for job in jobs:
        if job.exists and not force:
            summary.counts["skipped"] += 1
            clear_failure(results_dir, job.run_id)
        else:
            todo.append(job)

    def _finish(job: Job, status: str, detail: str) -> None:
        if status in ("done", "skipped"):
            summary.counts[status] += 1
            clear_failure(results_dir, job.run_id)
        elif status == "no_data":
            summary.no_data.add(job.overrides.get("data"))
            print(f"  [no-data] skip {job.tag}: {detail.splitlines()[0] if detail else ''}")
        else:
            summary.failed.append((job.tag, status))
            record_failure(results_dir, job, status, detail)
            print(f"  [FAILED:{status}] {job.tag}")
            if status != "timeout" and detail:
                print(detail.rstrip().splitlines()[-1])

    if not (isolate or n_jobs > 1 or timeout is not None):
        for job in todo:
            _finish(job, *_run_in_process(job, force))
        return summary

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    cpu_threads = max(1, (os.cpu_count() or 1) // max(n_jobs, 1))
    capacity = {"cpu": max(n_jobs, 1), "torch": max(torch_jobs, 1)}
    pending = {
        "cpu": [j for j in todo if j.lane == "cpu"],
        "torch": [j for j in todo if j.lane == "torch"],
    }
    active: list[_Active] = []
    total, finished = len(todo), 0

    while pending["cpu"] or pending["torch"] or active:
        for lane in ("torch", "cpu"):
            while pending[lane] and sum(a.job.lane == lane for a in active) < capacity[lane]:
                job = pending[lane].pop(0)
                parent, child = ctx.Pipe(duplex=False)
                threads = cpu_threads if lane == "cpu" else None
                proc = ctx.Process(
                    target=_child, args=(job.overrides, force, threads, plugins, child)
                )
                proc.start()
                child.close()
                active.append(_Active(job, proc, parent, time.monotonic()))
        time.sleep(poll_s)
        still: list[_Active] = []
        for a in active:
            worker, conn = a.proc, a.conn
            message = None
            if conn.poll():
                try:
                    message = conn.recv()
                except EOFError:  # 결과를 보내기 전에 워커가 죽어 파이프가 닫혔다
                    worker.join(5)
            if message is not None:
                status, detail = message
                worker.join()
            elif not worker.is_alive():
                code = worker.exitcode
                # SIGKILL(-9)은 대개 커널 OOM killer
                status = "oom" if code == -9 else "crashed"
                detail = f"worker exited with code {code} without reporting a result"
            elif timeout is not None and time.monotonic() - a.started > timeout:
                worker.terminate()
                worker.join(5)
                status, detail = "timeout", f"exceeded {timeout:.0f}s"
            else:
                still.append(a)
                continue
            finished += 1
            print(
                f"[{finished}/{total}] {status:<8} {a.job.tag}  ({time.monotonic() - a.started:.1f}s)"
            )
            _finish(a.job, status, detail)
        active = still
    return summary
