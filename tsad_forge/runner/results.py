"""결과 저장 규약 (CLAUDE.md §5).

- long-format parquet: model, generation, dataset, channel, seed, metric, value,
  runtime_s(=fit+score), runtime_fit_s, runtime_score_s, peak_mem_mb(호스트 RSS 피크
  증가분), commit_hash, config_hash, timestamp. CUDA 실행 시 peak_vram_mb는 JSON
  요약에 병기된다 (CPU 전용 실행에서는 존재하지 않는 값이므로 컬럼을 만들지 않는다).
- 실행별 JSON 요약 (재현 커맨드/설정 포함)
- 스코어 원본 npz (evaluation.protocol.save_scores)
- resume: 동일 (model, dataset, channel, seed, config_hash) 결과가 있으면 건너뜀
- 집계(load_all_results)는 동일 (model, dataset, channel, seed)에 config_hash가
  다른 결과가 공존하면 최신 timestamp만 취한다 — 기본 config 변경 후 재실행 시
  구결과가 리더보드에 이중 집계되는 것을 방지 (리뷰 P0-3)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "model",
    "generation",
    "dataset",
    "channel",
    "seed",
    "metric",
    "value",
    "runtime_s",
    "runtime_fit_s",
    "runtime_score_s",
    "peak_mem_mb",
    "data_hash",
    "commit_hash",
    "config_hash",
    "timestamp",
]


def config_hash(config: dict) -> str:
    """설정 dict의 안정적 해시 (재현 추적용, 리더보드 행 -> config 링크)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def get_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_id(model: str, dataset: str, channel: str, seed: int, cfg_hash: str) -> str:
    return f"{model}__{dataset}__{channel}__seed{seed}__{cfg_hash}"


def write_result(
    results_dir: str | Path,
    metrics: dict[str, float],
    *,
    model: str,
    generation: str,
    dataset: str,
    channel: str,
    seed: int,
    runtime_s: float,
    runtime_fit_s: float,
    runtime_score_s: float,
    peak_mem_mb: float,
    peak_vram_mb: float | None = None,
    data_hash: str,
    cfg_hash: str,
    config: dict,
) -> Path:
    """단일 실행 결과를 parquet(지표 long-format) + JSON(요약)으로 저장."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(model, dataset, channel, seed, cfg_hash)
    commit = get_commit_hash()
    ts = utc_now()

    rows = [
        {
            "model": model,
            "generation": generation,
            "dataset": dataset,
            "channel": channel,
            "seed": seed,
            "metric": metric,
            "value": float(value),
            "runtime_s": runtime_s,
            "runtime_fit_s": runtime_fit_s,
            "runtime_score_s": runtime_score_s,
            "peak_mem_mb": peak_mem_mb,
            "data_hash": data_hash,
            "commit_hash": commit,
            "config_hash": cfg_hash,
            "timestamp": ts,
        }
        for metric, value in metrics.items()
    ]
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    parquet_path = results_dir / f"{rid}.parquet"
    df.to_parquet(parquet_path, index=False)

    summary = {
        "run_id": rid,
        "model": model,
        "generation": generation,
        "dataset": dataset,
        "channel": channel,
        "seed": seed,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "runtime_s": runtime_s,
        "runtime_fit_s": runtime_fit_s,
        "runtime_score_s": runtime_score_s,
        "peak_mem_mb": peak_mem_mb,
        **({"peak_vram_mb": peak_vram_mb} if peak_vram_mb is not None else {}),
        "commit_hash": commit,
        "config_hash": cfg_hash,
        "config": config,
        "timestamp": ts,
    }
    (results_dir / f"{rid}.json").write_text(json.dumps(summary, indent=2, default=str))
    return parquet_path


def result_exists(
    results_dir: str | Path, model: str, dataset: str, channel: str, seed: int, cfg_hash: str
) -> bool:
    """resume 판정: 동일 조합의 parquet이 이미 있으면 True."""
    rid = run_id(model, dataset, channel, seed, cfg_hash)
    return (Path(results_dir) / f"{rid}.parquet").exists()


def load_all_results(results_dir: str | Path, dedup: bool = True) -> pd.DataFrame:
    """results 디렉터리의 모든 parquet을 합쳐 반환 (없으면 빈 DataFrame).

    dedup=True(기본): 동일 실험 정체성 (model, dataset, channel, seed, data_hash,
    metric)에 결과가 여러 개(기본 config 변경 → config_hash 변화 → 재실행)면 최신
    timestamp만 남긴다. 구결과가 남아 리더보드에 이중 집계되는 것을 막는다 (리뷰
    P0-3). data_hash는 (data, data_params)의 해시 — channel로 구분되지 않는 정당한
    데이터 변형(예: synthetic n_dims)을 실수로 합치지 않기 위한 정체성 축이다.
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.parquet")) if results_dir.exists() else []
    if not files:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if dedup:
        key = ["model", "dataset", "channel", "seed", "data_hash", "metric"]
        deduped = df.sort_values("timestamp", kind="stable").drop_duplicates(key, keep="last")
        n_dropped = len(df) - len(deduped)
        if n_dropped:
            stale = sorted(set(df["config_hash"]) - set(deduped["config_hash"]))
            warnings.warn(
                f"excluded {n_dropped} superseded result row(s) from aggregation "
                f"(same model/dataset/seed re-run under a newer config_hash; "
                f"stale hashes: {stale[:5]}{'...' if len(stale) > 5 else ''}). "
                "Delete the old parquet/json files to silence this warning.",
                stacklevel=2,
            )
        df = deduped.reset_index(drop=True)
    return df
