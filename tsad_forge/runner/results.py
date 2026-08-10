"""결과 저장 규약 (CLAUDE.md §5).

- long-format parquet: model, generation, dataset, channel, seed, metric, value,
  runtime_s, peak_vram_mb, commit_hash, config_hash, timestamp
- 실행별 JSON 요약 (재현 커맨드/설정 포함)
- 스코어 원본 npz (evaluation.protocol.save_scores)
- resume: 동일 (model, dataset, channel, seed, config_hash) 결과가 있으면 건너뜀
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
    "peak_vram_mb",
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
    peak_vram_mb: float,
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
            "peak_vram_mb": peak_vram_mb,
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
        "peak_vram_mb": peak_vram_mb,
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


def load_all_results(results_dir: str | Path) -> pd.DataFrame:
    """results 디렉터리의 모든 parquet을 합쳐 반환 (없으면 빈 DataFrame)."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.parquet")) if results_dir.exists() else []
    if not files:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
