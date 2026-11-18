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


#: 해시 대상에서 빠지는 실행 부수 설정 (결과 수치에 영향 없음)
HASH_EXCLUDED_KEYS = ("results_dir", "save_raw_scores")

#: config에 나중에 추가된 키의 "레거시 동작" 값. 값이 이것과 같으면 해시에서 뺀다 —
#: 새 키를 도입해도 그 키가 생기기 전의 결과(같은 동작)의 해시가 바뀌지 않게 하는
#: 하위호환 장치다. 새 config 키를 추가할 때는 반드시 여기에 레거시 값을 등록하라
#: (tests/test_integrity.py가 DEFAULT_CONFIG의 기존 해시 불변을 검사한다).
HASH_EXEMPT_DEFAULTS: dict = {
    "device": "cpu",  # 해석된 실행 장치; 기존 결과는 전부 CPU
    "normalize_params": {},  # std_floor/clip 미사용 = 레거시 z-score
    "model_version": 1,  # BaseDetector.model_version; 재구현 시 올라간다
    "model_alias": None,  # 결과 라벨 (ablation 변형 구분); 미지정 시 해시 불변
}


def experiment_hash(config: dict) -> str:
    """실험 정체성 해시: 실행 부수 키와 레거시 값의 신규 키를 제외하고 해시한다."""
    payload = {
        k: v
        for k, v in config.items()
        if k not in HASH_EXCLUDED_KEYS
        and not (k in HASH_EXEMPT_DEFAULTS and v == HASH_EXEMPT_DEFAULTS[k])
    }
    return config_hash(payload)


def get_commit_hash() -> str:
    from tsad_forge.runner.environment import git_state

    return str(git_state()["commit"])


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
    environment: dict | None = None,
) -> Path:
    """단일 실행 결과를 parquet(지표 long-format) + JSON(요약)으로 저장.

    environment: runner.environment.environment_fingerprint() 결과 (JSON에만 기록).
    """
    from tsad_forge.runner.environment import git_state

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(model, dataset, channel, seed, cfg_hash)
    git = git_state()
    commit = git["commit"]
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
        # dirty=True면 커밋 해시만으로는 이 결과를 재현할 수 없다 (리더보드 감사용)
        "git_dirty": git["dirty"],
        "config_hash": cfg_hash,
        "config": config,
        **({"environment": environment} if environment is not None else {}),
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


def add_entity_column(df: pd.DataFrame) -> pd.DataFrame:
    """평가 단위(entity) 라벨 열을 붙인다: "dataset" 또는 "dataset/channel".

    channel은 data_params의 식별 키(machine/channel/series...)에서만 만들어지므로,
    식별 키가 없는 데이터 변형(예: synthetic의 n_dims=1 vs n_dims=5)은 둘 다
    "synthetic"이 되어 리더보드에서 서로 다른 데이터가 평균으로 섞인다. 같은 라벨에
    data_hash가 여럿이면 "#<data_hash 앞 6자>"를 붙여 구분한다.
    """
    df = df.copy()
    df["entity"] = df["dataset"].where(df["channel"] == "all", df["dataset"] + "/" + df["channel"])
    if "data_hash" in df.columns and len(df):
        n_variants = df.groupby("entity")["data_hash"].transform("nunique")
        multi = n_variants > 1
        df.loc[multi, "entity"] = (
            df.loc[multi, "entity"] + "#" + df.loc[multi, "data_hash"].astype(str).str[:6]
        )
    return df


def shared_entities(pivot: pd.DataFrame) -> list:
    """pivot(index=모델, columns=entity)에서 모든 행이 값을 가진 entity 목록.

    모델마다 실행된 entity 집합이 다르면 단순 평균은 "어떤 데이터에서 돌았는가"를
    비교하게 된다 (예: DL 모델만 쉬운 synthetic에서 실행). 공정 비교는 교집합에서만.
    """
    return [c for c in pivot.columns if pivot[c].notna().all()]


def normalized_ranks(pivot: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """entity별 순위를 [0, 1]로 정규화 (0=최고, 1=최하). 결측은 NaN 유지.

    entity마다 참가 모델 수가 다르면 원시 순위의 척도가 달라진다 (21개 중 10위와
    36개 중 10위는 다른 의미). (rank-1)/(n-1)로 맞춘 뒤 평균해야 커버리지가 다른
    모델끼리 순위를 비교할 수 있다. 참가 모델이 1개뿐인 entity는 제외(NaN).
    """
    ranks = pivot.rank(ascending=not higher_is_better, axis=0)
    n = pivot.notna().sum(axis=0)
    denom = (n - 1).where(n > 1)
    return (ranks - 1) / denom
