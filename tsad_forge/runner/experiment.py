"""단일 실험 실행기: 데이터 로드 → 정규화 → fit → score → 임계값 → 지표 → 저장.

config(dict) 기반으로 동작하며 CLI(tsad_forge.cli)와 benchmarks/run_all.py가 공유한다.
"""

from __future__ import annotations

import time
from pathlib import Path

from tsad_forge.data.loaders.file import load_file
from tsad_forge.data.registry import load_dataset
from tsad_forge.data.schema import TSADDataset
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import save_scores, set_seed, zscore_normalize
from tsad_forge.evaluation.thresholding import NEEDS_CALIBRATION, apply_threshold
from tsad_forge.models.registry import get_model
from tsad_forge.runner import results as res
from tsad_forge.runner.measure import PeakRssSampler, cuda_peak_mb, reset_cuda_peak

DEFAULT_CONFIG = {
    "model": "dummy",
    "model_params": {},
    "data": "synthetic",
    "data_params": {},
    "seed": 0,
    "normalize": "zscore",
    "threshold": {"method": "quantile", "q": 0.99},
    "sliding_window": 100,  # VUS buffer 최대 폭 (TSB-AD 기본과 동일)
    "legacy_pa": False,
    "results_dir": "benchmarks/results",
    "save_raw_scores": True,
}


def resolve_config(overrides: dict | None = None) -> dict:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def load_data(cfg: dict) -> TSADDataset:
    data = cfg["data"]
    params = cfg.get("data_params", {})
    if isinstance(data, str) and (data.endswith((".csv", ".txt", ".parquet", ".pq"))):
        return load_file(data, **params)
    # 레지스트리 데이터셋: 시드가 있는 생성기(synthetic)에는 시드 전달
    if data == "synthetic" and "seed" not in params:
        params = {**params, "seed": cfg["seed"]}
    return load_dataset(data, **params)


def run_experiment(
    overrides: dict | None = None, *, force: bool = False, verbose: bool = True
) -> dict | None:
    """단일 (model, data, seed) 실험. resume 대상이면 None을 반환하고 건너뜀."""
    cfg = resolve_config(overrides)
    cfg_hash = res.config_hash(
        {k: v for k, v in cfg.items() if k not in ("results_dir", "save_raw_scores")}
    )
    results_dir = Path(cfg["results_dir"])
    dataset_name = Path(str(cfg["data"])).stem if "." in str(cfg["data"]) else str(cfg["data"])
    dp = cfg.get("data_params", {})
    # 데이터셋별 하위 단위 식별자 (smd: machine, smap/msl: channel, ucr: series ...)
    channel = str(
        dp.get("channel")
        or dp.get("machine")
        or dp.get("series")
        or dp.get("experiment")
        or dp.get("filename")
        or dp.get("rel_path")
        or "all"
    ).replace("/", "_")

    if not force and res.result_exists(
        results_dir, cfg["model"], dataset_name, channel, cfg["seed"], cfg_hash
    ):
        if verbose:
            print(
                f"[resume] skip {cfg['model']} on {dataset_name} "
                f"(seed={cfg['seed']}, config={cfg_hash}) — result exists. Use --force to rerun."
            )
        return None

    set_seed(cfg["seed"])
    ds = load_data(cfg)

    train, test = ds.train, ds.test
    if cfg["normalize"] == "zscore":
        train, test = zscore_normalize(train, test)

    model = get_model(cfg["model"], seed=cfg["seed"], **cfg.get("model_params", {}))

    # 계측 (measure.py): 타이머 구간 안에 프로파일러 없음, RSS는 스레드 샘플링,
    # CUDA 실행 시에만 정확한 피크 VRAM을 병기.
    cuda_tracking = reset_cuda_peak()
    with PeakRssSampler() as mem:
        t0 = time.perf_counter()
        model.fit(train)
        runtime_fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = model.score(test)
        runtime_score_s = time.perf_counter() - t0
    runtime_s = runtime_fit_s + runtime_score_s
    peak_vram = cuda_peak_mb() if cuda_tracking else None

    th_cfg = dict(cfg["threshold"])
    method = th_cfg.pop("method")
    train_scores = None
    if method in NEEDS_CALIBRATION:
        # 보정용 train(정상) 점수 — 임계값 비용이므로 fit/score 런타임에 불포함
        train_scores = model.score(train)
    threshold, preds = apply_threshold(scores, method=method, train_scores=train_scores, **th_cfg)

    has_labels = ds.meta.get("has_labels", True) and ds.labels.sum() > 0
    metrics = (
        compute_metrics(
            scores,
            ds.labels,
            threshold=threshold,
            legacy_pa=cfg["legacy_pa"],
            sliding_window=cfg.get("sliding_window", 100),
        )
        if has_labels
        else {}
    )
    metrics["threshold"] = threshold
    metrics["n_predicted_anomalies"] = int(preds.sum())

    if cfg.get("save_raw_scores", True):
        rid = res.run_id(cfg["model"], dataset_name, channel, cfg["seed"], cfg_hash)
        save_scores(scores, ds.labels, results_dir / "scores" / f"{rid}.npz")

    res.write_result(
        results_dir,
        metrics,
        model=cfg["model"],
        generation=type(model).generation,
        dataset=dataset_name,
        channel=channel,
        seed=cfg["seed"],
        runtime_s=round(runtime_s, 4),
        runtime_fit_s=round(runtime_fit_s, 4),
        runtime_score_s=round(runtime_score_s, 4),
        peak_mem_mb=round(mem.peak_mb, 2),
        peak_vram_mb=round(peak_vram, 2) if peak_vram is not None else None,
        data_hash=res.config_hash({"data": cfg["data"], "data_params": cfg.get("data_params", {})}),
        cfg_hash=cfg_hash,
        config=cfg,
    )

    if verbose:
        print(
            f"model={cfg['model']} ({type(model).generation})  data={dataset_name}  seed={cfg['seed']}"
        )
        print(f"runtime={runtime_s:.3f}s  config_hash={cfg_hash}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if not has_labels:
            print("  (no labels — score/threshold outputs only)")

    return {"config": cfg, "config_hash": cfg_hash, "metrics": metrics, "scores": scores}
