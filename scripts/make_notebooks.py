"""학습 트랙 노트북 생성기 (ch01–ch10) — notebooks/*.ipynb를 재생성한다.

노트북은 합성 데이터 기반이라 다운로드 없이 실행 가능하다 (일부 셀은 로컬 data/가
있으면 실데이터를 사용). 사용: python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }


SETUP = """
import numpy as np
import matplotlib.pyplot as plt
from tsad_forge.synthetic.generator import generate_synthetic
"""

NOTEBOOKS: dict[str, list[dict]] = {
    "ch01_problem": [
        md(
            "# ch01 — TSAD 문제 정의\n\n이상 유형(point/contextual/collective)을 직접 만들어 본다.\n"
            "이론: [docs/learn/ch01](../docs/learn/ch01_problem.md)"
        ),
        code(SETUP),
        code("""
# 이상 유형별 합성 데이터
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, kind in zip(axes, ["spike", "level_shift", "contextual"]):
    ds = generate_synthetic(n_test=800, n_events=3, anomaly_kinds=[kind], seed=1)
    ax.plot(ds.test[:, 0], lw=0.8)
    ax.fill_between(np.arange(800), *ax.get_ylim(), where=ds.labels.astype(bool),
                    alpha=0.25, color="red")
    ax.set_title(f"anomaly kind = {kind}  (rate={ds.anomaly_rate:.3f})")
plt.tight_layout()
"""),
        code("""
# contamination: train 오염이 zscore baseline에 미치는 영향
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics

for cont in [0.0, 1.0, 3.0]:
    ds = generate_synthetic(seed=3, contamination=cont)
    scores = get_model("zscore").fit(ds.train).score(ds.test)
    m = compute_metrics(scores, ds.labels)
    print(f"contamination={cont}: VUS-PR={m['vus_pr']:.3f}  AUC-PR={m['auc_pr']:.3f}")
"""),
    ],
    "ch02_gen1_statistical": [
        md(
            "# ch02 — Gen1 통계: PCA-T²/SPE\n\nSMD(로컬 data/ 필요) 또는 합성 다변량 데이터에 적용."
        ),
        code(SETUP),
        code("""
# SMD가 있으면 실데이터, 없으면 합성 다변량
try:
    from tsad_forge.data.registry import load_dataset
    ds = load_dataset("smd", machine="machine-1-1")
    print("using SMD machine-1-1")
except FileNotFoundError:
    ds = generate_synthetic(n_dims=8, n_events=6, seed=2)
    print("using synthetic (run `tsad-forge download smd` for real data)")
"""),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.protocol import zscore_normalize
train, test = zscore_normalize(ds.train, ds.test)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, mode in zip(axes, ["t2", "spe", "combined"]):
    scores = get_model("pca_t2spe", mode=mode).fit(train).score(test)
    ax.plot(scores, lw=0.6)
    ax.fill_between(np.arange(len(scores)), *ax.get_ylim(),
                    where=ds.labels.astype(bool), alpha=0.25, color="red")
    ax.set_title(f"PCA-{mode.upper()} — T²(주성분 공간) vs SPE(잔차 공간)는 다른 이상을 잡는다")
plt.tight_layout()
"""),
    ],
    "ch03_gen2_classical_ml": [
        md(
            "# ch03 — Gen2: Matrix Profile discord\n\nstumpy로 주기 신호 속 파형 왜곡(discord)을 찾는다.\n"
            "UCR 데이터가 있으면 (`tsad-forge download ucr`) 실데이터로 바꿔 실행해 보라."
        ),
        code(SETUP),
        code("""
import stumpy
t = np.arange(3000)
x = np.sin(2 * np.pi * t / 60) + 0.05 * np.random.default_rng(0).normal(size=3000)
x[1500:1560] = np.sin(2 * np.pi * t[1500:1560] / 17)  # 주기 붕괴 discord

m = 60
mp = stumpy.stump(x, m=m)[:, 0].astype(float)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
a1.plot(x, lw=0.5); a1.axvspan(1500, 1560, alpha=0.3, color="red"); a1.set_title("series")
a2.plot(mp, lw=0.7); a2.set_title(f"matrix profile (m={m}) — 최대값 위치가 discord")
print("discord at:", int(np.argmax(mp)))
"""),
        code("""
# IForest는 시드에 따라 결과가 다르다 — 그래서 시드 3개 평가 (CLAUDE.md §4)
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
ds = generate_synthetic(seed=0)
for seed in range(3):
    s = get_model("iforest", seed=seed).fit(ds.train).score(ds.test)
    print(f"seed={seed}: VUS-PR={compute_metrics(s, ds.labels)['vus_pr']:.3f}")
"""),
    ],
    "ch04_gen3_dl": [
        md("# ch04 — Gen3 DL: 재구성 vs 예측"),
        code(SETUP),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import zscore_normalize

ds = generate_synthetic(n_events=6, seed=4)
train, test = zscore_normalize(ds.train, ds.test)
for name in ["ae", "lstm_p", "usad"]:  # 재구성 / 예측 / 적대적 재구성
    scores = get_model(name, seed=0, epochs=5, window=32).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"{name:8s} VUS-PR={m['vus_pr']:.3f}  AUC-PR={m['auc_pr']:.3f}")
"""),
        code("""
# over-generalization: AE 용량(latent)이 크면 이상까지 복원한다
for latent in [2, 16, 64]:
    scores = get_model("ae", seed=0, epochs=5, window=32, latent=latent).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"latent={latent:3d}: VUS-PR={m['vus_pr']:.3f}")
"""),
    ],
    "ch05_gen4": [
        md("# ch05 — Gen4: 그래프(GDN)와 어텐션(Anomaly Transformer)"),
        code(SETUP),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import zscore_normalize

ds = generate_synthetic(n_dims=5, n_events=8, seed=5)
train, test = zscore_normalize(ds.train, ds.test)
for name in ["gdn", "anomaly_transformer", "timesnet", "sub_pca"]:  # sub_pca = 대조군
    kw = dict(seed=0, epochs=5, window=32) if name != "sub_pca" else {}
    scores = get_model(name, **kw).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"{name:20s} VUS-PR={m['vus_pr']:.3f}")
print("\\n단순 baseline(sub_pca)과의 격차를 보라 — 리더보드의 핵심 질문이다.")
"""),
    ],
    "ch06_gen5": [
        md(
            "# ch06 — Gen5: MambaTSAD faithful vs fixed\n\n구현 이슈 4개(상태 인덱싱, CPU 분기, "
            "HP filter 목적, AMA 전역 FFT)의 효과를 정량화한다."
        ),
        code(SETUP),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import zscore_normalize

ds = generate_synthetic(n_events=6, seed=6)
train, test = zscore_normalize(ds.train, ds.test)
for name in ["mamba_tsad_faithful", "mamba_tsad_fixed"]:
    vals = []
    for seed in range(3):
        s = get_model(name, seed=seed, epochs=3, window=32).fit(train).score(test)
        vals.append(compute_metrics(s, ds.labels)["vus_pr"])
    print(f"{name:22s} VUS-PR = {np.mean(vals):.3f} ± {np.std(vals):.3f} (3 seeds)")
"""),
        code("""
# HP filter 분해 시각화 — faithful은 trend를, fixed는 cycle을 모델 입력으로 쓴다
from tsad_forge.models.gen5_ssm_foundation.mamba_tsad import hp_filter
trend, cycle = hp_filter(ds.test)
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for ax, (y, ttl) in zip(axes, [(ds.test[:, 0], "original"), (trend[:, 0], "trend (faithful 입력)"),
                               (cycle[:, 0], "cycle (fixed 입력)")]):
    ax.plot(y, lw=0.6); ax.set_title(ttl)
plt.tight_layout()
"""),
    ],
    "ch07_evaluation": [
        md(
            "# ch07 — 평가 방법론: random score로 PA-F1 'SOTA' 만들기\n\n"
            "**이 노트북이 이 저장소의 존재 이유다** (Kim et al., AAAI 2022 재현)."
        ),
        code(SETUP),
        code("""
from sklearn.metrics import f1_score
from tsad_forge.evaluation.metrics import point_adjust, compute_metrics
import warnings

rng = np.random.default_rng(0)
labels = np.zeros(5000, dtype=int)
for s in range(200, 5000, 500):
    labels[s:s+100] = 1                      # 긴 이상 이벤트 (현실적)
scores = rng.random(5000)                    # 완전 무작위 점수!

th = np.quantile(scores, 0.99)
pred = (scores >= th).astype(int)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = compute_metrics(scores, labels, threshold=th, legacy_pa=True)
print(f"random score: PA-F1     = {m['pa_f1']:.3f}   <- 'SOTA급'")
print(f"random score: standard-F1= {m['standard_f1']:.3f}")
print(f"random score: VUS-PR    = {m['vus_pr']:.3f}   <- 주지표는 속지 않는다")
"""),
        code("""
# 이벤트가 길수록 부풀림이 커진다
import warnings
for ev_len in [10, 50, 100, 300]:
    labels = np.zeros(5000, dtype=int)
    for s in range(200, 5000, 1000):
        labels[s:s+ev_len] = 1
    scores = rng.random(5000)
    th = np.quantile(scores, 0.99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = compute_metrics(scores, labels, threshold=th, legacy_pa=True)
    print(f"event_len={ev_len:4d}: PA-F1={m['pa_f1']:.3f}  standard-F1={m['standard_f1']:.3f}")
"""),
    ],
    "ch08_thresholding": [
        md("# ch08 — 임계값: EVT(SPOT)와 conformal"),
        code(SETUP),
        code("""
from tsad_forge.evaluation.thresholding import spot_threshold, conformal_threshold, quantile_threshold
rng = np.random.default_rng(0)
cal = rng.normal(size=5000)                       # 정상(보정) 점수
test_scores = np.concatenate([rng.normal(size=2000), rng.normal(5, 1, size=20)])  # 이상 20개

for name, th in [
    ("quantile(0.99)", quantile_threshold(cal, 0.99)),
    ("SPOT(q=1e-3)", spot_threshold(test_scores, q=1e-3, calibration=cal)),
    ("conformal(a=0.01)", conformal_threshold(test_scores, alpha=0.01, calibration=cal)),
]:
    pred = test_scores >= th
    fp = pred[:2000].mean(); tp = pred[2000:].mean()
    print(f"{name:18s} th={th:6.3f}  오탐률={fp:.4f}  이상탐지율={tp:.2f}")
"""),
        code("""
# best-F1(oracle)이 배포에서 재현 불가능한 이유: test 라벨을 몰래 본 값이다
from tsad_forge.evaluation.metrics import compute_metrics
labels = np.concatenate([np.zeros(2000, dtype=int), np.ones(20, dtype=int)])
m = compute_metrics(test_scores, labels)
print(f"best_f1(oracle)={m['best_f1']:.3f} — 리더보드 참고용일 뿐, 운영 임계값이 아니다")
"""),
    ],
    "ch09_industrial": [
        md("# ch09 — 산업 적용: BYOD 워크플로\n\n자기 CSV를 즉시 평가하는 전체 흐름."),
        code(SETUP),
        code("""
# 예제 CSV 생성 (실무에서는 자기 데이터 경로 사용)
import pandas as pd
rng = np.random.default_rng(0)
n = 3000
values = np.sin(2*np.pi*np.arange(n)/100) + rng.normal(scale=0.1, size=n)
labels = np.zeros(n, dtype=int); values[2400:2450] += 2.5; labels[2400:2450] = 1
pd.DataFrame({"timestamp": np.arange(n), "value": values, "label": labels}).to_csv(
    "/tmp/my_sensor.csv", index=False)
"""),
        code("""
from tsad_forge.cli import main
# 라벨 열이 있으므로 전체 지표 산출. 라벨 열이 없으면 스코어+임계값만 출력된다.
main(["run", "--model", "sub_pca", "--data", "/tmp/my_sensor.csv",
      "--results-dir", "/tmp/byod-results"])
"""),
        code("""
# regime vs fault: 레짐 변화(계절성 전환)는 fault가 아니다 — drift 적응 임계값 비교
from tsad_forge.evaluation.thresholding import spot_threshold, dspot_threshold
drifting = np.arange(3000)/300 + rng.normal(scale=0.5, size=3000)
print("SPOT :", round(spot_threshold(drifting, q=1e-3), 2), "(드리프트를 이상으로 오인)")
print("DSPOT:", round(dspot_threshold(drifting, q=1e-3, depth=100), 2), "(마지막 drift 수준 반영)")
"""),
    ],
    "ch10_reading_benchmarks": [
        md("# ch10 — 벤치마크 읽는 법\n\n리더보드 parquet을 직접 열어 해석한다."),
        code(SETUP),
        code("""
from tsad_forge.runner.results import load_all_results
df = load_all_results("../benchmarks/results")
if df.empty:
    print("결과 없음 — 먼저: python benchmarks/run_all.py --profile configs/lite.yaml")
else:
    v = df[df["metric"] == "vus_pr"]
    print(v.groupby("generation")["value"].describe().round(3))
"""),
        code("""
# 세대별 분포 — '세대가 오르면 성능이 오르는가?'
if not df.empty:
    v.boxplot(column="value", by="generation", figsize=(9, 4))
    plt.suptitle(""); plt.title("VUS-PR by generation"); plt.ylabel("VUS-PR")
"""),
        code("""
# 재현: 각 행의 config_hash로 설정 역참조
import json, glob
js = sorted(glob.glob("../benchmarks/results/*.json"))
if js:
    cfg = json.load(open(js[0]))
    print(cfg["run_id"], "->", json.dumps(cfg["config"], indent=1)[:400])
"""),
    ],
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1))
        print("wrote", path)


if __name__ == "__main__":
    main()
