"""Learning-track notebook generator (ch01-ch10) — regenerates notebooks/*.ipynb.

Notebooks are synthetic-data based so they run without downloads (some cells switch
to real data when a local data/ directory exists). Usage: python scripts/make_notebooks.py
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
            "# ch01 — Defining the TSAD problem\n\nBuild the anomaly types "
            "(point / contextual / collective) by hand.\n"
            "Theory: [docs/learn/ch01](../docs/learn/ch01_problem.md)"
        ),
        code(SETUP),
        code("""
# Synthetic data per anomaly type
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
# Contamination: how a polluted train split hurts the zscore baseline
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
            "# ch02 — Gen1 statistics: PCA-T²/SPE\n\nApplied to SMD (needs local data/) "
            "or to synthetic multivariate data."
        ),
        code(SETUP),
        code("""
# Use SMD if available, otherwise synthetic multivariate data
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
    ax.set_title(f"PCA-{mode.upper()} — T² (PC subspace) and SPE (residual subspace) catch different anomalies")
plt.tight_layout()
"""),
    ],
    "ch03_gen2_classical_ml": [
        md(
            "# ch03 — Gen2: Matrix Profile discords\n\nFind a waveform-distortion discord "
            "in a periodic signal with stumpy.\n"
            "If UCR data is available (`tsad-forge download ucr`), swap in a real series."
        ),
        code(SETUP),
        code("""
import stumpy
t = np.arange(3000)
x = np.sin(2 * np.pi * t / 60) + 0.05 * np.random.default_rng(0).normal(size=3000)
x[1500:1560] = np.sin(2 * np.pi * t[1500:1560] / 17)  # period-break discord

m = 60
mp = stumpy.stump(x, m=m)[:, 0].astype(float)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
a1.plot(x, lw=0.5)
a1.axvspan(1500, 1560, alpha=0.3, color="red")
a1.set_title("series")
a2.plot(mp, lw=0.7)
a2.set_title(f"matrix profile (m={m}) — the argmax is the discord")
print("discord at:", int(np.argmax(mp)))
"""),
        code("""
# IForest depends on the seed — hence 3-seed evaluation (CLAUDE.md section 4)
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
ds = generate_synthetic(seed=0)
for seed in range(3):
    s = get_model("iforest", seed=seed).fit(ds.train).score(ds.test)
    print(f"seed={seed}: VUS-PR={compute_metrics(s, ds.labels)['vus_pr']:.3f}")
"""),
    ],
    "ch04_gen3_dl": [
        md("# ch04 — Gen3 deep learning: reconstruction vs forecasting"),
        code(SETUP),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import zscore_normalize

ds = generate_synthetic(n_events=6, seed=4)
train, test = zscore_normalize(ds.train, ds.test)
for name in ["ae", "lstm_p", "usad"]:  # reconstruction / forecasting / adversarial recon
    scores = get_model(name, seed=0, epochs=5, window=32).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"{name:8s} VUS-PR={m['vus_pr']:.3f}  AUC-PR={m['auc_pr']:.3f}")
"""),
        code("""
# Over-generalization: a large-capacity AE reconstructs the anomalies too
for latent in [2, 16, 64]:
    scores = get_model("ae", seed=0, epochs=5, window=32, latent=latent).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"latent={latent:3d}: VUS-PR={m['vus_pr']:.3f}")
"""),
    ],
    "ch05_gen4": [
        md("# ch05 — Gen4: graphs (GDN) and attention (Anomaly Transformer)"),
        code(SETUP),
        code("""
from tsad_forge.models.registry import get_model
from tsad_forge.evaluation.metrics import compute_metrics
from tsad_forge.evaluation.protocol import zscore_normalize

ds = generate_synthetic(n_dims=5, n_events=8, seed=5)
train, test = zscore_normalize(ds.train, ds.test)
for name in ["gdn", "anomaly_transformer", "timesnet", "sub_pca"]:  # sub_pca = control
    kw = dict(seed=0, epochs=5, window=32) if name != "sub_pca" else {}
    scores = get_model(name, **kw).fit(train).score(test)
    m = compute_metrics(scores, ds.labels)
    print(f"{name:20s} VUS-PR={m['vus_pr']:.3f}")
print("\\nMind the gap to the simple baseline (sub_pca) — the leaderboard's core question.")
"""),
    ],
    "ch06_gen5": [
        md(
            "# ch06 — Gen5: MambaTSAD faithful vs fixed\n\nQuantify the effect of the four "
            "implementation issues (state indexing, CPU branch, HP-filter objective, "
            "global-FFT AMA)."
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
    print(f"{name:22s} VUS-PR = {np.mean(vals):.3f} +/- {np.std(vals):.3f} (3 seeds)")
"""),
        code("""
# HP-filter decomposition — faithful feeds the trend, fixed feeds the cycle
from tsad_forge.models.gen5_ssm_foundation.mamba_tsad import hp_filter
trend, cycle = hp_filter(ds.test)
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for ax, (y, ttl) in zip(axes, [(ds.test[:, 0], "original"),
                               (trend[:, 0], "trend (faithful input)"),
                               (cycle[:, 0], "cycle (fixed input)")]):
    ax.plot(y, lw=0.6)
    ax.set_title(ttl)
plt.tight_layout()
"""),
    ],
    "ch07_evaluation": [
        md(
            "# ch07 — Evaluation methodology: making a random score 'SOTA' with PA-F1\n\n"
            "**This notebook is why this repository exists** (reproducing Kim et al., AAAI 2022)."
        ),
        code(SETUP),
        code("""
from tsad_forge.evaluation.metrics import compute_metrics
import warnings

rng = np.random.default_rng(0)
labels = np.zeros(5000, dtype=int)
for s in range(200, 5000, 500):
    labels[s:s+100] = 1                      # long anomaly events (realistic)
scores = rng.random(5000)                    # completely random scores!

th = np.quantile(scores, 0.99)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = compute_metrics(scores, labels, threshold=th, legacy_pa=True)
print(f"random score: PA-F1       = {m['pa_f1']:.3f}   <- looks 'SOTA'")
print(f"random score: standard-F1 = {m['standard_f1']:.3f}")
print(f"random score: VUS-PR      = {m['vus_pr']:.3f}   <- the primary metric is not fooled")
"""),
        code("""
# The longer the events, the bigger the inflation
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
        md("# ch08 — Thresholding: EVT (SPOT) and conformal"),
        code(SETUP),
        code("""
from tsad_forge.evaluation.thresholding import spot_threshold, conformal_threshold, quantile_threshold
rng = np.random.default_rng(0)
cal = rng.normal(size=5000)                       # normal (calibration) scores
test_scores = np.concatenate([rng.normal(size=2000), rng.normal(5, 1, size=20)])  # 20 anomalies

for name, th in [
    ("quantile(0.99)", quantile_threshold(cal, 0.99)),
    ("SPOT(q=1e-3)", spot_threshold(test_scores, q=1e-3, calibration=cal)),
    ("conformal(a=0.01)", conformal_threshold(test_scores, alpha=0.01, calibration=cal)),
]:
    pred = test_scores >= th
    fp = pred[:2000].mean()
    tp = pred[2000:].mean()
    print(f"{name:18s} th={th:6.3f}  false-positive rate={fp:.4f}  detection rate={tp:.2f}")
"""),
        code("""
# Why oracle best-F1 is not deployable: it peeked at the test labels
from tsad_forge.evaluation.metrics import compute_metrics
labels = np.concatenate([np.zeros(2000, dtype=int), np.ones(20, dtype=int)])
m = compute_metrics(test_scores, labels)
print(f"best_f1(oracle)={m['best_f1']:.3f} — a leaderboard reference, not an operating threshold")
"""),
    ],
    "ch09_industrial": [
        md("# ch09 — Industrial practice: the BYOD workflow\n\nEvaluate your own CSV end to end."),
        code(SETUP),
        code("""
# Create an example CSV (use your own file path in practice)
import pandas as pd
rng = np.random.default_rng(0)
n = 3000
values = np.sin(2*np.pi*np.arange(n)/100) + rng.normal(scale=0.1, size=n)
labels = np.zeros(n, dtype=int)
values[2400:2450] += 2.5
labels[2400:2450] = 1
pd.DataFrame({"timestamp": np.arange(n), "value": values, "label": labels}).to_csv(
    "/tmp/my_sensor.csv", index=False)
"""),
        code("""
from tsad_forge.cli import main
# With a label column the full metric suite is computed;
# without one you get scores + threshold decisions only.
main(["run", "--model", "sub_pca", "--data", "/tmp/my_sensor.csv",
      "--results-dir", "/tmp/byod-results"])
"""),
        code("""
# Regime vs fault: a regime change (drift) is not a fault — compare drift-aware thresholds
from tsad_forge.evaluation.thresholding import spot_threshold, dspot_threshold
drifting = np.arange(3000)/300 + rng.normal(scale=0.5, size=3000)
print("SPOT :", round(spot_threshold(drifting, q=1e-3), 2), "(mistakes the drift for anomaly)")
print("DSPOT:", round(dspot_threshold(drifting, q=1e-3, depth=100), 2), "(tracks the final drift level)")
"""),
    ],
    "ch10_reading_benchmarks": [
        md("# ch10 — How to read the benchmark\n\nOpen the results parquet directly."),
        code(SETUP),
        code("""
from tsad_forge.runner.results import load_all_results
df = load_all_results("../benchmarks/results")
if df.empty:
    print("no results — first run: python benchmarks/run_all.py --profile configs/lite.yaml")
else:
    v = df[df["metric"] == "vus_pr"]
    print(v.groupby("generation")["value"].describe().round(3))
"""),
        code("""
# Distribution by generation — 'does newer mean better?'
if not df.empty:
    v.boxplot(column="value", by="generation", figsize=(9, 4))
    plt.suptitle("")
    plt.title("VUS-PR by generation")
    plt.ylabel("VUS-PR")
"""),
        code("""
# Reproducibility: resolve any row's config_hash back to its full configuration
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
