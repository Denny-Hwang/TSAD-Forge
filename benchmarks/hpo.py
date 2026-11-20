"""하이퍼파라미터 탐색 실행기 — 프로토콜은 tsad_forge/runner/hpo.py 참조.

    python benchmarks/hpo.py --study configs/hpo/lite.yaml --dry-run
    python benchmarks/hpo.py --study configs/hpo/lite.yaml --jobs 6 --timeout 1800
    python benchmarks/run_all.py --profile configs/lite_tuned.yaml   # 선택값으로 평가

trial 결과는 study의 results_dir(리더보드와 분리)에 쌓이고 resume된다. 선택 결과는
study의 output 파일에 출처(provenance)와 함께 기록된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsad_forge.runner.hpo import (  # noqa: E402
    build_trial_jobs,
    load_study,
    select_best,
    write_selected,
)
from tsad_forge.runner.matrix import run_matrix  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--torch-jobs", type=int, default=1)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    study = load_study(args.study)
    jobs, trials = build_trial_jobs(study)
    n_ent = len(study["_tuning"])
    print(
        f"study={study['name']}  models={len(trials)}  trials/model={study['n_trials']}  "
        f"tuning entities={n_ent}  runs={len(jobs)}  pending={sum(not j.exists for j in jobs)}"
    )
    if args.dry_run:
        for model, ts in trials.items():
            for k, t in enumerate(ts):
                print(f"  {model}#t{k:02d}  {t or '(defaults)'}")
        return 0
    summary = run_matrix(
        jobs,
        results_dir=study["results_dir"],
        n_jobs=args.jobs,
        torch_jobs=args.torch_jobs,
        timeout=args.timeout,
    )
    print(summary.line())
    selected = select_best(study, trials)
    out = write_selected(study, selected)
    metric = study.get("metric", "vus_pr")
    for model, s in selected.items():
        print(
            f"  {model:20s} trial {s['trial']:2d}  tuning {metric}={s[f'tuning_{metric}']:.4f}"
            f"  (default {s[f'default_{metric}']})  {s['params']}"
        )
    print(f"selected -> {out}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
