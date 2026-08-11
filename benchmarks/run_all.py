"""벤치마크 매트릭스 실행기 (CLAUDE.md §5).

configs/의 프로파일(YAML)을 읽어 (model × dataset × seed) 매트릭스를 순차 실행한다.
- resume: 이미 결과가 있는 조합은 건너뜀 (--force로 재실행)
- 데이터 미존재(FileNotFoundError)는 실패가 아닌 'no_data'로 건너뜀
  (재배포 금지·방화벽 환경 대비 — 어떤 데이터가 빠졌는지 요약에 표시)
- 결정적 모델은 seeds[0]만, 확률적 모델(stochastic: true)은 전체 시드 실행

사용:
    python benchmarks/run_all.py --profile configs/lite.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsad_forge.runner.experiment import run_experiment  # noqa: E402


def expand_combos(profile: dict) -> list[dict]:
    seeds = profile.get("seeds", [0])
    combos = []
    for model_spec in profile["models"]:
        model_spec = {"model": model_spec} if isinstance(model_spec, str) else dict(model_spec)
        stochastic = model_spec.pop("stochastic", False)
        model_seeds = seeds if stochastic else seeds[:1]
        for data_spec in profile["datasets"]:
            data_spec = {"data": data_spec} if isinstance(data_spec, str) else dict(data_spec)
            for seed in model_seeds:
                combos.append({**model_spec, **data_spec, "seed": seed})
    return combos


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, default=Path("configs/lite.yaml"))
    p.add_argument("--results-dir", default=None, help="override profile results_dir")
    p.add_argument("--force", action="store_true", help="rerun existing results")
    args = p.parse_args(argv)

    profile = yaml.safe_load(args.profile.read_text())
    results_dir = args.results_dir or profile.get("results_dir", "benchmarks/results")
    defaults = profile.get("defaults", {})
    combos = expand_combos(profile)
    print(f"profile={args.profile}  combos={len(combos)}  results_dir={results_dir}")

    failed: list[tuple] = []
    no_data: set[str] = set()
    done = skipped = 0
    for combo in combos:
        overrides = {**defaults, **combo, "results_dir": results_dir}
        tag = (combo.get("model"), str(combo.get("data")), combo.get("seed"))
        try:
            result = run_experiment(overrides, force=args.force)
            if result is None:
                skipped += 1
            else:
                done += 1
        except FileNotFoundError as e:
            no_data.add(str(combo.get("data")))
            print(f"  [no-data] skip {tag}: {e}")
        except Exception:
            failed.append(tag)
            traceback.print_exc()

    print(f"\ndone={done}  skipped(resume)={skipped}  no_data={len(no_data)}  failed={len(failed)}")
    if no_data:
        print("  datasets without local data (download first):", sorted(no_data))
    for f in failed:
        print(f"  FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
