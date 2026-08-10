"""벤치마크 매트릭스 실행기 (CLAUDE.md §5).

configs/의 프로파일(YAML)을 읽어 (model × dataset × seed) 매트릭스를 순차 실행한다.
이미 결과가 있는 조합은 건너뛴다(resume). 실패한 조합은 기록하고 계속 진행한다.

사용:
    python benchmarks/run_all.py --profile configs/lite.yaml
    python benchmarks/run_all.py --profile configs/lite.yaml --force
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsad_forge.runner.experiment import run_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, default=Path("configs/lite.yaml"))
    p.add_argument("--results-dir", default=None, help="override profile results_dir")
    p.add_argument("--force", action="store_true", help="rerun existing results")
    args = p.parse_args(argv)

    profile = yaml.safe_load(args.profile.read_text())
    results_dir = args.results_dir or profile.get("results_dir", "benchmarks/results")
    seeds = profile.get("seeds", [0])
    defaults = profile.get("defaults", {})

    combos = [(m, d, s) for m in profile["models"] for d in profile["datasets"] for s in seeds]
    print(f"profile={args.profile}  combos={len(combos)}  results_dir={results_dir}")

    failed: list[tuple] = []
    done = skipped = 0
    for model_spec, data_spec, seed in combos:
        model_spec = {"model": model_spec} if isinstance(model_spec, str) else dict(model_spec)
        data_spec = {"data": data_spec} if isinstance(data_spec, str) else dict(data_spec)
        overrides = {
            **defaults,
            **model_spec,
            **data_spec,
            "seed": seed,
            "results_dir": results_dir,
        }
        try:
            result = run_experiment(overrides, force=args.force)
            if result is None:
                skipped += 1
            else:
                done += 1
        except Exception:
            failed.append((model_spec.get("model"), data_spec.get("data"), seed))
            traceback.print_exc()

    print(f"\ndone={done}  skipped(resume)={skipped}  failed={len(failed)}")
    for f in failed:
        print(f"  FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
