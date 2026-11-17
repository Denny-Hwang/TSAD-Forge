"""벤치마크 매트릭스 실행기 (CLAUDE.md §5).

configs/의 프로파일(YAML)을 읽어 (model × dataset × seed) 매트릭스를 실행한다.
- resume: 이미 결과가 있는 조합은 건너뜀 (--force로 재실행)
- 데이터 미존재(FileNotFoundError)는 실패가 아닌 'no_data'로 건너뜀
- 결정적 모델은 seeds[0]만, 확률적 모델(stochastic: true)은 전체 시드 실행
- `{ data: smd, entities: all }`은 로컬에 있는 entity 전부로 확장된다
- 병렬/격리 실행과 실패 기록: tsad_forge/runner/matrix.py 참조

사용:
    python benchmarks/run_all.py --profile configs/lite.yaml
    python benchmarks/run_all.py --profile configs/full.yaml --jobs 8 --timeout 3600
    python benchmarks/run_all.py --profile configs/full.yaml --models iforest,lof --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsad_forge.runner.matrix import (  # noqa: E402
    build_jobs,
    expand_combos,
    load_failures,
    run_matrix,
)

__all__ = ["expand_combos", "main"]


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--profile", type=Path, default=Path("configs/lite.yaml"))
    p.add_argument("--results-dir", default=None, help="override profile results_dir")
    p.add_argument("--force", action="store_true", help="rerun existing results")
    p.add_argument("--models", help="comma-separated model filter")
    p.add_argument("--datasets", help="comma-separated dataset-name filter (e.g. smd,ucr)")
    p.add_argument("--seeds", help="comma-separated seed filter")
    p.add_argument(
        "--jobs", type=int, default=1, help="parallel workers for non-torch models (isolated)"
    )
    p.add_argument(
        "--torch-jobs",
        type=int,
        default=1,
        help="concurrent torch-model runs (default 1 = one GPU)",
    )
    p.add_argument("--timeout", type=float, default=None, help="per-run timeout in seconds")
    p.add_argument("--isolate", action="store_true", help="one process per run even with --jobs 1")
    p.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    args = p.parse_args(argv)

    profile = yaml.safe_load(args.profile.read_text())
    results_dir = args.results_dir or profile.get("results_dir", "benchmarks/results")
    seeds = [int(s) for s in _csv(args.seeds)] if args.seeds else None
    jobs, empty = build_jobs(
        profile, results_dir, models=_csv(args.models), datasets=_csv(args.datasets), seeds=seeds
    )
    print(f"profile={args.profile}  combos={len(jobs)}  results_dir={results_dir}")
    if empty:
        print(f"  datasets with no local entities (download first): {sorted(set(empty))}")

    if args.dry_run:
        failed = {f["run_id"]: f["kind"] for f in load_failures(results_dir)}
        status = Counter()
        for j in jobs:
            s = "done" if j.exists and not args.force else failed.get(j.run_id, "pending")
            status[(j.lane, s)] += 1
            if s != "done":
                print(f"  {s:<12} {j.lane:<5} {j.tag}")
        for (lane, s), n in sorted(status.items()):
            print(f"  {lane:<5} {s:<12} {n}")
        return 0

    summary = run_matrix(
        jobs,
        results_dir=results_dir,
        force=args.force,
        n_jobs=args.jobs,
        torch_jobs=args.torch_jobs,
        timeout=args.timeout,
        isolate=args.isolate,
        plugins=profile.get("plugins"),
    )
    print(f"\n{summary.line()}")
    if summary.no_data:
        print("  datasets without local data (download first):", sorted(map(str, summary.no_data)))
    for tag, kind in summary.failed:
        print(f"  FAILED[{kind}]: {tag}")
    if summary.failed:
        print(f"  failure details: {Path(results_dir) / 'failures'}/")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
