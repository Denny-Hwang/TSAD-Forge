"""TSAD-Forge CLI.

tsad-forge run --model dummy --data synthetic
tsad-forge run --model zscore --data path/to/your.csv        # BYOD
tsad-forge download <dataset>
tsad-forge list models|datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tsad-forge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a single experiment (fit → score → threshold → metrics)")
    run.add_argument("--model", default="dummy", help="registered model name")
    run.add_argument(
        "--data", default="synthetic", help="registered dataset name or CSV/parquet path"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--config", type=Path, help="YAML config file (CLI flags override it)")
    run.add_argument("--results-dir", default="benchmarks/results")
    run.add_argument(
        "--threshold-q", type=float, default=None, help="quantile threshold q (default 0.99)"
    )
    run.add_argument(
        "--force", action="store_true", help="rerun even if a result exists (disable resume)"
    )
    run.add_argument(
        "--legacy-pa",
        action="store_true",
        help="ALSO compute PA-F1 (inflated legacy metric; a warning is attached)",
    )
    run.add_argument(
        "--data-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="loader kwarg, repeatable (e.g. --data-param machine=machine-1-1)",
    )

    dl = sub.add_parser("download", help="download a public dataset with checksum manifest")
    dl.add_argument("dataset")
    dl.add_argument(
        "--data-dir", default=None, help="data root (default: $TSAD_FORGE_DATA or ./data)"
    )
    dl.add_argument(
        "--subset",
        default=None,
        help="comma-separated subset (e.g. smd machines, skab groups, nab files)",
    )

    ls = sub.add_parser("list", help="list registered models or datasets")
    ls.add_argument("what", choices=["models", "datasets"])

    viz = sub.add_parser("viz", help="generate all charts + leaderboard from results")
    viz.add_argument("--results-dir", default="benchmarks/results")
    viz.add_argument("--out-dir", default="docs/assets/charts")
    viz.add_argument("--data-dir", default="data")
    viz.add_argument("--leaderboard-out", default="docs/leaderboard/lite.md")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        if args.what == "models":
            from tsad_forge.models.registry import list_models

            for name in list_models():
                print(name)
        else:
            from tsad_forge.data.registry import list_datasets

            for name in list_datasets():
                print(name)
        return 0

    if args.command == "download":
        from tsad_forge.data.download import download_dataset

        subset = args.subset.split(",") if args.subset else None
        download_dataset(args.dataset, data_dir=args.data_dir, subset=subset)
        return 0

    if args.command == "viz":
        from tsad_forge.viz.charts import generate_all
        from tsad_forge.viz.eda import generate_eda
        from tsad_forge.viz.leaderboard import save_leaderboard

        generate_all(args.results_dir, args.out_dir, args.data_dir)
        generate_eda(Path(args.out_dir).parent / "eda", args.data_dir)
        lb = save_leaderboard(args.results_dir, args.leaderboard_out)
        print(f"leaderboard: {lb}")
        return 0

    if args.command == "run":
        from tsad_forge.runner.experiment import run_experiment

        overrides: dict = {}
        if args.config:
            overrides.update(yaml.safe_load(args.config.read_text()) or {})
        overrides.update(
            {
                "model": args.model,
                "data": args.data,
                "seed": args.seed,
                "results_dir": args.results_dir,
                "legacy_pa": args.legacy_pa,
            }
        )
        if args.threshold_q is not None:
            overrides["threshold"] = {"method": "quantile", "q": args.threshold_q}
        if args.data_param:
            dp = {}
            for item in args.data_param:
                key, _, val = item.partition("=")
                dp[key] = int(val) if val.isdigit() else val
            overrides["data_params"] = dp
        run_experiment(overrides, force=args.force)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
