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

    dl = sub.add_parser("download", help="download a public dataset with checksum verification")
    dl.add_argument("dataset")
    dl.add_argument("--data-dir", default="data")

    ls = sub.add_parser("list", help="list registered models or datasets")
    ls.add_argument("what", choices=["models", "datasets"])

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

        download_dataset(args.dataset, data_dir=args.data_dir)
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
        run_experiment(overrides, force=args.force)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
