"""TSAD-Forge CLI.

tsad-forge run --model dummy --data synthetic
tsad-forge run --model zscore --data path/to/your.csv        # BYOD
tsad-forge download <dataset>
tsad-forge list models|datasets
tsad-forge results pack|verify|fetch   # 결과 스냅샷 (GitHub Release 자산)
tsad-forge archive validate|render|scan # 연구 아카이브 원장
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tsad-forge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a single experiment (fit → score → threshold → metrics)")
    # 기본값은 None: 명시한 플래그만 --config 값을 덮어쓴다 (미지정 플래그가 config의
    # model/data/seed를 기본값으로 조용히 덮던 버그 수정). 최종 기본값은
    # runner.experiment.DEFAULT_CONFIG (model=dummy, data=synthetic, seed=0).
    run.add_argument("--model", default=None, help="registered model name (default: dummy)")
    run.add_argument(
        "--data",
        default=None,
        help="registered dataset name or CSV/parquet path (default: synthetic)",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--config", type=Path, help="YAML config file (CLI flags override it)")
    run.add_argument("--results-dir", default=None, help="default: benchmarks/results")
    run.add_argument(
        "--device",
        default=None,
        help="auto | cpu | cuda | cuda:N | mps (torch models only; default: auto; "
        "auto never picks mps)",
    )
    run.add_argument(
        "--sliding-window",
        default=None,
        help="VUS buffer width: integer, or 'auto' for per-series ACF period (default: 100)",
    )
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

    rs = sub.add_parser("results", help="pack / verify / fetch result snapshots (release assets)")
    rsub = rs.add_subparsers(dest="action", required=True)
    pk = rsub.add_parser("pack", help="bundle a results dir into an immutable snapshot")
    pk.add_argument("--results-dir", default="benchmarks/results")
    pk.add_argument("--name", required=True, help="snapshot name, e.g. full-v0.3.0")
    pk.add_argument("--out", default="dist/snapshots")
    pk.add_argument("--with-scores", action="store_true", help="include raw score npz files")
    pk.add_argument(
        "--part-size-mb",
        type=int,
        default=1800,
        help="max size of each scores-NNN.zip in MiB (GitHub Release assets must be < 2 GiB)",
    )
    vf = rsub.add_parser("verify", help="check a snapshot against its MANIFEST sha256")
    vf.add_argument("snapshot")
    ft = rsub.add_parser("fetch", help="download + verify + unpack a published snapshot")
    ft.add_argument("--base-url", required=True, help=".../releases/download/<tag>")
    ft.add_argument("--dest", required=True, help="results dir to unpack into")

    ar = sub.add_parser("archive", help="research archive ledger (archive/papers.yaml)")
    asub = ar.add_subparsers(dest="action", required=True)
    asub.add_parser("validate", help="check ledger schema and model/card consistency")
    rd = asub.add_parser("render", help="write the archive page")
    rd.add_argument("--out", default="docs/archive.md")
    sc = asub.add_parser("scan", help="list recent arXiv TSAD papers missing from the ledger")
    sc.add_argument("--days", type=int, default=8)
    sc.add_argument("--max-results", type=int, default=100)
    sc.add_argument("--out", default=None, help="write markdown here (default: stdout)")

    viz = sub.add_parser("viz", help="generate all charts + leaderboard from results")
    viz.add_argument("--results-dir", default="benchmarks/results")
    viz.add_argument("--out-dir", default="docs/assets/charts")
    viz.add_argument("--data-dir", default="data")
    viz.add_argument("--leaderboard-out", default="docs/leaderboard/lite.md")
    viz.add_argument("--catalog-out", default="docs/models.md", help="model catalog page")

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

    if args.command == "archive":
        from tsad_forge import archive

        papers = archive.load_ledger()
        if args.action == "validate":
            errors = archive.validate_ledger(papers)
            print("ledger OK" if not errors else "\n".join(errors))
            return 1 if errors else 0
        if args.action == "render":
            print(f"archive page: {archive.save_render(papers, args.out)}")
            return 0
        entries = archive.parse_arxiv_atom(archive.fetch_arxiv(args.max_results))
        md = archive.candidates_markdown(
            archive.new_candidates(entries, papers, args.days), args.days
        )
        if args.out:
            Path(args.out).write_text(md)
        else:
            print(md or f"no new candidates in the last {args.days} days")
        return 0

    if args.command == "results":
        from tsad_forge.runner import snapshot as snap

        if args.action == "pack":
            path = snap.pack_snapshot(
                args.results_dir,
                args.out,
                args.name,
                args.with_scores,
                part_bytes=args.part_size_mb * 2**20,
            )
            print(f"snapshot: {path}")
            parts = json.loads((path / snap.MANIFEST).read_text())["score_parts"]
            if parts:
                print(f"scores:   {len(parts)} part(s), {parts[0]} … {parts[-1]}")
            print(f"publish:  gh release upload <tag> {path}/*")
            return 0
        if args.action == "verify":
            problems = snap.verify_snapshot(args.snapshot)
            print("OK" if not problems else "\n".join(problems))
            return 1 if problems else 0
        dest = snap.fetch_snapshot(args.base_url, args.dest)
        print(f"unpacked into {dest} — run: tsad-forge viz --results-dir {dest}")
        return 0

    if args.command == "viz":
        from tsad_forge.viz.charts import generate_all
        from tsad_forge.viz.eda import generate_eda
        from tsad_forge.viz.leaderboard import save_leaderboard

        generate_all(args.results_dir, args.out_dir, args.data_dir)
        generate_eda(Path(args.out_dir).parent / "eda", args.data_dir)
        lb = save_leaderboard(args.results_dir, args.leaderboard_out)
        print(f"leaderboard: {lb}")
        from tsad_forge.models.cards import save_catalog

        print(f"model catalog: {save_catalog(args.catalog_out)}")
        return 0

    if args.command == "run":
        from tsad_forge.runner.experiment import run_experiment

        overrides: dict = {}
        if args.config:
            overrides.update(yaml.safe_load(args.config.read_text()) or {})
        flags = {
            "model": args.model,
            "data": args.data,
            "seed": args.seed,
            "results_dir": args.results_dir,
            "device": args.device,
            "legacy_pa": True if args.legacy_pa else None,
        }
        if args.sliding_window is not None:
            sw = args.sliding_window
            flags["sliding_window"] = sw if sw == "auto" else int(sw)
        overrides.update({k: v for k, v in flags.items() if v is not None})
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
