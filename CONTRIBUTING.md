# Contributing to TSAD-Forge

Contributions are welcome — new detectors, new datasets, metric fixes,
documentation, and benchmark runs. This guide is the checklist the CI (and the
reviewers) will hold you to.

## Ground rules

- **License hygiene is non-negotiable.** Before importing any external code,
  fetch and read the upstream `LICENSE`. Only MIT / BSD / Apache-2.0 code may be
  vendored (copied). GPL/AGPL or unlicensed code must never be copied — connect
  via a pip dependency or reimplement from the paper, and record the decision in
  `THIRD_PARTY_NOTICES.md` either way.
- **Datasets are never committed.** Ship a download function + pinned SHA256
  checksums (`tsad_forge/data/checksums/`) + a dataset card. Restricted datasets
  get application instructions and a loader only.
- **No point adjustment by default.** VUS-PR is the primary metric; PA-F1 exists
  only behind `--legacy-pa` with a warning. PRs that report PA numbers as
  headline results will be asked to change.
- **Honest numbers only.** Performance claims need metric + dataset + seed count
  attached. "SOTA" without a leaderboard reference does not merge.
- **Language policy.** Docs are bilingual: every `docs/<page>.md` needs a
  `docs/<page>.ko.md` twin (CI enforces parity). Code comments/docstrings may be
  Korean or English — the project started Korean-first; either is acceptable,
  clarity wins. Public API names are always English.

## Dev setup

```bash
git clone https://github.com/Denny-Hwang/TSAD-Forge.git && cd TSAD-Forge
pip install -e ".[dev]"          # + ".[dl]" for Gen3-5 (torch)
pytest                            # must be green before any PR
ruff check . && black --check . && python -m mypy tsad_forge
```

## Checklist: adding a model

1. Subclass `BaseDetector` (`fit(train)`, `score(test) -> np.ndarray[T]`,
   continuous scores, larger = more anomalous). **No thresholding inside the
   model** — that belongs to `evaluation/thresholding.py`.
   Online detectors subclass `OnlineDetector` (`observe(x_t) -> float`) instead.
2. Register with `@register_model("name")`, set `generation = "genN"`, and add
   the import to `_ensure_builtin()` in `models/registry.py` if it's a new module.
3. License audit (see ground rules) → row in `THIRD_PARTY_NOTICES.md`.
   Reimplementations must state deviations from the paper in the module docstring.
4. Tests: contract test (shape/determinism with fixed seed) + a behavioral test
   (detects an injected anomaly on synthetic data). Torch models must run within
   8 GB VRAM with default config, and be guarded by `pytest.importorskip("torch")`.
5. Add the model to `configs/lite.yaml` (mark `stochastic: true` if seed-dependent),
   run `python benchmarks/run_all.py --profile configs/lite.yaml`, and commit the
   new results + regenerated leaderboard (`tsad-forge viz`).

## Checklist: adding a dataset

1. Loader in `tsad_forge/data/loaders/` returning the unified `TSADDataset`
   schema; register it in `data/registry.py` with source, license, and citation.
2. Download function in `data/download.py` + pinned SHA256 manifest under
   `tsad_forge/data/checksums/<name>.json` (run the download once, then copy the
   generated `MANIFEST.json`). Non-redistributable data: application
   instructions + loader only, no downloader.
3. Dataset card `docs/datasets/<name>.md` (+ `.ko.md`): what the data is, known
   flaws, and an honest note on label quality. Add EDA via `EDA_TARGETS` in
   `tsad_forge/viz/eda.py` if the data is publicly downloadable.
4. Loader test with a synthetic fixture mimicking the real file format
   (tests must not require the actual download).
5. Wire into `configs/lite.yaml` if it should be benchmarked by default.

## Results and reproducibility

- Every experiment must be reproducible from a config: fixed seeds, config-hash
  tracked, results as parquet+JSON under `benchmarks/results/`.
- Do not edit result files by hand. Re-run instead; aggregation dedups
  superseded runs automatically.
- The measurement methodology (timing, memory, aggregation, CI/Friedman) is
  documented in `docs/methodology.md` — read it before changing the runner.

## Releases

Tags `v*` trigger the release workflow (build + GitHub Release). Versioning is
semver-ish: schema-affecting changes bump minor pre-1.0. PyPI publishing uses
trusted publishing and requires a maintainer to enable it (see
`.github/workflows/release.yml`). Archiving releases on Zenodo for a citable
DOI is planned — maintainers: enable the Zenodo-GitHub integration, then update
`CITATION.cff` with the DOI.
