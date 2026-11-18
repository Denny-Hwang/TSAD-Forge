"""결과 스냅샷 — full/HPO 결과를 git 밖(GitHub Release 자산)으로 배포하기 위한 묶음.

왜: 실행당 parquet+JSON을 git에 커밋하는 방식은 lite(수백 개)에는 괜찮지만 full
매트릭스(수만 개)와 원본 스코어(npz)에는 맞지 않는다. 스냅샷은 한 번의 벤치마크
실행을 감사 가능한 단위로 묶는다:

    <name>/
      MANIFEST.json      스냅샷 메타 (커밋, dirty 여부, 개수, 파일별 sha256·크기)
      results.parquet    중복 제거된 전체 지표 (load_all_results와 같은 스키마)
      runs.jsonl         실행별 JSON 요약 (config·환경 지문 — 재현 커맨드의 근거)
      failures.jsonl     실패 기록 (있으면)
      scores.zip         원본 스코어 npz (--with-scores; 지표 재계산·사례 뷰어용)

배포: `gh release upload <tag> <name>/*` (TSAD-Forge 저장소의 릴리스에 첨부).
사용: `tsad-forge results fetch --base-url <릴리스 다운로드 URL> --dest <dir>` →
sha256 검증 후 <dir>/snapshot.parquet, runs.jsonl, scores/로 풀어 `tsad-forge viz`가
그대로 읽게 한다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = "MANIFEST.json"
FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_snapshot(
    results_dir: str | Path,
    out_dir: str | Path,
    name: str,
    with_scores: bool = False,
) -> Path:
    """results_dir를 스냅샷 디렉터리 out_dir/name 으로 묶는다. 반환: 스냅샷 경로."""
    from tsad_forge import __version__
    from tsad_forge.runner.environment import git_state
    from tsad_forge.runner.matrix import load_failures
    from tsad_forge.runner.results import load_all_results, utc_now

    results_dir = Path(results_dir)
    df = load_all_results(results_dir)
    if df.empty:
        raise FileNotFoundError(f"no results under {results_dir}")
    snap = Path(out_dir) / name
    if snap.exists():
        raise FileExistsError(f"{snap} already exists — snapshots are immutable, pick a new name")
    snap.mkdir(parents=True)

    df.to_parquet(snap / "results.parquet", index=False)
    kept = set(
        zip(df["model"], df["dataset"], df["channel"], df["seed"], df["config_hash"], strict=True)
    )
    n_runs = 0
    with open(snap / "runs.jsonl", "w") as f:
        for js in sorted(results_dir.glob("*.json")):
            run = json.loads(js.read_text())
            key = (run["model"], run["dataset"], run["channel"], run["seed"], run["config_hash"])
            if key in kept:  # 중복 제거에서 탈락한 구버전 실행은 넣지 않는다
                f.write(json.dumps(run, default=str) + "\n")
                n_runs += 1
    failures = load_failures(results_dir)
    if failures:
        with open(snap / "failures.jsonl", "w") as f:
            for rec in failures:
                f.write(json.dumps(rec, default=str) + "\n")
    if with_scores:
        scores = sorted((results_dir / "scores").glob("*.npz"))
        with zipfile.ZipFile(snap / "scores.zip", "w", zipfile.ZIP_STORED) as z:
            for p in scores:  # npz는 이미 압축돼 있다
                z.write(p, arcname=p.name)

    git = git_state()
    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "created": utc_now(),
        "tsad_forge_version": __version__,
        "packed_at_commit": git["commit"],
        "packed_with_dirty_tree": git["dirty"],
        "result_commits": sorted(set(df["commit_hash"].astype(str))),
        "counts": {
            "runs": n_runs,
            "metric_rows": int(len(df)),
            "models": int(df["model"].nunique()),
            "datasets": int(df["dataset"].nunique()),
            "failures": len(failures),
        },
        "files": {
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in sorted(snap.iterdir())
            if p.name != MANIFEST
        },
    }
    (snap / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return snap


def verify_snapshot(snap: str | Path) -> list[str]:
    """MANIFEST 대비 누락/크기/해시 불일치 목록 (빈 목록 = 무결)."""
    snap = Path(snap)
    manifest = json.loads((snap / MANIFEST).read_text())
    problems = []
    for fname, meta in manifest["files"].items():
        p = snap / fname
        if not p.exists():
            problems.append(f"missing: {fname}")
        elif p.stat().st_size != meta["bytes"] or _sha256(p) != meta["sha256"]:
            problems.append(f"checksum mismatch: {fname}")
    return problems


def unpack_snapshot(snap: str | Path, dest: str | Path) -> Path:
    """검증된 스냅샷을 results_dir 형태로 푼다 (viz/leaderboard가 그대로 읽는다)."""
    snap, dest = Path(snap), Path(dest)
    problems = verify_snapshot(snap)
    if problems:
        raise ValueError(f"snapshot failed verification: {problems}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snap / "results.parquet", dest / "snapshot.parquet")
    shutil.copy2(snap / "runs.jsonl", dest / "runs.jsonl")
    shutil.copy2(snap / MANIFEST, dest / "SNAPSHOT_MANIFEST.json")
    if (snap / "scores.zip").exists():
        with zipfile.ZipFile(snap / "scores.zip") as z:
            for member in z.namelist():
                # zip slip 방지: 평탄한 npz 이름만 허용
                if "/" in member or "\\" in member or not member.endswith(".npz"):
                    raise ValueError(f"unexpected entry in scores.zip: {member!r}")
            z.extractall(dest / "scores")
    return dest


def fetch_snapshot(base_url: str, dest: str | Path, download_dir: str | Path | None = None) -> Path:
    """base_url/MANIFEST.json과 그 파일들을 받아 검증 후 dest에 푼다.

    base_url 예: https://github.com/<owner>/<repo>/releases/download/<tag>
    """
    base_url = base_url.rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("snapshot base_url must be https")
    dl = Path(download_dir) if download_dir else Path(dest) / ".snapshot_download"
    dl.mkdir(parents=True, exist_ok=True)

    def _get(fname: str) -> Path:
        out = dl / fname
        req = urllib.request.Request(f"{base_url}/{fname}", headers={"User-Agent": "tsad-forge"})
        with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
        return out

    manifest = json.loads(_get(MANIFEST).read_text())
    for fname in manifest["files"]:
        if "/" in fname or fname.startswith("."):
            raise ValueError(f"unsafe file name in manifest: {fname!r}")
        _get(fname)
    return unpack_snapshot(dl, dest)
