"""결과 스냅샷 — full/HPO 결과를 git 밖(GitHub Release 자산)으로 배포하기 위한 묶음.

왜: 실행당 parquet+JSON을 git에 커밋하는 방식은 lite(수백 개)에는 괜찮지만 full
매트릭스(수만 개)와 원본 스코어(npz)에는 맞지 않는다. 스냅샷은 한 번의 벤치마크
실행을 감사 가능한 단위로 묶는다:

    <name>/
      MANIFEST.json      스냅샷 메타 (커밋, dirty 여부, 개수, 파일별 sha256·크기)
      results.parquet    중복 제거된 전체 지표 (load_all_results와 같은 스키마)
      runs.jsonl         실행별 JSON 요약 (config·환경 지문 — 재현 커맨드의 근거)
      failures.jsonl     실패 기록 (있으면)
      scores-000.zip …   원본 스코어 npz (--with-scores; 지표 재계산·사례 뷰어용).
                         GitHub Release 자산은 파일당 2 GiB 미만이어야 해서 part_bytes
                         (기본 1800 MiB) 단위로 나눈다. format 1의 단일 scores.zip도 읽는다.

배포: `gh release upload <tag> <name>/*` (TSAD-Forge 저장소의 릴리스에 첨부).
사용: `tsad-forge results fetch --base-url <릴리스 다운로드 URL> --dest <dir>` →
sha256 검증 후 <dir>/snapshot.parquet, runs.jsonl, scores/로 풀어 `tsad-forge viz`가
그대로 읽게 한다.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = "MANIFEST.json"
FORMAT_VERSION = 2  # 2: 스코어를 scores-NNN.zip 여러 개로 분할 (1: 단일 scores.zip)
# GitHub Release 자산 한도(파일당 2 GiB 미만)보다 여유 있게 잡는다
DEFAULT_PART_BYTES = 1800 * 2**20
_SCORE_PART = re.compile(r"scores(-\d{3})?\.zip")
_ZIP_EOCD = 22  # end-of-central-directory 레코드
_ZIP_ENTRY = 30 + 46  # 로컬 헤더 + 중앙 디렉터리 항목 (이름 길이는 별도로 두 번)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_part_name(i: int) -> str:
    return f"scores-{i:03d}.zip"


def _plan_score_parts(files: list[Path], part_bytes: int) -> list[list[Path]]:
    """npz 목록을 순서대로 나눠, 무압축(ZIP_STORED) zip 하나가 part_bytes를 넘지 않게 한다."""
    parts: list[list[Path]] = []
    size = part_bytes + 1  # 첫 파일에서 새 part를 연다
    for p in files:
        need = p.stat().st_size + _ZIP_ENTRY + 2 * len(p.name.encode())
        if _ZIP_EOCD + need > part_bytes:
            raise ValueError(
                f"{p.name} ({p.stat().st_size} bytes) alone exceeds the part size {part_bytes}"
            )
        if size + need > part_bytes:
            parts.append([])
            size = _ZIP_EOCD
        parts[-1].append(p)
        size += need
    return parts


def pack_snapshot(
    results_dir: str | Path,
    out_dir: str | Path,
    name: str,
    with_scores: bool = False,
    part_bytes: int = DEFAULT_PART_BYTES,
) -> Path:
    """results_dir를 스냅샷 디렉터리 out_dir/name 으로 묶는다. 반환: 스냅샷 경로.

    with_scores면 원본 스코어를 scores-000.zip, scores-001.zip … 으로 나눠 담는다.
    각 zip은 part_bytes 이하다 (Release 자산 한도 대응).
    """
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
    # run_model = 실행 JSON의 model (집계 이름의 `@mps` 같은 장치 접미사 없음)
    kept = set(
        zip(
            df["run_model"],
            df["dataset"],
            df["channel"],
            df["seed"],
            df["config_hash"],
            strict=True,
        )
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
    score_parts: list[str] = []
    n_scores = 0
    if with_scores:
        scores = sorted((results_dir / "scores").glob("*.npz"))
        for i, group in enumerate(_plan_score_parts(scores, part_bytes)):
            part = snap / _score_part_name(i)
            with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
                for p in group:  # npz는 이미 압축돼 있다
                    z.write(p, arcname=p.name)
            if part.stat().st_size > part_bytes:  # 크기 계산이 어긋나면 업로드 전에 멈춘다
                raise RuntimeError(f"{part.name} is {part.stat().st_size} bytes > {part_bytes}")
            score_parts.append(part.name)
        n_scores = len(scores)

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
            "scores": n_scores,
        },
        "score_parts": score_parts,
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
    manifest = json.loads((snap / MANIFEST).read_text())
    # 검증된 MANIFEST에 있는 part만 푼다 (format 1: scores.zip, format 2: scores-NNN.zip)
    parts = sorted(f for f in manifest["files"] if _SCORE_PART.fullmatch(f))
    members: dict[str, list[str]] = {}
    seen: set[str] = set()
    for fname in parts:
        with zipfile.ZipFile(snap / fname) as z:
            members[fname] = z.namelist()
        for member in members[fname]:
            # zip slip 방지: 평탄한 npz 이름만 허용
            if "/" in member or "\\" in member or not member.endswith(".npz"):
                raise ValueError(f"unexpected entry in {fname}: {member!r}")
            if member in seen:
                raise ValueError(f"duplicate score entry across parts: {member!r}")
            seen.add(member)
    expected = manifest.get("counts", {}).get("scores")
    if expected is not None and parts and len(seen) != expected:
        raise ValueError(f"score parts hold {len(seen)} npz files, MANIFEST says {expected}")

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snap / "results.parquet", dest / "snapshot.parquet")
    shutil.copy2(snap / "runs.jsonl", dest / "runs.jsonl")
    shutil.copy2(snap / MANIFEST, dest / "SNAPSHOT_MANIFEST.json")
    for fname in parts:
        with zipfile.ZipFile(snap / fname) as z:
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
