"""데이터셋 다운로드 (CLAUDE.md §2, §10-3).

원칙:
- 데이터셋은 절대 저장소에 커밋하지 않는다. 이 모듈은 로컬 `data/` 아래로만 받는다.
- 재배포 금지 데이터셋(Yahoo S5, SWaT/WADI)은 다운로드하지 않고 신청 안내를 출력한다.
- 받은 파일은 sha256을 기록/검증한다 (registry.checksums에 알려진 값이 있으면 대조).

사용: tsad-forge download smd [--subset machine-1-1,machine-1-2] [--data-dir data]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")

SMD_MACHINES = [f"machine-{g}-{i}" for g, n in [(1, 8), (2, 9), (3, 11)] for i in range(1, n + 1)]

# NAB lite 부분집합 (전체는 58개 파일; subset="all"로 전체 다운로드)
NAB_LITE_FILES = [
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "artificialWithAnomaly/art_daily_jumpsup.csv",
]

SKAB_GROUPS = {"valve1": 16, "valve2": 4, "other": 15}  # 그룹별 파일 수 (0..n-1.csv)

URLS = {
    "smd": "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset",
    "telemanom_labels": "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv",
    "telemanom_zip": "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
    "nab": "https://raw.githubusercontent.com/numenta/NAB/master",
    "psm": "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data",
    "skab": "https://raw.githubusercontent.com/waico/SKAB/master/data",
    "ucr_zip": (
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
        "UCR_TimeSeriesAnomalyDatasets2021.zip"
    ),
    "tsb_ad_u_zip": "https://www.thedatum.org/datasets/TSB-AD-U.zip",
    "tsb_ad_m_zip": "https://www.thedatum.org/datasets/TSB-AD-M.zip",
}


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected: str) -> None:
    actual = sha256sum(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The file may be corrupted or the upstream source changed."
        )


def _fetch(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """URL을 dest로 내려받고 sha256을 기록(.sha256 sidecar). 이미 있으면 건너뜀."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise ConnectionError(f"failed to fetch {url}: {e}") from e
    if expected_sha256:
        actual = sha256sum(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
    tmp.rename(dest)
    dest.with_suffix(dest.suffix + ".sha256").write_text(sha256sum(dest) + "\n")
    print(f"  fetched {dest} ({dest.stat().st_size:,} bytes)")
    return dest


PINNED_CHECKSUMS_DIR = Path(__file__).parent / "checksums"


def load_pinned_checksums(dataset_dirname: str) -> dict[str, str] | None:
    """저장소에 커밋된 known-good sha256 (파일 상대경로 -> 해시). 없으면 None."""
    path = PINNED_CHECKSUMS_DIR / f"{dataset_dirname}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_manifest(dataset_dir: Path, verify: bool = True) -> None:
    """디렉터리 내 데이터 파일들의 sha256 manifest 기록 + 고정 체크섬 대조.

    저장소에 커밋된 known-good 체크섬(tsad_forge/data/checksums/<name>.json)이 있으면
    받은 파일과 대조한다 (리뷰 P0-4 — 다운로드 시점 기록만으로는 업스트림 변조/오염을
    감지하지 못하는 TOFU 문제 해결). subset 다운로드를 지원해야 하므로 대조는
    '고정 목록과 실제 받은 파일의 교집합'에 대해서만 수행하고, 고정 목록에 없는
    파일은 MANIFEST.json에 기록만 한다.
    """
    manifest = {
        str(p.relative_to(dataset_dir)): sha256sum(p)
        for p in sorted(dataset_dir.rglob("*"))
        if p.is_file() and not p.name.endswith((".sha256", ".part")) and p.name != "MANIFEST.json"
    }
    (dataset_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    pinned = load_pinned_checksums(dataset_dir.name) if verify else None
    if pinned:
        mismatched = {
            rel: (pinned[rel], manifest[rel])
            for rel in pinned.keys() & manifest.keys()
            if pinned[rel] != manifest[rel]
        }
        if mismatched:
            detail = "\n".join(
                f"  {rel}: pinned {exp[:12]}..., got {act[:12]}..."
                for rel, (exp, act) in sorted(mismatched.items())[:5]
            )
            raise RuntimeError(
                f"checksum mismatch against pinned known-good values for "
                f"'{dataset_dir.name}' ({len(mismatched)} file(s)):\n{detail}\n"
                "The upstream source may have changed or the download is corrupted. "
                f"Delete {dataset_dir} and retry; if the mismatch persists, please "
                "open an issue — the pinned checksums may need a reviewed update."
            )
        n_checked = len(pinned.keys() & manifest.keys())
        print(f"[checksum] {dataset_dir.name}: {n_checked} file(s) verified against pinned sha256")


# --- 데이터셋별 다운로더 ---


def download_smd(data_dir: Path, subset: list[str] | None = None) -> None:
    machines = subset or SMD_MACHINES
    unknown = set(machines) - set(SMD_MACHINES)
    if unknown:
        raise ValueError(f"unknown SMD machines: {sorted(unknown)}")
    root = data_dir / "smd"
    for m in machines:
        for part in ("train", "test", "test_label"):
            _fetch(f"{URLS['smd']}/{part}/{m}.txt", root / part / f"{m}.txt")
    _write_manifest(root)


def download_smap_msl(data_dir: Path, subset: list[str] | None = None) -> None:
    """SMAP/MSL (NASA telemanom). 라벨 csv + 채널별 npy가 든 zip.

    1차 출처 s3(telemanom) zip. 네트워크 정책으로 zip을 받을 수 없는 환경에서는
    HuggingFace 미러(appleparan/telemanom)를 수동으로 받아 data/smap_msl/ 아래
    train/<chan>.npy, test/<chan>.npy 로 배치하면 로더가 동작한다.
    """
    root = data_dir / "smap_msl"
    _fetch(URLS["telemanom_labels"], root / "labeled_anomalies.csv")
    zip_path = root / "data.zip"
    if not (root / "train").exists():
        try:
            _fetch(URLS["telemanom_zip"], zip_path)
        except ConnectionError as e:
            raise ConnectionError(
                f"{e}\nS3 접근 불가 시 HuggingFace 미러 'appleparan/telemanom'에서 수동 다운로드 후 "
                f"{root}/train, {root}/test 아래에 채널별 .npy를 배치하세요 (docs/datasets/smap_msl.md)."
            ) from e
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        # zip 내부 구조 data/train/*.npy → root/train/*.npy 평탄화
        inner = root / "data"
        if inner.exists():
            for sub in ("train", "test"):
                (root / sub).mkdir(exist_ok=True)
                for f in (inner / sub).glob("*.npy"):
                    f.rename(root / sub / f.name)
            shutil.rmtree(inner)
        zip_path.unlink(missing_ok=True)
    _write_manifest(root)


def download_ucr(data_dir: Path, subset: list[str] | None = None) -> None:
    """UCR Anomaly Archive (250개 시계열, Keogh 공식 배포)."""
    root = data_dir / "ucr"
    zip_path = root / "ucr.zip"
    if not any(root.glob("*.txt")):
        _fetch(URLS["ucr_zip"], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                name = Path(info.filename).name
                if name.endswith(".txt") and "UCR_Anomaly" in name:
                    with z.open(info) as src, open(root / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
    _write_manifest(root)


def download_tsb_ad(data_dir: Path, variant: str = "u", subset: list[str] | None = None) -> None:
    """TSB-AD-U / TSB-AD-M (TheDatumOrg, Apache-2.0)."""
    variant = variant.lower()
    if variant not in ("u", "m"):
        raise ValueError("variant must be 'u' or 'm'")
    root = data_dir / f"tsb_ad_{variant}"
    zip_path = root / "data.zip"
    if not any(root.rglob("*.csv")):
        _fetch(URLS[f"tsb_ad_{variant}_zip"], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        zip_path.unlink(missing_ok=True)
    _write_manifest(root)


def download_nab(data_dir: Path, subset: list[str] | None = None) -> None:
    root = data_dir / "nab"
    files = NAB_LITE_FILES if subset is None else subset
    if subset == ["all"]:
        raise NotImplementedError(
            "full NAB download는 저장소 clone을 권장: git clone https://github.com/numenta/NAB "
            "(데이터 파일은 자유 이용; 코드는 AGPL이므로 코드 복사 금지)"
        )
    _fetch(f"{URLS['nab']}/labels/combined_windows.json", root / "labels" / "combined_windows.json")
    for rel in files:
        _fetch(f"{URLS['nab']}/data/{rel}", root / "data" / rel)
    _write_manifest(root)


def download_psm(data_dir: Path, subset: list[str] | None = None) -> None:
    root = data_dir / "psm"
    for name in ("train.csv", "test.csv", "test_label.csv"):
        _fetch(f"{URLS['psm']}/{name}", root / name)
    _write_manifest(root)


def download_skab(data_dir: Path, subset: list[str] | None = None) -> None:
    root = data_dir / "skab"
    _fetch(
        f"{URLS['skab']}/anomaly-free/anomaly-free.csv", root / "anomaly-free" / "anomaly-free.csv"
    )
    groups = subset or list(SKAB_GROUPS)
    for group in groups:
        if group not in SKAB_GROUPS:
            raise ValueError(f"unknown SKAB group '{group}'. Available: {sorted(SKAB_GROUPS)}")
        for i in range(SKAB_GROUPS[group]):
            _fetch(f"{URLS['skab']}/{group}/{i}.csv", root / group / f"{i}.csv")
    _write_manifest(root)


_DOWNLOADERS = {
    "smd": download_smd,
    "smap": download_smap_msl,
    "msl": download_smap_msl,
    "smap_msl": download_smap_msl,
    "ucr": download_ucr,
    "nab": download_nab,
    "psm": download_psm,
    "skab": download_skab,
}

_RESTRICTED = {
    "yahoo_s5": "docs/datasets/yahoo_s5.md",
    "swat": "docs/datasets/swat_wadi.md",
    "wadi": "docs/datasets/swat_wadi.md",
}


def download_dataset(
    name: str, data_dir: str | Path = DEFAULT_DATA_DIR, subset: list[str] | None = None
) -> None:
    name = name.lower().replace("-", "_")
    data_dir = Path(data_dir)
    if name == "synthetic":
        print("'synthetic' is generated in-process; nothing to download.")
        return
    if name in _RESTRICTED:
        raise SystemExit(
            f"'{name}'은(는) 재배포 금지 데이터셋입니다. 신청 방법과 로컬 배치 경로는 "
            f"{_RESTRICTED[name]} 문서를 참고하세요."
        )
    if name.startswith("tsb_ad"):
        variant = name.split("_")[-1] if name[-1] in ("u", "m") else "u"
        download_tsb_ad(data_dir, variant=variant, subset=subset)
        return
    if name not in _DOWNLOADERS:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(_DOWNLOADERS)} + tsb_ad_u/m")
    _DOWNLOADERS[name](data_dir, subset=subset)
    print(f"done: {name} → {data_dir / name}")


def download_mgab(data_dir: Path, subset: list[str] | None = None) -> None:
    """MGAB (Mackey-Glass Anomaly Benchmark, Thill et al.) — CC0 1.0 (public domain).

    10 univariate chaotic series (1..10.csv) fetched from the MarkusThill/MGAB repo.
    """
    root = data_dir / "mgab"
    series = subset or [str(i) for i in range(1, 11)]
    base = "https://raw.githubusercontent.com/MarkusThill/MGAB/master"
    for s in series:
        _fetch(f"{base}/{int(s)}.csv", root / f"{int(s)}.csv")
    _write_manifest(root)


def download_mba(data_dir: Path, subset: list[str] | None = None) -> None:
    """MBA (MIT-BIH Supraventricular Arrhythmia, 2-lead ECG) — processed copy from the
    TranAD repository (BSD-3-Clause); underlying PhysioNet data is ODC-BY (open).
    """
    root = data_dir / "mba"
    base = "https://raw.githubusercontent.com/imperial-qore/TranAD/main/data/MBA"
    for name in ("train.xlsx", "test.xlsx", "labels.xlsx"):
        _fetch(f"{base}/{name}", root / name)
    _write_manifest(root)


_DOWNLOADERS["mgab"] = download_mgab
_DOWNLOADERS["mba"] = download_mba
