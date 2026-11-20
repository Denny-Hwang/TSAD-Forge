"""로컬에 배치된 데이터셋의 평가 단위(entity) 열거 — full 프로파일의 매트릭스 확장용.

벤치마크 프로파일에서 `{ data: smd, entities: all }`처럼 쓰면 runner가 이 모듈로
로컬 파일을 훑어 entity별 data_params 목록으로 펼친다 (benchmarks/run_all.py).
열거는 **로컬 파일 기준**이다 — 다운로드되지 않은 entity는 나타나지 않으며,
아무것도 없으면 빈 목록(run_all이 no_data로 보고)을 반환한다.

반환값은 각 로더의 키워드 인자 dict 목록이며 정렬되어 있다 (재현 가능한 순서,
`limit`로 앞에서 N개를 고를 때 결정적).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from tsad_forge.data.paths import resolve_data_dir


def _smd(root: Path) -> list[dict]:
    return [{"machine": p.stem} for p in sorted((root / "smd" / "train").glob("machine-*.txt"))]


def _nasa(spacecraft: str) -> Callable[[Path], list[dict]]:
    def enum(root: Path) -> list[dict]:
        base = root / "smap_msl"
        csv = base / "labeled_anomalies.csv"
        if not csv.exists():
            return []
        import pandas as pd

        df = pd.read_csv(csv)
        chans = df.loc[df["spacecraft"] == spacecraft, "chan_id"].astype(str)
        return [
            {"channel": c} for c in sorted(set(chans)) if (base / "train" / f"{c}.npy").exists()
        ]

    return enum


_UCR_RE = re.compile(r"^(\d{3})_UCR_Anomaly_")


def _ucr(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / "ucr").glob("*_UCR_Anomaly_*.txt")):
        m = _UCR_RE.match(p.name)
        if m:
            out.append({"series": int(m.group(1))})
    return out


def _tsb_ad(variant: str) -> Callable[[Path], list[dict]]:
    def enum(root: Path) -> list[dict]:
        base = root / f"tsb_ad_{variant}"
        return [{"filename": p.name} for p in sorted(base.rglob("*.csv"))]

    return enum


def _nab(root: Path) -> list[dict]:
    base = root / "nab" / "data"
    return [{"rel_path": p.relative_to(base).as_posix()} for p in sorted(base.rglob("*.csv"))]


def _skab(root: Path) -> list[dict]:
    base = root / "skab"
    return [
        {"experiment": p.relative_to(base).with_suffix("").as_posix()}
        for p in sorted(base.rglob("*.csv"))
        if p.parent.name != "anomaly-free"
    ]


def _mgab(root: Path) -> list[dict]:
    files = [p for p in (root / "mgab").glob("*.csv") if p.stem.isdigit()]
    return [{"series": int(p.stem)} for p in sorted(files, key=lambda p: int(p.stem))]


#: 데이터셋 이름 -> 열거 함수. 여기 없는 데이터셋은 entity가 하나뿐이다 (빈 params).
_ENUMERATORS: dict[str, Callable[[Path], list[dict]]] = {
    "smd": _smd,
    "smap": _nasa("SMAP"),
    "msl": _nasa("MSL"),
    "ucr": _ucr,
    "tsb_ad_u": _tsb_ad("u"),
    "tsb_ad_m": _tsb_ad("m"),
    "nab": _nab,
    "skab": _skab,
    "mgab": _mgab,
}


def supports_enumeration(name: str) -> bool:
    return name in _ENUMERATORS


def list_entities(name: str, data_dir: str | Path | None = None) -> list[dict]:
    """로컬에 존재하는 `name` 데이터셋의 entity별 로더 인자 목록."""
    if name not in _ENUMERATORS:
        raise KeyError(
            f"dataset '{name}' has a single entity or no enumerator; "
            f"enumerable: {sorted(_ENUMERATORS)}"
        )
    return _ENUMERATORS[name](resolve_data_dir(data_dir))


__all__ = ["list_entities", "supports_enumeration"]
