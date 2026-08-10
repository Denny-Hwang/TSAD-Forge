"""데이터셋 레지스트리 (CLAUDE.md §2): 출처·라이선스·인용·취급 원칙 + 로더 바인딩.

데이터 파일 체크섬은 다운로드 시 각 데이터셋 디렉터리의 MANIFEST.json에 기록된다
(tsad_forge.data.download._write_manifest). 알려진 결함은 docs/datasets/ 카드 참조.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tsad_forge.data.schema import TSADDataset


@dataclass
class DatasetEntry:
    """레지스트리 항목: 출처·라이선스·체크섬·인용 + 로더."""

    name: str
    loader: Callable[..., TSADDataset]
    source_url: str = ""
    license: str = ""
    citation: str = ""
    redistributable: bool = True
    checksums: dict[str, str] = field(default_factory=dict)  # 파일명 -> sha256 (알려진 값)
    notes: str = ""  # 알려진 결함 요약 (상세는 데이터셋 카드)


_REGISTRY: dict[str, DatasetEntry] = {}


def register_dataset(entry: DatasetEntry) -> None:
    if entry.name in _REGISTRY:
        raise ValueError(f"dataset '{entry.name}' already registered")
    _REGISTRY[entry.name] = entry


def get_dataset_entry(name: str) -> DatasetEntry:
    _ensure_builtin()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset '{name}'. Available: {sorted(_REGISTRY)}. "
            "For your own data, pass a CSV/parquet file path instead."
        )
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    _ensure_builtin()
    return sorted(_REGISTRY)


def load_dataset(name: str, **kwargs) -> TSADDataset:
    return get_dataset_entry(name).loader(**kwargs)


_BUILTIN_LOADED = False


def _ensure_builtin() -> None:
    """내장 항목 등록 (import 순환 방지를 위해 지연 등록)."""
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True

    from tsad_forge.data.loaders.nab import load_nab
    from tsad_forge.data.loaders.psm import load_psm
    from tsad_forge.data.loaders.skab import load_skab
    from tsad_forge.data.loaders.smap_msl import load_msl, load_smap
    from tsad_forge.data.loaders.smd import load_smd
    from tsad_forge.data.loaders.swat_wadi import load_swat, load_wadi
    from tsad_forge.data.loaders.tsb_ad import load_tsb_ad
    from tsad_forge.data.loaders.ucr import load_ucr
    from tsad_forge.data.loaders.yahoo import load_yahoo
    from tsad_forge.synthetic.generator import generate_synthetic

    register_dataset(
        DatasetEntry(
            name="synthetic",
            loader=generate_synthetic,
            source_url="(generated in-process)",
            license="Apache-2.0",
            citation="TSAD-Forge synthetic generator",
            notes="스모크 테스트·교육용 합성 데이터. 실제 벤치마크 결론에 사용 금지.",
        )
    )
    register_dataset(
        DatasetEntry(
            name="smap",
            loader=load_smap,
            source_url="https://github.com/khundman/telemanom",
            license="NASA open data",
            citation="Hundman et al., KDD 2018",
            notes="준이진 채널 다수, 희소 라벨 — PA 기반 선행 보고와 비교 불가",
        )
    )
    register_dataset(
        DatasetEntry(
            name="msl",
            loader=load_msl,
            source_url="https://github.com/khundman/telemanom",
            license="NASA open data",
            citation="Hundman et al., KDD 2018",
            notes="SMAP과 동일한 결함 프로파일",
        )
    )
    register_dataset(
        DatasetEntry(
            name="smd",
            loader=load_smd,
            source_url="https://github.com/NetManAIOps/OmniAnomaly",
            license="MIT (repo)",
            citation="Su et al., KDD 2019",
            notes="머신 간 이벤트 길이 분산 큼, 준상수 채널 존재",
        )
    )
    register_dataset(
        DatasetEntry(
            name="ucr",
            loader=load_ucr,
            source_url="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/",
            license="free for research",
            citation="Wu & Keogh, TKDE 2021",
            notes="시계열당 이상 1개 규약 — event-level 지표 해석 주의",
        )
    )
    register_dataset(
        DatasetEntry(
            name="tsb_ad_u",
            loader=lambda **kw: load_tsb_ad(variant="u", **kw),
            source_url="https://github.com/TheDatumOrg/TSB-AD",
            license="Apache-2.0",
            citation="Liu & Paparrizos, NeurIPS 2024",
        )
    )
    register_dataset(
        DatasetEntry(
            name="tsb_ad_m",
            loader=lambda **kw: load_tsb_ad(variant="m", **kw),
            source_url="https://github.com/TheDatumOrg/TSB-AD",
            license="Apache-2.0",
            citation="Liu & Paparrizos, NeurIPS 2024",
        )
    )
    register_dataset(
        DatasetEntry(
            name="nab",
            loader=load_nab,
            source_url="https://github.com/numenta/NAB",
            license="data free to use (code AGPL — not used)",
            citation="Lavin & Ahmad, ICMLA 2015",
            notes="train(probationary 15%)에 이상 포함 가능",
        )
    )
    register_dataset(
        DatasetEntry(
            name="psm",
            loader=load_psm,
            source_url="https://github.com/eBay/RANSynCoders",
            license="Apache-2.0 (repo)",
            citation="Abdulaal et al., KDD 2021",
            notes="train 결측 보간 필요, 분 단위 라벨 경계",
        )
    )
    register_dataset(
        DatasetEntry(
            name="skab",
            loader=load_skab,
            source_url="https://github.com/waico/SKAB",
            license="repo AGPL — 데이터 파일만 사용, 코드 미사용",
            citation="Katser & Kozitsin, 2020",
        )
    )
    register_dataset(
        DatasetEntry(
            name="yahoo_s5",
            loader=load_yahoo,
            source_url="https://webscope.sandbox.yahoo.com/catalog.php?datatype=s",
            license="Yahoo Webscope — 신청 필요",
            citation="Laptev et al., 2015",
            redistributable=False,
            notes="재배포 금지 — 신청 안내: docs/datasets/yahoo_s5.md",
        )
    )
    register_dataset(
        DatasetEntry(
            name="swat",
            loader=load_swat,
            source_url="https://itrust.sutd.edu.sg/itrust-labs_datasets/",
            license="iTrust — 신청 필요",
            citation="Goh et al., CRITIS 2016",
            redistributable=False,
            notes="재배포 금지 — 신청 안내: docs/datasets/swat_wadi.md",
        )
    )
    register_dataset(
        DatasetEntry(
            name="wadi",
            loader=load_wadi,
            source_url="https://itrust.sutd.edu.sg/itrust-labs_datasets/",
            license="iTrust — 신청 필요",
            citation="Ahmed et al., CySWATER 2017",
            redistributable=False,
            notes="재배포 금지 — 신청 안내: docs/datasets/swat_wadi.md",
        )
    )
