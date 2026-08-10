"""데이터셋 레지스트리 (CLAUDE.md §2).

M0에서는 스캐폴딩만 제공한다: 메타 항목 구조 + synthetic 항목.
M1에서 SMAP/MSL, SMD, UCR, TSB-AD 등 실제 데이터셋 항목·체크섬을 채운다.
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
    checksums: dict[str, str] = field(default_factory=dict)  # 파일명 -> sha256
    notes: str = ""  # 알려진 결함 등 (데이터셋 카드에도 반영)


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
