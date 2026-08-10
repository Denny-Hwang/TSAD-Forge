"""데이터셋별 로더 — 모두 통일 스키마 TSADDataset을 반환한다 (CLAUDE.md §2)."""

from tsad_forge.data.loaders.file import load_file
from tsad_forge.data.loaders.nab import load_nab
from tsad_forge.data.loaders.psm import load_psm
from tsad_forge.data.loaders.skab import load_skab
from tsad_forge.data.loaders.smap_msl import load_msl, load_smap
from tsad_forge.data.loaders.smd import load_smd
from tsad_forge.data.loaders.swat_wadi import load_swat, load_wadi
from tsad_forge.data.loaders.tsb_ad import load_tsb_ad
from tsad_forge.data.loaders.ucr import load_ucr
from tsad_forge.data.loaders.yahoo import load_yahoo

__all__ = [
    "load_file",
    "load_msl",
    "load_nab",
    "load_psm",
    "load_skab",
    "load_smap",
    "load_smd",
    "load_swat",
    "load_tsb_ad",
    "load_ucr",
    "load_wadi",
    "load_yahoo",
]
