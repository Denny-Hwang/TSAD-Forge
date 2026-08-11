"""Gen 5 — SSM/Foundation (2023–).

- MambaTSAD (Chen et al., IEEE SPL 2024, arXiv:2405.19823) 재현: `faithful`(원 구현의
  알려진 이슈 재현) vs `fixed`(수정판) 두 변형 — 수정 효과를 벤치마크로 정량화.
- MOMENT / Chronos / TimesFM 어댑터: 선택 의존성 (extras: foundation).
- DADA (ICLR 2025): 공개 저장소 라이선스 미확정 → 도입 보류 (THIRD_PARTY_NOTICES).
- mamba-ssm 미설치 환경 대비: 순수 PyTorch selective-scan fallback (기본 경로).
"""

from tsad_forge.models.gen5_ssm_foundation.foundation import (
    ChronosResidualDetector,
    MOMENTDetector,
    TimesFMResidualDetector,
)
from tsad_forge.models.gen5_ssm_foundation.mamba_tsad import (
    MambaTSADFaithful,
    MambaTSADFixed,
)

__all__ = [
    "ChronosResidualDetector",
    "MOMENTDetector",
    "MambaTSADFaithful",
    "MambaTSADFixed",
    "TimesFMResidualDetector",
]
