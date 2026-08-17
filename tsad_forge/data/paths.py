"""데이터 디렉터리 해석 (리뷰 P1) — cwd 의존 제거.

우선순위:
1. 호출자가 명시한 data_dir 인자
2. 환경변수 TSAD_FORGE_DATA
3. 저장소 루트의 ./data (기존 기본값 — 하위 호환)

로더/다운로더는 `data_dir: str | Path | None = None` 시그니처를 쓰고
`resolve_data_dir(data_dir)`로 실제 경로를 얻는다.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "TSAD_FORGE_DATA"


def default_data_dir() -> Path:
    """환경변수 TSAD_FORGE_DATA가 있으면 그 경로, 없으면 ./data."""
    return Path(os.environ.get(ENV_VAR) or "data")


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    """명시 인자 > 환경변수 > ./data 순으로 데이터 루트를 결정한다."""
    return Path(data_dir) if data_dir is not None else default_data_dir()
