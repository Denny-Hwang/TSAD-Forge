"""데이터셋 다운로드 CLI 백엔드 (CLAUDE.md §2).

M0에서는 뼈대만 제공한다: sha256 검증 유틸 + 안내 메시지.
M1에서 데이터셋별 다운로드 로직과 체크섬이 채워진다.

원칙 (CLAUDE.md §10-3): 데이터셋은 절대 저장소에 커밋하지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")


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


def download_dataset(name: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    from tsad_forge.data.registry import get_dataset_entry

    entry = get_dataset_entry(name)
    if name == "synthetic":
        print("'synthetic' is generated in-process; nothing to download.")
        return
    if not entry.redistributable:
        raise SystemExit(
            f"'{name}' cannot be redistributed. See docs/datasets/{name}.md for how to "
            "apply for access and where to place the files locally."
        )
    raise NotImplementedError(
        f"downloader for '{name}' lands in milestone M1 (see CLAUDE.md §8)."
    )
