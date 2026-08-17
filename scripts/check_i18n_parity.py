#!/usr/bin/env python
"""docs/ 영/한 문서 쌍 검사 (리뷰 P2 — 수동 병행 유지의 드리프트 방지).

규칙: docs/ 아래 모든 콘텐츠 페이지 <name>.md 는 <name>.ko.md 짝이 있어야 하고,
고아 .ko.md(영어 원본 없음)도 금지. 자동 생성물(assets/)과 리더보드 테이블
(결과에서 재생성되는 lite.md — 숫자 표라 언어 중립)은 제외한다.

CI에서 실행: python scripts/check_i18n_parity.py  (불일치 시 exit 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# 언어 쌍을 요구하지 않는 경로 (자동 생성물 / 숫자 표)
EXEMPT = {
    "leaderboard/lite.md",  # results에서 재생성되는 숫자 표 (언어 중립)
}
EXEMPT_DIRS = ("assets/",)


def main() -> int:
    missing_ko: list[str] = []
    orphan_ko: list[str] = []
    for p in sorted(DOCS.rglob("*.md")):
        rel = str(p.relative_to(DOCS))
        if rel in EXEMPT or rel.startswith(EXEMPT_DIRS):
            continue
        if rel.endswith(".ko.md"):
            if not (DOCS / rel.removesuffix(".ko.md")).with_suffix(".md").exists():
                orphan_ko.append(rel)
        else:
            ko = p.with_name(p.stem + ".ko.md")
            if not ko.exists():
                missing_ko.append(rel)

    if missing_ko:
        print("English pages missing a Korean counterpart (.ko.md):")
        for rel in missing_ko:
            print(f"  docs/{rel}")
    if orphan_ko:
        print("Korean pages whose English original is gone:")
        for rel in orphan_ko:
            print(f"  docs/{rel}")
    if missing_ko or orphan_ko:
        return 1
    print(f"i18n parity OK ({sum(1 for _ in DOCS.rglob('*.md'))} markdown files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
