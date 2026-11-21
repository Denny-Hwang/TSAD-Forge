"""모델 카드 — 출처·라이선스·재현 충실도 메타데이터 (tsad_forge/models/cards.yaml).

리더보드는 이 카드의 `fidelity`를 행마다 표시한다. "simplified" 행의 수치는 원 방법의
성능이 아니라 이 저장소의 단순화 재구현의 성능이다 — 아카이브로서 이를 숨기지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CARDS_PATH = Path(__file__).with_name("cards.yaml")

FIDELITY_LEVELS = {
    "library": "maintained library / official package",
    "reference": "classical algorithm implemented from its definition",
    "upstream_verified": "behavior checked against the official code",
    "paper": "follows the paper (protocol-level omissions only)",
    "simplified": "paper-based reimplementation with documented simplifications",
    "baseline": "sanity / proposed baseline of this repository",
}
IMPLEMENTATIONS = ("own", "library", "adapter")
REQUIRED = ("title", "citation", "year", "implementation", "fidelity", "deviations")


@lru_cache(maxsize=1)
def load_cards() -> dict[str, dict]:
    return yaml.safe_load(CARDS_PATH.read_text()) or {}


def get_card(name: str) -> dict | None:
    """모델 이름의 카드. `<model>@tuned`는 기반 모델의 카드를 쓴다.

    그 밖의 model_alias 변형(예: mamba_tsad_fix_a)은 카드가 없을 수 있다 (None).
    """
    cards = load_cards()
    return cards.get(name) or cards.get(name.split("@", 1)[0])


def validate_card(name: str, card: dict) -> list[str]:
    """카드 스키마 오류 목록 (빈 목록이면 유효)."""
    errors = [f"{name}: missing '{k}'" for k in REQUIRED if k not in card]
    if card.get("fidelity") not in FIDELITY_LEVELS:
        errors.append(f"{name}: fidelity '{card.get('fidelity')}' not in {sorted(FIDELITY_LEVELS)}")
    if card.get("implementation") not in IMPLEMENTATIONS:
        errors.append(
            f"{name}: implementation '{card.get('implementation')}' not in {IMPLEMENTATIONS}"
        )
    if card.get("fidelity") == "simplified" and not card.get("deviations"):
        errors.append(f"{name}: simplified fidelity requires listed deviations")
    if not isinstance(card.get("deviations", []), list):
        errors.append(f"{name}: deviations must be a list")
    return errors


def catalog_markdown() -> str:
    """모델 카탈로그 페이지 (docs/models.md, 자동 생성)."""
    from tsad_forge.models.registry import get_model_class, list_models

    lines = [
        "# Model catalog",
        "",
        "Auto-generated from `tsad_forge/models/cards.yaml` by `tsad-forge viz`.",
        "Fidelity says how close each implementation is to the cited method:",
        "",
        *[f"- **{k}** — {v}" for k, v in FIDELITY_LEVELS.items()],
        "",
        "Rows marked **simplified** measure *this repository's* simplified reimplementation,"
        " not the original method — read their leaderboard numbers accordingly.",
        "",
        "| model | gen | method | citation | fidelity | upstream (license) | deviations |",
        "|---|---|---|---|---|---|---|",
    ]
    cards = load_cards()
    rows = []
    for name in list_models():
        if name.startswith("_") or name not in cards:
            continue
        c = cards[name]
        gen = get_model_class(name).generation
        cite = c["citation"] + (
            f" ([arXiv](https://arxiv.org/abs/{c['arxiv']}))" if c.get("arxiv") else ""
        )
        up = f"[repo]({c['upstream']})" if c.get("upstream") else "—"
        if c.get("upstream_license"):
            up += f" ({c['upstream_license']})"
        dev = "<br>".join(c.get("deviations") or []) or "—"
        rows.append(
            (
                gen,
                name,
                f"| `{name}` | {gen} | {c['title']} | {cite} | {c['fidelity']} | {up} | {dev} |",
            )
        )
    lines += [r[2] for r in sorted(rows)]
    return "\n".join(lines) + "\n"


def save_catalog(out_path: str | Path = "docs/models.md") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(catalog_markdown())
    return out_path
