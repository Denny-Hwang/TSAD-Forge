"""모델 카드: 등록된 모든 모델에 유효한 카드가 있어야 한다 (아카이브 메타데이터 강제)."""

import pytest

from tsad_forge.models.cards import catalog_markdown, load_cards, validate_card
from tsad_forge.models.registry import get_model_class, list_models


def _public_models():
    return [m for m in list_models() if not m.startswith("_")]


def test_every_registered_model_has_a_valid_card():
    cards = load_cards()
    missing = [m for m in _public_models() if m not in cards]
    assert not missing, f"add cards for: {missing}"
    errors = [e for m in _public_models() for e in validate_card(m, cards[m])]
    assert not errors, errors


def test_no_orphan_cards():
    # torch 미설치 환경에서는 Gen3-5가 등록되지 않으므로 고아 검사를 건너뛴다
    pytest.importorskip("torch")
    orphans = sorted(set(load_cards()) - set(_public_models()))
    assert not orphans, f"cards without a registered model: {orphans}"


def test_card_generation_consistency_and_catalog():
    text = catalog_markdown()
    for m in _public_models():
        assert f"`{m}`" in text
        assert f"| {get_model_class(m).generation} |" in text


def test_validate_card_rejects_bad_entries():
    assert validate_card("x", {"fidelity": "great"})  # 필수 키 누락 + 잘못된 등급
    bad = {
        "title": "t",
        "citation": "c",
        "year": 2020,
        "implementation": "own",
        "fidelity": "simplified",
        "deviations": [],
    }
    assert any("requires listed deviations" in e for e in validate_card("x", bad))
