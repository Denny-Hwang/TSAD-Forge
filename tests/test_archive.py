"""연구 아카이브 원장: 스키마/정합성, 렌더링, arXiv 파싱·중복 제거 (네트워크 없음)."""

from datetime import date

from tsad_forge.archive import (
    candidates_markdown,
    load_ledger,
    new_candidates,
    parse_arxiv_atom,
    render_markdown,
    validate_ledger,
)

ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2611.00001v2</id>
    <published>2026-11-15T10:00:00Z</published>
    <title>A New   Time Series
      Anomaly Detector</title>
    <summary>We propose X. Code: https://github.com/someone/x-tsad .</summary>
    <author><name>A. One</name></author><author><name>B. Two</name></author>
    <author><name>C. Three</name></author><author><name>D. Four</name></author>
    <arxiv:comment>Accepted at KDD 2027</arxiv:comment>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2611.00002v1</id>
    <published>2026-10-01T10:00:00Z</published>
    <title>Old Paper</title><summary>s</summary><author><name>E</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2611.00003v1</id>
    <published>2026-11-16T10:00:00Z</published>
    <title>Robust Random Cut Forest</title><summary>s</summary><author><name>F</name></author>
  </entry>
</feed>"""


def test_shipped_ledger_is_valid():
    assert validate_ledger(load_ledger()) == []


def test_validator_catches_problems():
    bad = [
        {"id": "a", "title": "t", "year": 2020, "status": "rejected"},
        {"id": "a", "title": "t", "year": 2020, "status": "done"},
        {"id": "b", "title": "t", "year": 2020, "status": "implemented", "models": ["nope"]},
    ]
    errors = "\n".join(validate_ledger(bad, results_dir=None))
    assert "need a reason" in errors and "duplicate id" in errors
    assert "not in" in errors and "'nope' is not registered" in errors
    assert "is not in any implemented/benchmarked entry" in errors  # 커버리지 누락


def test_render_groups_by_status():
    text = render_markdown(load_ledger())
    assert "## Benchmarked" in text and "## Candidates" in text
    assert "`sub_pca` (reference)" in text  # 카드의 fidelity가 함께 표시된다


def test_parse_and_filter_arxiv():
    entries = parse_arxiv_atom(ATOM)
    e = entries[0]
    assert e["arxiv"] == "2611.00001"  # 버전 접미사 제거
    assert e["title"] == "A New Time Series Anomaly Detector"
    assert e["code_urls"] == ["https://github.com/someone/x-tsad"]
    assert e["comment"] == "Accepted at KDD 2027" and e["category"] == "cs.LG"
    ledger = [
        {"id": "rrcf", "title": "Robust Random Cut Forest", "year": 2016, "status": "candidate"}
    ]
    cands = new_candidates(entries, ledger, days=8, today=date(2026, 11, 18))
    # 오래된 논문(기간 밖)과 원장에 이미 있는 제목은 빠진다
    assert [c["arxiv"] for c in cands] == ["2611.00001"]
    md = candidates_markdown(cands, 8)
    assert "A. One, B. Two, C. Three et al." in md and "- [ ]" in md
    assert candidates_markdown([], 8) == ""
