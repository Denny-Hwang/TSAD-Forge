"""연구 아카이브 원장 (archive/papers.yaml) — 검증·렌더링·arXiv 신규 논문 스캔.

    tsad-forge archive validate          # 원장 스키마 + 모델/카드 정합성
    tsad-forge archive render            # docs/archive.md 생성
    tsad-forge archive scan --days 8     # 최근 arXiv TSAD 논문 중 원장에 없는 것 (markdown)

스캔은 후보를 *보고*만 한다 — 원장에 추가하는 것은 사람이 판단한다 (주간 워크플로가
이슈로 올린다: .github/workflows/archive-scan.yml).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

LEDGER = Path("archive/papers.yaml")
STATUSES = (
    "candidate",
    "license_audit",
    "implementing",
    "implemented",
    "benchmarked",
    "rejected",
)
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
#: 스캔 쿼리 — 제목/초록에 이상탐지와 시계열 표현이 함께 있는 논문
ARXIV_QUERY = (
    'abs:"anomaly detection" AND (abs:"time series" OR abs:"time-series" '
    'OR abs:"multivariate" OR abs:"sensor")'
)


def load_ledger(path: str | Path = LEDGER) -> list[dict]:
    return list((yaml.safe_load(Path(path).read_text()) or {}).get("papers", []))


def validate_ledger(
    papers: list[dict], results_dir: str | Path | None = "benchmarks/results"
) -> list[str]:
    """원장 오류 목록 (빈 목록 = 유효).

    - id 유일, status 유효, rejected에는 reason 필수
    - implemented/benchmarked는 등록된 모델(카드 보유)을 가리켜야 한다
    - benchmarked는 results_dir에 커밋된 결과가 하나 이상 있어야 한다
    - 공개 등록 모델은 모두 원장의 어떤 항목에 속해야 한다 (아카이브 누락 방지)
    """
    import importlib.util

    from tsad_forge.models.cards import get_card
    from tsad_forge.models.registry import list_models

    torch_missing = importlib.util.find_spec("torch") is None
    errors: list[str] = []
    seen: set[str] = set()
    registered = set(list_models())
    covered: set[str] = set()
    result_files = (
        {p.name.split("__", 1)[0] for p in Path(results_dir).glob("*.parquet")}
        if results_dir and Path(results_dir).exists()
        else None
    )
    for p in papers:
        pid = p.get("id")
        if not pid:
            errors.append(f"entry without id: {p.get('title')!r}")
            continue
        if pid in seen:
            errors.append(f"{pid}: duplicate id")
        seen.add(pid)
        for key in ("title", "year", "status"):
            if key not in p:
                errors.append(f"{pid}: missing '{key}'")
        status = p.get("status")
        if status not in STATUSES:
            errors.append(f"{pid}: status '{status}' not in {STATUSES}")
        if status == "rejected" and not p.get("reason"):
            errors.append(f"{pid}: rejected entries need a reason")
        if status in ("implemented", "benchmarked"):
            models = p.get("models") or []
            if not models:
                errors.append(f"{pid}: {status} entries must list models")
            for m in models:
                covered.add(m)
                if m not in registered:
                    # torch 없는 환경에서는 Gen3-5가 등록되지 않는다 — 카드가 있으면 통과
                    if not (torch_missing and get_card(m) is not None):
                        errors.append(f"{pid}: model '{m}' is not registered")
                elif get_card(m) is None:
                    errors.append(f"{pid}: model '{m}' has no card")
            if status == "benchmarked" and result_files is not None:
                if not set(models) & result_files:
                    errors.append(f"{pid}: marked benchmarked but no committed results found")
    public = {m for m in registered if not m.startswith("_")}
    for m in sorted(public - covered):
        errors.append(f"registered model '{m}' is not in any implemented/benchmarked entry")
    return errors


def render_markdown(papers: list[dict]) -> str:
    from tsad_forge.models.cards import get_card

    lines = [
        "# Research archive",
        "",
        "Auto-generated from `archive/papers.yaml` (`tsad-forge archive render`).",
        "Every method the project tracks, from first sighting to benchmark. New arXiv",
        "candidates are proposed weekly as GitHub issues by the archive-scan workflow;",
        "a maintainer decides what enters the ledger.",
        "",
    ]
    titles = {
        "benchmarked": "Benchmarked",
        "implemented": "Implemented (no published results yet)",
        "implementing": "In progress",
        "license_audit": "License audit",
        "candidate": "Candidates",
        "rejected": "Rejected",
    }
    for status in (
        "benchmarked",
        "implemented",
        "implementing",
        "license_audit",
        "candidate",
        "rejected",
    ):
        group = [p for p in papers if p.get("status") == status]
        if not group:
            continue
        lines += [f"## {titles[status]} ({len(group)})", ""]
        if status in ("benchmarked", "implemented"):
            lines += ["| method | year | models (fidelity) | notes |", "|---|---|---|---|"]
            for p in sorted(group, key=lambda x: x["year"]):
                models = ", ".join(
                    f"`{m}` ({(get_card(m) or {}).get('fidelity', '?')})"
                    for m in p.get("models", [])
                )
                lines.append(f"| {p['title']} | {p['year']} | {models} | {p.get('notes') or ''} |")
        else:
            lines += [
                "| method | venue | year | gen | code (license) | notes |",
                "|---|---|---|---|---|---|",
            ]
            for p in sorted(group, key=lambda x: -x["year"]):
                title = p["title"]
                if p.get("arxiv"):
                    title = f"[{title}](https://arxiv.org/abs/{p['arxiv']})"
                elif p.get("paper_url"):
                    title = f"[{title}]({p['paper_url']})"
                code = f"[repo]({p['code_url']})" if p.get("code_url") else "—"
                if p.get("code_license"):
                    code += f" ({p['code_license']})"
                note = p.get("reason") or p.get("notes") or ""
                lines.append(
                    f"| {title} | {p.get('venue') or ''} | {p['year']} | {p.get('generation') or ''}"
                    f" | {code} | {note} |"
                )
        lines.append("")
    return "\n".join(lines)


def save_render(papers: list[dict], out: str | Path = "docs/archive.md") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(papers))
    return out


# --- arXiv 스캔 ---------------------------------------------------------------


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def parse_arxiv_atom(xml_text: str) -> list[dict]:
    """arXiv API Atom 응답 → 논문 dict 목록 (id는 버전 접미사 제거)."""
    root = ET.fromstring(xml_text)  # noqa: S314 — arXiv 공식 API 응답
    out = []
    for e in root.findall(f"{_ATOM}entry"):
        raw_id = (e.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        summary = " ".join((e.findtext(f"{_ATOM}summary") or "").split())
        prim = e.find(f"{_ARXIV}primary_category")
        out.append(
            {
                "arxiv": arxiv_id,
                "title": " ".join((e.findtext(f"{_ATOM}title") or "").split()),
                "authors": [a.findtext(f"{_ATOM}name") for a in e.findall(f"{_ATOM}author")],
                "published": (e.findtext(f"{_ATOM}published") or "")[:10],
                "category": prim.get("term") if prim is not None else None,
                "comment": " ".join((e.findtext(f"{_ARXIV}comment") or "").split()) or None,
                "code_urls": sorted(
                    set(re.findall(r"https?://github\.com/[\w.-]+/[\w.-]+", summary))
                ),
            }
        )
    return out


def fetch_arxiv(max_results: int = 100, query: str = ARXIV_QUERY) -> str:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
    )
    req = urllib.request.Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "tsad-forge-archive-scan"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — 고정 https 엔드포인트
        return r.read().decode("utf-8")


def new_candidates(entries: list[dict], papers: list[dict], days: int, today=None) -> list[dict]:
    """원장에 없는(arXiv id·정규화 제목 기준) 최근 days일 이내 논문."""
    today = today or datetime.now(UTC).date()
    cutoff = (today - timedelta(days=days)).isoformat()
    known_ids = {str(p["arxiv"]) for p in papers if p.get("arxiv")}
    known_titles = {_norm_title(p["title"]) for p in papers}
    seen: set[str] = set()
    out = []
    for e in entries:
        if e["published"] < cutoff or e["arxiv"] in known_ids or e["arxiv"] in seen:
            continue
        if _norm_title(e["title"]) in known_titles:
            continue
        seen.add(e["arxiv"])
        out.append(e)
    return out


def candidates_markdown(cands: list[dict], days: int) -> str:
    if not cands:
        return ""
    lines = [
        f"{len(cands)} new arXiv paper(s) in the last {days} days mention time-series anomaly"
        " detection and are not in `archive/papers.yaml`. Triage each: add it as a `candidate`"
        " (or `rejected` with a reason), or ignore.",
        "",
    ]
    for c in cands:
        who = ", ".join(a for a in c["authors"][:3] if a) + (
            " et al." if len(c["authors"]) > 3 else ""
        )
        lines.append(f"- [ ] **[{c['title']}](https://arxiv.org/abs/{c['arxiv']})** — {who}")
        meta = [c["published"], c.get("category") or ""]
        if c.get("comment"):
            meta.append(c["comment"][:120])
        lines.append(f"  {' · '.join(m for m in meta if m)}")
        if c["code_urls"]:
            lines.append(f"  code: {', '.join(c['code_urls'])}")
    lines += [
        "",
        "Ledger snippet for an accepted candidate:",
        "```yaml",
        "  - id: <slug>",
        "    title: <title>",
        "    year: <year>",
        "    status: candidate",
        '    arxiv: "<id>"',
        "```",
    ]
    return "\n".join(lines)
