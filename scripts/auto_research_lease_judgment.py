#!/usr/bin/env python3
"""Research lease-underwriting knowledge and save decision-ready Obsidian notes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
DEFAULT_OUTPUT_DIR = "Projects/tune_lease_55/Research/Auto Research"


@dataclass(frozen=True)
class ResearchTopic:
    key: str
    title: str
    query: str
    validity_days: int
    tags: tuple[str, ...]


TOPICS: tuple[ResearchTopic, ...] = (
    ResearchTopic(
        "cash-flow",
        "返済余力と資金繰りの異常兆候",
        "中小企業 リース審査 キャッシュフロー 資金繰り 返済余力 異常兆候",
        180,
        ("信用リスク", "資金繰り", "財務"),
    ),
    ResearchTopic(
        "residual-value",
        "物件別の残価・中古流動性・陳腐化",
        "設備 リース 残価 中古市場 流動性 技術陳腐化 査定",
        180,
        ("残価", "物件", "中古市場"),
    ),
    ResearchTopic(
        "lease-accounting-tax",
        "リース会計・税務・制度変更の審査影響",
        "日本 リース会計基準 税務 制度変更 借手 審査 2026",
        180,
        ("会計基準", "税務", "制度"),
    ),
    ResearchTopic(
        "contract-ownership",
        "契約・所有権・検収・詐欺リスク",
        "リース契約 所有権 検収 架空物件 二重譲渡 詐欺 審査",
        365,
        ("契約", "所有権", "不正防止"),
    ),
    ResearchTopic(
        "industry-risk",
        "業種別の倒産要因と先行指標",
        "日本 中小企業 業種別 倒産要因 先行指標 リース審査",
        90,
        ("業種分析", "倒産", "先行指標"),
    ),
    ResearchTopic(
        "asset-operation",
        "設備稼働率・保守・更新投資の確認方法",
        "設備投資 稼働率 保守 更新投資 リース 審査 確認",
        365,
        ("設備投資", "稼働率", "保守"),
    ),
    ResearchTopic(
        "subsidy-timing",
        "補助金・税制優遇と支払タイミング",
        "設備投資 補助金 税制優遇 リース 対象要件 支払時期 2026",
        90,
        ("補助金", "税制優遇", "資金繰り"),
    ),
    ResearchTopic(
        "pricing-rate",
        "金利・料率・競合条件の組み立て",
        "日本 リース料率 金利 設備金融 信用スプレッド 競合条件",
        60,
        ("金利", "料率", "競合"),
    ),
    ResearchTopic(
        "bank-support",
        "銀行支援・保証・条件変更の実効性",
        "中小企業 銀行支援 保証 条件変更 リース審査 実効性",
        180,
        ("銀行支援", "保証", "条件変更"),
    ),
    ResearchTopic(
        "collection-default",
        "延滞・回収・倒産時の物件保全",
        "リース 延滞 回収 倒産 物件引揚げ 保全 日本",
        365,
        ("延滞", "回収", "物件保全"),
    ),
)


def _get_gemini_key() -> str:
    try:
        from secret_manager import get_gemini_api_key

        value = get_gemini_api_key()
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "").strip()


def _safe_path(vault: Path, relative: str) -> Path:
    vault = vault.expanduser().resolve()
    target = (vault / relative).resolve()
    if target != vault and vault not in target.parents:
        raise ValueError("refusing to write outside the Obsidian vault")
    return target


def _yaml_string(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _existing_topic_dates(output_dir: Path) -> dict[str, dt.date]:
    dates: dict[str, dt.date] = {}
    if not output_dir.exists():
        return dates
    for path in output_dir.glob("*.md"):
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:3000]
        except Exception:
            continue
        key_match = re.search(r"^research_topic:\s*[\"']?([^\"'\n]+)", head, re.MULTILINE)
        date_match = re.search(r"^date:\s*(\d{4}-\d{2}-\d{2})", head, re.MULTILINE)
        if not key_match or not date_match:
            continue
        try:
            researched = dt.date.fromisoformat(date_match.group(1))
        except ValueError:
            continue
        key = key_match.group(1).strip()
        if key not in dates or researched > dates[key]:
            dates[key] = researched
    return dates


def choose_topic(output_dir: Path, requested: str = "") -> ResearchTopic:
    if requested:
        normalized = requested.strip().lower()
        for topic in TOPICS:
            if normalized in {topic.key.lower(), topic.title.lower()}:
                return topic
        return ResearchTopic(
            key=re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "custom",
            title=requested.strip(),
            query=f"{requested.strip()} リース審査 判断",
            validity_days=180,
            tags=("リース審査", "個別調査"),
        )
    dates = _existing_topic_dates(output_dir)
    minimum = dt.date.min
    return min(TOPICS, key=lambda item: (dates.get(item.key, minimum), TOPICS.index(item)))


def _extract_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        metadata = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            url = str(getattr(web, "uri", "") or "").strip()
            title = str(getattr(web, "title", "") or url).strip()
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": title, "url": url})
    return sources[:12]


def _source_quality(title: str, url: str) -> str:
    text = f"{title} {url}".lower()
    authoritative = (
        ".go.jp",
        "asb-j.jp",
        "boj.or.jp",
        "jftc.go.jp",
        "jla.or.jp",
        "leasing.or.jp",
        "smrj.go.jp",
        "jfc.go.jp",
        "zeiken.co.jp",
        "meti.go.jp",
        "mlit.go.jp",
        "fsa.go.jp",
        "nta.go.jp",
        "chusho.meti.go.jp",
        "stat.go.jp",
        "e-stat.go.jp",
    )
    recognized = (
        "ntt.com",
        "mufg.jp",
        "smbc.co.jp",
        "mizuho",
        "deloitte",
        "pwc",
        "ey.com",
        "kpmg",
    )
    if any(domain in text for domain in authoritative):
        return "primary"
    if any(domain in text for domain in recognized):
        return "recognized"
    return "supplementary"


_REQUIRED_SECTION_TITLES = (
    "結論",
    "根拠品質",
    "判断に使える確認済み事実",
    "リース審査への適用",
    "担当者が確認する質問",
    "承認条件を変える兆候",
    "反証・過信してはいけない点",
    "更新が必要になる条件",
)


def _normalize_required_headings(body: str) -> str:
    normalized = body.strip()
    for title in _REQUIRED_SECTION_TITLES:
        patterns = (
            rf"^#{{1,6}}\s*{re.escape(title)}\s*$",
            rf"^\*\*{re.escape(title)}\*\*\s*$",
            rf"^{re.escape(title)}\s*$",
        )
        for pattern in patterns:
            normalized = re.sub(pattern, f"## {title}", normalized, flags=re.MULTILINE)
    return normalized


def _required_headings_present(body: str) -> bool:
    return all(f"## {title}" in body for title in _REQUIRED_SECTION_TITLES)


def _fallback_decision_body(topic: ResearchTopic, raw_research: str, sources: list[dict[str, str]]) -> str:
    excerpt = raw_research.strip()
    if len(excerpt) > 2400:
        excerpt = excerpt[:2400].rstrip() + "..."
    source_summary = "\n".join(
        f"- {item.get('title') or item.get('url')} ({item.get('quality', 'supplementary')})"
        for item in sources[:6]
    ) or "- 参照URLなし"
    return f"""## 結論
- Geminiの整形出力が必須見出しを満たさなかったため、このノートは自動フォールバックで保存しています。
- テーマ「{topic.title}」はリース審査の補助知識として扱い、個別案件へ使う前に担当者確認を必須にします。

## 根拠品質
- 参照URLは取得済みですが、本文の再構成品質は要確認です。
- 参照元の一次情報・専門機関・補助情報の区別は、下部の情報源とあわせて担当者が確認してください。
{source_summary}

## 判断に使える確認済み事実
- 以下はGemini検索の調査原文からの抜粋です。事実として採用する前に参照元で再確認してください。

```text
{excerpt or "要確認"}
```

## リース審査への適用
- 審査では、自動否決・自動承認ではなく、確認質問、承認条件、保全条件の検討材料として使います。
- 財務数値、契約条件、物件稼働、保全可能性に影響する点だけを案件判断へ変換します。

## 担当者が確認する質問
- この情報は対象業種・対象設備・対象時期に本当に該当するか。
- 顧客の資金繰り、物件稼働、契約条件、銀行支援のどれに影響するか。
- 参照元が一次情報か、専門機関か、補助情報か。

## 承認条件を変える兆候
- 返済原資、物件価値、稼働率、制度変更、補助金入金時期に直接影響する事実が確認できた場合。
- 補助情報だけでなく、一次情報または専門機関の根拠で裏取りできた場合。

## 反証・過信してはいけない点
- Geminiの要約は誤読や過剰一般化を含む可能性があります。
- ニュースや民間記事だけを根拠に、スコアや承認可否を直接変更しないでください。

## 更新が必要になる条件
- 制度改正、金利環境、業界統計、倒産動向、補助金要件が更新された場合。
- 参照元URLが古くなった、またはより一次情報に近い資料が見つかった場合。"""


def research_topic(topic: ResearchTopic) -> tuple[str, list[dict[str, str]], str]:
    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google import genai
    from google.genai import types

    today = dt.date.today().isoformat()
    search_prompt = f"""
あなたは日本のリース会社の審査企画担当です。次のテーマをWeb検索し、ニュース要約ではなく、
個別案件の審査判断に再利用できる実務知識へ変換してください。

調査日: {today}
テーマ: {topic.title}
検索観点: {topic.query}

要件:
- 官公庁、法令・基準設定主体、公的統計、業界団体、メーカー等の一次情報を優先する。
- 検索語に site:go.jp、site:asb-j.jp、site:boj.or.jp、site:smrj.go.jp 等を活用する。
- 現在も有効か確認し、日付・適用時期・対象範囲を明示する。
- 事実、実務上の推論、未確認事項を混同しない。
- ニュース記事の紹介ではなく「どの案件で、何を確認し、何なら承認条件を変えるか」に落とす。
- 自動否決や単純なスコア減点を提案しない。
- 顧客名や架空の数値を作らない。
- この段階では根拠候補を最大12件に絞り、一次情報と補助情報を区別する。
"""
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_RESEARCH_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    search_response = None
    raw_research = ""
    sources: list[dict[str, str]] = []
    for attempt in range(2):
        attempt_prompt = search_prompt
        if attempt:
            attempt_prompt += (
                "\n前回は参照URLを取得できませんでした。必ずGoogle検索を実行し、"
                "官公庁・公的機関・基準設定主体を最低1件参照してから回答してください。"
            )
        search_response = client.models.generate_content(
            model=model,
            contents=attempt_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4500,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                http_options=types.HttpOptions(timeout=60000),
            ),
        )
        raw_research = str(getattr(search_response, "text", "") or "").strip()
        sources = _extract_sources(search_response)
        if raw_research and sources:
            break
    if not raw_research:
        raise RuntimeError("Gemini research returned no text")
    if not sources:
        raise RuntimeError("Gemini research returned no verifiable source URLs; note was not saved")
    source_catalog = [
        {
            **source,
            "quality": _source_quality(source["title"], source["url"]),
        }
        for source in sources
    ]
    synthesis_prompt = f"""
次のWeb調査を、リース審査で使える短い判断ノートに再構成してください。
一次情報または認知度の高い専門機関で確認できない数値基準は削除してください。
民間ブログだけを根拠にした内容は「未確認」または「推論」と明記してください。
各節は最大5項目、1項目は2文以内にしてください。

テーマ: {topic.title}
調査日: {today}
参照元一覧:
{json.dumps(source_catalog, ensure_ascii=False)}

調査原文:
{raw_research[:18000]}

以下のMarkdown見出しを、この順序で全て出してください。
## 結論
## 根拠品質
## 判断に使える確認済み事実
## リース審査への適用
## 担当者が確認する質問
## 承認条件を変える兆候
## 反証・過信してはいけない点
## 更新が必要になる条件

「根拠品質」には、一次情報の有無、補助情報への依存、担当者が再確認すべき点を書いてください。
"""
    synthesis_response = client.models.generate_content(
        model=model,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=3500,
            http_options=types.HttpOptions(timeout=60000),
        ),
    )
    body = _normalize_required_headings(str(getattr(synthesis_response, "text", "") or ""))
    if not body or not _required_headings_present(body):
        repair_prompt = f"""
次の文章を内容を増やさず、8つの見出しへ再配置してください。
各見出しを一度だけ、完全に `## 見出し名` の形式で出してください。
不足情報は「要確認」と書き、見出しを省略しないでください。

見出し:
{chr(10).join(f"## {title}" for title in _REQUIRED_SECTION_TITLES)}

文章:
{body[:14000] or raw_research[:14000]}
"""
        repair_response = client.models.generate_content(
            model=model,
            contents=repair_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=3500,
                http_options=types.HttpOptions(timeout=60000),
            ),
        )
        body = _normalize_required_headings(str(getattr(repair_response, "text", "") or ""))
    if not body or not _required_headings_present(body):
        body = _fallback_decision_body(topic, raw_research, source_catalog)
    return body, source_catalog, model


def build_note(topic: ResearchTopic, body: str, sources: list[dict[str, str]], model: str) -> str:
    today = dt.date.today()
    valid_until = today + dt.timedelta(days=topic.validity_days)
    tags = json.dumps(["リース審査", "autoresearch", *topic.tags], ensure_ascii=False)
    quality_counts = {
        quality: sum(1 for item in sources if item.get("quality") == quality)
        for quality in ("primary", "recognized", "supplementary")
    }
    source_lines = "\n".join(
        f"- `{item.get('quality', 'supplementary')}` [{item['title']}]({item['url']})"
        for item in sources
    ) or "- Gemini検索結果に参照URLが返されなかったため、本文の事実は要再確認"
    return f"""---
date: {today.isoformat()}
research_topic: {topic.key}
title: {_yaml_string(topic.title)}
tags: {tags}
knowledge_type: lease-judgment-research
source: gemini-google-search
model: {_yaml_string(model)}
valid_until: {valid_until.isoformat()}
review_status: needs_human_review
primary_source_count: {quality_counts["primary"]}
recognized_source_count: {quality_counts["recognized"]}
supplementary_source_count: {quality_counts["supplementary"]}
---
# {topic.title} - リース判断Auto Research

> このノートは審査判断の補助知識です。個別案件の事実確認を省略せず、自動否決・自動承認には使用しません。

{body}

## 情報源
{source_lines}

## 検索用分類
- 調査テーマ: {topic.title}
- 対象: リース審査、信用判断、条件設計
- 有効期限: {valid_until.isoformat()}
- レビュー状態: needs_human_review
"""


def _index_note(path: Path) -> None:
    try:
        from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
        from api.knowledge.vector_store import get_store

        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        chunks = _chunk_by_h2(body, str(path), path.name, meta, path.stat().st_mtime)
        if chunks:
            get_store().upsert_chunks(chunks)
            print(f"[rag] indexed {len(chunks)} chunks")
    except Exception as exc:
        print(f"[rag] index skipped: {exc}", file=sys.stderr)


def run(vault: Path, output_dir: str, requested_topic: str = "", dry_run: bool = False) -> dict[str, Any]:
    target_dir = _safe_path(vault, output_dir)
    topic = choose_topic(target_dir, requested_topic)
    result: dict[str, Any] = {
        "topic": topic.key,
        "title": topic.title,
        "query": topic.query,
        "target_dir": str(target_dir),
    }
    if dry_run:
        return result

    body, sources, model = research_topic(topic)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dt.date.today().isoformat()}_{topic.key}.md"
    path = target_dir / filename
    path.write_text(build_note(topic, body, sources, model), encoding="utf-8")
    _index_note(path)
    result.update({"path": str(path), "source_count": len(sources), "model": model})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-research lease judgment knowledge into Obsidian.")
    parser.add_argument("--vault", default=os.environ.get("OBSIDIAN_VAULT", str(DEFAULT_VAULT)))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", default="", help="Topic key or a custom research theme.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = run(
            Path(args.vault),
            args.output_dir.strip("/") or DEFAULT_OUTPUT_DIR,
            requested_topic=args.topic,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
