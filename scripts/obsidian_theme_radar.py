#!/usr/bin/env python3
"""Find recurring vault themes, check which already have real-world buzz, and
propose one content angle where a recurring interest overlaps with buzz.

Recurring interest = top 3 vault themes by weighted tag + keyword frequency.
Buzz = Vertex AI Google Search grounding (api.vertex_agent_search) returns
multiple distinct web sources for that theme. When one of the top 3 themes
also shows buzz, Vertex text generation proposes a single content angle built
on that theme + evidence.

Read-only: writes nothing to Obsidian. Every Vertex call failure degrades to
an "unavailable" status for that step rather than raising.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._obsidian_common import (  # noqa: E402
    iter_vault_markdown_files,
    note_tags,
    parse_note_frontmatter,
    strip_frontmatter,
)
from scripts._vertex_text_gen import generate_text  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "obsidian_theme_radar_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "obsidian_theme_radar_latest.md"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[一-龥ぁ-んァ-ン]{2,}")
STOPWORDS = {
    "こと", "ため", "よう", "する", "した", "ある", "ない", "これ", "それ",
    "メモ", "今日", "今回", "自分", "内容", "とても", "the", "and", "for",
    "with", "from", "that", "this",
}
TAG_WEIGHT = 3
MIN_BUZZ_SOURCES = 2


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _keyword_counts(text: str) -> Counter[str]:
    tokens = TOKEN_RE.findall(text)
    return Counter(token for token in tokens if token not in STOPWORDS)


def compute_theme_frequency(vault: Path) -> Counter[str]:
    """Weighted frequency: explicit tags count TAG_WEIGHT-x, free-text keywords count 1x.

    Merges labels by NFKC-normalized lowercase form so 'AI' and 'ai' aggregate,
    keeping the most common original-casing spelling as the display label.
    """
    normalized_scores: Counter[str] = Counter()
    display_labels: dict[str, Counter[str]] = {}

    def _add(label: str, weight: int) -> None:
        normalized = unicodedata.normalize("NFKC", label).strip().lower()
        if not normalized:
            return
        normalized_scores[normalized] += weight
        display_labels.setdefault(normalized, Counter())[label] += 1

    for path in iter_vault_markdown_files(vault):
        text = _read_text(path)
        if not text:
            continue
        frontmatter = parse_note_frontmatter(text)
        for tag in note_tags(frontmatter):
            _add(tag, TAG_WEIGHT)
        for token, count in _keyword_counts(strip_frontmatter(text)).items():
            _add(token, count)

    result: Counter[str] = Counter()
    for normalized, score in normalized_scores.items():
        label = display_labels[normalized].most_common(1)[0][0]
        result[label] = score
    return result


def top_recurring_themes(vault: Path, limit: int = 3) -> list[dict[str, Any]]:
    freq = compute_theme_frequency(vault)
    return [{"theme": theme, "score": score} for theme, score in freq.most_common(limit)]


def check_theme_buzz(theme: str, *, search_fn=None) -> dict[str, Any]:
    if search_fn is None:
        from api.vertex_agent_search import google_search_grounding as search_fn  # noqa: PLC0415

    result = search_fn(f"{theme} 話題 反響 SNS まとめ")
    if not result.get("used") or result.get("status") != "ok":
        return {"theme": theme, "status": result.get("status", "unavailable"), "buzzing": False, "sources": []}

    sources = result.get("sources") or []
    buzzing = len(sources) >= MIN_BUZZ_SOURCES
    return {
        "theme": theme,
        "status": "ok",
        "buzzing": buzzing,
        "source_count": len(sources),
        "sources": sources[:5],
        "summary": result.get("text", ""),
    }


def propose_angle(theme: str, buzz: dict[str, Any], *, generate_fn=generate_text) -> dict[str, Any]:
    summary = str(buzz.get("summary") or "")
    source_titles = ", ".join(str(s.get("title") or "") for s in buzz.get("sources") or [] if s.get("title"))
    prompt = (
        "あなたはコンテンツ企画のアシスタントです。\n"
        f"ユーザーが繰り返し興味を示しているテーマ「{theme}」は、実際に世の中でも反応が大きいテーマです。\n"
        f"世の中の反応の要約: {summary[:800] or '(要約なし)'}\n"
        f"参考にされている記事タイトル: {source_titles or '(なし)'}\n\n"
        "このテーマとユーザー自身の関心を軸にした、独自性のあるコンテンツの切り口を1つだけ、"
        "1〜2文の日本語で簡潔に提案してください。説明文や前置きは不要です。"
    )
    result = generate_fn(prompt)
    if not result.get("used"):
        return {"angle": "", "status": result.get("status", "unavailable")}
    return {"angle": result.get("text", "").strip(), "status": "ok"}


def build_theme_radar(vault: Path, *, search_fn=None, generate_fn=generate_text) -> dict[str, Any]:
    themes = top_recurring_themes(vault, limit=3)
    buzz_checks = [check_theme_buzz(item["theme"], search_fn=search_fn) for item in themes]
    overlap = next((buzz for buzz in buzz_checks if buzz.get("buzzing")), None)

    angle: dict[str, Any] = {"angle": "", "status": "no_overlap"}
    if overlap:
        angle = propose_angle(overlap["theme"], overlap, generate_fn=generate_fn)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "recurring_themes": themes,
        "buzz_checks": buzz_checks,
        "overlap_theme": overlap["theme"] if overlap else "",
        "proposed_angle": angle,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Theme Radar",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Recurring Themes (top 3)",
    ]
    for item in report["recurring_themes"]:
        lines.append(f"- {item['theme']} (score={item['score']})")
    lines += ["", "## Buzz Check"]
    for buzz in report["buzz_checks"]:
        lines.append(f"### {buzz['theme']}")
        lines.append(f"- status: `{buzz['status']}`")
        lines.append(f"- buzzing: {buzz.get('buzzing')}")
        for source in buzz.get("sources") or []:
            lines.append(f"  - {source.get('title')}: {source.get('uri')}")
    lines += [
        "",
        "## Overlap Theme",
        f"- {report['overlap_theme'] or 'なし'}",
        "",
        "## Proposed Angle",
        f"- {report['proposed_angle'].get('angle') or '(提案なし: ' + report['proposed_angle'].get('status', '') + ')'}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else resolve_obsidian_vault()
    if not vault.exists():
        print(f"[ERROR] Vault not found: {vault}")
        return 1

    report = build_theme_radar(vault)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
