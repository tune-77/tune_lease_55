from __future__ import annotations

from pathlib import Path

from api.knowledge.news_classifier import (
    build_classified_news_summary,
    build_classified_news_summary_from_vault,
    classify_note_axes,
)


def test_classify_note_axes_matches_required_news_dimensions():
    note = {
        "title": "建設業で資材価格と人手不足が続き、日銀金利上昇も負担",
        "summary_lines": ["建設資材の価格高騰と人手不足が工期に影響する。"],
        "usage_memo": "返済余力と工期遅延を確認する。",
        "tags": ["建設/不動産", "金利"],
        "industries": ["建設業"],
        "text": "建設 資材 価格高騰 人手不足 日銀 金利",
    }

    axes = classify_note_axes(note)

    assert "建設業" in axes["industry"]
    assert "物価高" in axes["social"]
    assert "人手不足" in axes["social"]
    assert "金利動向" in axes["finance"]


def test_build_classified_news_summary_generates_lease_implications():
    notes = [
        {
            "date": "2026-08-24",
            "title": "運送業の倒産件数が増加",
            "summary_lines": ["燃料費と人手不足で運送業の倒産が増えている。"],
            "usage_memo": "資金繰りを確認する。",
            "source": "Example",
            "article_url": "https://example.com/news",
            "file_path": "news.md",
            "importance": "高",
            "impact_direction": "negative",
            "source_reliability": "medium",
            "text": "運送 物流 トラック 燃料費 人手不足 倒産 件数 資金繰り",
        }
    ]

    summary = build_classified_news_summary(notes, generated_at="2026-08-24T00:00:00+00:00")

    assert summary["available"] is True
    assert summary["article_count"] == 1
    finance = next(axis for axis in summary["axes"] if axis["axis"] == "finance")
    bankruptcy = next(category for category in finance["categories"] if category["category"] == "倒産件数")
    assert bankruptcy["lease_implications"]["direction"] == "negative"
    assert "返済能力" not in bankruptcy["trend"]
    assert bankruptcy["recommended_checks"]
    assert bankruptcy["articles"][0]["article_url"] == "https://example.com/news"


def test_build_classified_news_summary_from_vault_reads_news_notes(tmp_path: Path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    (news_dir / "2026-08-24_news.md").write_text(
        """---
date: 2026-08-24
tags: ["製造/DX", "設備投資"]
industries: "製造業"
lease_assets: "生産設備"
impact_direction: positive
source: Example
importance: 中
canonical_url: "https://example.com/manufacturing"
---
# 製造業が省力化設備を導入

## 3行要約
- 製造業で省力化設備の投資が増える。
- 人手不足への対応として自動化が進む。
- 補助金も活用されている。

## 活用メモ
投資回収期間と生産性改善効果を確認する。
""",
        encoding="utf-8",
    )

    summary = build_classified_news_summary_from_vault(vault, limit=5, days=60)

    assert summary["available"] is True
    industry = next(axis for axis in summary["axes"] if axis["axis"] == "industry")
    assert industry["categories"][0]["category"] == "製造業"
    assert industry["categories"][0]["articles"][0]["file_path"].endswith("2026-08-24_news.md")
