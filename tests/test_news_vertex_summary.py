from __future__ import annotations

from api.knowledge.news_classifier import build_classified_news_summary
from api.knowledge.news_vertex_summary import (
    build_news_trend_prompt,
    build_vertex_assisted_news_trend_summary,
)


def _sample_summary() -> dict:
    return build_classified_news_summary(
        [
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
            },
            {
                "date": "2026-08-24",
                "title": "製造業で省力化投資が増加",
                "summary_lines": ["人手不足対応で自動化設備への投資が増えている。"],
                "usage_memo": "投資回収期間を確認する。",
                "source": "Example",
                "article_url": "https://example.com/manufacturing",
                "file_path": "manufacturing.md",
                "importance": "中",
                "impact_direction": "positive",
                "source_reliability": "medium",
                "text": "製造業 省力化 自動化 設備投資 人手不足",
            },
        ],
        generated_at="2026-08-24T00:00:00+00:00",
    )


def test_build_news_trend_prompt_contains_classified_context():
    summary = _sample_summary()

    prompt = build_news_trend_prompt(summary)

    assert "最近の業界ニュース分類" in prompt
    assert "運輸業" in prompt
    assert "倒産件数" in prompt
    assert "返済能力" in prompt


def test_vertex_trend_summary_falls_back_without_vertex():
    result = build_vertex_assisted_news_trend_summary(_sample_summary(), use_vertex=False)

    assert result["available"] is True
    assert result["source"] == "classified_news_rules"
    assert result["vertex"]["used"] is False
    assert result["trend_lines"]
    assert result["caution_points"]
    assert result["screening_actions"]
    assert result["watch_categories"]
    assert result["daily_focus"]
    assert result["daily_focus"]["title"] in result["overall_summary"]
    assert result["daily_focus"]["quoted_summary"]
    assert result["daily_focus"]["screening_note"]


def test_vertex_trend_summary_uses_vertex_answer_when_available():
    def fake_answer_fn(*args, **kwargs):
        return {
            "used": True,
            "status": "ok",
            "answer_text": "Vertex補助では、燃料費と人手不足は返済余力確認を厚くする論点です。",
            "grounding_score": 0.8,
            "search_results": [
                {
                    "title": "運送業リスクメモ",
                    "uri": "obsidian://risk",
                    "snippet": "燃料費と人手不足は資金繰り確認へつなげる。",
                }
            ],
        }

    result = build_vertex_assisted_news_trend_summary(_sample_summary(), answer_fn=fake_answer_fn)

    assert result["source"] == "vertex_ai_search"
    assert result["trend_title"] == "Vertex 最近ニュース診断"
    assert "Vertex補助所見" in result["overall_summary"]
    assert result["vertex"]["answer_text"].startswith("Vertex補助")
    assert result["vertex"]["used"] is True
    assert result["vertex"]["refs"][0]["title"] == "運送業リスクメモ"
