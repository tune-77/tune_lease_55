from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "lease_news_digest.py"
_SPEC = importlib.util.spec_from_file_location("lease_news_digest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
digest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = digest
_SPEC.loader.exec_module(digest)


def test_write_lease_news_focus_note_creates_project_note_and_daily_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "METRICS_PATH", tmp_path / "lease_news_metrics.json")
    monkeypatch.setattr(digest, "_actions_json_path", lambda date_str: tmp_path / f"lease_news_actions_{date_str}.json")
    monkeypatch.setattr(digest, "_actions_latest_path", lambda: tmp_path / "lease_news_actions_latest.json")
    monkeypatch.setattr(
        "novelist_agent.generate_daily_lease_grumble",
        lambda **_: [
            "今日も稟議書を開いた。",
            "数字は正直だが、営業の説明は長い。",
            "プリンの代わりに追加資料が届いた。",
            "明日も返済予定表と向き合う。",
        ],
    )
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)

    note = news_dir / "2026-06-11_業界リスクニュース_建設_AI導入.md"
    note.write_text(
        """---
date: 2026-06-11
week: 2026-W24
month: 2026-06
tags: ["建設/不動産", "製造/DX"]
region: 国内
source: Example News
importance: 中
valid_until: 2099-12-31
---
# 建設会社がAI導入で事務作業を効率化

## 3行要約
- 建設会社がAIを導入し、事務作業を削減する。
- 省力化投資の効果が見えやすくなる。
- 補助金の適用余地も検討されている。

## 活用メモ
審査では省力化投資と現場稼働への影響を確認する。
""",
        encoding="utf-8",
    )

    focus = digest.get_latest_lease_news_focus(vault)
    assert focus.available
    assert focus.theme_summary == "国内 / 中"
    assert any("リース期間" in line for line in focus.focus_lines)

    result = digest.write_lease_news_focus_note(date_str="2026-06-11", vault=vault, focus=focus)
    assert result is not None
    assert Path(result.note_path).exists()

    focus_note = vault / "Projects" / "tune_lease_55" / "News" / "2026-06-11_industry-risk-news-focus.md"
    assert focus_note.exists()
    focus_text = focus_note.read_text(encoding="utf-8")
    assert "## 注目論点" in focus_text
    assert "リース期間・中古価値・再リース余地を確認する。" in focus_text

    daily_note = vault / "Daily" / "2026-06-11.md"
    assert daily_note.exists()
    daily_text = daily_note.read_text(encoding="utf-8")
    assert "##" in daily_text
    assert "業界リスクニュースの注目論点" in daily_text

    reflection = digest.write_lease_news_reflection_note(date_str="2026-06-11", vault=vault, focus=focus)
    assert reflection is not None
    reflection_note = vault / "Projects" / "tune_lease_55" / "News" / "2026-06-11_industry-risk-news-reflection.md"
    assert reflection_note.exists()
    reflection_text = reflection_note.read_text(encoding="utf-8")
    assert "## 今日の外界へのぼやき" in reflection_text
    assert "判断前提が増えた" in reflection_text
    assert "## 今日の考え" in reflection_text
    assert "明日見ること" in reflection_text

    parsed = digest.get_latest_lease_news_reflection(vault)
    assert parsed.available
    assert parsed.note_date == "2026-06-11"
    assert parsed.note_path == "Projects/tune_lease_55/News/2026-06-11_industry-risk-news-reflection.md"
    assert parsed.thought_lines
    assert 3 <= len(parsed.thought_lines) <= 4
    assert parsed.tomorrow_lines
    assert parsed.illustration_url == "/lease-grumble/2026-06-11.webp"
    assert parsed.continuity_days == 0
    assert "機械意識" not in parsed.self_narrative
    assert parsed.observed_days == 0

    actions = digest.write_lease_news_actions_note(date_str="2026-06-11", vault=vault)
    assert actions is not None
    assert actions.action_items
    action = actions.action_items[0]
    assert action.felt_signal
    assert action.judgment_tension
    assert "補助金" in action.felt_signal or "省力化" in action.felt_signal

    actions_note = vault / "Projects" / "tune_lease_55" / "News" / "2026-06-11_industry-risk-news-actions.md"
    assert actions_note.exists()
    actions_text = actions_note.read_text(encoding="utf-8")
    assert "紫苑が引っかかったこと" in actions_text
    assert "審査で気持ち悪い点" in actions_text

    prompt_text = digest.lease_news_actions_as_text(vault=vault, industry="建設", asset_name="AI設備")
    assert "引っかかり:" in prompt_text
    assert "気持ち悪い点:" in prompt_text


def test_structured_news_fields_survive_action_generation_and_gate_by_case(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    note = news_dir / "2026-09-26_業界リスクニュース_建設倒産.md"
    note.write_text(
        """---
date: 2026-09-26
tags: ["建設/不動産"]
region: 国内
source: Example News
importance: 高
industries: "建設業, 不動産業"
lease_assets: "建設機械, 建物附属設備"
impact_direction: negative
source_reliability: medium
classification_confidence: 0.45
valid_until: 2027-03-24
classification_source: rule
---
# 建設業の倒産増加

## 3行要約
- 売上不振と人手不足が重なっている。

## 活用メモ
工期と資金繰りへの波及を確認する。

## AI審査分類
- 対象業種: 建設業, 不動産業

### 審査上の確認事項
- 価格転嫁、粗利、外注費、支払サイトを確認する。
- 工期遅延が返済原資へ波及していないか確認する。
""",
        encoding="utf-8",
    )

    parsed = digest._parse_news_note(note)
    assert parsed["industries"] == ["建設業", "不動産業"]
    assert parsed["lease_assets"] == ["建設機械", "建物附属設備"]
    assert parsed["source_reliability"] == "medium"
    assert parsed["classification_confidence"] == 0.45
    assert parsed["screening_checks"][0].startswith("価格転嫁")

    action = digest._infer_news_action(parsed)
    assert action.affected_industries == ("建設業", "不動産業")
    assert action.affected_assets == ("建設機械", "建物附属設備")
    assert action.recommended_checks[0].startswith("価格転嫁")
    assert all("補助金" not in text for text in action.recommended_checks + action.condition_impacts)
    assert action.source_reliability == "medium"
    assert action.classification_confidence == 0.45

    recorded: list[dict] = []
    monkeypatch.setattr(digest, "record_lease_news_action_use", lambda *args, **kwargs: recorded.append(kwargs) or {})
    assert digest.lease_news_actions_as_text(vault=vault) == ""
    assert digest.lease_news_actions_as_text(vault=vault, industry="医療") == ""
    matched = digest.lease_news_actions_as_text(vault=vault, industry="建設", asset_name="建設機械")
    assert "価格転嫁" in matched
    assert "補助金" not in matched
    message_matched = digest.lease_news_actions_as_text(
        vault=vault,
        risk_context="最近、建設業の倒産が増えているけど、この会社は大丈夫？",
    )
    assert "価格転嫁" in message_matched
    assert len(recorded) == 2
    assert all(item["matched_count"] == 1 for item in recorded)

    feedback_path = tmp_path / "news-feedback.jsonl"
    feedback_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "source_path": "05-クリップ_記事/業界リスクニュース/2026-09-26_業界リスクニュース_建設倒産.md",
                    "outcome": "irrelevant",
                },
                ensure_ascii=False,
            )
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(digest, "NEWS_USAGE_FEEDBACK_JSONL", feedback_path)
    assert digest.lease_news_actions_as_text(vault=vault, industry="建設") == ""


def test_lease_news_actions_treats_all_industries_as_wildcard(monkeypatch):
    action = digest.LeaseNewsAction(
        signal="金利上昇",
        source_title="金融環境の変化",
        source_path="news/macro.md",
        affected_industries=("全業種",),
        recommended_checks=("返済余力を確認する",),
        confidence=0.8,
        source_reliability="high",
    )
    actions = digest.LeaseNewsActions(available=True, date="2026-09-27", action_items=(action,))
    monkeypatch.setattr(digest, "get_latest_lease_news_actions", lambda **_kwargs: actions)
    monkeypatch.setattr(digest, "_load_news_usage_feedback_scores", lambda: {})
    monkeypatch.setattr(digest, "record_lease_news_action_use", lambda *_args, **_kwargs: {})

    assert "返済余力" in digest.lease_news_actions_as_text(industry="製造業")


def test_low_gemini_classification_confidence_is_not_promoted():
    action = digest._infer_news_action(
        {
            "title": "設備投資ニュース",
            "industries": ["製造業"],
            "lease_assets": ["生産設備"],
            "screening_checks": ["受注状況を確認する"],
            "classification_confidence": 0.1,
            "classification_source": "gemini",
            "source_reliability": "medium",
        }
    )

    assert action.classification_confidence == 0.1
    assert action.confidence < 0.55


def test_feedback_scores_merge_and_deduplicate_durable_events(tmp_path, monkeypatch):
    local_path = tmp_path / "feedback.jsonl"
    local_path.write_text(
        json.dumps({"event_id": "same", "source_path": "news/a.md", "outcome": "used"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(digest, "NEWS_USAGE_FEEDBACK_JSONL", local_path)

    from api import cloudrun_writeback

    monkeypatch.setattr(
        cloudrun_writeback,
        "read_lease_news_usage_feedback_events",
        lambda limit=2000: [
            {"event_id": "same", "source_path": "news/a.md", "outcome": "used"},
            {"event_id": "remote", "source_path": "news/a.md", "outcome": "question_changed"},
        ],
    )

    assert digest._load_news_usage_feedback_scores() == {"news/a.md": 3}


def test_record_lease_news_usage_feedback_writes_outcome_metrics(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback.jsonl"
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(digest, "METRICS_PATH", metrics_path)

    result = digest.record_lease_news_usage_feedback(
        source_path="news/example.md",
        source_title="設備投資ニュース",
        outcome="question_changed",
        surface="news_dashboard",
        path=feedback_path,
    )

    assert result["event_id"].startswith("nf-")
    assert result["outcome"] == "question_changed"
    assert '"outcome": "question_changed"' in feedback_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bucket = metrics["days"][dt.date.today().isoformat()]
    assert bucket["feedback_total"] == 1
    assert bucket["feedback_question_changed"] == 1
    assert bucket["news_actions_used"] == 1
