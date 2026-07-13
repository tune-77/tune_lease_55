from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

from scripts import auto_research_lease_judgment as research


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_autoresearch_judgment_asset_candidates.py"
_SPEC = importlib.util.spec_from_file_location("build_autoresearch_judgment_asset_candidates", _SCRIPT)
assert _SPEC and _SPEC.loader
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)


def _note(body: str) -> str:
    return f"""---
date: 2026-07-13
research_topic: industry-risk
title: "業種別の倒産要因と先行指標"
review_status: needs_human_review
---
# 業種別の倒産要因と先行指標 - リース判断Auto Research

{body}

## 情報源
- `primary` [Source](https://example.go.jp/source)
"""


def _substantive_body() -> str:
    values = {
        "結論": "- 倒産要因は業種ごとに違うため、資金繰りと稼働率を分けて確認する。",
        "根拠品質": "- 公的統計と業界団体資料を併用し、民間記事は補助情報として扱う。",
        "判断に使える確認済み事実": "- 人件費と仕入価格の上昇は、小口先の利益率を圧迫しやすい。",
        "リース審査への適用": "- 売上増加だけでなく、粗利率と固定費吸収後の返済余力を見る。",
        "担当者が確認する質問": "- 直近3か月の仕入価格と販売価格への転嫁状況を確認する。",
        "承認条件を変える兆候": "- 価格転嫁できず短期借入が増えている場合は、追加保全を検討する。",
        "反証・過信してはいけない点": "- 業界全体の悪化だけで個社の稼働状況を決めつけない。",
        "更新が必要になる条件": "- 倒産統計、物価、金利、補助金要件が更新された場合に見直す。",
    }
    return "\n\n".join(f"## {title}\n{text}" for title, text in values.items())


def test_extract_candidates_rejects_youkakunin_only_note(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    target = vault / research.DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True)
    weak = "\n\n".join(f"## {title}\n要確認" for title in research._REQUIRED_SECTION_TITLES)
    (target / "2026-07-13_industry-risk.md").write_text(_note(weak), encoding="utf-8")

    candidates = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
    )

    assert candidates == []


def test_extract_candidates_marks_autoresearch_items_as_review_only(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    target = vault / research.DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True)
    (target / "2026-07-13_industry-risk.md").write_text(_note(_substantive_body()), encoding="utf-8")

    candidates = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
    )

    assert {item["candidate_type"] for item in candidates} == {
        "application_rule",
        "confirmation_question",
        "condition_signal",
        "caution",
    }
    assert all(item["review_status"] == "candidate" for item in candidates)
    assert all(item["promotion_status"] == "not_promoted" for item in candidates)
    assert all(item["use_count"] == 0 for item in candidates)
    assert all(item["useful_count"] == 0 for item in candidates)
    assert all(item["rejected_count"] == 0 for item in candidates)
    assert all(item["verified_status"] == "unverified" for item in candidates)
    assert all(item["requires_human_use_feedback"] for item in candidates)
    assert all(item["requires_result_verification"] for item in candidates)


def test_extract_candidates_merges_feedback_state(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    target = vault / research.DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True)
    (target / "2026-07-13_industry-risk.md").write_text(_note(_substantive_body()), encoding="utf-8")

    first_pass = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
    )
    candidate_id = next(item["id"] for item in first_pass if item["candidate_type"] == "confirmation_question")
    state_path = tmp_path / "candidate_state.json"
    state_path.write_text(
        f"""{{
  "{candidate_id}": {{
    "use_count": 2,
    "useful_count": 1,
    "rejected_count": 0,
    "neutral_count": 1,
    "last_used_at": "2026-07-13T09:00:00",
    "last_feedback_at": "2026-07-13T09:05:00",
    "verified_status": "supported",
    "verification_note": "案件結果でも確認観点が妥当だった",
    "edited_claim": "直近3か月の仕入価格と価格転嫁を月次で確認する。",
    "edit_count": 1,
    "last_edited_at": "2026-07-13T09:04:00"
  }}
}}
""",
        encoding="utf-8",
    )

    candidates = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
        state_path=state_path,
    )

    updated = next(item for item in candidates if item["id"] == candidate_id)
    assert updated["use_count"] == 2
    assert updated["useful_count"] == 1
    assert updated["neutral_count"] == 1
    assert updated["verified_status"] == "supported"
    assert updated["promotion_status"] == "ready_for_promotion"
    assert updated["asset_quality"] == "actionable"
    assert updated["edited_claim"] == "直近3か月の仕入価格と価格転嫁を月次で確認する。"
    assert updated["edit_count"] == 1


def test_textbook_general_candidate_cannot_be_promoted_even_with_feedback(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    target = vault / research.DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True)
    textbook_claim = "財務内容を確認し、返済原資を確認する。"
    body = _substantive_body().replace(
        "- 直近3か月の仕入価格と販売価格への転嫁状況を確認する。",
        f"- {textbook_claim}",
    )
    (target / "2026-07-13_industry-risk.md").write_text(_note(body), encoding="utf-8")

    first_pass = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
    )
    textbook = next(item for item in first_pass if item["claim"] == textbook_claim)
    assert textbook["asset_quality"] == "textbook_general"
    assert textbook["promotion_status"] == "not_promoted_textbook_general"

    state_path = tmp_path / "candidate_state.json"
    state_path.write_text(
        f"""{{
  "{textbook['id']}": {{
    "use_count": 3,
    "useful_count": 2,
    "rejected_count": 0,
    "neutral_count": 1,
    "last_used_at": "2026-07-13T09:00:00",
    "last_feedback_at": "2026-07-13T09:05:00",
    "verified_status": "supported",
    "verification_note": "人間は良いと言ったが一般論"
  }}
}}
""",
        encoding="utf-8",
    )

    candidates = builder.extract_candidates(
        vault=vault,
        output_dir=research.DEFAULT_OUTPUT_DIR,
        end_date=dt.date(2026, 7, 13),
        days=1,
        state_path=state_path,
    )

    updated = next(item for item in candidates if item["id"] == textbook["id"])
    assert updated["useful_count"] == 2
    assert updated["verified_status"] == "supported"
    assert updated["asset_quality"] == "textbook_general"
    assert updated["promotion_status"] == "not_promoted_textbook_general"
    assert "textbook_general_marker" in updated["quality_reasons"]


def test_dedupe_similar_candidates_keeps_one_representative():
    base = {
        "candidate_type": "confirmation_question",
        "research_topic": "industry-risk",
        "research_title": "業種別の倒産要因と先行指標",
        "research_date": "2026-07-13",
        "source_section": "担当者が確認する質問",
        "review_status": "candidate",
        "promotion_status": "not_promoted",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "verified_status": "unverified",
    }
    candidates = [
        {
            **base,
            "id": "a",
            "claim": "直近3か月の仕入価格と販売価格への転嫁状況を確認する。",
            "evidence_path": "note-a.md",
        },
        {
            **base,
            "id": "b",
            "claim": "直近3か月の仕入価格と販売価格への転嫁状況を確認します。",
            "evidence_path": "note-b.md",
        },
    ]

    deduped = builder.dedupe_similar_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0]["deduped_count"] == 1
    assert deduped[0]["similar_candidates"][0]["id"] in {"a", "b"}
    assert set(deduped[0]["evidence_paths"]) == {"note-a.md", "note-b.md"}


def test_dedupe_prefers_edited_candidate_as_representative():
    base = {
        "candidate_type": "confirmation_question",
        "research_topic": "industry-risk",
        "research_title": "業種別の倒産要因と先行指標",
        "research_date": "2026-07-13",
        "source_section": "担当者が確認する質問",
        "review_status": "candidate",
        "promotion_status": "not_promoted",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "verified_status": "unverified",
    }
    candidates = [
        {
            **base,
            "id": "a",
            "claim": "直近3か月の仕入価格と販売価格への転嫁状況を確認する。",
            "evidence_path": "note-a.md",
            "edit_count": 0,
        },
        {
            **base,
            "id": "b",
            "claim": "直近3か月の仕入価格と販売価格への転嫁状況を確認します。",
            "edited_claim": "直近3か月の仕入価格と価格転嫁を月次で確認する。",
            "evidence_path": "note-b.md",
            "edit_count": 1,
        },
    ]

    deduped = builder.dedupe_similar_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0]["id"] == "b"
    assert deduped[0]["edit_count"] == 1


def test_write_report_describes_promotion_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "REPORTS_DIR", tmp_path / "reports")
    candidates = [
        {
            "candidate_type": "confirmation_question",
            "research_date": "2026-07-13",
            "research_topic": "industry-risk",
            "claim": "価格転嫁状況を確認する。",
            "source_section": "担当者が確認する質問",
            "evidence_path": "Projects/tune_lease_55/Research/Auto Research/2026-07-13_industry-risk.md",
            "review_status": "candidate",
            "promotion_status": "not_promoted",
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "neutral_count": 0,
            "verified_status": "unverified",
            "deduped_count": 2,
            "similar_candidates": [],
        }
    ]

    paths = builder.write_report(
        candidates,
        end_date=dt.date(2026, 7, 13),
        days=1,
        output_jsonl=tmp_path / "candidates.jsonl",
    )

    md = Path(paths["latest_markdown"]).read_text(encoding="utf-8")
    assert "Auto Research is material, not memory" in md
    assert "当たり前なこと言ってやった気になるな" in md
    assert "Edited candidates are prioritized" in md
    assert "not_promoted" in md
    assert "Asset quality:" in md
    assert "Metrics: use=0, useful=0, rejected=0, neutral=0, verified=unverified" in md
    assert "Deduped similar candidates: 2" in md
    assert "Deduped similar: 2" in md
