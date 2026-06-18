from __future__ import annotations

import json


def test_operational_trust_summary_flags_review_items_without_raw_questions(tmp_path):
    from operational_trust import build_operational_trust_summary

    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "case_memory_usage_log.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2099-01-01T10:00:00",
                "surface": "next_chat_rag",
                "question_hash": "abcdef1234567890",
                "question_preview": "株式会社テストの質問本文",
                "knowledge_refs": ["Knowledge/A.md", "Knowledge/B.md"],
                "pdca_applied": True,
                "judgment_learning_used": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "pdca_ai_rules.json").write_text(
        json.dumps(
            {
                "ai_prompt_addons": ["期限切れルール"],
                "pdca_rule_meta": [
                    {
                        "rule": "期限切れルール",
                        "source": "manual",
                        "expires_at": "2000-01-01",
                        "status": "active",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vault = tmp_path / "vault"
    corrections = vault / "Projects/tune_lease_55/Lease Intelligence/Knowledge Corrections"
    corrections.mkdir(parents=True)
    (corrections / "asset_2099-01-01.md").write_text(
        "---\nstatus: needs_review\n---\n# correction\n",
        encoding="utf-8",
    )

    summary = build_operational_trust_summary(repo, vault=vault)

    assert summary["status"] == "attention"
    assert "knowledge_corrections_need_review" in summary["attention"]
    assert "pdca_rules_expired" in summary["attention"]
    assert summary["memory_usage"]["recent_total"] == 1
    assert summary["memory_usage"]["recent_items"][0]["question_hash"] == "abcdef123456"
    assert "question_preview" not in summary["memory_usage"]["recent_items"][0]
    assert summary["pdca_rules"]["expired"] == 1
    assert summary["knowledge_corrections"]["needs_review"] == 1


def test_operational_trust_summary_ok_when_clean(tmp_path):
    from operational_trust import build_operational_trust_summary

    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "case_memory_usage_log.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2099-01-01T10:00:00",
                "surface": "score_full",
                "question_hash": "hash",
                "knowledge_refs": ["scoring_core"],
                "pdca_applied": False,
                "judgment_learning_used": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "pdca_ai_rules.json").write_text(
        json.dumps(
            {
                "ai_prompt_addons": ["有効ルール"],
                "pdca_rule_meta": [
                    {
                        "rule": "有効ルール",
                        "source": "manual",
                        "expires_at": "2099-12-31",
                        "status": "active",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = build_operational_trust_summary(repo, vault=None)

    assert summary["status"] == "ok"
    assert summary["attention"] == []
    assert summary["pdca_rules"]["active"] == 1
