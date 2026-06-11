from __future__ import annotations

import importlib
import json


def test_build_pdca_prompt_block_dedupes_and_limits(tmp_path, monkeypatch):
    rules_path = tmp_path / "pdca_ai_rules.json"
    rules_path.write_text(
        json.dumps(
                {
                    "last_run": "2026-06-12T09:00:00",
                    "analyzed_count": 12,
                    "reflection_summary": "自己資本比率の低い案件で失注が増えている。",
                    "ai_prompt_addons": [
                        "自己資本比率が低い案件は資金繰りを重視する。",
                        "自己資本比率が低い案件は資金繰りを重視する。",
                        "年商に対するリース料比率を確認する。",
                        "競合条件との差を明確にする。",
                        "主取引銀行の支援姿勢を確認する。",
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    import prompt_feedback

    monkeypatch.setattr(prompt_feedback, "DEFAULT_PDCA_RULES_FILE", str(rules_path))
    block = prompt_feedback.build_pdca_prompt_block()

    assert "自動学習システムからの特記事項" in block
    assert "自己資本比率が低い案件は資金繰りを重視する。" in block
    assert block.count("自己資本比率が低い案件は資金繰りを重視する。") == 1
    assert "競合条件との差を明確にする。" in block
    assert "主取引銀行の支援姿勢を確認する。" not in block
    assert "自己資本比率の低い案件で失注が増えている。" in block


def test_lease_advisor_prompt_includes_pdca_block(tmp_path, monkeypatch):
    rules_path = tmp_path / "pdca_ai_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "last_run": "2026-06-12T09:00:00",
                "analyzed_count": 8,
                "reflection_summary": "製造業では追加資料の確認が重要。",
                "ai_prompt_addons": [
                    "製造業は資金繰り表と受注残を確認する。",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    import prompt_feedback
    monkeypatch.setattr(prompt_feedback, "DEFAULT_PDCA_RULES_FILE", str(rules_path))

    lease_advisor_logic = importlib.import_module("lease_advisor_logic")
    importlib.reload(lease_advisor_logic)

    prompt = lease_advisor_logic.build_lease_advisor_prompt(
        "次の打ち手を教えて",
        {"業種": "製造業", "売上高": 100000},
        "general",
    )

    assert "自動学習システムからの特記事項" in prompt
    assert "製造業は資金繰り表と受注残を確認する。" in prompt


def test_record_prompt_feedback_writes_previous_response_diff(tmp_path):
    import prompt_feedback

    log_path = tmp_path / "prompt_feedback_log.jsonl"
    first = prompt_feedback.record_prompt_feedback(
        surface="consultation",
        question="条件付き承認の具体的な方法を教えて",
        base_prompt="base v1",
        final_prompt="base v1\n\nPDCA A",
        response="追加資料を求める。",
        log_path=str(log_path),
    )
    second = prompt_feedback.record_prompt_feedback(
        surface="consultation",
        question="条件付き承認の具体的な方法を教えて",
        base_prompt="base v2",
        final_prompt="base v2\n\nPDCA B",
        response="追加資料と期間短縮を求める。",
        log_path=str(log_path),
    )

    assert first["pdca_applied"] is True
    assert "PDCA A" in first["prompt_diff"]
    assert "response_diff_from_previous" in second
    assert "期間短縮" in second["response_diff_from_previous"]
    saved = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(saved) == 2


def test_summarize_prompt_feedback_builds_metrics():
    from scripts import summarize_prompt_feedback as summarizer

    rows = [
        {
            "surface": "consultation",
            "pdca_applied": True,
            "response_len": 120,
            "prompt_base_len": 100,
            "prompt_final_len": 150,
            "prompt_diff": "--- base\n+++ pdca\n+rule a\n-rule b",
            "response_diff_from_previous": "--- prev\n+++ cur\n+more detail\n-old detail",
            "timestamp": "2026-06-12T10:00:00",
            "question": "Q1",
        },
        {
            "surface": "lease_advisor",
            "pdca_applied": False,
            "response_len": 80,
            "prompt_base_len": 90,
            "prompt_final_len": 90,
            "prompt_diff": "",
            "response_diff_from_previous": "",
            "timestamp": "2026-06-12T11:00:00",
            "question": "Q2",
        },
    ]

    summary = summarizer.build_summary(rows)
    assert summary["total"] == 2
    assert summary["pdca_count"] == 1
    assert summary["by_surface"]["consultation"]["count"] == 1
    assert summary["avg_prompt_base_len"] == 95.0
    md = summarizer.render_markdown(summary, summarizer.DEFAULT_LOG_PATH)
    assert "Prompt Feedback Summary" in md
    assert "Largest Prompt Changes" in md


def test_prompt_feedback_metrics_summary_shape():
    from prompt_feedback_metrics import build_summary

    summary = build_summary([
        {
            "surface": "next_chat_general",
            "pdca_applied": True,
            "response_len": 50,
            "prompt_base_len": 80,
            "prompt_final_len": 120,
            "prompt_diff": "+PDCA",
            "response_diff_from_previous": "",
        }
    ])

    assert summary["total"] == 1
    assert summary["by_surface"]["next_chat_general"]["count"] == 1
    assert summary["avg_prompt_final_len"] == 120.0


def test_append_pdca_rule_dedupes_and_persists(tmp_path):
    import prompt_feedback

    rules_path = tmp_path / "pdca_ai_rules.json"
    first = prompt_feedback.append_pdca_rule(
        "境界案件では条件提示を先に出す。",
        path=str(rules_path),
        source="manual",
    )
    second = prompt_feedback.append_pdca_rule(
        "境界案件では条件提示を先に出す。",
        path=str(rules_path),
        source="manual",
    )

    assert first["ok"] is True
    assert first["appended"] is True
    assert second["appended"] is False
    data = prompt_feedback.load_pdca_rules(str(rules_path))
    assert data["ai_prompt_addons"] == ["境界案件では条件提示を先に出す。"]
    assert data["manual_rule_count"] == 1
