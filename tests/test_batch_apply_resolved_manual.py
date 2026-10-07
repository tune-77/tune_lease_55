"""解決済みの manual ルールを「手動対応必要」に数えない（REV-493）。

analyze_pipeline_health が復旧を確認して stale_resolved にした障害検出（REV-399a 等）が、
毎朝「失敗率62%」のまま手動対応一覧に出続け、実際は直っているのに失敗中に見えていた。
"""
from api.rule_engine import batch_apply


def _rules():
    return [
        {
            "rev_id": "REV-399a",
            "type": "manual",
            "status": "stale_resolved",
            "resolved_at": "2026-09-20T19:03:46Z",
            "description": "[パイプライン自動検出] write_daily_brief が過去7日で失敗率62%（5/8件, 5日失敗）",
        },
        {
            "rev_id": "REV-900a",
            "type": "manual",
            "status": "pending_review",
            "description": "[パイプライン自動検出] some_step が過去7日で失敗率50%",
        },
        {"rev_id": "REV-901a", "type": "manual", "description": "status なしの手動ルール"},
    ]


def test_resolved_manual_rules_are_not_listed_as_needing_action(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch_apply, "_LEDGER_PATH", tmp_path / "ledger_rules.json")

    batch_apply.run_batch(_rules(), dry_run=True, rev_filter=None)

    out = capsys.readouterr().out
    assert "REV-399a" not in out
    assert "REV-900a" in out and "REV-901a" in out
    assert "手動対応必要 [manual]        :   2 件" in out
    assert "解決済み manual スキップ      :   1 件" in out
