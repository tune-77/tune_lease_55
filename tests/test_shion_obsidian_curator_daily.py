from __future__ import annotations

from scripts import shion_obsidian_curator_daily as daily


def _health(status: str, **summary_overrides):
    summary = {
        "graph_notes": None,
        "graph_edges": None,
        "graph_buckets": {},
        "retrieval_notes": None,
        "retrieval_edges": None,
        "degree0_count": 0,
        "top_degree0_dirs": {},
    }
    summary.update(summary_overrides)
    return {
        "mode": "read_only_obsidian_curator",
        "vault": "/tmp/vault",
        "status": status,
        "summary": summary,
        "monitor_checks": {},
        "proposals": [],
        "guardrail": "read_only_no_vault_write_no_chroma_reindex_no_prompt_change",
    }


def test_main_exits_nonzero_when_source_reports_missing(monkeypatch, tmp_path, capsys):
    # graph_effect / retrieval_graph の両方が読めないと review_obsidian_vault_health は
    # status="missing_reports" を返す。これを無視すると上流レポート生成の破損が
    # 「top_actions=0」の静かな成功として見過ごされる。
    monkeypatch.setattr(daily, "review_obsidian_vault_health", lambda limit: _health("missing_reports"))
    monkeypatch.setattr(
        daily,
        "suggest_obsidian_curation_actions",
        lambda theme, limit: {"theme": theme, "status": "missing_retrieval_graph", "suggestions": []},
    )

    exit_code = daily.main(
        [
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "警告" in captured.err


def test_main_exits_zero_when_healthy_with_no_actions(monkeypatch, tmp_path):
    # レポートは正常に読めていて、単に今日は提案すべきアクションが0件なだけの
    # 場合は誤検知せず exit 0 のままであること。
    monkeypatch.setattr(daily, "review_obsidian_vault_health", lambda limit: _health("ok"))
    monkeypatch.setattr(
        daily,
        "suggest_obsidian_curation_actions",
        lambda theme, limit: {"theme": theme, "status": "ok", "suggestions": []},
    )

    exit_code = daily.main(
        [
            "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"),
        ]
    )

    assert exit_code == 0
