import datetime as dt
import json

import obsidian_ai_context as oac


def test_collect_obsidian_ai_context_applies_retrieval_budget(monkeypatch):
    hits = [
        {"path": "note-a.md", "snippet": "A" * 1200},
        {"path": "note-b.md", "snippet": "B" * 1200},
        {"path": "note-c.md", "snippet": "C" * 1200},
    ]

    def fake_collect(_query, limit=4):
        return hits[:limit]

    def fake_digest(_query, selected_hits):
        return {"digest": f"selected={len(selected_hits)}"}

    monkeypatch.setattr(oac, "_load_obsidian_bridge", lambda: (fake_collect, fake_digest))

    result = oac.collect_obsidian_ai_context("資金繰り", limit=3, max_tokens=160)

    assert result["source_count"] == 1
    assert result["retrieval_boundary"]["candidate_count"] == 3
    assert result["retrieval_boundary"]["selected_count"] == 1
    assert result["retrieval_boundary"]["token_budget"] == 160
    assert "note-a.md" in result["block"]
    assert "note-b.md" not in result["block"]


def test_build_obsidian_ai_context_block_passes_budget(monkeypatch):
    hits = [
        {"path": "short.md", "snippet": "短いノート"},
    ]

    monkeypatch.setattr(
        oac,
        "_load_obsidian_bridge",
        lambda: (lambda _query, limit=4: hits, lambda _query, _hits: {"digest": "digest"}),
    )

    block = oac.build_obsidian_ai_context_block("期待使用期間", max_tokens=20)

    assert "short.md" in block
    assert "digest" in block


def test_collect_obsidian_ai_context_applies_optional_typesafe_filter(monkeypatch):
    hits = [
        {"path": "first.md", "snippet": "first"},
        {"path": "second.md", "snippet": "second"},
    ]

    monkeypatch.setattr(
        oac,
        "_load_obsidian_bridge",
        lambda: (lambda _query, limit=4: hits[:limit], lambda _query, _hits: {"digest": "digest"}),
    )
    monkeypatch.setattr(
        oac,
        "_load_typesafe_rag_filter",
        lambda _query: lambda _query, _hits: (
            [{**hits[1], "typesafe_route": "include"}],
            {"status": "applied", "accepted_count": 1},
        ),
    )

    result = oac.collect_obsidian_ai_context("query", limit=2)

    assert [hit["path"] for hit in result["hits"]] == ["second.md"]
    assert result["retrieval_boundary"]["typesafe"] == {
        "status": "applied",
        "accepted_count": 1,
    }


def test_shared_context_requires_separate_external_processing_opt_in(monkeypatch):
    monkeypatch.setenv("TYPESAFE_RAG_ENABLED", "1")
    monkeypatch.delenv("TYPESAFE_ALLOW_SHARED_CONTEXT", raising=False)

    assert oac._load_typesafe_rag_filter("一般的なリース知識") is None


def test_recent_worklog_context_filters_private_old_and_unrelated_items(tmp_path):
    report = tmp_path / "worklogs.json"
    report.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "date": "2026-09-28",
                        "time": "10:00",
                        "agent": "Codex",
                        "source_path": "/vault/Daily/2026-09-28.md",
                        "summary": ["公開要約"],
                        "decisions": ["公開判断"],
                        "changes": ["公開変更"],
                        "private": ["これは出してはいけない"],
                        "raw_note": "秘密の原文",
                    },
                    {
                        "date": "2026-08-01",
                        "time": "12:00",
                        "agent": "Codex",
                        "source_path": "/vault/Daily/2026-08-01.md",
                        "summary": ["古い判断"],
                    },
                    {
                        "date": "2026-09-28",
                        "time": "11:00",
                        "agent": "Agent",
                        "source_path": "/vault/Projects/other/notes.md",
                        "summary": ["無関係ノート"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    block = oac.build_recent_worklog_ai_context_block(
        report_path=report,
        today=dt.date(2026, 9, 28),
        days=14,
    )

    assert "公開要約" in block
    assert "公開判断" in block
    assert "これは出してはいけない" not in block
    assert "秘密の原文" not in block
    assert "古い判断" not in block
    assert "無関係ノート" not in block


def test_recent_worklog_context_accepts_project_worklog_folder(tmp_path):
    report = tmp_path / "worklogs.json"
    report.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "date": "2026-09-28",
                        "time": "15:30",
                        "agent": "Agent",
                        "source_path": "/vault/Projects/tune_lease_55/Work Logs/2026-09-28.md",
                        "summary": ["復旧作業"],
                        "decisions": [],
                        "changes": ["安全策を追加"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    block = oac.build_recent_worklog_ai_context_block(
        report_path=report,
        today=dt.date(2026, 9, 28),
    )

    assert "復旧作業" in block
    assert "安全策を追加" in block
