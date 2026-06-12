import lease_intelligence_knowledge as lik
from lease_intelligence_knowledge import build_lease_intelligence_knowledge


def _reset_count_cache(monkeypatch):
    monkeypatch.setattr(lik, "_count_cache", {"at": 0.0, "counts": None})


def test_knowledge_access_uses_shared_obsidian_context(monkeypatch):
    _reset_count_cache(monkeypatch)
    documents = [
        {"path": "リース知識/残価.md", "source_type": "knowledge"},
        {"path": "Daily/2026-06-13.md", "source_type": "chat_log"},
    ]
    monkeypatch.setattr(
        "mobile_app.obsidian_bridge.iter_indexed_obsidian_documents",
        lambda **kwargs: documents,
    )
    monkeypatch.setattr(
        "obsidian_ai_context.collect_obsidian_ai_context",
        lambda query, **kwargs: {
            "block": "【リース知性体が参照したObsidian知識】\n残価は中古市場を確認する。",
            "hits": [{"path": "リース知識/残価.md", "snippet": "残価"}],
            "source_count": 1,
        },
    )

    knowledge = build_lease_intelligence_knowledge(
        theme="設備投資",
        focus_lines=["残価と中古売却を確認する。"],
        user_interests=[{"label": "車・移動"}],
    )

    assert knowledge.available is True
    assert knowledge.indexed_notes == 2
    assert knowledge.knowledge_notes == 1
    assert knowledge.chat_log_notes == 1
    assert knowledge.source_paths == ("リース知識/残価.md",)
    assert "残価" in knowledge.query
    assert "Obsidian知識" in knowledge.context_block


def test_document_counts_are_cached_within_ttl(monkeypatch):
    _reset_count_cache(monkeypatch)
    calls = {"count": 0}

    def fake_loader():
        calls["count"] += 1
        return {"indexed_notes": 5, "knowledge_notes": 4, "chat_log_notes": 1}

    monkeypatch.setattr(lik, "_load_document_counts", fake_loader)

    first = lik._document_counts(now=1000.0)
    second = lik._document_counts(now=1000.0 + lik._COUNT_TTL_SECONDS - 1)
    assert first == second
    assert calls["count"] == 1, "TTL内は全件列挙を繰り返さない"

    lik._document_counts(now=1000.0 + lik._COUNT_TTL_SECONDS + 1)
    assert calls["count"] == 2, "TTLを過ぎたら数え直す"
