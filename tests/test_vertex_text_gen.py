from scripts._vertex_text_gen import generate_text


def test_empty_prompt_short_circuits():
    result = generate_text("")
    assert result["used"] is False
    assert result["status"] == "empty_prompt"


def test_disabled_config_returns_not_configured(monkeypatch):
    from api import vertex_agent_search

    def fake_get_config():
        return vertex_agent_search.VertexSearchConfig(
            enabled=False,
            project_id="",
            engine_id="",
            location="global",
            collection="default_collection",
            page_size=3,
            timeout_seconds=8,
            cache_ttl_seconds=0,
        )

    monkeypatch.setattr(vertex_agent_search, "get_config", fake_get_config)

    result = generate_text("hello")
    assert result["used"] is False
    assert result["status"] == "not_configured"
