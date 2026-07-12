from pathlib import Path


def test_agent_platform_adapter_keeps_cloud_run_as_core():
    text = Path("docs/agent_platform_adapter.md").read_text(encoding="utf-8")

    assert "Gemini Enterprise Agent Platform" in text
    assert "ADK" in text
    assert "Cloud Run / FastAPI" in text
    assert "本体を移植しない" in text


def test_agent_platform_adapter_preserves_judgment_rule_review_gate():
    text = Path("docs/agent_platform_adapter.md").read_text(encoding="utf-8")

    assert "active判断基準は自動昇格しない" in text
    assert "promote_canonical_judgment_rules.py" in text
    assert "人間レビュー後だけ実行" in text
    assert "canonical_judgment_rules_preview" in text


def test_readme_links_agent_platform_adapter_plan():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Gemini Enterprise Agent Platform" in text
    assert "api/shion_agent.py" in text
    assert "docs/agent_platform_adapter.md" in text
