from __future__ import annotations

from pathlib import Path


def test_classified_summary_api_refreshes_vault_before_cache(monkeypatch, tmp_path: Path):
    from api import main

    calls: list[str] = []

    def fake_find_vault() -> Path:
        calls.append("find_vault")
        return tmp_path

    def fake_build_summary(vault: Path, *, limit: int, days: int) -> dict:
        calls.append(f"build:{vault}:{limit}:{days}")
        return {"available": True, "article_count": 1, "axes": [], "top_insights": [], "articles": []}

    def fail_cache_load() -> dict:
        raise AssertionError("cache should not be used before refreshing the Vault")

    monkeypatch.setattr(main, "find_vault", fake_find_vault)
    monkeypatch.setattr(main, "build_classified_news_summary_from_vault", fake_build_summary)
    monkeypatch.setattr(main, "load_latest_classified_news_summary", fail_cache_load)

    result = main.get_lease_news_classified_summary_api(limit=999, days=999)

    assert result["article_count"] == 1
    assert calls == ["find_vault", f"build:{tmp_path}:80:60"]
