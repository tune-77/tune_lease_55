from pathlib import Path


def test_debate_save_to_obsidian_accepts_decimal_score(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_OBSIDIAN_VAULT_PATH", str(tmp_path))

    req = main.SaveDebateToObsidianRequest(
        company_name="デモ精密工業",
        score=56.5,
        grade="C",
        cautious=None,
        aggressive=None,
        arbiter_summary="財務と物件保全を確認する。",
        final_decision="条件付承認",
        conditions=["資金繰り資料を確認する"],
        debate_log="紫苑（懐疑）: 条件付承認",
    )

    result = main.save_debate_to_obsidian(req)

    saved = tmp_path / result["path"]
    assert saved.exists()
    text = saved.read_text(encoding="utf-8")
    assert "score: 56.5" in text
    assert "**スコア**: 56.5点" in text
    assert "資金繰り資料を確認する" in text
