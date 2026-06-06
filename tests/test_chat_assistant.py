from __future__ import annotations

import importlib
from pathlib import Path

from obsidian_query import split_query_terms


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    return vault


def test_collect_obsidian_context_includes_recent_notes(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    note = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-16.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# memo\n\n条件付き承認は、追加資料・期間短縮・前受金で進める。",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.collect_obsidian_context("条件付き承認", limit=3)
    assert hits, "Obsidianの保存メモを拾えていない"
    assert any("条件付き承認" in h["snippet"] for h in hits)


def test_build_prompt_mentions_condition_playbook(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    note = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-16.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "## memo\n\n承認条件は追加資料と期間短縮。",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge, chat_assistant
    importlib.reload(obsidian_bridge)
    importlib.reload(chat_assistant)

    prompt = chat_assistant._build_prompt(
        "条件付き承認の具体的な方法を教えて",
        [{"role": "user", "content": "前の相談"}],
        {"score": 66, "judgment": "条件付"},
        obsidian_bridge.collect_obsidian_context("条件付き承認", limit=2),
        [],
        humor_style="standard",
    )
    assert "条件付き承認の説明方針" in prompt
    assert "Obsidianの過去メモ" in prompt
    assert "1. 追加資料 2. 期間短縮" in prompt


def test_web_context_parses_search_results(monkeypatch):
    html = """
    <html><body>
      <a class="result__a" href="https://example.com/alpha">Alpha Result</a>
      <a class="result__snippet">Alpha snippet text.</a>
      <a class="result__a" href="https://example.com/bravo">Bravo Result</a>
      <a class="result__snippet">Bravo snippet text.</a>
    </body></html>
    """

    class Resp:
        status_code = 200
        text = html
        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Resp()

    from mobile_app import web_bridge
    monkeypatch.setattr(web_bridge.requests, "get", fake_get)

    hits = web_bridge.collect_web_context("最新情報", limit=2)
    assert len(hits) == 2
    assert hits[0]["title"] == "Alpha Result"
    assert hits[0]["url"] == "https://example.com/alpha"
    assert "snippet" in hits[0]


def test_build_prompt_mentions_web_section(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge, chat_assistant
    importlib.reload(obsidian_bridge)
    importlib.reload(chat_assistant)

    prompt = chat_assistant._build_prompt(
        "Gemini の最新モデルを教えて",
        [],
        None,
        [],
        [{"title": "Gemini", "url": "https://example.com", "snippet": "info", "domain": "example.com"}],
        humor_style="standard",
    )
    assert "Web参照の方針" in prompt
    assert "Web検索結果" in prompt


def test_append_improvement_note_writes_daily_log(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_improvement_note(
        "改善候補",
        "## 抽出された改善候補\n\n- **入力導線** [high]\n  - ユーザー要望: Enterで誤送信しないようにしたい\n  - 改善案: チャット欄のみEnter送信にする\n  - 根拠: 送信と審査が競合しやすい",
    )
    assert result["status"] == "saved"
    assert "Improvement Log" in result["path"]
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "入力導線" in saved


def test_append_web_note_writes_daily_log(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_web_note(
        "Web参照メモ",
        "## Web要点\n\n- Gemini 2.5 Flash は公式ブログで更新\n- 公式情報を優先する",
    )
    assert result["status"] == "saved"
    assert "Web Research" in result["path"]
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "Gemini 2.5 Flash" in saved


def test_build_obsidian_digest_combines_multiple_notes(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    for idx, text in enumerate([
        "## memo\n\n条件付き承認は追加資料と期間短縮が基本。",
        "## memo\n\n条件付き承認は前受金も候補。",
    ], start=1):
        note = vault / "Projects" / "tune_lease_55" / "AI Chat" / f"2026-05-16-{idx}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(text, encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.collect_obsidian_context("条件付き承認", limit=4)
    digest = obsidian_bridge.build_obsidian_digest("条件付き承認", hits)
    assert "Obsidian統合要約" in digest["digest"]
    assert "関連ノート数" in digest["digest"]
    assert "追加資料" in digest["digest"] or "前受金" in digest["digest"]
    assert "[[" in digest["digest"]


def test_split_query_terms_handles_chatty_japanese_questions():
    assert split_query_terms("補助金について教えて") == ["補助金"]
    assert split_query_terms("期待使用期間とリース期間の関係を教えて") == ["期待使用期間", "リース期間", "関係"]
    assert split_query_terms("格付の見方を教えて") == ["格付", "見方"]
    assert split_query_terms("格付８−２先について教えて") == ["格付8-2先"]


def test_search_notes_prioritizes_subsidy_knowledge_over_chat_logs(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    knowledge = vault / "Projects" / "tune_lease_55" / "2026-05-13_補助金まとめ.md"
    knowledge.parent.mkdir(parents=True, exist_ok=True)
    knowledge.write_text(
        "# 補助金まとめ\n\n- 中小企業省力化投資補助金\n- ものづくり補助金\n",
        encoding="utf-8",
    )
    for rel in [
        "Projects/tune_lease_55/AI Chat/2026-05-17.md",
        "Projects/tune_lease_55/AI Chat/Weekly Review/2026-W20.md",
        "Projects/tune_lease_55/AI Chat/Improvement Log/2026-05-17.md",
        "Daily/2026-05-17.md",
    ]:
        note = vault / rel
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("補助金について教えて、というチャットログ。", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.collect_obsidian_context("補助金について教えて", limit=4)
    assert hits[0]["path"] == "Projects/tune_lease_55/2026-05-13_補助金まとめ.md"
    assert any("AI Chat" in hit["path"] for hit in hits[1:])


def test_search_notes_splits_japanese_chat_query_for_obsidian_search(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    knowledge = vault / "Projects" / "tune_lease_55" / "期待使用期間まとめ.md"
    knowledge.parent.mkdir(parents=True, exist_ok=True)
    knowledge.write_text(
        "# 期待使用期間まとめ\n\n- 期待使用期間とリース期間の関係を整理する。\n",
        encoding="utf-8",
    )
    chat_log = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-17.md"
    chat_log.parent.mkdir(parents=True, exist_ok=True)
    chat_log.write_text("期待使用期間とリース期間の関係を教えて、というチャットログ。", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    terms = obsidian_bridge._expand_query_terms("期待使用期間とリース期間の関係を教えて")
    assert "期待使用期間" in terms
    assert "リース期間" in terms
    hits = obsidian_bridge.collect_obsidian_context("期待使用期間とリース期間の関係を教えて", limit=3)
    assert hits[0]["path"] == "Projects/tune_lease_55/期待使用期間まとめ.md"


def test_obsidian_ai_context_block_uses_split_search_terms(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    knowledge = vault / "Projects" / "tune_lease_55" / "2026-05-13_補助金まとめ.md"
    knowledge.parent.mkdir(parents=True, exist_ok=True)
    knowledge.write_text(
        "# 補助金まとめ\n\n- 中小企業省力化投資補助金\n- ものづくり補助金\n",
        encoding="utf-8",
    )
    chat_log = vault / "Projects" / "tune_lease_55" / "AI Chat" / "2026-05-17.md"
    chat_log.parent.mkdir(parents=True, exist_ok=True)
    chat_log.write_text("補助金について教えて、というチャットログ。", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    from obsidian_ai_context import build_obsidian_ai_context_block
    importlib.reload(obsidian_bridge)

    block = build_obsidian_ai_context_block("補助金について教えて")
    assert "Projects/tune_lease_55/2026-05-13_補助金まとめ.md" in block
    assert "中小企業省力化投資補助金" in block


def test_search_notes_reranks_knowledge_and_suppresses_unrelated_humor(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    notes = {
        "リース知識/格付８−２先への対応.md": "# 資金繰り\n\nキャッシュフローと手元資金を確認する。",
        "Projects/tune_lease_55/AI Chat/2026-06-06.md": "資金繰りが厳しい会社の確認事項を質問した。",
        "Humor/八奈見.md": "資金繰りをユーモア口調で説明する。",
    }
    for rel, body in notes.items():
        note = vault / rel
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(body, encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.search_notes("資金繰りが厳しい会社にリースを出す時の確認事項は？", limit=3)
    assert hits[0]["path"] == "リース知識/格付８−２先への対応.md"
    assert all("Humor/" not in hit["path"] for hit in hits[:2])


def test_search_notes_allows_humor_when_query_requests_it(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    humor = vault / "Humor" / "審査コメント口調.md"
    humor.parent.mkdir(parents=True, exist_ok=True)
    humor.write_text("審査コメントのユーモア口調ルール。", encoding="utf-8")
    normal = vault / "リース知識" / "審査コメント.md"
    normal.parent.mkdir(parents=True, exist_ok=True)
    normal.write_text("審査コメントの基本ルール。", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    hits = obsidian_bridge.search_notes("審査コメントのユーモア口調ルールを確認したい", limit=2)
    assert hits[0]["path"] == "Humor/審査コメント口調.md"


def test_append_wiki_note_writes_hub(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_wiki_note(
        "条件付き承認",
        "## 要点\n\n- 追加資料\n- 期間短縮\n- 前受金",
        related_paths=[
            "Projects/tune_lease_55/AI Chat/2026-05-16.md",
            "Projects/tune_lease_55/2026-05-12_ファイナンスリース_autoresearch.md",
        ],
    )
    assert result["status"] == "saved"
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "関連ノート" in saved
    assert "[[Projects/tune_lease_55/AI Chat/2026-05-16|2026-05-16]]" in saved
    assert "条件付き承認" in saved


def test_append_weekly_review_note_writes_weekly_file(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setenv("OBSIDIAN_VAULT", str(vault))

    from mobile_app import obsidian_bridge
    importlib.reload(obsidian_bridge)

    result = obsidian_bridge.append_weekly_review_note(
        "週次改善レビュー",
        "## 今週の改善候補まとめ\n\n- 条件付き承認の推奨アクション自動表示 (accept)\n- 入力導線の明確化 (review)",
    )
    assert result["status"] == "saved"
    assert "Weekly Review" in result["path"]
    saved = (vault / result["path"]).read_text(encoding="utf-8")
    assert "今週の改善候補まとめ" in saved
    assert "条件付き承認" in saved


def test_fallback_chat_packet_builds_review_and_wiki():
    from mobile_app import chat_assistant

    packet = chat_assistant._fallback_chat_packet(
        "条件付き承認の具体的な方法を教えて",
        {"score": 66, "judgment": "条件付"},
        [{"path": "Projects/tune_lease_55/AI Chat/2026-05-16.md", "snippet": "条件付き承認は追加資料と期間短縮"}],
        [],
        humor_style="standard",
    )
    assert packet["wiki_should_save"] is True
    assert packet["weekly_should_save"] is True
    assert packet["improvement_items"][0]["decision"] == "accept"


def test_build_strategy_advice_includes_indicator_takeaways():
    from mobile_app import advisor_strategy

    score_result = {
        "score": 58,
        "judgment": "条件付",
        "base_rate": 2.15,
        "recommended_rate": 2.45,
        "spread_pred": 0.31,
        "indicator_analysis": {
            "summary": "業界平均より営業利益率は上、自己資本比率は下回る。",
            "detail": "detail",
            "indicators": [
                {"name": "営業利益率", "value": 5.0, "bench": 4.5, "unit": "%"},
                {"name": "自己資本比率", "value": 33.3, "bench": 35.0, "unit": "%"},
            ],
        },
        "aurion": {"q_risk": {"score": 12}, "competitor_pressure": {"score": 7}},
        "streamlit": {"credit_risk_group_score": 22, "credit_risk_group_level": "ok"},
    }
    case = {
        "industry_sub": "06 総合工事業",
        "customer_type": "既存先",
        "op_profit": 5,
        "nenshu": 100,
        "acquisition_cost": 10,
    }

    advice = advisor_strategy.build_strategy_advice(score_result=score_result, case=case)

    assert advice["indicator_summary"] == "業界平均より営業利益率は上、自己資本比率は下回る。"
    assert advice["indicator_takeaways"]
    assert any("営業利益率" in x for x in advice["indicator_takeaways"])
    assert any("自己資本比率" in x for x in advice["indicator_takeaways"])
    assert advice["additional_guidance"]
    assert any("追加資料" in x or "期間調整" in x for x in advice["additional_guidance"])
    assert advice["probability_uplifts"]
    assert any(item["gain_pct"] > 0 for item in advice["probability_uplifts"])
