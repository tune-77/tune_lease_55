"""検索でハブが当たった時の最近のニュース永続メモ（REV-502・既定オフ）。"""
import datetime as dt
import json

import pytest

from api import news_zettel_context as ctx

TODAY = dt.date(2026, 10, 9)
MEMO_DIR = "05-クリップ_記事/業界リスクニュース/永続メモ"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_NEWS_ZETTEL_CONTEXT", "1")
    memos = tmp_path / MEMO_DIR
    memos.mkdir(parents=True)
    state = {}

    def add(name, *, hubs, fit, status="written", body="運送業は燃料高で資金繰りが苦しいかも。荷主への転嫁力を確かめたい。"):
        (memos / f"{name}.md").write_text(f"---\ntype: news_zettel\n---\n# {name}の見出し\n\n{body}\n\n- 元記事: [[x]]\n", encoding="utf-8")
        state[f"clip/{name}.md"] = {"status": status, "memo": f"{MEMO_DIR}/{name}.md", "hubs": hubs, "hub_fit": fit}

    add("2026-10-08_新しい倒産", hubs=["倒産率とリスク"], fit=0.92)
    add("2026-10-07_少し前の倒産", hubs=["倒産率とリスク"], fit=0.81)
    add("2026-10-06_三件目", hubs=["倒産率とリスク"], fit=0.88)
    add("2026-10-08_未接続", hubs=[], fit=0.3)
    add("2026-10-08_判定なし", hubs=["倒産率とリスク"], fit=None)  # REV-501 より前（Jev 判定を通っていない）
    add("2026-08-01_古い", hubs=["倒産率とリスク"], fit=0.95)
    add("2026-10-08_別ハブ", hubs=["補助金の制度全体像"], fit=0.9)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return tmp_path, state_path


def _build(setup, refs):
    vault, state_path = setup
    return ctx.build_news_zettel_context(refs, vault=vault, state_path=state_path, today=TODAY)


def test_off_by_default(setup, monkeypatch):
    monkeypatch.delenv("SHION_NEWS_ZETTEL_CONTEXT")
    assert _build(setup, ["倒産率とリスク.md"]) == ""


def test_hub_hit_adds_two_recent_jev_passed_memos_as_reference(setup):
    block = _build(setup, ["[[倒産率とリスク#節]]", "2026-10-03_industry-risk-news-actions.md"])

    assert block.startswith("【最近のニュースから（参考。審査の根拠にしない）】")
    assert "新しい倒産" in block and "少し前の倒産" in block
    assert "三件目" not in block  # 2件まで
    assert "未接続" not in block and "判定なし" not in block and "古い" not in block and "別ハブ" not in block
    assert "最近のニュースでは〜という見方もある" in block and "根拠には使わない" in block


def test_no_hub_in_hits_means_no_block(setup):
    assert _build(setup, ["業種別詳細調査.md", "2026-08-28.md"]) == ""


def test_hit_hubs_reads_ref_and_file_name_forms():
    assert ctx.hit_hubs(["[[Q-Risk#概要]]", "業種別傾向.md", "Q-Risk.md", "関係ないノート.md"]) == ["Q-Risk", "業種別傾向"]


def test_wired_into_dialogue_and_chat_rag_path():
    from pathlib import Path

    from api.chat_prompt_budget import SPECS

    assert "news_zettel_context" in SPECS
    main = Path("api/main.py").read_text(encoding="utf-8")
    assert '("news_zettel_context", block_with_spacing(_news_zettel_from_hits(_rag_hits)))' in main
    assert '("news_zettel_context", _news_zettel_from_refs(rag_refs))' in main


def test_helpers_build_from_hits_and_refs(setup, monkeypatch):
    vault, state_path = setup
    monkeypatch.setattr(ctx, "STATE_PATH", state_path)
    monkeypatch.setattr(ctx, "build_news_zettel_context",
                        lambda refs, **_: "BLOCK" if "倒産率とリスク.md" in list(refs) else "")
    assert ctx.context_from_hits([{"file_name": "倒産率とリスク.md"}]) == "BLOCK"
    assert ctx.context_from_refs(["倒産率とリスク.md"]) == "\n\nBLOCK"
    assert ctx.context_from_refs([]) == ""
