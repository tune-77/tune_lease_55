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
    # REV-535 質問文を渡して、質問に近いメモを選ぶ
    assert '("news_zettel_context", block_with_spacing(_news_zettel_from_hits(_rag_hits, question=message)))' in main
    assert '("news_zettel_context", _news_zettel_from_refs(rag_refs, question=search_text))' in main


def test_helpers_build_from_hits_and_refs(setup, monkeypatch):
    vault, state_path = setup
    monkeypatch.setattr(ctx, "STATE_PATH", state_path)
    monkeypatch.setattr(ctx, "build_news_zettel_context",
                        lambda refs, **_: "BLOCK" if "倒産率とリスク.md" in list(refs) else "")
    assert ctx.context_from_hits([{"file_name": "倒産率とリスク.md"}]) == "BLOCK"
    assert ctx.context_from_refs(["倒産率とリスク.md"]) == "\n\nBLOCK"
    assert ctx.context_from_refs([]) == ""


def test_existing_memos_on_stricter_hubs_need_085(setup):
    """REV-534: 既存メモは書き換えず、読む時に機械受注統計・補助金だけ 0.85 で絞る。"""
    vault, state_path = setup
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["clip/2026-10-08_別ハブ.md"]["hub_fit"] = 0.8
    assert ctx.recent_memos(["補助金の制度全体像"], state=state, today=TODAY) == []
    state["clip/2026-10-08_別ハブ.md"]["hub_fit"] = 0.86
    assert len(ctx.recent_memos(["補助金の制度全体像"], state=state, today=TODAY)) == 1
    assert len(ctx.recent_memos(["倒産率とリスク"], state=state, today=TODAY)) == 2  # 0.81 でも従来どおり


# ── REV-535: 質問との近さで選ぶ（AI 呼び出しなし） ──────────────────────
@pytest.fixture
def topical(tmp_path, monkeypatch):
    monkeypatch.setenv("SHION_NEWS_ZETTEL_CONTEXT", "1")
    memos = tmp_path / MEMO_DIR
    memos.mkdir(parents=True)
    state = {}
    rows = [
        ("2026-10-09_半導体", "半導体工場の設備投資が加速", "半導体関連の設備投資が増えそう。製造装置の需要を見たい。"),
        ("2026-10-08_建設倒産", "建設業の倒産が前年比2割増", "建設業の倒産が増えている。受注残と資金繰りを確かめたい。"),
        ("2026-10-07_運送倒産", "燃料高で運送業の倒産が増加", "運送業は燃料高を転嫁できず倒産が増えているかも。運賃の転嫁を確かめたい。"),
        ("2026-10-06_金利", "中小企業の調達金利が上昇", "金利上昇で利払いが重くなりそう。返済余力を確かめたい。"),
        ("2026-10-05_アパレル", "アパレル卸の売上が減少", "アパレル卸は在庫が重いかも。在庫回転を確かめたい。"),
    ]
    for name, title, body in rows:
        (memos / f"{name}.md").write_text(f"---\ntype: news_zettel\n---\n# {title}\n\n{body}\n\n- 元記事: [[x]]\n", encoding="utf-8")
        state[f"clip/{name}.md"] = {"status": "written", "memo": f"{MEMO_DIR}/{name}.md", "hubs": ["倒産率とリスク"], "hub_fit": 0.9}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    ctx._INDEX_CACHE.clear()
    return tmp_path, state_path


def _ask(topical, question):
    vault, state_path = topical
    return ctx.build_news_zettel_context(["倒産率とリスク.md"], question=question, vault=vault, state_path=state_path, today=TODAY)


def test_question_picks_closest_memos_not_newest(topical):
    block = _ask(topical, "運送業や建設業の倒産が増えてるけど、リース審査で気をつけることは？")
    assert "建設業の倒産" in block and "運送業の倒産" in block
    assert "半導体" not in block  # 一番新しいが質問と関係ない


def test_unrelated_question_adds_nothing(topical):
    assert _ask(topical, "こんにちは、今日は寒いね") == ""


def test_without_question_keeps_newest_first_for_compatibility(topical):
    vault, state_path = topical
    block = ctx.build_news_zettel_context(["倒産率とリスク.md"], vault=vault, state_path=state_path, today=TODAY)
    assert "半導体" in block  # question を渡さない呼び出しは従来どおり新しい順


def test_helpers_pass_question_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(ctx, "build_news_zettel_context", lambda refs, **kw: seen.update(kw) or "B")
    ctx.context_from_hits([{"file_name": "x.md"}], question="質問")
    assert seen["question"] == "質問"
    ctx.context_from_refs(["x.md"])
    assert seen["question"] is None
