"""ニュースの永続メモ（ツェッテルカステン試行・REV-499）。"""
import datetime as dt
import json

import pytest

from scripts import build_news_zettel as zettel

CLIP = """---
date: 2026-10-08
industries: "運送業, 物流業"
lease_assets: "車両, 物流設備"
importance: 高
---
# 人手不足倒産は225件、年度上半期として4年連続で過去最多 - 東京商工リサーチ

## 3行要約
- 人手不足倒産は225件、年度上半期として4年連続で過去最多
- （詳細なし）
- 稼働率・車両更新・保守費用の見通しを確認する。
"""


@pytest.fixture(autouse=True)
def _no_real_jev(monkeypatch):
    # 既存テストは本物の Jev を呼ばない。ハブ判定のテストだけ偽の判定器と NEWS_ZETTEL_HUB_CHECK=1 を使う
    monkeypatch.setenv("NEWS_ZETTEL_HUB_CHECK", "0")


@pytest.fixture
def vault(tmp_path):
    news = tmp_path / zettel.NEWS_DIR
    news.mkdir(parents=True)
    for day, name in [("2026-10-08", "A"), ("2026-10-07", "B"), ("2026-09-01", "C"), ("2026-08-01", "D")]:
        text = CLIP.replace("2026-10-08", day).replace("過去最多 - 東京商工リサーチ", f"過去最多（{name}） - 東京商工リサーチ")
        (news / f"{day}_業界リスクニュース_{name}.md").write_text(text, encoding="utf-8")
    hub = tmp_path / "03-知識_業界/業種分析/倒産率とリスク.md"
    hub.parent.mkdir(parents=True)
    hub.write_text("# 倒産率とリスク\n", encoding="utf-8")
    return tmp_path


def test_only_existing_hubs_are_offered(vault):
    assert [hub["label"] for hub in zettel.available_hubs(vault)] == ["倒産率とリスク"]


def test_selects_recent_first_and_backfills_newest_first_only_when_enabled(vault):
    today = dt.date(2026, 10, 8)
    kwargs = dict(today=today, new_days=2, new_limit=20, backfill_limit=1)

    recent = zettel.select_clips(vault, {}, backfill=False, **kwargs)
    assert [p.name[:10] for p in recent] == ["2026-10-08", "2026-10-07"]

    with_backfill = zettel.select_clips(vault, {}, backfill=True, **kwargs)
    assert [p.name[:10] for p in with_backfill] == ["2026-10-08", "2026-10-07", "2026-09-01"]

    done = {str(zettel.NEWS_DIR / recent[0].name): {"status": "written"}}
    assert [p.name[:10] for p in zettel.select_clips(vault, done, backfill=False, **kwargs)] == ["2026-10-07"]


def test_parse_clip_drops_placeholder_and_title_echo(vault):
    path = next((vault / zettel.NEWS_DIR).glob("2026-10-08*.md"))
    item = zettel.parse_clip(path, path.read_text(encoding="utf-8"))

    assert item["industries"] == "運送業, 物流業"
    assert "詳細なし" not in item["summary"]
    assert item["summary"] == "稼働率・車両更新・保守費用の見通しを確認する。"


def test_process_writes_one_memo_per_article_with_links_and_never_overwrites(vault):
    clips = sorted((vault / zettel.NEWS_DIR).glob("2026-10-0*.md"), reverse=True)
    calls: list[str] = []

    def fake_model(prompt: str) -> dict:
        calls.append(prompt)
        return {"items": [
            {"i": 0, "idea": "運送業は人手不足で稼働が落ちると返済原資が細るので、ドライバー確保の見通しを確かめたい。", "hubs": ["h2", "h99"]},
            {"i": 1, "idea": "", "hubs": []},
        ]}

    state: dict = {}
    summary = zettel.process(vault, clips, state, model_call=fake_model, model_name="flash-lite", now="2026-10-08T10:00:00")

    assert len(calls) == 1  # 1回の呼び出しにまとめる
    assert summary["written"] == 1 and summary["connected"] == 1 and summary["no_idea"] == 1
    memo = next((vault / zettel.MEMO_DIR).glob("*.md"))
    text = memo.read_text(encoding="utf-8")
    assert "[[03-知識_業界/業種分析/倒産率とリスク|倒産率とリスク]]" in text  # 実在するハブだけ
    assert "[[05-クリップ_記事/業界リスクニュース/2026-10-08_業界リスクニュース_A|" in text
    assert "東京商工リサーチ" not in text.splitlines()[text.splitlines().index("---", 1) + 1]  # 見出しから配信元を外す
    assert state[str(zettel.NEWS_DIR / clips[1].name)]["status"] == "no_idea"

    # 同名の永続メモがあれば上書きしない
    memo.write_text("手で直したメモ", encoding="utf-8")
    zettel.process(vault, clips[:1], {}, model_call=fake_model, model_name="flash-lite")
    assert memo.read_text(encoding="utf-8") == "手で直したメモ"


def test_unconnected_when_no_hub_fits(vault):
    clips = sorted((vault / zettel.NEWS_DIR).glob("2026-10-08*.md"))
    idea = "物流設備の更新需要は続きそうなので、中古車両の相場も合わせて見ておきたい。"
    zettel.process(vault, clips, {}, model_call=lambda _p: {"items": [{"i": 0, "idea": idea, "hubs": []}]})

    text = next((vault / zettel.MEMO_DIR).glob("*.md")).read_text(encoding="utf-8")
    assert "connection: unconnected" in text and "- 関連: 未接続" in text


def test_model_failure_stops_and_leaves_items_unprocessed(vault):
    clips = sorted((vault / zettel.NEWS_DIR).glob("2026-10-0*.md"))

    def blocked(_prompt):
        raise RuntimeError("AI予算ガードで停止")

    state: dict = {}
    summary = zettel.process(vault, clips, state, model_call=blocked)

    assert summary["stopped"].startswith("RuntimeError") and state == {}
    assert not (vault / zettel.MEMO_DIR).exists()


def test_validate_rejects_out_of_range_and_bad_lengths():
    rows = zettel.validate({"items": [{"i": 5, "idea": "x" * 30}, {"i": 0, "idea": "短い"}]}, 2, {"h1"})
    assert rows == {0: {"idea": "", "hubs": []}}


def test_feature_is_a_proactive_budget_class():
    import ai_budget

    assert ai_budget.call_class("news_zettel", {"source": "build_news_zettel.py"}) == ai_budget.PROACTIVE


def test_run_script_wires_zettel_after_collection():
    from pathlib import Path

    script = Path("scripts/run_lease_news_collection.sh").read_text(encoding="utf-8")
    assert script.index("collect_lease_news_to_obsidian.py") < script.index("build_news_zettel.py")
    assert '"$SCRIPT_DIR/build_news_zettel.py" ||' in script


def test_read_text_works_without_st_flags(tmp_path, monkeypatch):
    # CI（Linux）には st_flags が無い。退避判定ができなくても普通に読む
    import os
    from types import SimpleNamespace

    path = tmp_path / "a.md"
    path.write_text("本文", encoding="utf-8")
    monkeypatch.setattr(zettel.os, "stat", lambda _p: SimpleNamespace(st_size=6))

    assert zettel._read_text(path) == "本文"


# ── REV-500: 品質の修正 ────────────────────────────────────────────────
def _clip(vault, day, name, title, topic=None):
    text = CLIP.replace("2026-10-08", day).replace(
        "# 人手不足倒産は225件、年度上半期として4年連続で過去最多 - 東京商工リサーチ", f"# {title}"
    )
    if topic:
        text = text.replace("importance: 高\n", f'importance: 高\ncanonical_topic: "{topic}"\n')
    path = vault / zettel.NEWS_DIR / f"{day}_業界リスクニュース_{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_promo_titles_are_skipped_without_calling_model(vault):
    promo = _clip(vault, "2026-10-08", "P", "中古建設機械市場の成長予測(2026年から2033年、年平均成長率5.9%) - PR")
    calls = []
    state: dict = {}

    summary = zettel.process(vault, [promo], state, model_call=lambda p: calls.append(p) or {"items": []})

    assert summary["promo"] == 1 and calls == []
    assert state[str(zettel.NEWS_DIR / promo.name)]["status"] == "skipped_promo"


def test_nikkei_sponsored_feature_is_promo():
    assert zettel.is_promo_title("コストから価値創造へ 物流を経営の中核に|日本経済新聞 電子版特集(PR) - ps.nikkei.com")
    assert zettel.is_promo_title("物流DXの最前線|日本経済新聞 電子版特集（PR）")
    assert not zettel.is_promo_title("物流2024年問題、運送業の倒産が増加 - 日本経済新聞")
    assert not zettel.is_promo_title("新宇都宮本社始動!物流の効率化 - PR TIMES")


def test_same_topic_from_another_source_is_not_written_twice(vault):
    first = _clip(vault, "2026-10-08", "X1", "東京港・大井コンテナふ頭再編の全体像 - LOGISTICS TODAY", "東京港・大井コンテナふ頭再編の全体像")
    second = _clip(vault, "2026-10-08", "X2", "東京港・大井コンテナふ頭再編の全体像 - Yahoo!ニュース", "東京港・大井コンテナふ頭再編の全体像")
    idea = "港の再編で荷役機器の入れ替えが出てくるかも。荷役機器の案件なら残価の見方を確かめたい。"
    calls = []

    def model(prompt):
        calls.append(prompt)
        return {"items": [{"i": 0, "idea": idea, "hubs": ["h2"]}]}

    state: dict = {}
    summary = zettel.process(vault, [first, second], state, model_call=model)

    assert summary["written"] == 1 and summary["duplicate_topic"] == 1
    assert "[1]" not in calls[0]  # 重複は呼び出しに含めない
    # 翌日、同じ話題の別クリップが来ても作らない（state の topic で判定）
    third = _clip(vault, "2026-10-09", "X3", "東京港・大井コンテナふ頭再編の全体像 - 47NEWS", "東京港・大井コンテナふ頭再編の全体像")
    assert zettel.process(vault, [third], state, model_call=model)["duplicate_topic"] == 1


def test_old_state_without_topic_is_backfilled_from_clip(vault):
    path = next((vault / zettel.NEWS_DIR).glob("2026-10-08*.md"))
    state = {str(zettel.NEWS_DIR / path.name): {"status": "written"}}

    topics = zettel.processed_topics(vault, state)

    assert topics and state[str(zettel.NEWS_DIR / path.name)]["topic"] in topics


def test_at_most_one_hub_is_kept():
    rows = zettel.validate({"items": [{"i": 0, "idea": "x" * 40, "hubs": ["h1", "h2"]}]}, 1, {"h1", "h2"})
    assert rows[0]["hubs"] == ["h1"]


def test_prompt_forbids_assertive_phrases_and_asks_single_hub():
    prompt = zettel.build_prompt([{"title": "t", "industries": "", "lease_assets": ""}], [])
    assert "直結する" in prompt and "急増中" in prompt
    assert "id を1個" in prompt


def test_news_collector_plist_enables_backfill_with_daily_limit():
    import plistlib
    from pathlib import Path

    env = plistlib.loads(Path("launchd/com.tunelease.lease-news-collector.plist").read_bytes())["EnvironmentVariables"]
    assert env["NEWS_ZETTEL_BACKFILL"] == "1"
    assert env["NEWS_ZETTEL_BACKFILL_DAILY_LIMIT"] == "200"



# ── REV-501: ハブ判定（案B）と統計記事の重複（案C） ──────────────────────
IDEA = "運送業は人手不足で稼働が落ちると返済原資が細るかも。ドライバー確保の見通しを確かめたい。"


def _one_clip(vault):
    return sorted((vault / zettel.NEWS_DIR).glob("2026-10-08*.md"))


def test_weak_hub_is_dropped_and_memo_left_unconnected(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_HUB_CHECK", "1")
    seen: list[str] = []

    def checker(texts):
        seen.extend(texts)
        return [0.3]

    state: dict = {}
    summary = zettel.process(vault, _one_clip(vault), state,
                             model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": ["h2"]}]},
                             hub_checker=checker)

    assert summary["hub_checked"] == 1 and summary["hub_dropped"] == 1 and summary["unconnected"] == 1
    assert "ハブ: 倒産率とリスク" in seen[0]
    text = next((vault / zettel.MEMO_DIR).glob("*.md")).read_text(encoding="utf-8")
    assert "connection: unconnected" in text and "hub_check: 0.3" in text and "倒産率とリスク" in text.split("---")[1]
    entry = next(iter(state.values()))
    assert entry["hubs"] == [] and entry["proposed_hubs"] == ["倒産率とリスク"] and entry["hub_fit"] == 0.3


def test_fitting_hub_is_kept(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_HUB_CHECK", "1")
    zettel.process(vault, _one_clip(vault), {},
                   model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": ["h2"]}]},
                   hub_checker=lambda texts: [0.9])

    text = next((vault / zettel.MEMO_DIR).glob("*.md")).read_text(encoding="utf-8")
    assert "connection: connected" in text and "[[03-知識_業界/業種分析/倒産率とリスク|倒産率とリスク]]" in text


def test_jev_unavailable_keeps_link_marked_unchecked(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_HUB_CHECK", "1")
    summary = zettel.process(vault, _one_clip(vault), {},
                             model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": ["h2"]}]},
                             hub_checker=lambda texts: [None])

    assert summary["hub_unchecked"] == 1 and summary["connected"] == 1
    assert "hub_check: unchecked" in next((vault / zettel.MEMO_DIR).glob("*.md")).read_text(encoding="utf-8")


def test_stat_key_matches_same_numbers_only_with_units():
    a = zettel.stat_key("2026-10-08", "8月の工作機械受注 64%増 北米アジア伸び歴代2位 - 日刊工業")
    b = zettel.stat_key("2026-10-08", "工作機械受注、8月64%増 北米アジア伸び歴代2位 - 日経")
    assert a and a == b
    assert zettel.stat_key("2026-10-08", "8月の工作機械受注65%増 AI関連が好調") != a
    assert zettel.stat_key("2026-10-08", "日銀短観は製造業改善も設備投資は鈍化") == ""
    assert zettel.stat_key("2026-10-09", "工作機械受注、8月64%増 北米アジア伸び歴代2位") != a  # 日付が違えば別


def test_same_stat_requires_overlap_and_same_place():
    def item(title):
        return {"topic": zettel.normalize_topic(title), "stat_key": zettel.stat_key("2026-10-08", title)}

    machine = item("8月の工作機械受注 64%増 北米アジア伸び歴代2位")
    assert zettel.is_same_stat(item("工作機械受注、8月64%増 北米アジア伸び歴代2位"), {machine["stat_key"]: [machine["topic"]]})
    aomori = item("9月の青森県内企業倒産は3件 負債総額は小規模")
    assert not zettel.is_same_stat(item("9月の岡山県内企業倒産は3件 建設業が中心"), {aomori["stat_key"]: [aomori["topic"]]})


def test_same_statistic_from_another_outlet_is_skipped_before_model(vault):
    first = _clip(vault, "2026-10-08", "S1", "8月の工作機械受注 64%増 北米アジア伸び歴代2位 - 日刊工業")
    second = _clip(vault, "2026-10-08", "S2", "工作機械受注、8月64%増 北米アジア伸び歴代2位 - 日経")
    calls: list[str] = []

    def model(prompt):
        calls.append(prompt)
        return {"items": [{"i": 0, "idea": IDEA, "hubs": []}]}

    state: dict = {}
    summary = zettel.process(vault, [first, second], state, model_call=model)

    assert summary["duplicate_stat"] == 1 and summary["written"] == 1 and "[1]" not in calls[0]
    assert state[str(zettel.NEWS_DIR / second.name)]["status"] == "duplicate_stat"


# ── REV-502: リースニュースへの拡大（既定オフ） ─────────────────────────────
def _lease_clip(vault, day, name, title):
    folder = vault / zettel.LEASE_NEWS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day}_リースニュース_{name}.md"
    path.write_text(CLIP.replace("2026-10-08", day).replace(
        "# 人手不足倒産は225件、年度上半期として4年連続で過去最多 - 東京商工リサーチ", f"# {title}"), encoding="utf-8")
    return path


def test_lease_news_is_ignored_unless_enabled(vault, monkeypatch):
    _lease_clip(vault, "2026-10-08", "L1", "建機レンタル大手が中古建機の再販を強化 - リース事業協会")
    monkeypatch.delenv("NEWS_ZETTEL_LEASE_NEWS", raising=False)
    kwargs = dict(today=dt.date(2026, 10, 8), new_days=2, new_limit=20, backfill=True, backfill_limit=20)

    assert all(p.parent.name == "業界リスクニュース" for p in zettel.select_clips(vault, {}, **kwargs))

    monkeypatch.setenv("NEWS_ZETTEL_LEASE_NEWS", "1")
    picked = zettel.select_clips(vault, {}, **kwargs)
    assert {p.parent.name for p in picked} == {"業界リスクニュース", "リースニュース"}
    assert [p.name[:10] for p in picked] == sorted([p.name[:10] for p in picked], reverse=True)  # 日付の新しい順に混ぜる


def test_lease_news_memo_goes_to_its_own_folder_with_correct_link(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_LEASE_NEWS", "1")
    clip = _lease_clip(vault, "2026-10-08", "L1", "建機レンタル大手が中古建機の再販を強化 - リース事業協会")
    state: dict = {}

    zettel.process(vault, [clip], state, model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": []}]})

    memo = next((vault / zettel.LEASE_NEWS_DIR / "永続メモ").glob("*.md"))
    assert "[[05-クリップ_記事/リースニュース/2026-10-08_リースニュース_L1|" in memo.read_text(encoding="utf-8")
    assert str(zettel.LEASE_NEWS_DIR / clip.name) in state
    assert not (vault / zettel.MEMO_DIR).exists()


def test_same_topic_across_risk_and_lease_news_is_written_once(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_LEASE_NEWS", "1")
    risk = _clip(vault, "2026-10-08", "R1", "東京港・大井コンテナふ頭再編の全体像 - LOGISTICS TODAY", "東京港・大井コンテナふ頭再編の全体像")
    lease = _lease_clip(vault, "2026-10-08", "L2", "東京港・大井コンテナふ頭再編の全体像 - リース事業協会")
    summary = zettel.process(vault, [risk, lease], {},
                             model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": []}]})
    assert summary["written"] == 1 and summary["duplicate_topic"] == 1


# ── REV-503: 経産省フィードの要約を本文の要点に使う ───────────────────────────
FEED = """<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>令和8年度物流パートナーシップ優良事業者を募集します</title>
<link rel="alternate" type="text/html" href="https://www.meti.go.jp/press/2026/10/20261007001/20261007001.html"/>
<updated>2026-10-07T05:00:00Z</updated>
<summary>経済産業省・国土交通省では、物流の生産性向上や構造改革等に向けた取組に顕著な功績のあった事業者を表彰します。</summary></entry>
<entry><title>古い発表</title><link rel="alternate" href="https://www.meti.go.jp/press/2026/06/x.html"/>
<updated>2026-06-19T05:00:00Z</updated><summary>六月の発表の要約です。十分な長さの文章がここに入ります。</summary></entry>
</feed>"""
TODAY = dt.date(2026, 10, 8)


def test_parse_and_select_only_recent_unprocessed_entries():
    entries = zettel.parse_meti_feed(FEED)
    assert [e["date"] for e in entries] == ["2026-10-07", "2026-06-19"]
    items = zettel.meti_items(entries, {}, today=TODAY)
    assert [i["title"] for i in items] == ["令和8年度物流パートナーシップ優良事業者を募集します"]
    assert items[0]["body"].startswith("経済産業省・国土交通省では") and items[0]["source_label"] == "経済産業省"
    assert zettel.meti_items(entries, {items[0]["key"]: {"status": "written"}}, today=TODAY) == []


def test_stale_feed_does_nothing_and_records_latest_date(vault, tmp_path, monkeypatch):
    status = tmp_path / "meti.json"
    monkeypatch.setattr(zettel, "METI_STATUS_PATH", status)
    stale = FEED.replace("2026-10-07T05", "2026-06-18T05")
    calls = []

    summary = zettel.process_meti(vault, {}, fetch=lambda: stale, today=TODAY, model_call=lambda p: calls.append(p))

    assert summary["meti_new"] == 0 and calls == []
    saved = json.loads(status.read_text(encoding="utf-8"))
    assert saved["latest_entry_date"] == "2026-06-19" and saved["days_since_latest"] == 111 and saved["stale"] is True
    line = zettel.morning_report_lines(status)[0]
    assert "⚠️" in line and "最新 2026-06-19" in line and "111日前" in line


def test_new_entry_uses_feed_summary_as_body_and_cites_meti_without_saving_it(vault, tmp_path, monkeypatch):
    monkeypatch.setattr(zettel, "METI_STATUS_PATH", tmp_path / "meti.json")
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return {"items": [{"i": 0, "idea": IDEA, "hubs": []}]}

    state: dict = {}
    summary = zettel.process_meti(vault, state, fetch=lambda: FEED, today=TODAY, model_call=model)

    assert summary["meti_written"] == 1
    assert "本文の要点（経済産業省の発表より）: 経済産業省・国土交通省では" in prompts[0]
    memo = next((vault / zettel.MEMO_DIR).glob("*.md")).read_text(encoding="utf-8")
    assert "（出典: 経済産業省）" in memo and "https://www.meti.go.jp/press/2026/10/20261007001/20261007001.html" in memo
    assert "顕著な功績のあった事業者を表彰します" not in memo  # 要約そのものは保存しない
    assert "source_label: 経済産業省" in memo
    assert state["meti:https://www.meti.go.jp/press/2026/10/20261007001/20261007001.html"]["status"] == "written"


def test_feed_error_is_recorded_for_morning_report(vault, tmp_path, monkeypatch):
    status = tmp_path / "meti.json"
    monkeypatch.setattr(zettel, "METI_STATUS_PATH", status)

    def broken():
        raise OSError("HTTP Error 403: Forbidden")

    assert zettel.process_meti(vault, {}, fetch=broken, today=TODAY)["meti_stopped"] == "OSError"
    assert "取得失敗" in zettel.morning_report_lines(status)[0]


def test_never_requests_article_pages_or_robots():
    from pathlib import Path

    source = Path("scripts/build_news_zettel.py").read_text(encoding="utf-8")
    assert source.count("urlopen(") == 1 and "METI_FEED_URL" in source.split("urlopen(")[0].rsplit("def ", 1)[1]
    assert "robots.txt\", " not in source


def test_morning_report_includes_feed_line():
    from pathlib import Path

    assert "*news_zettel_feed_lines()," in Path("scripts/aurion_core_daily.py").read_text(encoding="utf-8")


# ── REV-534: 機械受注統計・補助金のハブだけ説明を絞り、しきい値を 0.85 に上げる ──────────
def test_stricter_threshold_only_for_machine_orders_and_subsidy_hubs(vault, monkeypatch):
    monkeypatch.setenv("NEWS_ZETTEL_HUB_CHECK", "1")
    by_label = {hub["label"]: hub for hub in zettel.HUBS}
    # REV-539 機械受注統計は説明を絞った分、今の説明での判定は 0.70・古い説明の点数は 0.85
    assert zettel.hub_fit_min(by_label["機械受注統計"]) == 0.70
    assert zettel.hub_legacy_fit_min(by_label["機械受注統計"]) == 0.85
    assert zettel.hub_fit_min(by_label["補助金の制度全体像"]) == 0.85
    assert zettel.hub_legacy_fit_min(by_label["補助金の制度全体像"]) == 0.85
    assert zettel.hub_fit_min(by_label["倒産率とリスク"]) == zettel.HUB_FIT_MIN  # 他のハブは変えない
    # REV-539 さらに内閣府の機械受注統計・工作機械受注そのものに絞る
    assert by_label["機械受注統計"]["use"].startswith("内閣府の機械受注統計・工作機械受注（日工会）そのもの")
    assert "法人企業統計" in by_label["機械受注統計"]["use"] and "海外の資本財受注" in by_label["機械受注統計"]["use"]
    assert "融資・保証・相談窓口・セミナーは除く" in by_label["補助金の制度全体像"]["use"]

    hub = vault / "03-知識_業界/市場分析データ/機械受注統計_2023-2026.md"
    hub.parent.mkdir(parents=True)
    hub.write_text("# 機械受注統計\n", encoding="utf-8")
    seen: list[str] = []
    state: dict = {}
    summary = zettel.process(vault, _one_clip(vault), state,
                             model_call=lambda _p: {"items": [{"i": 0, "idea": IDEA, "hubs": ["h15"]}]},
                             hub_checker=lambda texts: seen.extend(texts) or [0.65])
    assert "内閣府の機械受注統計" in seen[0]
    entry = next(iter(state.values()))
    assert summary["hub_dropped"] == 1 and entry["hubs"] == []
    assert entry["hub_use"] == by_label["機械受注統計"]["use"]  # 判定に使った説明を残す（REV-539）


def test_recheck_writes_side_file_without_touching_memos_or_state(vault, tmp_path):
    """REV-539: 説明を変えたハブの既存接続を判定し直し、別ファイルにだけ保存する。"""
    memo_dir = vault / zettel.MEMO_DIR
    memo_dir.mkdir(parents=True)
    memo = memo_dir / "2026-10-08_機械受注.md"
    memo.write_text("---\ntype: news_zettel\n---\n# 7月の機械受注3.7%減\n\n機械受注が減ったかも。\n\n- 元記事: [[x]]\n", encoding="utf-8")
    before = memo.read_text(encoding="utf-8")
    state = {"clip/a.md": {"status": "written", "memo": str(memo.relative_to(vault)), "hubs": ["機械受注統計"], "hub_fit": 0.79},
             "clip/b.md": {"status": "written", "memo": "x.md", "hubs": ["倒産率とリスク"], "hub_fit": 0.9}}
    snapshot = json.dumps(state, sort_keys=True)
    seen: list[str] = []
    out = tmp_path / "recheck.json"
    result = zettel.recheck_hubs(vault, state, ["機械受注統計"], checker=lambda texts: seen.extend(texts) or [0.77], path=out)
    assert result == {"targets": 1, "checked": 1} and "内閣府の機械受注統計" in seen[0]
    item = zettel.load_recheck(out)["clip/a.md"]
    assert item["hub_fit"] == 0.77 and item["hub"] == "機械受注統計" and item["hub_use"].startswith("内閣府")
    assert memo.read_text(encoding="utf-8") == before and json.dumps(state, sort_keys=True) == snapshot


def test_industry_trend_hub_is_also_stricter():
    """REV-536: 業種別傾向も説明を絞り 0.85（個社の事例・地域の設備投資・海外の動きを寄せていた）。"""
    by_label = {hub["label"]: hub for hub in zettel.HUBS}
    hub = by_label["業種別傾向"]
    assert zettel.hub_fit_min(hub) == 0.85
    assert hub["use"].startswith("特定業種の業界全体の") and "個社の事例" in hub["use"]
    assert zettel.hub_fit_min(by_label["Q-Risk"]) == zettel.HUB_FIT_MIN
