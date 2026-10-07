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


@pytest.fixture
def vault(tmp_path):
    news = tmp_path / zettel.NEWS_DIR
    news.mkdir(parents=True)
    for day, name in [("2026-10-08", "A"), ("2026-10-07", "B"), ("2026-09-01", "C"), ("2026-08-01", "D")]:
        (news / f"{day}_業界リスクニュース_{name}.md").write_text(CLIP.replace("2026-10-08", day), encoding="utf-8")
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
