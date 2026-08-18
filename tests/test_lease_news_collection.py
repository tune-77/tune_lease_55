from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import lease_news_digest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_lease_news_to_obsidian.py"
_SPEC = importlib.util.spec_from_file_location("collect_lease_news_to_obsidian", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
news = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = news
_SPEC.loader.exec_module(news)


def _article(**overrides):
    values = {
        "title": "建設会社がAI導入で事務作業を効率化",
        "link": "https://example.com/news/1?utm_source=test",
        "source": "Example News",
        "published": dt.datetime(2026, 6, 6, tzinfo=dt.timezone.utc),
        "summary": "建設会社がAIを導入し、事務作業の削減を進める。",
        "query": "建設業 倒産",
        "theme": "製造・DX",
        "tags": ("建設/不動産", "製造/DX"),
        "score": 2,
    }
    values.update(overrides)
    return news.Article(**values)


def test_rule_classification_populates_searchable_fields():
    article = _article()

    news.classify_articles([article], use_ai=False)

    assert "建設業" in article.industries
    assert "建設機械" in article.lease_assets
    assert article.impact_direction == "positive"
    assert article.source_reliability == "medium"
    assert article.valid_until == "2027-06-06"
    assert article.classification_source == "rule"


def test_article_content_contains_ai_screening_classification():
    article = _article()
    news.classify_articles([article], use_ai=False)

    content = news._build_article_content(
        article,
        date_str="2026-06-06",
        week="2026-W23",
        month="2026-06",
        profile="industry-watch",
    )

    assert 'industries: "建設業, 不動産業, 製造業"' in content
    assert "impact_direction: positive" in content
    assert "valid_until: 2027-06-06" in content
    assert 'canonical_url: "https://example.com/news/1"' in content
    assert "## AI審査分類" in content
    assert "### 審査上の確認事項" in content


def test_duplicate_article_is_merged_into_related_reports(tmp_path):
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    first = _article()
    second = _article(
        link="https://another.example.com/story/99",
        source="Second Source",
    )
    news.classify_articles([first, second], use_ai=False)

    saved_first = news._save_articles_to_obsidian(
        [first],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )
    saved_second = news._save_articles_to_obsidian(
        [second],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )

    files = list((vault / news_dir).glob("*.md"))
    assert len(saved_first) == 1
    assert saved_second == saved_first
    assert len(files) == 1
    merged = files[0].read_text(encoding="utf-8")
    assert "## 関連報道" in merged
    assert "Second Source" in merged
    assert "https://another.example.com/story/99" in merged


def test_exact_duplicate_does_not_append_twice(tmp_path):
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    article = _article()
    news.classify_articles([article], use_ai=False)

    news._save_articles_to_obsidian([article], vault, news_dir, "2026-06-06", "industry-watch")
    second_save = news._save_articles_to_obsidian(
        [article],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )

    assert second_save == []
    content = next((vault / news_dir).glob("*.md")).read_text(encoding="utf-8")
    assert "## 関連報道" not in content


def test_daily_digest_shows_fresh_content_after_related_report_merge(tmp_path):
    """続報がマージされた日、ダイジェストは新しい記事内容を返すべき（date更新だけでは不十分）。

    続報かどうかの類似度判定（_find_duplicate）は別テストで担保済みのため、ここでは
    実際に本番で使われる _load_existing_news / _merge_related_report /
    build_daily_news_digest を直結してマージ後の見え方だけを検証する。
    """
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"

    first = _article(
        title="設備投資が拡大",
        link="https://example.com/story/1",
        summary="設備投資が拡大している。背景に円安がある。",
    )
    news.classify_articles([first], use_ai=False)
    news._save_articles_to_obsidian([first], vault, news_dir, "2026-06-06", "industry-watch")

    # 数日後、同一トピックの続報が来て既存ノートへマージされる。
    second = _article(
        title="設備投資が一段と加速、補助金追い風",
        link="https://another.example.com/story/2",
        summary="設備投資が一段と加速している。補助金の後押しが大きい。",
    )
    news.classify_articles([second], use_ai=False)

    existing = news._load_existing_news(vault, news_dir)
    assert len(existing) == 1
    record = existing[0]
    merged = news._merge_related_report(record, second, "2026-06-09", "2026-W24", "2026-06")
    assert merged is True

    files = list((vault / news_dir).glob("*.md"))
    assert len(files) == 1, "続報は新規ノートではなく既存ノートへマージされるべき"

    digest = lease_news_digest.build_daily_news_digest(date_str="2026-06-09", vault=vault, limit=5)

    assert digest["available"] is True
    assert digest["date"] == "2026-06-09"
    item = digest["items"][0]
    assert item["title"] == second.title
    assert any("補助金の後押し" in line for line in item["summary_lines"])
    assert "円安" not in item["title"]
    assert not any("円安" in line for line in item["summary_lines"])


def test_news_action_reflects_escalated_risk_after_related_report_merge(tmp_path):
    """続報でトピックの実質的なリスク種別が変わったら、region/importance/tags/活用メモ/
    リンクも当日内容へ更新され、審査アクション推論(_infer_news_action)が新しいリスクを
    拾うべき（タイトル/3行要約の更新だけでは不十分）。
    """
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"

    first = _article(
        title="設備投資が拡大",
        link="https://example.com/story/10",
        source="日経",
        summary="設備投資が拡大している。背景に円安がある。",
        tags=("設備投資",),
        score=1,
    )
    news.classify_articles([first], use_ai=False)
    news._save_articles_to_obsidian([first], vault, news_dir, "2026-08-10", "industry-watch")

    # 続報で、実は金利上昇による資金繰り悪化という信用リスクの強い話に発展した。
    second = _article(
        title="設備投資向け融資に急ブレーキ、金利上昇で資金繰り悪化",
        link="https://reuters.example.com/story/11",
        source="Reuters",
        summary="金利上昇で資金繰りが悪化している。倒産増加の懸念も出ている。",
        tags=("金利", "与信"),
        score=2,
    )
    news.classify_articles([second], use_ai=False)

    existing = news._load_existing_news(vault, news_dir)
    assert len(existing) == 1
    record = existing[0]
    merged = news._merge_related_report(record, second, "2026-08-15", "2026-W33", "2026-08")
    assert merged is True

    parsed = lease_news_digest._parse_news_note(Path(record["path"]))
    assert parsed["region"] == "米国", "続報の発信元(Reuters)を踏まえた地域へ更新されるべき"
    assert parsed["importance"] == "高", "続報のスコア・タグを踏まえた重要度へ更新されるべき"
    assert parsed["tags"] == ["金利", "与信"]
    assert "金利" in parsed["usage_memo"], "活用メモが続報のタグに基づく内容へ更新されるべき"
    assert parsed["article_url"] == second.link, "詳細セクションのリンクが最新記事へ更新されるべき"

    action = lease_news_digest._infer_news_action(parsed)
    assert "金利負担・返済余力" in action.risk_flags, (
        "region/importance/tagsが古いままだと、審査アクション推論が続報の信用リスクを拾えない"
    )


def _collect_day(vault, news_dir, articles, date_str, week, month):
    """1日分の収集を本番と同じ経路（重複判定→新規作成 or 続報マージ）で再現する。"""
    news.classify_articles(articles, use_ai=False)
    return news._save_articles_to_obsidian(articles, vault, news_dir, date_str, "industry-watch")


def _merge_followup(vault, news_dir, article, date_str, week, month):
    """既存ノートへの続報マージを明示的に起こす。

    見出しが大きく変わる続報は _find_duplicate の類似度しきい値に届かず新規ノートに
    なるため、マージ経路そのものを検証したいテストでは明示的に呼ぶ
    （類似度判定側は test_duplicate_article_is_merged_into_related_reports が担保）。
    """
    news.classify_articles([article], use_ai=False)
    existing = news._load_existing_news(vault, news_dir)
    assert len(existing) == 1, "マージ対象の既存ノートが1件だけの前提"
    assert news._merge_related_report(existing[0], article, date_str, week, month) is True
    return Path(existing[0]["path"])


def test_digest_content_changes_across_days_including_merge_only_day(tmp_path):
    """原症状「ニュースダイジェストが毎日同じ」の直接的な回帰テスト。

    3日分を実際に収集し、日ごとにダイジェストの中身が変わることを確認する。
    2日目は全記事が既存ノートへの続報マージになる日（新規ファイルが1つも増えない日）で、
    ここが過去2回のバグの震源地だった。
    """
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"

    # 1日目: 新規ノートが作られる
    _collect_day(
        vault, news_dir,
        [_article(title="設備投資が拡大", link="https://a.example.com/1",
                  summary="設備投資が拡大している。背景に円安がある。")],
        "2026-06-01", "2026-W23", "2026-06",
    )
    day1 = lease_news_digest.build_daily_news_digest(date_str="2026-06-01", vault=vault, limit=5)

    # 2日目: 同一トピックの続報が既存ノートへマージされる（新規ファイルは増えない）
    _merge_followup(
        vault, news_dir,
        _article(title="設備投資が一段と加速、補助金追い風", link="https://b.example.com/2",
                 summary="設備投資が一段と加速している。補助金の後押しが大きい。"),
        "2026-06-02", "2026-W23", "2026-06",
    )
    assert len(list((vault / news_dir).glob("*.md"))) == 1, "2日目は続報マージで新規ファイルが増えない"
    day2 = lease_news_digest.build_daily_news_digest(date_str="2026-06-02", vault=vault, limit=5)

    # 3日目: さらに続報が重なる
    _merge_followup(
        vault, news_dir,
        _article(title="設備投資に急ブレーキ、金利上昇で資金繰り悪化", link="https://c.example.com/3",
                 summary="金利上昇で資金繰りが悪化している。倒産増加の懸念も出ている。"),
        "2026-06-03", "2026-W23", "2026-06",
    )
    assert len(list((vault / news_dir).glob("*.md"))) == 1
    day3 = lease_news_digest.build_daily_news_digest(date_str="2026-06-03", vault=vault, limit=5)

    for day, expected in ((day1, "2026-06-01"), (day2, "2026-06-02"), (day3, "2026-06-03")):
        assert day["available"] is True
        assert day["date"] == expected
        assert day["is_stale"] is False, "各日とも当日分が存在するのでフォールバックしない"

    titles = [d["items"][0]["title"] for d in (day1, day2, day3)]
    assert len(set(titles)) == 3, f"日ごとにダイジェストの内容が変わるべき: {titles}"


def test_digest_marks_stale_when_todays_news_missing(tmp_path):
    """当日分が無い日は、過去ノートを当日分のように黙って返さない。

    収集が止まっていても「毎日同じ内容」が普通に表示されるため、原因が
    収集停止なのか更新漏れなのか区別できなかった。
    """
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    _collect_day(
        vault, news_dir,
        [_article(title="設備投資が拡大", link="https://a.example.com/1",
                  summary="設備投資が拡大している。背景に円安がある。")],
        "2026-06-01", "2026-W23", "2026-06",
    )

    # 4日後、収集が動いていない状態でダイジェストを引く
    digest = lease_news_digest.build_daily_news_digest(date_str="2026-06-05", vault=vault, limit=5)

    assert digest["available"] is True
    assert digest["is_stale"] is True, "当日分が無いことを明示すべき"
    assert digest["requested_date"] == "2026-06-05"
    assert digest["date"] == "2026-06-01"
    assert digest["stale_days"] == 4

    text = lease_news_digest.daily_news_digest_as_text(date_str="2026-06-05", vault=vault, limit=5)
    assert "収集されていません" in text, "朝報テキストでも古いデータであることを伝えるべき"
    assert "4日前" in text


def test_latest_news_note_prefers_frontmatter_date_over_filename(tmp_path):
    """続報マージ後のノートはファイル名が古いままでも「最新ノート」として選ばれる。

    _latest_news_note がファイル名の日付で並べていたため、当日更新された
    （ファイル名は古い）ノートが、数日前に作られた別ノートに負けていた。
    ホーム画面と注目論点(/api/lease-news/focus)がこの経路を使う。
    """
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    _collect_day(
        vault, news_dir,
        [_article(title="継続トピック", link="https://a.example.com/1",
                  summary="継続しているトピック。初回の内容。")],
        "2026-06-01", "2026-W23", "2026-06",
    )
    # 別トピックのノートが後日作られる（ファイル名は新しい）
    _collect_day(
        vault, news_dir,
        [_article(title="別トピックの単発ニュース", link="https://z.example.com/9",
                  summary="別トピックの単発。これは続報が来ない。")],
        "2026-06-02", "2026-W23", "2026-06",
    )
    # 継続トピックに当日の続報が来る → 古いファイル名のノートの frontmatter だけ当日へ
    news.classify_articles([_followup := _article(
        title="継続トピックに新展開", link="https://a2.example.com/2",
        summary="継続しているトピック。新しい展開があった。")], use_ai=False)
    target = next(
        record for record in news._load_existing_news(vault, news_dir)
        if Path(record["path"]).name.startswith("2026-06-01")
    )
    assert news._merge_related_report(target, _followup, "2026-06-03", "2026-W23", "2026-06") is True

    latest = lease_news_digest._latest_news_note(vault)

    assert latest is not None
    assert latest.name.startswith("2026-06-01"), "ファイル名は初回作成日のまま据え置かれる"
    assert lease_news_digest._note_frontmatter_date(latest) == "2026-06-03"
    focus = lease_news_digest.get_latest_lease_news_focus(vault=vault)
    assert focus.note_date == "2026-06-03", "注目論点が当日更新されたノートを掴むべき"
