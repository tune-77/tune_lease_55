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


def _fake_gemini(monkeypatch, finish_reason: str, text: str) -> dict:
    from google import genai

    captured: dict = {}

    class _Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            candidate = type("Candidate", (), {"finish_reason": finish_reason})()
            return type("Response", (), {"candidates": [candidate], "text": text, "usage_metadata": None})()

    class _Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    monkeypatch.setattr(genai, "Client", _Client)
    monkeypatch.setattr(news, "_get_gemini_key", lambda: "test-key")
    monkeypatch.setattr(news, "_news_guard_actions", lambda articles: ["send"] * len(articles))
    return captured


def test_gemini_classification_disables_thinking_budget(monkeypatch):
    captured = _fake_gemini(
        monkeypatch,
        "STOP",
        '{"classifications": [{"article_index": 0, "industries": ["建設業"], "lease_assets": [],'
        ' "credit_risk_impact": "x", "screening_checks": [], "impact_direction": "neutral",'
        ' "classification_confidence": 0.8, "source_reliability": "medium",'
        ' "valid_until": "2027-01-01", "canonical_topic": "t"}]}',
    )
    article = _article()

    news.classify_articles([article])

    assert captured["config"].thinking_config.thinking_budget == 0
    assert article.classification_source == "gemini"


def test_truncated_gemini_output_keeps_rule_fallback(monkeypatch, capsys):
    _fake_gemini(monkeypatch, "FinishReason.MAX_TOKENS", '{"classifications": [{"article_index": 0, "industr')
    article = _article()

    news.classify_articles([article])

    assert article.classification_source == "rule"
    assert "truncated" in capsys.readouterr().err


def test_rule_classification_distinguishes_official_and_weak_sources():
    official = _article(
        source="pref.example.lg.jp",
        link="https://www.pref.example.lg.jp/subsidy",
        title="中小企業向け設備補助金",
    )
    weak = _article(
        source="比較サイト",
        title="おすすめカーリースランキング",
    )

    news.classify_articles([official, weak], use_ai=False)

    assert official.source_reliability == "high"
    assert weak.source_reliability == "low"


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


def test_daily_digest_prefers_note_date_over_gcs_download_mtime(tmp_path):
    """GCS一括同期で古いファイルのmtimeが新しくなっても、日付で今日分を選ぶ。"""
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)

    fresh = news_dir / "2026-08-24_業界リスクニュース_今日の設備投資.md"
    fresh.write_text(
        "\n".join(
            [
                "---",
                "date: 2026-08-24",
                "title: 今日の設備投資",
                "source: 日刊工業新聞",
                "tags: [設備投資]",
                "---",
                "## 3行要約",
                "- 今日分の設備投資ニュース。",
            ]
        ),
        encoding="utf-8",
    )

    for idx in range(50):
        old = news_dir / f"2026-06-06_業界リスクニュース_古いニュース{idx:02d}.md"
        old.write_text(
            "\n".join(
                [
                    "---",
                    "date: 2026-06-06",
                    f"title: 古いニュース{idx:02d}",
                    "source: 古い媒体",
                    "---",
                    "## 3行要約",
                    "- 古いニュース。",
                ]
            ),
            encoding="utf-8",
        )

    digest = lease_news_digest.build_daily_news_digest(date_str="2026-08-24", vault=vault, limit=3)

    assert digest["available"] is True
    assert digest["is_stale"] is False
    assert digest["date"] == "2026-08-24"
    assert "今日の設備投資" in digest["items"][0]["title"]


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
    assert parsed["industries"] == list(second.industries)
    assert parsed["lease_assets"] == list(second.lease_assets)
    assert parsed["impact_direction"] == second.impact_direction
    assert parsed["source_reliability"] == second.source_reliability
    assert parsed["classification_confidence"] == second.classification_confidence
    assert parsed["valid_until"] == second.valid_until
    assert parsed["screening_checks"] == list(second.screening_checks)
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


def test_find_vault_refreshes_cloudrun_gcs_vault(monkeypatch, tmp_path):
    vault = tmp_path / "gcs_vault"
    vault.mkdir()
    calls: list[Path] = []

    def fake_download_vault(*, dest_dir):
        calls.append(dest_dir)
        return vault

    monkeypatch.setenv("USE_GCS_VAULT", "true")
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.setattr(lease_news_digest, "_GCS_VAULT_LAST_SYNC", 0.0)
    monkeypatch.setattr("scripts.gcs_vault_loader.download_vault", fake_download_vault)

    result = lease_news_digest.find_vault()

    assert result == vault
    assert calls == [lease_news_digest._GCS_VAULT_LOCAL_DIR]
    assert lease_news_digest.find_vault() == vault


# 見出しは 2026-09〜10 の lease-news-collector 実ログから採った同一出来事／別記事の組
_SAME_EVENT_PAIRS = [
    (
        "中小企業の資金繰り支援へ 三重県信用保証協会と三十三銀行が提携商品を創設(三重テレビ放送) - Yahoo!ニュース",
        "中小企業の資金繰り支援 超長期保証商品を創設 県信用保証協会と三十三銀行 - 伊勢新聞",
    ),
    ("大勝建設(株)ほか1社 | TSR速報 | 倒産・注目企業情報 - 東京商工リサーチ", "大勝建設株式会社など2社 - tdb.co.jp"),
    (
        "地域小規模運送業者・振興物産株式会社(鳥取県岩美町)の破産手続き開始 2026.09.25",
        "貨物自動車運送業者「振興物産」破産手続き開始決定 負債は約1億8000万円 鳥取県岩美町 (日本海テレビ)",
    ),
    ("工作機械受注、8月64%増 北米アジア伸び歴代2位 - 日刊工業新聞", "8月の工作機械受注 64%増 北米アジア伸び歴代2位 - 日刊工業新聞"),
    ("8月工作機械受注は前年比64.7%増、14カ月連続プラス=工作機械工業会", "8月の工作機械受注 64.7%増 14カ月連続プラス"),
]
_DIFFERENT_EVENT_PAIRS = [
    (
        "特集・食品工場のスマートファクトリー化と省力化・自動化2026:FOOMAセミナー - 日本食糧新聞",
        "特集・食品工場のスマートファクトリー化と省力化・自動化2026:解説2=OTセキュリティー - 日本食糧新聞",
    ),
    ("車向け工作機械受注、8月は44%増 老朽機更新で大型案件 - 日本経済新聞", "工作機械受注、8月64%増 北米アジア伸び歴代2位 - 日刊工業新聞"),
    ("8月の工作機械受注65%増 AI関連・航空宇宙が好調 - 日本経済新聞", "8月の工作機械受注 64%増 北米アジア伸び歴代2位 - 日刊工業新聞"),
    (
        "【倒産情報】負債総額は45億円...創業79年の総合建設業が破産申請へ",
        "【倒産速報】売上高が10.5億円→3.1億円に... 土地売買・建築工事会社が事業停止→自己破産申請へ",
    ),
    ("100年経営「老舗企業」の倒産動向調査(2026年1-8月) - tdb.co.jp", "老舗倒産112件“過去最多”ペース...「100年企業」に何が? 製造・卸売・小売で相次ぐ"),
]


def test_same_event_headlines_with_different_wording_are_merged():
    for a, b in _SAME_EVENT_PAIRS:
        assert news._same_event(_article(title=a, source=""), _article(title=b, source="")), (a, b)


def test_distinct_events_and_series_installments_are_not_merged():
    for a, b in _DIFFERENT_EVENT_PAIRS:
        assert not news._same_event(_article(title=a, source=""), _article(title=b, source="")), (a, b)


def test_merge_keeps_first_as_representative_and_other_urls_as_related(tmp_path):
    rep = _article(title=_SAME_EVENT_PAIRS[0][0], link="https://news.yahoo.co.jp/a/1", source="Yahoo!ニュース")
    dup = _article(title=_SAME_EVENT_PAIRS[0][1], link="https://www.isenp.co.jp/b/2", source="伊勢新聞")
    other = _article(title=_DIFFERENT_EVENT_PAIRS[0][0], link="https://example.com/c/3")

    merged = news.merge_same_event_articles([rep, dup, other])

    assert merged == [rep, other]
    assert rep.related == (dup,)
    news.classify_articles(merged, use_ai=False)
    news._save_articles_to_obsidian(merged, tmp_path, "news", "2026-10-01", "industry-watch")
    note = next(p for p in (tmp_path / "news").glob("*.md") if "三重" in p.name or "資金繰り" in p.name)
    text = note.read_text(encoding="utf-8")
    assert "## 関連報道" in text
    assert "https://www.isenp.co.jp/b/2" in text


def _fake_jev(probabilities):
    def request(payload):
        count = len(payload["state"]["pairs"])
        return {"model": "jev-test", "answers": {f"pair{n}_same": {"type": "noul", "noul": probabilities(payload["state"]["pairs"][n])} for n in range(count)}}

    return request


def test_same_event_shadow_records_without_changing_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(tmp_path / "jev.jsonl"))
    a = _article(title="建設業の倒産/廃業、リーマン超えで過去最多 2026年上半期:調査レポート - ITmedia", source="")
    b = _article(title="建設業の倒産・廃業が最多の5937件 1〜6月、帝国データ調べ - 日本経済新聞", source="")
    reps = news.merge_same_event_articles([a, b])
    assert reps == [a, b]  # 既存ルールは別記事のまま

    result = news.shadow_same_event_jev(reps, request_fn=_fake_jev(lambda pair: 0.9))

    assert result["status"] == "applied" and result["would_merge"] == 1
    assert reps == [a, b] and a.related == ()  # shadow なので統合はしない
    record = __import__("json").loads((tmp_path / "jev.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["guard"] == "news_same_event" and record["mode"] == "shadow" and record["route"] == "would_merge"
    assert "建設業" not in (tmp_path / "jev.jsonl").read_text(encoding="utf-8")  # 見出しはハッシュのみ


def test_same_event_shadow_is_off_with_news_guard_and_fails_open(monkeypatch):
    reps = [_article(title=_SAME_EVENT_PAIRS[0][0], source=""), _article(title=_DIFFERENT_EVENT_PAIRS[0][0], source="")]
    monkeypatch.delenv("TYPESAFE_NEWS_MODE", raising=False)
    assert news.shadow_same_event_jev(reps)["status"] == "disabled"

    def broken(payload):
        raise TimeoutError

    assert news.shadow_same_event_jev(reps, request_fn=broken)["status"] in {"fallback", "skipped"}
