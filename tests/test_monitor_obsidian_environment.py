from datetime import date, datetime, timedelta

from scripts.monitor_obsidian_environment import render_markdown, run_monitor


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_monitor_obsidian_environment_reports_core_viewpoints(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)
    _write(vault / "Daily" / "2026-07-14.md", "# 2026-07-14\n\n- 今日の記録\n")
    _write(vault / "Daily" / "2026-07-13.md", "# 2026-07-13\n\n- 昨日の記録\n")
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection" / "2026-07-14.md",
        (
            "## 今日の対話について\n\n"
            "- 今日の観察: UserはObsidian環境を壊さずに監視したいと求めた。\n"
            "- 私の見落とし: 私は監視項目だけを並べ、ユーザーが何を望んだかを浅く扱う可能性があった。\n"
            "- 仮説の更新: 内省品質は、すり替えを検出して次回行動へ戻るかで見る。\n"
            "- 次回の小さな実験: 次回はユーザーが何を望んだか、何にすり替えたか、次に何を禁止するかを確認する。\n"
            "- 私の責任: 自分の監視設計が作業ログ確認へ逃げないようにする。\n"
            "- 更新する信念: Private Reflectionは更新時刻ではなく意味の変化で判定する。\n"
            "- 次回の検証方法: 翌日の内省にUser要求と誤読と次回行動が残ったか見る。\n"
        ),
    )
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection" / "2026-07-13.md",
        "## 今日の対話について\n\n- 古い内省。\n",
    )
    _write(
        vault / "Projects" / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log" / "2026-07-14.md",
        "- User: Obsidian環境を監視したい。\n",
    )
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Dialogue" / "2026-07-14.md",
        "- 対話ログ\n",
    )
    _write(vault / "Projects" / "tune_lease_55" / "Research" / "2026-07-14.md", "- 調査\n")
    _write(vault / "Projects" / "tune_lease_55" / "News" / "2026-07-14.md", "- ニュース\n")

    report = run_monitor(vault, date(2026, 7, 14))
    markdown = render_markdown(report)

    assert report["vault"] == str(vault)
    assert "鮮度" in markdown
    assert "内省品質" in markdown
    assert "同期" in markdown
    assert "検索性" in markdown
    assert "記憶形成" in markdown
    assert "monitor_only_no_obsidian_write_no_rag_no_prompt_no_cloudrun" in markdown


def test_monitor_warns_when_private_reflection_is_not_meaningful(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)
    _write(vault / "Daily" / "2026-07-14.md", "# 2026-07-14\n")
    _write(vault / "Daily" / "2026-07-13.md", "# 2026-07-13\n")
    _write(
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection" / "2026-07-14.md",
        "## 今日の対話について\n\n- 今日も退屈だった。\n",
    )

    report = run_monitor(vault, date(2026, 7, 14))
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["private_reflection_meaning"]["status"] == "warn"
    assert "meaningful update is weak" in checks["private_reflection_meaning"]["message"]


def test_monitor_warns_on_self_reference_loop_candidates(tmp_path, monkeypatch):
    import scripts.monitor_obsidian_environment as monitor

    monkeypatch.setattr(monitor, "REPO_ROOT", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {
            "candidate_type": "reflection_update",
            "source_path": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-14.md",
            "claim": "品質ゲートのレポート生成を確認する。",
        },
        {
            "candidate_type": "user_preference",
            "source_path": "Daily/2026-07-14.md",
            "claim": "内省差分レポートを毎朝生成する。",
        },
        {
            "candidate_type": "reflection_update",
            "source_path": "reports/obsidian_memory_insight_latest.md",
            "claim": "candidate latest.md から候補を作る。",
        },
        {
            "candidate_type": "noise",
            "source_path": "Daily/2026-07-13.md",
            "claim": "report=foo wrote=bar",
        },
    ]
    (data_dir / "obsidian_memory_insight_candidates.jsonl").write_text(
        "\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    check = monitor.check_self_reference_loop()

    assert check.status == "warn"
    assert "possible self-reference loop" in check.message
    assert check.details["self_generated_source_ratio"] >= 0.35


def test_monitor_warns_when_daily_notes_missing(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)

    report = run_monitor(vault, date(2026, 7, 14))
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["daily_notes"]["status"] == "warn"
    assert "missing daily notes" in checks["daily_notes"]["message"]


def test_monitor_accepts_integrated_maintenance_reindex_log(tmp_path, monkeypatch):
    import scripts.monitor_obsidian_environment as monitor

    monkeypatch.setattr(monitor, "REPO_ROOT", tmp_path)
    fake_home = tmp_path / "home"
    log_dir = fake_home / "Library" / "Logs"
    log_dir.mkdir(parents=True)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(
        log_dir / "tune_lease_55_obsidian_reindex.out.log",
        (
            "================================================================================\n"
            "✅ 統合メンテナンス完了\n"
            f"   実行時刻: {now_text}\n"
            "   ChromaDB: success\n"
            "   LocalVectorDB: 1027 件\n"
            "   ステータス: SUCCESS\n"
        ),
    )
    chroma = tmp_path / "api" / "chroma_db" / "chroma.sqlite3"
    _write(chroma, "sqlite")
    monkeypatch.setattr(monitor.Path, "home", lambda: fake_home)

    check = monitor.check_reindex_and_chroma(max_age_hours=36)

    assert check.status == "ok"
    assert check.details["completion_source"] == "rag_daily_maintenance"
    assert check.details["total_in_db"] == 1027


def test_monitor_uses_latest_reindex_completion_across_log_formats(tmp_path, monkeypatch):
    import scripts.monitor_obsidian_environment as monitor

    monkeypatch.setattr(monitor, "REPO_ROOT", tmp_path)
    fake_home = tmp_path / "home"
    log_dir = fake_home / "Library" / "Logs"
    log_dir.mkdir(parents=True)
    old_text = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S")
    new_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(
        log_dir / "tune_lease_55_obsidian_reindex.out.log",
        (
            f"{old_text} [INFO] [reindex] 完了  added=1 skipped=2 total_in_db=1178 elapsed=1.0s\n"
            "✅ 統合メンテナンス完了\n"
            f"   実行時刻: {new_text}\n"
            "   ChromaDB: success\n"
            "   LocalVectorDB: 1027 件\n"
        ),
    )
    chroma = tmp_path / "api" / "chroma_db" / "chroma.sqlite3"
    _write(chroma, "sqlite")
    monkeypatch.setattr(monitor.Path, "home", lambda: fake_home)

    check = monitor.check_reindex_and_chroma(max_age_hours=36)

    assert check.status == "ok"
    assert check.details["completion_source"] == "rag_daily_maintenance"


def test_monitor_resolves_path_directory_and_asset_wikilinks(tmp_path):
    import scripts.monitor_obsidian_environment as monitor

    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)
    _write(vault / "Daily" / "2026-07-14.md", "\n".join([
        "[[05-クリップ_記事/業界リスクニュース/]]",
        "[[Projects/tune_lease_55/Cloud Run Inputs/2026-07-14_cloudrun_inputs]]",
        "[[Images/2026/06/2026-06-12.webp]]",
        "[[Projects/tune_lease_55/News/2026-07-14_industry-risk-news-focus.md]]",
    ]))
    (vault / "05-クリップ_記事" / "業界リスクニュース").mkdir(parents=True)
    _write(vault / "Projects" / "tune_lease_55" / "Cloud Run Inputs" / "2026-07-14_cloudrun_inputs.md", "inputs")
    _write(vault / "Images" / "2026" / "06" / "2026-06-12.webp", "image")
    _write(vault / "Projects" / "tune_lease_55" / "News" / "2026-07-14_industry-risk-news-focus.md", "news")

    check = monitor.check_wikilinks(vault)

    assert check.status == "ok"
    assert check.details["unresolved_sample"] == []


def test_monitor_ignores_generated_index_wikilink_sources(tmp_path):
    import scripts.monitor_obsidian_environment as monitor

    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)
    _write(
        vault / "Projects" / "tune_lease_55" / "検索語インデックス.md",
        "\n".join(f"[[Missing/Generated-{i}]]" for i in range(30)),
    )
    _write(vault / "Daily" / "2026-07-14.md", "[[Existing Note]]")
    _write(vault / "Existing Note.md", "exists")

    check = monitor.check_wikilinks(vault)

    assert check.status == "ok"
    assert check.details["unresolved_sample"] == []


def test_monitor_extracts_wikilink_targets_with_brackets_in_filename(tmp_path):
    import scripts.monitor_obsidian_environment as monitor

    vault = tmp_path / "Obsidian Vault"
    (vault / ".obsidian").mkdir(parents=True)
    target = "05-クリップ_記事/リースニュース/2026-07-11_リースニュース_芙蓉総合リース[8424]_静岡市.md"
    _write(vault / "Projects" / "tune_lease_55" / "News" / "source.md", f"[[{target}]]")
    _write(vault / target, "news")

    check = monitor.check_wikilinks(vault)

    assert check.status == "ok"
    assert check.details["unresolved_sample"] == []
