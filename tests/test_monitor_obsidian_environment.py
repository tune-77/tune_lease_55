from datetime import date

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
