from __future__ import annotations

import json
import datetime as dt
import os


def test_agent_worklog_digest_extracts_public_summary_sections(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    daily.mkdir(parents=True)
    note_date = dt.date.today().isoformat()
    note = daily / f"{note_date}.md"
    note.write_text(
        """
# {note_date}

## 10:15 Codex Work Log

### Summary
- FAQ導線をサポートハブへ寄せた

### Chat Summary
- Userは低利用を不要ではなく文脈不足として見たいと判断した

### Decisions
- FAQは削除せず、審査画面と紫苑対話室から導線を追加する

### Changes
- frontend/src/app/help/page.tsx

### Verification
- npm run typecheck

### Git
- commit: abc

## 11:30 Claude Work Log

### Summary
- 自己提案の判定軸をレビュー

### Private
- これは出してはいけない
""".format(note_date=note_date).strip(),
        encoding="utf-8",
    )

    parsed = digest.parse_work_logs(note)
    result = digest.build_digest(vault, days=1, limit=5)

    assert len(parsed) == 2
    assert result["count"] == 2
    first = result["items"][0]
    assert first["agent"] == "Claude"
    assert first["summary"] == ["自己提案の判定軸をレビュー"]
    second = result["items"][1]
    assert second["agent"] == "Codex"
    assert "文脈不足" in second["chat_summary"][0]
    assert "これは出してはいけない" not in json.dumps(result, ensure_ascii=False)
    assert result["policy"]["raw_chat_logs_excluded"] is True


def test_build_digest_reports_note_files_scanned(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    daily.mkdir(parents=True)
    for offset in range(3):
        day = dt.date.today() - dt.timedelta(days=offset)
        (daily / f"{day.isoformat()}.md").write_text("# no work logs here\n", encoding="utf-8")

    result = digest.build_digest(vault, days=3, limit=5)

    assert result["note_files_scanned"] == 3
    assert result["source_count"] == 0


def test_build_digest_reads_project_work_log_format(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    worklogs = vault / "Projects" / "tune_lease_55" / "Work Logs"
    worklogs.mkdir(parents=True)
    note_date = dt.date.today().isoformat()
    (worklogs / f"{note_date}.md").write_text(
        """
---
date: 2026-09-28
type: work_log
---

## 作業: パイプライン障害の復旧

### 何をしたか

記憶ヘルスチェックと作業録ダイジェストを修正した。

### 検証

- 対象テストが成功した
""".strip(),
        encoding="utf-8",
    )

    result = digest.build_digest(vault, days=1, limit=5)

    assert result["note_files_scanned"] == 1
    assert result["source_count"] == 1
    assert "パイプライン障害" in result["items"][0]["summary"][0]
    assert result["items"][0]["verification"] == ["対象テストが成功した"]


def test_project_work_log_keeps_appended_tasks_separate(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    note = tmp_path / "2026-09-28.md"
    note.write_text(
        """
## 作業: 最初の修正

### 何をしたか
最初の変更を実装した。

## 作業: 二番目の修正

### 何をしたか
二番目の変更を実装した。

### 次回どう切り分けるか
次はログから確認する。
""".strip(),
        encoding="utf-8",
    )

    logs = digest.parse_project_work_log(note)

    assert len(logs) == 2
    assert logs[0]["sections"]["Summary"] == ["作業: 最初の修正", "最初の変更を実装した。"]
    assert logs[1]["sections"]["Summary"] == ["作業: 二番目の修正", "二番目の変更を実装した。"]
    assert logs[1]["sections"]["Open Items"] == ["次はログから確認する。"]


def test_build_digest_prefers_newest_appended_project_tasks(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    worklogs = vault / "Projects" / "tune_lease_55" / "Work Logs"
    worklogs.mkdir(parents=True)
    note_date = dt.date.today().isoformat()
    blocks = [f"## 作業: task-{index}\n\n### 何をしたか\nchange-{index}" for index in range(13)]
    (worklogs / f"{note_date}.md").write_text("\n\n".join(blocks), encoding="utf-8")

    result = digest.build_digest(vault, days=1, limit=12)

    assert result["source_count"] == 13
    assert result["items"][0]["summary"][0] == "作業: task-12"
    assert result["items"][-1]["summary"][0] == "作業: task-1"


def test_build_digest_compares_project_mtime_with_daily_log_time(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    worklogs = vault / "Projects" / "tune_lease_55" / "Work Logs"
    daily.mkdir(parents=True)
    worklogs.mkdir(parents=True)
    note_date = dt.date.today().isoformat()
    (daily / f"{note_date}.md").write_text(
        "## 10:00 Codex Work Log\n\n### Summary\n- Daily側",
        encoding="utf-8",
    )
    project_note = worklogs / f"{note_date}.md"
    project_note.write_text(
        "## 作業: Project側\n\n### 何をしたか\n新しい作業",
        encoding="utf-8",
    )
    project_timestamp = dt.datetime.combine(dt.date.today(), dt.time(11, 0)).timestamp()
    os.utime(project_note, (project_timestamp, project_timestamp))

    result = digest.build_digest(vault, days=1, limit=1)

    assert result["items"][0]["summary"][0] == "作業: Project側"
    assert result["items"][0]["time"] == "11:00"


def test_main_detects_project_format_drift_even_when_daily_parser_succeeds(tmp_path, monkeypatch, capsys):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    worklogs = vault / "Projects" / "tune_lease_55" / "Work Logs"
    daily.mkdir(parents=True)
    worklogs.mkdir(parents=True)
    for offset in range(3):
        day = dt.date.today() - dt.timedelta(days=offset)
        (worklogs / f"{day.isoformat()}.md").write_text("## 別形式の作業ログ\n", encoding="utf-8")
    (daily / f"{dt.date.today().isoformat()}.md").write_text(
        "## 10:00 Codex Work Log\n\n### Summary\n- Daily側は正常",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_agent_worklog_digest.py",
            "--vault", str(vault),
            "--days", "3",
            "--json", str(tmp_path / "out.json"),
            "--md", str(tmp_path / "out.md"),
        ],
    )

    assert digest.main() == 1
    assert "Projects/tune_lease_55/Work Logs" in capsys.readouterr().err


def test_main_warns_and_exits_nonzero_when_heading_format_drifts(tmp_path, monkeypatch, capsys):
    """WORKLOG_HEADING_RE がドリフトして日次ノートはあるのに作業録が0件の場合、
    無条件exit 0のまま無音停止しないことを確認する回帰テスト。"""
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    daily.mkdir(parents=True)
    for offset in range(3):
        day = dt.date.today() - dt.timedelta(days=offset)
        (daily / f"{day.isoformat()}.md").write_text(
            "## 10:00 何らかの別形式ログ\n- 中身\n", encoding="utf-8"
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_agent_worklog_digest.py",
            "--vault", str(vault),
            "--days", "3",
            "--json", str(tmp_path / "out.json"),
            "--md", str(tmp_path / "out.md"),
        ],
    )

    exit_code = digest.main()

    assert exit_code == 1
    assert "書式ドリフト" in capsys.readouterr().err


def test_main_returns_zero_when_too_few_notes_to_judge_drift(tmp_path, monkeypatch, capsys):
    """日次ノートが閾値未満なら、単に静かな期間として扱い誤検知しない。"""
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    daily.mkdir(parents=True)
    (daily / f"{dt.date.today().isoformat()}.md").write_text("# 何もなし\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_agent_worklog_digest.py",
            "--vault", str(vault),
            "--days", "1",
            "--json", str(tmp_path / "out.json"),
            "--md", str(tmp_path / "out.md"),
        ],
    )

    assert digest.main() == 0
