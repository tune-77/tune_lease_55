from __future__ import annotations

import json
import datetime as dt


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
