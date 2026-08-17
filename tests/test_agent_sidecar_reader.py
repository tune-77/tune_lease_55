from pathlib import Path


def test_agent_sidecar_reader_extracts_report_sections(tmp_path, monkeypatch):
    from scripts import agent_sidecar_reader as reader

    project = tmp_path / "project"
    report_root = project / ".claude" / "reports"
    report = report_root / "scoring-audit" / "latest.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        """---
agent: scoring-auditor
task: score audit
timestamp: 2026-06-08 09:00
status: success
---

## サマリー
スコアの根拠を確認した。

## 詳細
- detail

## 課題・リスク
- テストデータ混入。

## 後続エージェントへの申し送り
- data-quality-checker が確認。
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(reader, "PROJECT_ROOT", project)
    monkeypatch.setattr(reader, "REPORT_ROOT", report_root)

    reports = reader.load_sidecar_reports()
    assert len(reports) == 1
    assert reports[0].agent == "scoring-auditor"
    assert "スコアの根拠" in reports[0].summary
    assert "テストデータ" in reports[0].risks

    markdown = reader.build_markdown(reports)
    assert "read-only advisory" in markdown
    assert ".claude/reports/scoring-audit/latest.md" in markdown


def test_agent_sidecar_reader_writes_outputs(tmp_path, monkeypatch):
    from scripts import agent_sidecar_reader as reader

    project = tmp_path / "project"
    report_root = project / ".claude" / "reports"
    report = report_root / "security" / "latest.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        """---
agent: security-checker
task: security review
timestamp: 2026-06-08 10:00
status: partial
---

## サマリー
外部送信は見つからない。
""",
        encoding="utf-8",
    )
    out_md = project / "reports" / "agent_sidecar_brief.md"
    out_json = project / "reports" / "agent_sidecar_brief.json"
    monkeypatch.setattr(reader, "PROJECT_ROOT", project)
    monkeypatch.setattr(reader, "REPORT_ROOT", report_root)
    monkeypatch.setattr(reader, "OUT_MD", out_md)
    monkeypatch.setattr(reader, "OUT_JSON", out_json)

    reader.write_outputs()

    assert out_md.exists()
    assert out_json.exists()
    assert "security-checker" in out_md.read_text(encoding="utf-8")


def _report(reader, *, agent, path, stale, summary="所見テキストです。", risks=""):
    return reader.SidecarReport(
        path=path,
        agent=agent,
        task="task",
        timestamp="2026-08-16 09:00" if not stale else "2026-03-01 09:00",
        status="success",
        summary=summary,
        risks=risks,
        handoff="",
        stale=stale,
    )


def test_stale_reports_are_demoted_out_of_findings_section():
    """GAP-003: 古いレポートは所見ではなく再確認TODOへ落とす（削除はしない）。"""
    from scripts import agent_sidecar_reader as reader

    markdown = reader.build_markdown(
        [
            _report(reader, agent="scoring-auditor", path="a/latest.md", stale=False,
                    summary="最新の所見テキスト。"),
            _report(reader, agent="security-checker", path="b/latest.md", stale=True,
                    summary="古い所見テキスト。", risks="古いリスク記述。"),
        ]
    )

    findings, _, recheck = markdown.partition(reader.RECHECK_HEADING)

    # 新しい所見は所見セクションに残る
    assert "最新の所見テキスト。" in findings
    # 古い所見は所見セクションから消え、再確認TODO側にだけ現れる
    assert "古い所見テキスト。" not in findings
    assert "security-checker" not in findings
    assert "security-checker" in recheck
    # チェックリストとしての価値（当時の指摘）は保持する
    assert "古いリスク記述。" in recheck
    assert "- [ ]" in recheck


def test_all_stale_yields_empty_findings_section():
    from scripts import agent_sidecar_reader as reader

    markdown = reader.build_markdown(
        [_report(reader, agent="a", path="a/latest.md", stale=True)]
    )
    findings, _, recheck = markdown.partition(reader.RECHECK_HEADING)
    assert "No fresh reports" in findings
    assert "a/latest.md" in recheck


def test_recheck_heading_appears_only_as_a_heading():
    """見出し文字列を本文へ埋めない（素朴な分割による誤マッチを防ぐ）。"""
    from scripts import agent_sidecar_reader as reader

    for reports in (
        [_report(reader, agent="a", path="a/latest.md", stale=True)],
        [_report(reader, agent="b", path="b/latest.md", stale=False)],
        [
            _report(reader, agent="a", path="a/latest.md", stale=True),
            _report(reader, agent="b", path="b/latest.md", stale=False),
        ],
    ):
        markdown = reader.build_markdown(reports)
        assert markdown.count(reader.RECHECK_HEADING) <= 1
