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
