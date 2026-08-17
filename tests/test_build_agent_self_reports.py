"""決定論的セルフレポートのテスト。

LLM も外部プロセスも使わないことが前提なので、監査本体は素直に検証できる。
"""

import pytest

from scripts import build_agent_self_reports as selfrep


def test_report_dirs_are_documented_in_agents_md():
    """出力先が .claude/AGENTS.md の記載と一致すること（新設ディレクトリを作らない）。"""
    import re

    agents_md = (selfrep.PROJECT_ROOT / ".claude" / "AGENTS.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`([a-z-]+)/latest\.md`", agents_md))

    assert documented, "AGENTS.md からレポートディレクトリを抽出できていない"
    assert set(selfrep.CHECKS) - documented == set()


def test_render_report_follows_report_schema():
    result = selfrep.CheckResult(
        agent="rule-validator",
        task="整合性チェック",
        status="partial",
        summary="逸脱を1件検出した。",
        details=["カテゴリ数: 4"],
        risks=["IT機器: ウェイト合計 95"],
        handoff="定義を確認すること。",
    )
    md = selfrep.render_report(result)

    assert md.startswith("---\n")
    for key in ("agent:", "task:", "timestamp:", "status:", "reads_from:"):
        assert key in md
    for section in ("## サマリー", "## 詳細", "## 課題・リスク", "## 後続エージェントへの申し送り"):
        assert section in md
    assert "IT機器: ウェイト合計 95" in md
    # 自動生成であることを機械判定できる印を必ず入れる（退避判定に使う）
    assert selfrep.SELF_MARKER in md


def test_render_report_is_parseable_by_sidecar_reader():
    """書式が sidecar のパーサと噛み合うこと（片方だけ直る事故を防ぐ）。"""
    from scripts import agent_sidecar_reader as reader

    result = selfrep.CheckResult(
        agent="scoring-auditor",
        task="スコア範囲チェック",
        status="success",
        summary="全カテゴリが範囲内。",
        details=["検査カテゴリ数: 4"],
        risks=[],
        handoff="",
    )
    meta, body = reader._parse_frontmatter(selfrep.render_report(result))

    assert meta["agent"] == "scoring-auditor"
    assert meta["status"] == "success"
    assert meta["timestamp"]
    assert reader._extract_section(body, "サマリー") == "全カテゴリが範囲内。"
    assert reader._extract_section(body, "課題・リスク") == "- なし"


def test_empty_risks_render_as_none():
    result = selfrep.CheckResult(agent="a", task="t", summary="ok")
    md = selfrep.render_report(result)
    assert "## 課題・リスク\n- なし" in md


def test_fail_marks_failure_and_records_reason():
    result = selfrep.CheckResult(agent="a", task="t").fail("import できない")
    assert result.status == "failure"
    assert "import できない" in result.risks[0]
    assert "import できない" in result.summary


def test_import_failure_is_reported_not_swallowed(monkeypatch):
    """依存欠落時に「異常なし」と書かないこと。偽の成功が最悪の失敗。"""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "category_config":
            raise ModuleNotFoundError("No module named 'category_config'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    result = selfrep.check_rules()

    assert result.status == "failure"
    assert "成功" not in result.summary
    assert any("category_config" in r for r in result.risks)


def test_missing_db_is_reported_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(selfrep, "DB_PATH", tmp_path / "absent.db")
    result = selfrep.check_data_quality()
    assert result.status == "failure"
    assert result.risks


def test_missing_logs_dir_is_not_an_error(monkeypatch, tmp_path):
    """ログが無いのは異常ではない。無い事実だけ書く。"""
    monkeypatch.setattr(selfrep, "LOGS_DIR", tmp_path / "absent")
    result = selfrep.check_logs()
    assert result.status == "success"
    assert result.risks == []


def test_logs_check_counts_errors(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "api.log").write_text(
        "INFO ok\nERROR boom\nWARNING careful\nERROR again\n", encoding="utf-8"
    )
    monkeypatch.setattr(selfrep, "LOGS_DIR", logs)
    result = selfrep.check_logs()

    assert result.status == "partial"
    assert "ERROR: 2 行" in result.details[0]
    assert "WARNING: 1 行" in result.details[0]


def test_write_report_creates_directory(tmp_path):
    result = selfrep.CheckResult(agent="a", task="t", summary="s")
    path = selfrep.write_report("scoring-audit", result, root=tmp_path)
    assert path == tmp_path / "scoring-audit" / "latest.md"
    assert "agent: a" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("dirname", sorted(selfrep.CHECKS))
def test_every_check_is_callable(dirname):
    assert callable(selfrep.CHECKS[dirname])


AGENT_WRITTEN = """---
agent: scoring-auditor
task: 過去の監査
timestamp: 2026-04-04 11:30
status: partial
---

## サマリー
係数の飽和を検出した。

## 詳細

### 1. 異常値の検出
lease_credit_log 係数が飽和し 90点に張り付く。
"""


def test_agent_report_is_preserved_before_overwrite(tmp_path):
    """決定論的監査が再現できない指摘を、黙って消さないこと。"""
    d = tmp_path / "scoring-audit"
    d.mkdir()
    (d / "latest.md").write_text(AGENT_WRITTEN, encoding="utf-8")

    result = selfrep.CheckResult(agent="scoring-auditor", task="t", summary="ok")
    selfrep.write_report("scoring-audit", result, root=tmp_path)

    kept = d / selfrep.PREVIOUS_NAME
    assert kept.exists()
    # 構造に依存せず全文が残る（`## 課題・リスク` を持たない形式でも欠落しない）
    assert "lease_credit_log" in kept.read_text(encoding="utf-8")
    # 新しい latest.md から辿れる
    latest = (d / "latest.md").read_text(encoding="utf-8")
    assert selfrep.CARRIED_HEADING in latest
    assert selfrep.PREVIOUS_NAME in latest
    assert "2026-04-04 11:30" in latest


def test_rerun_does_not_clobber_the_preserved_report(tmp_path):
    d = tmp_path / "scoring-audit"
    d.mkdir()
    (d / "latest.md").write_text(AGENT_WRITTEN, encoding="utf-8")
    result = selfrep.CheckResult(agent="scoring-auditor", task="t", summary="ok")

    selfrep.write_report("scoring-audit", result, root=tmp_path)
    selfrep.write_report("scoring-audit", result, root=tmp_path)
    selfrep.write_report("scoring-audit", result, root=tmp_path)

    kept = (d / selfrep.PREVIOUS_NAME).read_text(encoding="utf-8")
    assert "lease_credit_log" in kept
    assert selfrep.SELF_MARKER not in kept  # 自動生成で塗り潰されていない


def test_no_previous_section_when_nothing_to_preserve(tmp_path):
    d = tmp_path / "api-health"
    d.mkdir()
    result = selfrep.CheckResult(agent="api-health-checker", task="t", summary="ok")
    selfrep.write_report("api-health", result, root=tmp_path)

    latest = (d / "latest.md").read_text(encoding="utf-8")
    assert selfrep.CARRIED_HEADING not in latest
    assert not (d / selfrep.PREVIOUS_NAME).exists()


def test_preserved_file_is_not_picked_up_as_a_report(tmp_path, monkeypatch):
    """退避ファイルが sidecar のブリーフに混ざらないこと。"""
    from scripts import agent_sidecar_reader as reader

    d = tmp_path / "scoring-audit"
    d.mkdir()
    (d / "latest.md").write_text(AGENT_WRITTEN, encoding="utf-8")
    selfrep.write_report(
        "scoring-audit",
        selfrep.CheckResult(agent="scoring-auditor", task="t", summary="ok"),
        root=tmp_path,
    )

    monkeypatch.setattr(reader, "REPORT_ROOT", tmp_path)
    monkeypatch.setattr(reader, "PROJECT_ROOT", tmp_path)
    paths = [p.name for p in reader._candidate_paths()]
    assert paths == ["latest.md"]
