"""再確認キューのテスト。

本物の `claude --print` サブプロセスは絶対に起動しない
（tests/test_execute_chat_quick_fix.py と同じ方針）。
"""

import json

import pytest

from scripts import recheck_stale_agent_reports as recheck


@pytest.fixture(autouse=True)
def _never_spawn_claude(monkeypatch):
    """保険: 取りこぼしがあっても実プロセスを起動させない。"""

    def _boom(*args, **kwargs):  # pragma: no cover - 発火したらテストの誤り
        raise AssertionError(f"claude を実行しようとした: {args!r}")

    monkeypatch.setattr(recheck.subprocess, "run", _boom)


def _write_report(root, dirname, timestamp):
    path = root / dirname / "latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nagent: x\ntask: t\ntimestamp: {timestamp}\nstatus: success\n---\n\n## サマリー\nok\n",
        encoding="utf-8",
    )
    return path


def _agent_dir(tmp_path, names):
    d = tmp_path / "agents"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / f"{name}.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
    return d


def test_agent_report_dir_map_covers_every_defined_agent():
    """定義が増えたのに対応表が追随していない状態を検出する。"""
    defined = set(recheck.defined_agents())
    assert defined, "エージェント定義が読めていない"
    assert defined - set(recheck.AGENT_REPORT_DIRS) == set()


def test_fresh_reports_are_not_rerun(tmp_path):
    import datetime as dt

    agents = _agent_dir(tmp_path, ["scoring-auditor"])
    reports = tmp_path / "reports"
    fresh = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    _write_report(reports, "scoring-audit", fresh)

    picked = recheck.select_stale_agents(3, agent_dir=agents, report_root=reports)
    assert picked == []


def test_stale_reports_are_selected(tmp_path):
    agents = _agent_dir(tmp_path, ["scoring-auditor"])
    reports = tmp_path / "reports"
    _write_report(reports, "scoring-audit", "2026-01-01 09:00")

    picked = recheck.select_stale_agents(3, agent_dir=agents, report_root=reports)
    assert [p["agent"] for p in picked] == ["scoring-auditor"]
    assert picked[0]["reason"] == "stale"


def test_missing_reports_take_priority_over_stale(tmp_path):
    agents = _agent_dir(tmp_path, ["scoring-auditor", "api-health-checker"])
    reports = tmp_path / "reports"
    _write_report(reports, "scoring-audit", "2026-01-01 09:00")
    # api-health は未作成 = 一度も走っていない

    picked = recheck.select_stale_agents(2, agent_dir=agents, report_root=reports)
    assert picked[0]["agent"] == "api-health-checker"
    assert picked[0]["reason"] == "missing"


def test_limit_is_respected(tmp_path):
    names = ["scoring-auditor", "api-health-checker", "rule-validator"]
    agents = _agent_dir(tmp_path, names)
    reports = tmp_path / "reports"

    picked = recheck.select_stale_agents(2, agent_dir=agents, report_root=reports)
    assert len(picked) == 2


def test_skip_agents_are_never_selected(tmp_path):
    agents = _agent_dir(tmp_path, ["test-result-analyzer"])
    reports = tmp_path / "reports"

    picked = recheck.select_stale_agents(3, agent_dir=agents, report_root=reports)
    assert picked == []


def test_kill_switch_stops_execution(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_RECHECK_DISABLED", "1")
    monkeypatch.setattr("sys.argv", ["recheck", "--dry-run"])
    assert recheck.main() == 0
    assert "AGENT_RECHECK_DISABLED" in capsys.readouterr().out


def test_daily_limit_zero_stops_execution(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_RECHECK_DAILY_LIMIT", "0")
    monkeypatch.setattr("sys.argv", ["recheck", "--dry-run"])
    assert recheck.main() == 0
    assert "AGENT_RECHECK_DAILY_LIMIT=0" in capsys.readouterr().out


def test_count_executed_today_ignores_dry_runs(tmp_path):
    (tmp_path / "agent_recheck_result_20260817.json").write_text(
        json.dumps(
            {
                "results": [
                    {"agent": "a", "dry_run": True},
                    {"agent": "b", "dry_run": False},
                    {"agent": "c", "dry_run": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert recheck.count_executed_today("20260817", result_dir=tmp_path) == 2


def test_dry_run_does_not_invoke_claude():
    result = recheck.run_candidate(
        {"agent": "scoring-auditor", "report": "r.md", "timestamp": "", "reason": "missing"},
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["exit_code"] == 0


def test_empty_stdout_is_treated_as_failure(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "   "
        stderr = ""

    monkeypatch.setattr(recheck.subprocess, "run", lambda *a, **k: _Proc())
    result = recheck.run_candidate(
        {"agent": "scoring-auditor", "report": "r.md", "timestamp": "", "reason": "stale"},
        dry_run=False,
    )
    assert result["exit_code"] != 0
    assert "empty output" in result["stderr"]


def test_prompt_forbids_code_changes():
    prompt = recheck.build_prompt(
        {"agent": "scoring-auditor", "report": ".claude/reports/scoring-audit/latest.md",
         "timestamp": "", "reason": "stale"}
    )
    assert "scoring-auditor" in prompt
    assert ".claude/reports/scoring-audit/latest.md" in prompt
    assert "REPORT_SCHEMA.md" in prompt
    assert "変更しないこと" in prompt


def _candidate(agent="scoring-auditor"):
    return {
        "agent": agent,
        "report": ".claude/reports/scoring-audit/latest.md",
        "timestamp": "",
        "reason": "stale",
    }


def test_command_selects_agent_explicitly():
    """エージェント選択をプロンプト頼みにしない（--agent で明示する）。"""
    cmd = recheck.build_command(_candidate(), "prompt")
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert cmd[cmd.index("--agent") + 1] == "scoring-auditor"


def test_command_sets_non_interactive_permission_mode(monkeypatch):
    """--print には許可を尋ねる相手が居ないため、権限モードの指定が必須。"""
    monkeypatch.delenv("AGENT_RECHECK_PERMISSION_MODE", raising=False)
    cmd = recheck.build_command(_candidate(), "prompt")
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"


def test_permission_mode_is_overridable(monkeypatch):
    monkeypatch.setenv("AGENT_RECHECK_PERMISSION_MODE", "dontAsk")
    assert recheck._permission_mode() == "dontAsk"


def test_unknown_permission_mode_falls_back_to_safe_default(monkeypatch):
    monkeypatch.setenv("AGENT_RECHECK_PERMISSION_MODE", "rm -rf /")
    assert recheck._permission_mode() == "acceptEdits"


def test_prompt_is_last_argument():
    """プロンプトがフラグとして解釈されないこと。"""
    cmd = recheck.build_command(_candidate(), "これはプロンプトです")
    assert cmd[-1] == "これはプロンプトです"
