import os
import sys
import time

from scripts import ops_friction_doctor as doctor


def test_scan_logs_groups_known_friction(tmp_path):
    memory = tmp_path / "memory"
    reports = tmp_path / "reports"
    memory.mkdir()
    reports.mkdir()
    (memory / "2026-08-25.md").write_text(
        "Gitship dirty worktree generated/runtime artifact\n"
        "LaunchAgent missing and cloudflared tunnel lost\n",
        encoding="utf-8",
    )
    (reports / "shion_memory_sentinel_latest.md").write_text(
        "needs_feedback and human approval remain in promotion_queue\n",
        encoding="utf-8",
    )

    hits = doctor.scan_logs(tmp_path, ("memory/*.md", "reports/*_latest.md"))

    assert hits["gitship_noise"][0].count == 1
    assert hits["local_deploy_restart"][0].count == 1
    assert hits["memory_pipeline_review"][0].count == 1


def test_build_findings_adds_generated_dirty_weight(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("Gitship dirty worktree\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 12, "generated_like": 10})

    findings = doctor.build_findings(tmp_path, ("memory/*.md",))

    gitship = next(item for item in findings if item.id == "gitship_noise")
    assert gitship.score == 11
    assert gitship.severity == "medium"
    assert "classify_git_ship_candidates.py" in gitship.command


def test_render_includes_next_command(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("Cloud Run GCS writeback materialize\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})

    text = doctor.render(doctor.build_findings(tmp_path, ("memory/*.md",)))

    assert "# Ops Friction Doctor" in text
    assert "sync_cloudrun_inputs_from_gcs.py" in text


def test_apply_safe_runs_only_allowlisted_actions(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text(
        "Gitship dirty worktree\nneeds_feedback remains in promotion_queue\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 12, "generated_like": 10})
    calls = []

    def fake_runner(command, root):
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    findings = doctor.build_findings(tmp_path, ("memory/*.md",))
    results = doctor.apply_safe_findings(findings, root=tmp_path, runner=fake_runner)

    assert calls == ["python scripts/build_shion_memory_sentinel_report.py"]
    assert results[0].finding_id == "memory_pipeline_review"
    assert results[0].applied is True


def test_write_report_creates_json_and_markdown(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("needs_feedback remains in promotion_queue\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})
    report = doctor.OpsFrictionReport(doctor.build_findings(tmp_path, ("memory/*.md",)))
    json_path = tmp_path / "reports" / "ops.json"
    md_path = tmp_path / "reports" / "ops.md"

    doctor.write_report(report, json_path=json_path, md_path=md_path)

    assert json_path.exists()
    assert md_path.exists()
    assert "Ops Friction Doctor" in md_path.read_text(encoding="utf-8")


def test_main_warns_and_exits_1_when_no_logs_scanned(tmp_path, monkeypatch, capsys):
    """走査対象ログが1件も無い＝log-patternがドリフトした疑いを検知する。"""
    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ops_friction_doctor.py", "--log-pattern", "nomatch/*.md"],
    )

    exit_code = doctor.main()

    assert exit_code == 1
    assert "log-pattern" in capsys.readouterr().err


def test_main_returns_0_when_logs_scanned_but_no_friction_found(tmp_path, monkeypatch, capsys):
    """findings=0（摩擦なし）は良い意味のゼロなので検知しない。"""
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("今日は静かな一日でした\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ops_friction_doctor.py", "--log-pattern", "memory/*.md"],
    )

    exit_code = doctor.main()

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_cloudrun_sync_gap_ignores_plain_cloudrun_mentions(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("Cloud Run Web URL is useful context\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})

    findings = doctor.build_findings(tmp_path, ("memory/*.md",))

    assert "cloudrun_sync_gap" not in {item.id for item in findings}


def test_scan_logs_records_report_age_in_days(tmp_path):
    """REV-420: 週次生成レポートは最長6日間ヒットし続けるので鮮度を持たせる。"""
    reports = tmp_path / "reports"
    reports.mkdir()
    stale = reports / "instruction_debt_latest.md"
    stale.write_text("Gitship dirty worktree generated/runtime artifact\n", encoding="utf-8")
    old = time.time() - 5 * 86400
    os.utime(stale, (old, old))

    hits = doctor.scan_logs(tmp_path, ("reports/*_latest.md",))

    assert hits["gitship_noise"][0].age_days == 5


def test_render_marks_stale_hits_but_not_fresh_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})
    fresh = doctor.LogHit(path="reports/a_latest.md", count=1, samples=[], age_days=0)
    stale = doctor.LogHit(path="reports/b_latest.md", count=1, samples=[], age_days=6)
    finding = doctor.Finding(
        id="gitship_noise",
        title="Gitship dirty noise",
        severity="low",
        score=2,
        scope="Gitship前",
        reason="test",
        command="python scripts/classify_git_ship_candidates.py",
        auto_command="",
        hits=[fresh, stale],
    )

    text = doctor.render([finding])

    assert "reports/a_latest.md (1 hits)" in text
    assert "日前のレポート" not in text.split("reports/b_latest.md")[0]
    assert "reports/b_latest.md (1 hits)（6日前のレポート）" in text


def test_stale_only_hits_are_dropped_from_findings(tmp_path, monkeypatch):
    """REV-420: 鮮度切れのヒットだけの項目は score 0 になり「今日の課題」から外れる。"""
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})
    reports = tmp_path / "reports"
    reports.mkdir()
    stale = reports / "instruction_debt_latest.md"
    stale.write_text("Gitship dirty worktree generated/runtime artifact\n", encoding="utf-8")
    old = time.time() - 5 * 86400
    os.utime(stale, (old, old))

    findings = doctor.build_findings(tmp_path, ("reports/*_latest.md",))

    assert [item.id for item in findings] == []


def test_fresh_hits_are_scored_and_stale_ones_are_reported_but_not_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})
    reports = tmp_path / "reports"
    reports.mkdir()
    fresh = reports / "ops_friction_source_latest.md"
    fresh.write_text("Gitship dirty worktree\n", encoding="utf-8")
    stale = reports / "instruction_debt_latest.md"
    stale.write_text("Gitship dirty worktree\nanother dirty worktree line\n", encoding="utf-8")
    old = time.time() - 5 * 86400
    os.utime(stale, (old, old))

    findings = doctor.build_findings(tmp_path, ("reports/*_latest.md",))

    assert len(findings) == 1
    assert findings[0].score == 1
    assert "2 more hits are from reports older than" in findings[0].reason
