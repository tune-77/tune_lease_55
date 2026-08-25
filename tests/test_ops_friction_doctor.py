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


def test_cloudrun_sync_gap_ignores_plain_cloudrun_mentions(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "2026-08-25.md").write_text("Cloud Run Web URL is useful context\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "git_dirty_counts", lambda root: {"total": 0, "generated_like": 0})

    findings = doctor.build_findings(tmp_path, ("memory/*.md",))

    assert "cloudrun_sync_gap" not in {item.id for item in findings}
