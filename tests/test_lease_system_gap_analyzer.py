import json


def test_gap_analyzer_detects_backlog_and_sidecar_risks(tmp_path, monkeypatch):
    from scripts import lease_system_gap_analyzer as analyzer

    project = tmp_path / "project"
    reports = project / "reports"
    reports.mkdir(parents=True)
    latest = reports / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "needs_review_count": 21,
                "needs_review": [
                    {
                        "id": "REV-X",
                        "title": "ポートフォリオリスク管理",
                        "auto_fix_policy": {"risk": "high"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sidecar = reports / "agent_sidecar_brief.json"
    sidecar.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "agent": "scoring-auditor",
                        "risks": "テストデータ混在と過学習の可能性",
                        "stale": True,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rag_eval = project / "api" / "knowledge" / "rag_eval_set.json"
    rag_eval.parent.mkdir(parents=True)
    rag_eval.write_text("[]", encoding="utf-8")
    tests = project / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text("def test_x(): pass\n", encoding="utf-8")
    specs = project / "specs"
    specs.mkdir()
    (specs / "P0.md").write_text("# spec\n", encoding="utf-8")

    monkeypatch.setattr(analyzer, "PROJECT_ROOT", project)
    monkeypatch.setattr(analyzer, "REPORTS_DIR", reports)
    monkeypatch.setattr(analyzer, "LATEST_REPORT", latest)
    monkeypatch.setattr(analyzer, "SIDECAR_JSON", sidecar)
    monkeypatch.setattr(analyzer, "RAG_EVAL_SET", rag_eval)
    monkeypatch.setattr(analyzer, "TESTS_DIR", tests)
    monkeypatch.setattr(analyzer, "SPECS_DIR", specs)

    gaps = analyzer.collect_gaps(run_rag_eval=False)
    ids = {gap.id for gap in gaps}
    assert {"GAP-001", "GAP-002", "GAP-003", "GAP-004", "GAP-005", "GAP-007"} <= ids
    assert gaps[0].priority == "critical"


def test_traceability_flags_rev_missing_spec(tmp_path, monkeypatch):
    from scripts import lease_system_gap_analyzer as analyzer

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "improvement_ledger.jsonl").write_text(
        '{"rev_id": "REV-100", "title": "テスト改善", "status": "applied"}\n',
        encoding="utf-8",
    )
    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_rev100.py").write_text("# REV-100 の回帰テスト\n", encoding="utf-8")
    specs_dir = project / "specs"
    specs_dir.mkdir()
    reports_dir = project / "reports"

    monkeypatch.setattr(analyzer, "PROJECT_ROOT", project)
    monkeypatch.setattr(analyzer, "TESTS_DIR", tests_dir)
    monkeypatch.setattr(analyzer, "SPECS_DIR", specs_dir)
    monkeypatch.setattr(analyzer, "REPORTS_DIR", reports_dir)

    rows = analyzer.build_rev_traceability_rows()
    row = next(r for r in rows if r["rev_id"] == "REV-100")
    assert row["missing_spec"] is True
    assert row["test_files"] == ["tests/test_rev100.py"]

    analyzer.write_traceability_outputs(rows)
    assert (reports_dir / "rev_spec_test_traceability.md").exists()
    data = json.loads((reports_dir / "rev_spec_test_traceability.json").read_text(encoding="utf-8"))
    assert any(r["rev_id"] == "REV-100" for r in data["rows"])


def test_recheck_queue_written_for_stale_sidecar_reports(tmp_path, monkeypatch):
    from scripts import lease_system_gap_analyzer as analyzer

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(analyzer, "REPORTS_DIR", reports_dir)

    analyzer._write_recheck_queue([{"id": "scoring-auditor", "title": "scoring-auditor", "reason": "stale"}])

    assert (reports_dir / "recheck_queue.md").exists()
    data = json.loads((reports_dir / "recheck_queue.json").read_text(encoding="utf-8"))
    assert data["items"][0]["id"] == "scoring-auditor"


def test_spec_gate_generates_stub_for_high_risk_item_without_spec(tmp_path, monkeypatch):
    from scripts import lease_system_gap_analyzer as analyzer

    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    monkeypatch.setattr(analyzer, "SPECS_DIR", specs_dir)

    written = analyzer._spec_gate_stub_for_high_risk_items(
        [{"id": "REV-200", "title": "高リスク改善項目のテスト"}]
    )

    assert len(written) == 1
    stub_path = specs_dir / "_generated_stubs" / "REV-200.md"
    assert stub_path.exists()
    content = stub_path.read_text(encoding="utf-8")
    assert "PENDING-REV-200" in content
    assert "高リスク改善項目のテスト" in content

    # 二回目は既存スタブを上書きしない
    written_again = analyzer._spec_gate_stub_for_high_risk_items(
        [{"id": "REV-200", "title": "高リスク改善項目のテスト"}]
    )
    assert written_again == []


def test_frontend_scan_flags_only_unsanitized_dangerous_html(tmp_path):
    from scripts import lease_system_gap_analyzer as analyzer

    src = tmp_path / "frontend_src"
    src.mkdir()
    (src / "Safe.tsx").write_text(
        "const x = <span dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }} />;\n",
        encoding="utf-8",
    )
    (src / "Unsafe.tsx").write_text(
        "const y = <span dangerouslySetInnerHTML={{ __html: rawHtml }} />;\n",
        encoding="utf-8",
    )

    findings = analyzer._frontend_unsanitized_dangerous_html(src)

    assert len(findings) == 1
    assert "Unsafe.tsx" in findings[0]


def test_gap_analyzer_writes_markdown_and_json(tmp_path):
    from scripts import lease_system_gap_analyzer as analyzer

    item = analyzer.GapItem(
        id="GAP-T",
        title="テスト",
        priority="high",
        category="quality",
        evidence=["evidence"],
        impact="impact",
        recommended_action="action",
        suggested_program="program.py",
    )
    out_md = tmp_path / "gap.md"
    out_json = tmp_path / "gap.json"

    analyzer.write_outputs([item], out_md, out_json)

    assert "Lease System Gap Analysis" in out_md.read_text(encoding="utf-8")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["gaps"][0]["id"] == "GAP-T"
