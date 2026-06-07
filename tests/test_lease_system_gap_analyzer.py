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
