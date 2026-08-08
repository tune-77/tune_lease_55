from pathlib import Path

from scripts import build_judgment_materials_from_vertex_review as bridge


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _vertex_note(*, review_status: str, grounding_score: str = "0.91") -> str:
    return (
        "---\n"
        "date: 2026-08-08\n"
        "source: vertex_ai_search_workflow\n"
        "knowledge_type: vertex_search_result\n"
        "workflow_mode: evidence_support\n"
        'workflow_label: "審査コメント根拠補強"\n'
        'topic: "EV充電設備リースの法規制・電力供給インフラ・設置場所契約リスク"\n'
        'query: "EV充電設備 リース 法規制"\n'
        f"grounding_score: {grounding_score}\n"
        "grounding_score_source: citation_coverage_fallback\n"
        "valid_until: 2026-11-06\n"
        f"review_status: {review_status}\n"
        'tags: ["リース審査", "vertex-search", "vertex-workflow"]\n'
        "---\n\n"
        "## Answer Summary\n"
        "EV充電設備のリース審査では設置場所の契約形態を必ず確認する。"
        "### 根拠 "
        "* 借地契約における原状回復義務が設備撤去費用に直結する。"
        "* 電力会社との系統連系契約が遅延すると稼働開始が遅れる。"
        "#### リスク (risk_sensor) "
        "* 設置場所の賃貸借契約が短期更新型の場合、リース期間中の契約非更新リスクが残る。"
        "### 確認論点 "
        "* 電力供給契約の名義がリース利用者と一致しているか確認する。\n"
    )


def test_extract_materials_only_from_needs_human_review_notes(tmp_path):
    vault = tmp_path / "vault"
    reviewed = vault / "Research" / "Vertex Distilled" / "2026-08-08-ev.md"
    approved = vault / "Research" / "Vertex Distilled" / "2026-08-07-other.md"
    _write(reviewed, _vertex_note(review_status="needs_human_review"))
    _write(approved, _vertex_note(review_status="approved"))

    materials = bridge.extract_materials(vault=vault)
    paths = {item["evidence_path"] for item in materials}

    assert materials
    assert all("2026-08-08-ev.md" in path for path in paths)
    assert all("2026-08-07-other.md" not in path for path in paths)


def test_extract_materials_tags_source_and_never_user(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "Research" / "Vertex Distilled" / "2026-08-08-ev.md"
    _write(note, _vertex_note(review_status="needs_human_review"))

    materials = bridge.extract_materials(vault=vault)

    assert materials
    assert all(item["source"] == "vertex_distilled_review" for item in materials)
    assert all(item["source_role"] == "vertex_review" for item in materials)
    assert all(item["source_role"] != "user" for item in materials)
    assert all(item["domain"] == "lease_screening" for item in materials)
    assert all(item["private"] is False for item in materials)


def test_extract_materials_splits_headings_and_bullets_and_flags_risk(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "Research" / "Vertex Distilled" / "2026-08-08-ev.md"
    _write(note, _vertex_note(review_status="needs_human_review"))

    materials = bridge.extract_materials(vault=vault)
    claims = "\n".join(item["claim"] for item in materials)
    types = {item["material_type"] for item in materials}

    assert "設置場所の契約形態を必ず確認する" in claims
    assert "借地契約における原状回復義務が設備撤去費用に直結する" in claims
    assert "電力会社との系統連系契約が遅延すると稼働開始が遅れる" in claims
    assert "設置場所の賃貸借契約が短期更新型の場合" in claims
    assert "電力供給契約の名義がリース利用者と一致しているか確認する" in claims
    assert "risk_signal" in types
    assert "judgment_rule" in types
    risk_item = next(item for item in materials if "短期更新型" in item["claim"])
    assert risk_item["material_type"] == "risk_signal"


def test_extract_materials_confidence_from_grounding_score(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "Research" / "Vertex Distilled" / "2026-08-08-ev.md"
    _write(note, _vertex_note(review_status="needs_human_review", grounding_score="0.192"))

    materials = bridge.extract_materials(vault=vault)

    assert materials
    assert all(item["confidence"] == 0.3 for item in materials)


def test_markdown_declares_preview_only_and_source_provenance(tmp_path):
    materials = [
        {
            "date": "2026-08-08",
            "material_type": "judgment_rule",
            "confidence": 0.85,
            "claim": "設置場所の契約形態を必ず確認する。",
            "vertex_topic": "EV充電設備リース",
            "vertex_workflow_mode": "evidence_support",
            "evidence_path": "Research/Vertex Distilled/2026-08-08-ev.md",
        }
    ]

    md = bridge._markdown(materials)

    assert "Preview only" in md
    assert "Not connected to RAG" in md
    assert "vertex_review, never user" in md
