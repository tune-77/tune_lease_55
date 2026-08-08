from api.chat_external_research import (
    build_external_research_suggestion,
    capture_vertex_materials_for_research,
    external_research_permission_reply,
    external_research_topic_from_message,
)


def test_external_research_topic_strips_command_words():
    topic = external_research_topic_from_message("ネットで調べてから、補助金付き工作機械リースの注意点")

    assert topic == "から、補助金付き工作機械リースの注意点"


def test_external_research_suggestion_triggers_when_no_data_for_researchable_topic():
    suggestion = build_external_research_suggestion(
        "補助金付き工作機械リースの注意点を教えて",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="shion",
    )

    assert suggestion == {
        "needed": True,
        "topic": "補助金付き工作機械リースの注意点を教えて",
        "reason": "手元のObsidian/RAGに該当データがありません / Obsidian/RAGの参照が薄い論点です",
        "output_dir": "Projects/tune_lease_55/Research/Auto Research/",
    }


def test_external_research_suggestion_skips_basic_stable_question_without_refs():
    assert build_external_research_suggestion(
        "ファイナンスリースとは？",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="shion",
    ) == {"needed": False}


def test_external_research_suggestion_triggers_for_explicit_research_request():

    suggestion = build_external_research_suggestion(
        "ネットで調べて補助金付き工作機械リースの注意点を教えて",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="shion",
    )

    assert suggestion == {
        "needed": True,
        "topic": "補助金付き工作機械リースの注意点を教えて",
        "reason": "ユーザーが外部調査を求めています / 手元のObsidian/RAGに該当データがありません",
        "output_dir": "Projects/tune_lease_55/Research/Auto Research/",
    }


def test_external_research_suggestion_disabled_for_general_mode():
    suggestion = build_external_research_suggestion(
        "ネットで調べて補助金の期限を確認して",
        question_category="lease_knowledge",
        knowledge_ref_count=0,
        response_mode="general",
    )

    assert suggestion == {"needed": False}


def test_external_research_permission_reply_keeps_consent_wording():
    reply = external_research_permission_reply(
        {"topic": "補助金期限", "reason": "ユーザーが外部調査を求めています"}
    )

    assert "この件は手元のデータがありません。ネットで調査してもいいですか？" in reply
    assert "- 足りない情報: 補助金期限" in reply
    assert "- 理由: ユーザーが外部調査を求めています" in reply
    assert "ObsidianのResearchノートに保存してから回答に使います" in reply


def test_capture_vertex_materials_for_research_saves_two_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_RESEARCH_MATERIAL_CAPTURE_ENABLED", "true")
    calls = []

    def fake_workflow(topic, *, mode, page_size):
        calls.append((topic, mode, page_size))
        return {
            "topic": topic,
            "mode": mode,
            "query": f"{topic} {mode}",
            "refs": [f"Projects/tune_lease_55/Research/{mode}.md"],
            "search": {"used": True},
            "answer": {"used": True},
        }

    def fake_capture(workflow, *, vault_path, state_path):
        return {
            "captured": True,
            "duplicate": False,
            "note_path": f"Projects/tune_lease_55/Research/Vertex Distilled/{workflow['mode']}.md",
        }

    result = capture_vertex_materials_for_research(
        "補助金付き工作機械",
        vault_path=tmp_path,
        workflow_runner=fake_workflow,
        capture_runner=fake_capture,
        state_path=tmp_path / "state.json",
    )

    assert result["used"] is True
    assert result["modes"] == ["evidence_support", "judgment_candidates"]
    assert len(result["captures"]) == 2
    assert len(result["saved_paths"]) == 2
    assert calls == [
        ("補助金付き工作機械", "evidence_support", 5),
        ("補助金付き工作機械", "judgment_candidates", 5),
    ]


def test_capture_vertex_materials_for_research_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_RESEARCH_MATERIAL_CAPTURE_ENABLED", "false")

    result = capture_vertex_materials_for_research("補助金", vault_path=tmp_path)

    assert result == {"used": False, "reason": "disabled", "captures": []}


def test_capture_vertex_materials_for_research_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_RESEARCH_MATERIAL_CAPTURE_ENABLED", "true")

    def fake_workflow(topic, *, mode, page_size):
        if mode == "judgment_candidates":
            raise RuntimeError("vertex timeout")
        return {"topic": topic, "mode": mode, "query": topic, "refs": [], "search": {"used": True}}

    def fake_capture(workflow, *, vault_path, state_path):
        return {"captured": True, "note_path": "ok.md"}

    result = capture_vertex_materials_for_research(
        "補助金",
        vault_path=tmp_path,
        workflow_runner=fake_workflow,
        capture_runner=fake_capture,
        state_path=tmp_path / "state.json",
    )

    assert result["used"] is True
    assert result["saved_paths"] == ["ok.md"]
    assert result["errors"][0]["mode"] == "judgment_candidates"
