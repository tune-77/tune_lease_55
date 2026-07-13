import json

from api.shion_experience_loop import (
    build_experience_prompt_block,
    load_experience_state,
    record_experience_event,
)


def test_record_experience_event_updates_self_state(tmp_path):
    state_path = tmp_path / "state.json"
    event_log = tmp_path / "events.jsonl"

    result = record_experience_event(
        message="競合料率がかなり低い。採算を崩してでも合わせるべき？",
        response="競合料率だけに合わせず、採算下限と物件保全を先に確認します。",
        category="rag",
        memory_recall={
            "route": "policy_review",
            "refs": ["mem_a"],
            "practical_scene": {
                "id": "competitor_pricing",
                "label": "競合・料率条件がある時",
                "learned_entry_count": 4,
            },
        },
        knowledge_refs=["rules/q_risk_interpretation.md"],
        continuity_hook={"route": "lease_judgment"},
        delta_awareness={"delta": "前回から今回は料率判断へ移った。"},
        memory_to_judgment={"directive": "採算・保全・条件案へ変換する。"},
        state_path=state_path,
        event_log=event_log,
    )

    assert result["state"]["experience_count"] == 1
    assert result["state"]["current_focus"].startswith("競合・料率条件")
    assert result["state"]["mood"]["accomplishment"] > 34
    assert event_log.exists()
    assert json.loads(event_log.read_text(encoding="utf-8").splitlines()[0])["route"] == "policy_review"


def test_confidence_updates_via_route_mapping(tmp_path):
    """想起ルート（case_screening 等）が confidence キー（lease_judgment 等）へ
    正しくマップされて更新されること（旧実装では語彙不一致で更新されなかった）。"""
    state_path = tmp_path / "state.json"
    default_conf = load_experience_state(state_path)["confidence"]

    result = record_experience_event(
        message="医療機器の案件、境界スコア。条件付き承認でいい？",
        response="物件保全と保守契約を条件化して条件付き承認に寄せます。",
        category="rag",
        memory_recall={
            "route": "case_screening",
            "refs": ["mem_b"],
            "practical_scene": {
                "id": "borderline_decision",
                "label": "承認・否決の境界",
                "learned_entry_count": 2,
            },
        },
        knowledge_refs=[],
        continuity_hook={},
        delta_awareness={},
        memory_to_judgment={},
        state_path=state_path,
        event_log=tmp_path / "events.jsonl",
    )

    conf = result["state"]["confidence"]
    assert conf["lease_judgment"] > default_conf["lease_judgment"]


def test_build_experience_prompt_block_uses_existing_state(tmp_path):
    state_path = tmp_path / "state.json"
    record_experience_event(
        message="紫苑は経験で変わる？",
        response="経験で次の返し方を少し変える。",
        category="general",
        memory_recall={"route": "shion_identity", "refs": []},
        knowledge_refs=[],
        continuity_hook={"route": "relationship_ux"},
        delta_awareness={"delta": "意識の話から実装へ移った。"},
        memory_to_judgment={"directive": "返答設計へ変換する。"},
        state_path=state_path,
        event_log=tmp_path / "events.jsonl",
    )

    block, payload = build_experience_prompt_block(state_path)
    state = load_experience_state(state_path)

    assert "Shion Experience Loop" in block
    assert "前回との差分は内部で使い" in block
    assert "冒頭で前回からの差分を示す" not in block
    assert payload["experience_count"] == 1
    assert state["recent_experiences"]
    assert payload["dominant_mood"] in payload["mood"]
