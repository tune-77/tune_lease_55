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
    assert payload["experience_count"] == 1
    assert state["recent_experiences"]
    assert payload["dominant_mood"] in payload["mood"]
