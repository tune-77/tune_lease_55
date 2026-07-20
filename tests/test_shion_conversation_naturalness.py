import api.main as main


def test_delta_awareness_suppressed_without_explicit_continuation():
    history = [
        {"role": "user", "content": "動画の2分案を考えて"},
        {"role": "assistant", "content": "判断資産の流れに絞るのがよいです。"},
    ]

    block, payload = main._build_delta_awareness_prompt_block(
        "リースシステムの紫苑の会話が不自然だから直して",
        history,
    )

    assert block == ""
    assert payload["used"] is False
    assert payload["reason"] == "no_explicit_continuation_request"


def test_delta_awareness_kept_for_explicit_continuation():
    history = [
        {"role": "user", "content": "リースシステムの紫苑の会話が不自然だから直して"},
        {"role": "assistant", "content": "前回接続の圧を弱めます。"},
    ]

    block, payload = main._build_delta_awareness_prompt_block(
        "その件の続きでもう少し直して",
        history,
    )

    assert "Delta Awareness" in block
    assert payload["used"] is True
    assert payload["explicit_continuation"] is True


def test_reflection_gate_does_not_force_previous_context_when_delta_unused():
    block, payload = main._build_reflection_gate_prompt_block(
        continuity_hook={"route": "lease_judgment"},
        delta_awareness={"used": False},
        memory_to_judgment={"route": "lease_judgment"},
    )

    assert "Userが続きと明示した時だけ前回差分" in block
    assert payload["explicit_continuation"] is False
