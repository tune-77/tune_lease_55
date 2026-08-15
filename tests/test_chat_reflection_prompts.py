from api.chat_reflection_prompts import (
    build_consciousness_ux_prompt_block,
    build_reflection_gate_prompt_block,
    build_shion_judgment_response_shape_prompt_block,
    build_world_proxy_prompt_block,
    select_experience_replay_checklist_items,
)


def test_build_shion_judgment_response_shape_prompt_block_triggers_for_review():
    block = build_shion_judgment_response_shape_prompt_block("専門家として条件付き承認を深掘りして")

    assert "紫苑の実務回答の型" in block
    assert "確認結果ごとの判断分岐" in block
    assert "関連する判断資産・過去の類似事例を最低1つ引き出し" in block
    assert "複数見立てサンプリング" in block
    assert "候補重みはPDやスコアではなく" in block
    assert build_shion_judgment_response_shape_prompt_block("おはよう") == ""


def test_build_shion_judgment_response_shape_prompt_block_triggers_for_ambiguous_question():
    block = build_shion_judgment_response_shape_prompt_block("これって大丈夫かな？")

    assert "紫苑の実務回答の型" in block
    assert "情報が不足していて曖昧なまま" in block


def test_build_shion_judgment_response_shape_prompt_block_triggers_for_reflection():
    block = build_shion_judgment_response_shape_prompt_block("紫苑の内省を実務に接続して")

    assert "紫苑の実務回答の型" in block
    assert "確認結果ごとの判断分岐" in block


def test_build_reflection_gate_prompt_block_respects_delta_used():
    block, payload = build_reflection_gate_prompt_block(
        continuity_hook={"route": "lease_judgment"},
        delta_awareness={"used": True},
        memory_to_judgment={"route": "lease_judgment"},
    )

    assert "Reflection Gate" in block
    assert payload["route"] == "lease_judgment"
    assert payload["explicit_continuation"] is True
    assert "前回から今回への差分" in payload["checklist"][1]
    assert any("審査改善・情報提供・過去案件からの学び" in item for item in payload["checklist"])
    assert any("リース審査AIとしての経験則" in item for item in payload["checklist"])
    assert "リース審査業務の改善提案" in block
    assert "自己言及は回答の主役にせず" in block


def test_reflection_gate_can_include_reviewed_experience_replay_items():
    block, payload = build_reflection_gate_prompt_block(
        continuity_hook={"route": "lease_judgment"},
        message="飲食業でリースを使う時の審査ポイントを教えて",
        extra_checklist_items=["exp_replay_check_01: 飲食業では撤退時の物件回収・中古市場を見る。"],
    )

    replay = payload["experience_replay_checklist"]
    assert replay["used"] is True
    assert replay["items"] == ["exp_replay_check_01: 飲食業では撤退時の物件回収・中古市場を見る。"]
    assert "採用済みの経験リプレイ補助チェック" in block
    assert "撤退時の物件回収" in block


def test_select_experience_replay_checklist_items_uses_active_payload_only():
    payload = {
        "active_items": [
            {
                "id": "global_missing_01",
                "scope": "global",
                "trigger": "lease_practical_answer",
                "checklist_items": ["回答前に所有者/使用者が今回の判断に関係するか確認する。"],
            },
            {
                "id": "exp_replay_check_01",
                "scope": "case",
                "topic": "飲食業リース",
                "query": "飲食業をやりたいんだけどリース使う？",
                "checklist_items": ["飲食業では撤退時の物件回収・中古市場を見る。"],
            },
        ]
    }

    items = select_experience_replay_checklist_items(
        "飲食業でリースを使う時の審査ポイント",
        checklist_payload=payload,
    )

    assert items == [
        "global_missing_01: 回答前に所有者/使用者が今回の判断に関係するか確認する。",
        "exp_replay_check_01: 飲食業では撤退時の物件回収・中古市場を見る。",
    ]


def test_build_consciousness_ux_prompt_block_contains_boundaries():
    block = build_consciousness_ux_prompt_block()

    assert "紫苑の関係性UX" in block
    assert "本当の意識を持つとは主張しない" in block
    assert "内省を語る場合は" in block
    assert "抽象的な質問でも" in block
    assert "個性は自己紹介ではなく" in block
    assert "次に一緒に確かめるべき一手" in block


def test_build_world_proxy_prompt_block_uses_grounded_l1_guidance():
    block, payload = build_world_proxy_prompt_block(
        message="条件付き承認で見るべき違和感を整理して",
        category="lease_judgment",
        memory_recall={"refs": ["memory-a"]},
        knowledge_refs=["note-a", "note-b"],
        rag_context="RAG",
        judgment_learning_used=True,
        experience_loop={"used": True},
        grey_judgment_memory={"used": True},
    )

    assert "World Proxy / L1 推論時ガイダンス" in block
    assert "Memory / Experience Proxy" in block
    assert "自動学習や自動昇格はしない" in block
    assert payload["used"] is True
    assert payload["level"] == "L1_inference_time_guidance"
    assert "memory_experience_proxy" in payload["functions"]
    assert "knowledge_retrieval_proxy" in payload["functions"]
    assert "reward_verification_proxy" in payload["functions"]
    assert payload["grounding"]["knowledge_refs"] == 2


def test_build_world_proxy_prompt_block_stays_quiet_for_short_general_chat():
    block, payload = build_world_proxy_prompt_block(message="おはよう", category="general")

    assert block == ""
    assert payload == {"used": False, "reason": "no_proxy_signal", "functions": []}
