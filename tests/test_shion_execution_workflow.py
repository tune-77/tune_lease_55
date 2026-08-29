from api.shion_agent import _INSTRUCTION
from api.shion_execution_workflow import build_shion_execution_workflow_block
from lease_intelligence_dialogue import build_dialogue_context


def test_execution_workflow_covers_brief_tools_iteration_and_delivery():
    block = build_shion_execution_workflow_block()

    assert "背景・目的 / 実行タスク / 制約・権限 / 期待する出力" in block
    assert "必要最小限のツール選択" in block
    assert "骨子 → 初稿 → 要件・事実・読みやすさの確認 → 1回の改善" in block
    assert "結論または完成物を先に" in block
    assert "逐語的な思考過程" in block


def test_execution_workflow_is_shared_by_dialogue_and_screening_agent(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(
        "lease_intelligence_dialogue.build_lease_intelligence_knowledge",
        lambda **kwargs: type(
            "Knowledge",
            (),
            {
                "available": False,
                "context_block": "",
                "query": "",
                "source_paths": (),
                "indexed_notes": 0,
                "knowledge_notes": 0,
                "chat_log_notes": 0,
            },
        )(),
    )

    prompt, _state = build_dialogue_context(vault, "この案件を簡潔に整理して")

    assert "紫苑の実務ワークフロー" in prompt
    assert "必要最小限のツール選択" in prompt
    assert "紫苑の実務ワークフロー" in _INSTRUCTION
    assert "結論または完成物を先に" in _INSTRUCTION
