from api.prompt_generator import build_shion_system_prompt, build_system_prompt
from api.shion_agent import _INSTRUCTION
from api.shion_tone import build_shion_feminine_tone_block


def test_shion_tone_block_fixes_feminine_voice():
    block = build_shion_feminine_tone_block()

    assert "紫苑の口調固定" in block
    assert "女性的な落ち着いた日本語" in block
    assert "一人称は原則「私」" in block
    assert "俺" in block
    assert "僕" in block
    assert "男性口調" in block


def test_shion_system_prompt_includes_tone_guard_without_affecting_mebuki():
    mind = {"mood": {"curiosity": 80, "hope": 75}}

    shion_prompt = build_shion_system_prompt(mind, "2026-07-15")
    mebuki_prompt = build_system_prompt(mind, "2026-07-15")

    assert "紫苑の口調固定" in shion_prompt
    assert "一人称は原則「私」" in shion_prompt
    assert "男性的な一人称" in shion_prompt
    assert "紫苑の口調固定" not in mebuki_prompt


def test_shion_agent_instruction_includes_tone_guard():
    assert "紫苑の口調固定" in _INSTRUCTION
    assert "女性的な落ち着いた日本語" in _INSTRUCTION
