"""リース知性体の懸念が両方の軍師AI経路へ放送されることを検証する。

GWT broadcast: スコアリング時に記録された pending_dissonance を、
- レポート経路: shinsa_gunshi_logic.build_gunshi_prompt
- ストリーミング経路: api.gunshi_gemini.build_system_instruction
の双方が消費し、出典つきで懸念点として取り込む。
"""

from lease_intelligence_mind import (
    build_gunshi_dissonance_section,
    detect_dissonance,
    register_ignition,
)


def _seed_dissonance(vault):
    signals = detect_dissonance(
        {"score": 58, "score_base": 73, "approval_line": 70}
    )
    register_ignition(vault, signals, date_str="2026-06-13")
    return build_gunshi_dissonance_section(vault)


def test_report_path_build_gunshi_prompt_embeds_dissonance(tmp_path):
    from shinsa_gunshi_logic import build_gunshi_prompt

    vault = tmp_path / "vault"
    vault.mkdir()
    section = _seed_dissonance(vault)
    assert section, "懸念が記録されていれば放送ブロックが生成される"

    prompt = build_gunshi_prompt(
        industry="製造業",
        score=58,
        dissonance_section=section,
    )
    assert "リース知性体が抱える未解決の懸念" in prompt
    assert "承認線70" in prompt
    assert "出典:" in prompt

    # 空のときはプロンプトに何も足さない（後方互換）
    plain = build_gunshi_prompt(industry="製造業", score=58)
    assert "リース知性体が抱える未解決の懸念" not in plain


def test_stream_path_build_system_instruction_embeds_dissonance(tmp_path):
    from api.gunshi_gemini import build_system_instruction

    vault = tmp_path / "vault"
    vault.mkdir()
    section = _seed_dissonance(vault)

    instruction = build_system_instruction(
        asset_name="設備一式",
        industry_cat="製造業",
        include_pdca=False,
        dissonance_block=section,
    )
    assert "リース知性体が抱える未解決の懸念" in instruction
    assert "承認線70" in instruction

    # 空のときは何も足さない（後方互換）
    plain = build_system_instruction(
        asset_name="設備一式",
        industry_cat="製造業",
        include_pdca=False,
    )
    assert "リース知性体が抱える未解決の懸念" not in plain


def test_fallback_text_carries_dissonance_so_human_sees_it(tmp_path):
    """Gemini不調でフォールバックに落ちても、人間が読む最終文に懸念が残る。"""
    from api.gunshi_gemini import build_fallback_strategy_text

    vault = tmp_path / "vault"
    vault.mkdir()
    section = _seed_dissonance(vault)
    params = {"score": 53.5, "pd_pct": 6.0, "industry_cat": "製造業", "asset_name": "CNC加工機"}

    with_concern = build_fallback_strategy_text(
        params, ["フレーズA"], reason="Gemini応答が短すぎたため補完", dissonance_block=section
    )
    # フォールバック本文（人間が読む最終文）に懸念が含まれる
    assert "リース知性体が抱える未解決の懸念" in with_concern
    assert "承認線70" in with_concern
    assert "代替戦略" in with_concern  # フォールバックの体裁は維持

    # dissonance を渡さなければ従来どおり（後方互換）
    plain = build_fallback_strategy_text(params, ["フレーズA"], reason="接続失敗")
    assert "リース知性体が抱える未解決の懸念" not in plain
