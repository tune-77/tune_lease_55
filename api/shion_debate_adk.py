"""ADK マルチエージェント討論（本番討論のフォールバック）。

本番の作り込んだ討論（api/multi_agent_screening.py）が主経路。それが失敗したときに、
**独立実装である ADK マルチエージェント討論**が受けて審査を止めない（レジリエンス層）。

同時に、ADK 本来の用途である「マルチエージェント・オーケストレーション」を実演する:
    SequentialAgent(
        ParallelAgent([懐疑派, 楽観派, 革新派]),  # 3ペルソナが並行に意見
        arbiter,                                    # 統合派が3意見を裁定
    )

設計方針（多層フォールバックで「主経路失敗時に必ず有効な結果を返す」）:
  1. run_debate_adk_fallback() が ADK 討論を試みる。
  2. google.adk 未導入・実行失敗など**あらゆる例外**は握り、スコア由来の最小結果へ劣化する。
  3. スコア判定は scoring_core.APPROVAL_LINE / CONDITIONAL_LINE を単一ソースとする
     （CLAUDE.md: 承認ラインを複製・ハードコードしない）。

注意: このモジュールは google.adk / scoring_core を**モジュール先頭で import しない**
（どの環境でも import 可能にし、フォールバックの劣化経路をテストできるようにするため）。
"""

from __future__ import annotations

import os
from typing import Any

_VALID_FINALS = {"承認", "条件付承認", "否決"}

# ── ペルソナ指示（本番の役割・温度設計に対応。簡潔版）───────────────────────────
_SKEPTIC_INSTRUCTION = (
    "あなたはリース審査の懐疑派です。財務の弱さ・返済原資・担保保全の観点から"
    "リスクを厳しく指摘してください。結論は『承認 / 条件付承認 / 否決』のいずれかで述べます。"
)
_OPTIMIST_INSTRUCTION = (
    "あなたはリース審査の楽観派です。事業の成長性・好材料・取引関係の観点から"
    "前向きな可能性を示してください。結論は『承認 / 条件付承認 / 否決』のいずれかで述べます。"
)
_INNOVATOR_INSTRUCTION = (
    "あなたはリース審査の革新派です。条件付承認を活かす実務的な工夫（保証・期間短縮等）を"
    "提案してください。結論は『承認 / 条件付承認 / 否決』のいずれかで述べます。"
)
_ARBITER_INSTRUCTION = (
    "あなたはリース審査の統合派（裁定役）です。懐疑派・楽観派・革新派の意見を統合し、"
    "最終判定を下してください。回答の最後に必ず『判定：承認』『判定：条件付承認』"
    "『判定：否決』のいずれか1行を明記してください。条件付承認の場合は条件を列挙します。"
)


def _case_text(params: dict) -> str:
    """討論者へ渡す案件テキスト（裁定役のみスコアを見る）。"""
    lines = [
        "【案件情報】",
        f"会社名: {params.get('company_name', '不明')}",
        f"業種: {params.get('industry_cat', params.get('industry_major', '不明'))}",
        f"物件: {params.get('asset_name', '不明')}",
        f"取得価格: {params.get('acquisition_cost', 0)}",
        f"リース期間(月): {params.get('lease_term', 0)}",
    ]
    return "\n".join(lines)


def _final_from_score(params: dict) -> str:
    """スコアから判定を導く。scoring_core を単一ソースにし、引けない時は安全側。"""
    try:
        score = float(params.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        from scoring_core import APPROVAL_LINE, CONDITIONAL_LINE

        if score >= APPROVAL_LINE:
            return "承認"
        if score >= CONDITIONAL_LINE:
            return "条件付承認"
        return "否決"
    except Exception:
        # 閾値が引けない極限時は安全側（人手確認前提の条件付）に倒す。
        return "条件付承認"


def _score_derived_minimal(params: dict, note: str = "") -> dict:
    """ADK が使えない/失敗した時の最小・有効な結果。常に妥当な dict を返す。"""
    final = _final_from_score(params)
    conditions = [] if final == "承認" else ["担当者による最終確認（フォールバック裁定のため）"]
    summary = "主経路の討論が利用できないため、スコアに基づく安全側の暫定裁定を返しました。"
    if note:
        summary += f"（{note}）"
    return {
        "final": final,
        "score": params.get("score", 0),
        "mode": "adk_fallback_minimal",
        "summary": summary,
        "conditions": conditions,
        "_fallback": "score_minimal",
    }


def _parse_final(text: str) -> str | None:
    """裁定テキストから最終判定を抽出する（否決 > 条件付承認 > 承認 の優先で判定）。"""
    if not text:
        return None
    if "否決" in text:
        return "否決"
    if "条件付" in text:
        return "条件付承認"
    if "承認" in text:
        return "承認"
    return None


def build_adk_debate_agent():
    """ADK のマルチエージェント討論を構築して返す（google.adk が必要）。

    構造: SequentialAgent[ ParallelAgent[懐疑派, 楽観派, 革新派], 統合派(arbiter) ]
    テストが構造を検証できるよう、実行とは分けて公開する。
    """
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
    from google.genai.types import GenerateContentConfig

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    skeptic = LlmAgent(
        name="skeptic", model=model, instruction=_SKEPTIC_INSTRUCTION,
        output_key="skeptic_opinion",
        generate_content_config=GenerateContentConfig(temperature=0.3),
    )
    optimist = LlmAgent(
        name="optimist", model=model, instruction=_OPTIMIST_INSTRUCTION,
        output_key="optimist_opinion",
        generate_content_config=GenerateContentConfig(temperature=0.9),
    )
    innovator = LlmAgent(
        name="innovator", model=model, instruction=_INNOVATOR_INSTRUCTION,
        output_key="innovator_opinion",
        generate_content_config=GenerateContentConfig(temperature=0.6),
    )
    panel = ParallelAgent(name="debate_panel", sub_agents=[skeptic, optimist, innovator])

    arbiter = LlmAgent(
        name="arbiter", model=model, instruction=_ARBITER_INSTRUCTION,
        generate_content_config=GenerateContentConfig(temperature=0.3),
    )
    return SequentialAgent(name="shion_debate_adk", sub_agents=[panel, arbiter])


def _run_adk_debate(params: dict) -> dict:
    """ADK 討論を実行して結果 dict を返す（失敗時は例外を送出、呼び出し側で握る）。"""
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    agent = build_adk_debate_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="tune_lease_debate", session_service=session_service)

    async def _run() -> str:
        import uuid

        sid = str(uuid.uuid4())
        await session_service.create_session(
            app_name="tune_lease_debate", user_id="fallback", session_id=sid
        )
        prompt = _case_text(params) + "\n\n各ペルソナの意見を統合し、最終判定を下してください。"
        message = Content(role="user", parts=[Part(text=prompt)])
        final_text = ""
        try:
            async for event in runner.run_async(
                user_id="fallback", session_id=sid, new_message=message
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        t = getattr(part, "text", None)
                        if t:
                            final_text = t  # 最後の（＝arbiterの）テキストを保持
        finally:
            try:
                await session_service.delete_session(
                    app_name="tune_lease_debate", user_id="fallback", session_id=sid
                )
            except Exception:
                pass
        return final_text

    final_text = asyncio.run(_run())
    final = _parse_final(final_text) or _final_from_score(params)
    conditions = [] if final == "承認" else ["担当者による最終確認"]
    return {
        "final": final,
        "score": params.get("score", 0),
        "mode": "adk_fallback_debate",
        "summary": final_text or "ADKマルチエージェント討論による裁定。",
        "conditions": conditions,
        "_fallback": "adk_debate",
    }


def run_debate_adk_fallback(params: dict) -> dict:
    """本番討論のフォールバック本体。

    ADK マルチエージェント討論を試み、**あらゆる失敗**（google.adk 未導入・実行時例外）を
    握ってスコア由来の最小結果へ劣化する。したがって常に妥当な結果 dict を返す。
    """
    try:
        return _run_adk_debate(params)
    except Exception as e:  # noqa: BLE001 — フォールバックは全例外を握って劣化する
        return _score_derived_minimal(params, note=f"ADK不可: {type(e).__name__}")
