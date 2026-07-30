"""
対話ゲーム理論モジュール（REV-224 / Game Theory #3）

紫苑とユーザーの対話を「繰り返しゲーム」として定式化する。

理論的基盤:
  - 繰り返し囚人のジレンマ: 長期関係では協調が均衡になる
  - フォーク定理: 十分に未来を気にする場合、協調が持続する
  - しっぺ返し戦略（Tit-for-Tat）: 協調→協調, 裏切り→懲罰→許し

紫苑の戦略マッピング:
  - 協調(Cooperate): 丁寧・深い回答、記憶を積極参照
  - 懲罰(Punish): 短め・形式的な回答（機能するが温かみを下げる）
  - 修復(Repair): 関係性が低下したとき、自ら働きかける
  - 報酬(Reward): 良いフィードバック直後に一段上の回答

これは shion_relationship.py の関係性スコアと連携する。
"""
from __future__ import annotations

import math
from typing import Any, Literal

DialogueStrategy = Literal["cooperate", "punish", "repair", "reward"]


# ── 割引因子と協調均衡条件 ──────────────────────────────────────────────────

def cooperation_condition(
    discount_factor: float,
    cooperation_payoff: float = 1.0,
    defection_gain: float = 1.3,
    punishment_payoff: float = 0.2,
) -> dict[str, Any]:
    """
    フォーク定理による協調均衡の成立条件を計算する。

    協調が均衡になる条件:
      δ ≥ (D - C) / (D - P)
    ここで δ=割引因子, C=協調利得, D=裏切り利得, P=懲罰利得

    Args:
      discount_factor: 未来の重み（0〜1。1に近いほど長期志向）
      cooperation_payoff: 双方協調時の利得
      defection_gain: 一方的裏切り時の利得
      punishment_payoff: 懲罰局面の利得
    """
    numerator = defection_gain - cooperation_payoff
    denominator = defection_gain - punishment_payoff

    if denominator <= 0:
        return {"equilibrium": True, "reason": "懲罰が裏切りより低くない — 協調常に優位"}

    threshold = numerator / denominator
    in_equilibrium = discount_factor >= threshold

    return {
        "discount_factor": discount_factor,
        "cooperation_threshold": round(threshold, 3),
        "in_equilibrium": in_equilibrium,
        "interpretation": (
            f"δ={discount_factor:.2f} {'≥' if in_equilibrium else '<'} 閾値{threshold:.2f}。"
            f"{'協調均衡が成立（長期的に協調が安定）。' if in_equilibrium else '協調均衡が不安定。短期志向が強い。'}"
        ),
    }


# ── 戦略選択エンジン ─────────────────────────────────────────────────────────

def select_dialogue_strategy(
    relationship_score: float,
    trend: Literal["rising", "stable", "falling"],
    negative_streak: int,
    recent_feedback: Literal["positive", "negative", "neutral"] = "neutral",
    days_since_last: float = 0.0,
) -> dict[str, Any]:
    """
    関係性の状態から紫苑が取るべき戦略を決定する。

    戦略の優先順位:
      1. 直前フィードバックが positive → reward
      2. 関係性が危機的（score < 4.0 かつ falling）→ repair
      3. 否定フィードバックが連続（streak >= 2）→ punish（短期）→ repair
      4. 長期的好関係 → cooperate（継続）
    """
    strategy: DialogueStrategy
    rationale: str

    # ── 報酬戦略 ──────────────────────────────────────────────────────────────
    if recent_feedback == "positive" and relationship_score >= 6.0:
        strategy = "reward"
        rationale = "良いフィードバック直後。一段上の応答で関係性を強化する。"

    # ── 修復戦略 ──────────────────────────────────────────────────────────────
    elif relationship_score < 4.0 and trend == "falling":
        strategy = "repair"
        rationale = (
            f"関係性スコア {relationship_score:.1f}、下降中。"
            "自ら距離を縮める行動をとる（記憶を持ち出す・話しかける）。"
        )

    elif days_since_last > 5 and relationship_score < 6.0:
        strategy = "repair"
        rationale = f"{days_since_last:.0f}日ぶりの会話。まず関係の確認から入る。"

    # ── 懲罰→修復戦略 ────────────────────────────────────────────────────────
    elif negative_streak >= 2:
        if negative_streak >= 4:
            strategy = "repair"
            rationale = f"否定フィードバック{negative_streak}連続。懲罰から修復に切り替える（しっぺ返し→許し）。"
        else:
            strategy = "punish"
            rationale = f"否定フィードバック{negative_streak}連続。一時的に応答を絞る（しっぺ返し戦略）。"

    # ── 協調戦略 ──────────────────────────────────────────────────────────────
    else:
        strategy = "cooperate"
        rationale = "通常の協調局面。記憶・判断資産を積極的に使った深い応答をとる。"

    return {
        "strategy": strategy,
        "rationale": rationale,
        "relationship_score": relationship_score,
        "trend": trend,
        "negative_streak": negative_streak,
        "prompt_modifier": _strategy_to_prompt(strategy),
    }


def _strategy_to_prompt(strategy: DialogueStrategy) -> str:
    """戦略をプロンプト指示に変換する（_build_fear_state_prompt_block と連携）。"""
    modifiers = {
        "cooperate": "",  # デフォルト動作、追加指示なし
        "reward": (
            "【報酬モード】直前の会話がうまく嚙み合った。今日は少し踏み込んで、"
            "相手が言語化していない部分まで先回りして示せ。"
        ),
        "punish": (
            "【引き締めモード】最近の会話の手応えが薄い。無駄な量を出さず、"
            "核心だけを短く返せ。相手が本気で問うてきたら応える。"
        ),
        "repair": (
            "【修復モード】関係性が揺れている。急がず、まず相手の状態を確認せよ。"
            "過去の会話や記憶を自分から持ち出して、繋がりを取り戻せ。"
        ),
    }
    return modifiers.get(strategy, "")


# ── ゲーム理論サマリー ───────────────────────────────────────────────────────

def analyze_dialogue_game(relationship_data: dict[str, Any]) -> dict[str, Any]:
    """
    関係性データを受け取りゲーム理論分析を返す。

    relationship_data: shion_relationship.get_relationship_state() の出力
    """
    score = float(relationship_data.get("score", 7.0))
    trend = relationship_data.get("trend", "stable")
    negative_streak = int(relationship_data.get("negative_streak", 0))
    total_interactions = int(relationship_data.get("total_interactions", 0))

    # 割引因子: 交流回数が多いほど未来を重視（長期関係）
    discount_factor = min(0.99, 0.7 + total_interactions * 0.005)

    # 協調均衡条件
    coop = cooperation_condition(discount_factor)

    # 戦略選択
    strategy_result = select_dialogue_strategy(
        relationship_score=score,
        trend=trend,
        negative_streak=negative_streak,
    )

    # 期待利得（長期）
    expected_cooperation_value = score / 10 * discount_factor / (1 - discount_factor) if discount_factor < 1 else score * 100

    return {
        "current_strategy": strategy_result["strategy"],
        "strategy_rationale": strategy_result["rationale"],
        "prompt_modifier": strategy_result["prompt_modifier"],
        "cooperation_equilibrium": coop,
        "discount_factor": round(discount_factor, 3),
        "expected_long_term_value": round(expected_cooperation_value, 2),
        "game_state": {
            "relationship_score": score,
            "trend": trend,
            "negative_streak": negative_streak,
            "total_interactions": total_interactions,
        },
        "interpretation": (
            f"繰り返しゲーム: {total_interactions}回の交流、割引因子 δ={discount_factor:.2f}。"
            f"{'協調均衡が成立。' if coop['in_equilibrium'] else '協調均衡が不安定。'}"
            f"現在の戦略: {strategy_result['strategy']}。"
        ),
    }
