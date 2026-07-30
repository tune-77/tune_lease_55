"""
ゲーム理論 API エンドポイント（REV-224）

GET/POST /api/game-theory/screening   — 審査ゲーム（情報操作検知）
POST     /api/game-theory/negotiation  — 交渉ゲーム（ナッシュ交渉解）
GET      /api/game-theory/dialogue     — 対話ゲーム（繰り返しゲーム分析）
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

router = APIRouter(tags=["game-theory"])


# ── 1. 審査ゲーム ────────────────────────────────────────────────────────────

class ScreeningGameRequest(BaseModel):
    industry_major: str = ""
    nenshu: Optional[float] = None          # 売上高（千円）
    op_profit: Optional[float] = None       # 営業利益（千円）
    net_income: Optional[float] = None      # 純利益（千円）
    net_assets: Optional[float] = None      # 純資産（千円）
    total_assets: Optional[float] = None    # 総資産（千円）
    user_equity_ratio: Optional[float] = None  # 自己資本比率(%)


@router.post("/api/game-theory/screening")
def analyze_screening(req: ScreeningGameRequest):
    """
    審査ゲーム理論分析。
    申告財務データから「情報操作の疑い」スコアと「支配戦略」分析を返す。
    """
    try:
        from api.game_theory.screening import analyze_screening_game
        result = analyze_screening_game(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. 交渉ゲーム ────────────────────────────────────────────────────────────

class NegotiationRequest(BaseModel):
    lease_amount: float                     # リース金額（千円）
    credit_score: float                     # 審査スコア 0〜100
    monthly_cashflow: float                 # 借り手の月次CF（千円）
    collateral_ratio: float = 0.5          # 担保率 0.0〜1.0
    asset_useful_life_months: int = 60     # 物件耐用年数（月）
    interest_rate_min: float = 2.0         # 検索金利下限(%)
    interest_rate_max: float = 7.5         # 検索金利上限(%)
    term_min: int = 24                     # 検索期間下限（月）
    term_max: int = 60                     # 検索期間上限（月）
    borrower_urgency: float = 0.90         # 借り手の割引率（低いほど急いでいる）


@router.post("/api/game-theory/negotiation")
def analyze_negotiation(req: NegotiationRequest):
    """
    交渉ゲーム（ナッシュ交渉解）。
    双方の効用を最大化する金利・期間・担保条件を返す。
    """
    try:
        from api.game_theory.negotiation import solve_nash_bargaining, rubinstein_equilibrium
        nash = solve_nash_bargaining(
            lease_amount=req.lease_amount,
            credit_score=req.credit_score,
            monthly_cashflow=req.monthly_cashflow,
            collateral_ratio=req.collateral_ratio,
            asset_useful_life_months=req.asset_useful_life_months,
            interest_rate_range=(req.interest_rate_min, req.interest_rate_max),
            term_range=(req.term_min, req.term_max),
        )
        rubinstein = rubinstein_equilibrium(
            borrower_discount=req.borrower_urgency,
        )
        return {**nash, "rubinstein": rubinstein}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. 対話ゲーム ────────────────────────────────────────────────────────────

@router.get("/api/game-theory/dialogue")
def analyze_dialogue():
    """
    繰り返しゲームとしての対話分析。
    現在の関係性状態を読み、紫苑が取るべき戦略と期待長期価値を返す。
    """
    try:
        from api.shion_relationship import get_relationship_state
        from api.game_theory.dialogue import analyze_dialogue_game
        state = get_relationship_state()
        return analyze_dialogue_game(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. 統合サマリー ──────────────────────────────────────────────────────────

@router.get("/api/game-theory/summary")
def get_game_theory_summary():
    """
    3つのゲームの現状サマリーを返す（ダッシュボード用）。
    """
    try:
        from api.shion_relationship import get_relationship_state, get_fear_context
        from api.game_theory.dialogue import analyze_dialogue_game

        rel_state = get_relationship_state()
        fear_ctx = get_fear_context()
        dialogue = analyze_dialogue_game(rel_state)

        return {
            "screening_game": {
                "description": "申込者の情報操作を検知するシグナリングゲーム",
                "endpoint": "POST /api/game-theory/screening",
                "status": "ready",
            },
            "negotiation_game": {
                "description": "貸し手・借り手の最適合意条件（ナッシュ交渉解）",
                "endpoint": "POST /api/game-theory/negotiation",
                "status": "ready",
            },
            "dialogue_game": {
                "description": "紫苑との繰り返しゲーム（協調・懲罰・修復戦略）",
                "current_strategy": dialogue["current_strategy"],
                "discount_factor": dialogue["discount_factor"],
                "in_cooperation_equilibrium": dialogue["cooperation_equilibrium"]["in_equilibrium"],
                "relationship_score": fear_ctx["score"],
                "trend": fear_ctx["trend"],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
