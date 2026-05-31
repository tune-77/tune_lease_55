"""
two_stage_model.py — 2段階スコアリングモデル

Stage 1 (信用リスク評価):
    借手の財務データからデフォルト確率を計算し、閾値超過なら即時否決。
    既存モデル: data/lgbm_model.pkl (retraining_pipeline.py が学習)

Stage 2 (成約予測評価):
    Stage 1 を通過した案件について案件特性から成約確率を計算。
    既存モデル: data/lgbm_contract_model.pkl または data/ml_rf_v4.pkl

既存の scoring_core.run_quick_scoring() を置き換えるものではなく、
2段階の評価フローを明示的に実行するサブコンポーネントとして使う。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# モデルロード・特徴量構築のラッパー（テスト時にモック可能）
# ---------------------------------------------------------------------------


def _load_default_risk_model() -> Optional[Any]:
    """Stage 1 用デフォルトリスクモデルをロードする。"""
    try:
        from scoring_core import _load_lgbm_default_model
        return _load_lgbm_default_model()
    except Exception as exc:
        logger.warning("default risk model load failed: %s", exc)
        return None


def _build_stage1_feature_row(inputs: dict) -> List[float]:
    """Stage 1 モデル用の特徴量ベクトルを構築する。"""
    from scoring_core import _build_default_model_feature_row
    return _build_default_model_feature_row(inputs)


def _run_predict_one(inputs: dict) -> Optional[Dict[str, Any]]:
    """Stage 2 用の predict_one を呼ぶ。"""
    try:
        from scoring_core import _safe_float
        from scoring.predict_one import predict_one, map_industry_major_to_scoring

        industry_major = (inputs.get("industry_major") or "D 建設業").strip()
        nenshu       = max(0.0, _safe_float(inputs.get("nenshu")))
        op_profit    = _safe_float(inputs.get("op_profit") or inputs.get("rieki"))
        net_income   = _safe_float(inputs.get("net_income"))
        net_assets   = _safe_float(inputs.get("net_assets"))
        total_assets = _safe_float(inputs.get("total_assets"))

        ctx = dict(inputs)
        ctx["industry"] = map_industry_major_to_scoring(industry_major)

        return predict_one(
            revenue=nenshu * 1000.0,
            total_assets=max(total_assets, 1.0) * 1000.0,
            equity=net_assets * 1000.0,
            operating_profit=op_profit * 1000.0,
            net_income=net_income * 1000.0,
            machinery_equipment=_safe_float(inputs.get("machines")) * 1000.0,
            other_fixed_assets=_safe_float(inputs.get("other_assets")) * 1000.0,
            depreciation=_safe_float(inputs.get("depreciation")) * 1000.0,
            rent_expense=_safe_float(inputs.get("rent_expense")) * 1000.0,
            industry=ctx["industry"],
            context=ctx,
        )
    except Exception as exc:
        logger.warning("predict_one call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class Stage1Result:
    """Stage 1（信用リスク評価）の結果。"""

    default_prob: float
    passed: bool
    threshold: float
    model_used: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class Stage2Result:
    """Stage 2（成約予測評価）の結果。"""

    approval_prob: float
    decision: str
    model_used: str
    top_factors: List[str] = field(default_factory=list)


@dataclass
class TwoStageResult:
    """2段階モデル全体の結果。"""

    stage1: Stage1Result
    stage2: Optional[Stage2Result]
    final_score: float
    final_decision: str
    skipped_stage2: bool


# ---------------------------------------------------------------------------
# TwoStageScorer
# ---------------------------------------------------------------------------


class TwoStageScorer:
    """
    2段階スコアリングモデル。

    Stage 1 でデフォルトリスクを評価し、閾値を超えた場合は Stage 2 をスキップして
    即時否決する。Stage 1 を通過した案件のみ Stage 2 で成約確率を評価する。

    使用例::

        scorer = TwoStageScorer()
        result = scorer.score(inputs)
        print(result.final_decision, result.final_score)
    """

    DEFAULT_STAGE1_THRESHOLD = 0.5

    def __init__(
        self,
        stage1_threshold: float = DEFAULT_STAGE1_THRESHOLD,
        data_dir: Optional[Path] = None,
    ) -> None:
        if not 0.0 < stage1_threshold < 1.0:
            raise ValueError(
                f"stage1_threshold は 0〜1 の開区間で指定してください: {stage1_threshold}"
            )
        self.stage1_threshold = stage1_threshold
        self._data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")

    # ------------------------------------------------------------------
    # パブリック API
    # ------------------------------------------------------------------

    def score(self, inputs: Dict[str, Any]) -> TwoStageResult:
        """
        2段階スコアリングを実行する。

        Args:
            inputs: scoring_core.run_quick_scoring() と同じ入力辞書。

        Returns:
            TwoStageResult — stage1 / stage2 の詳細と最終判定を含む。
        """
        stage1 = self._run_stage1(inputs)

        if not stage1.passed:
            final_score = round(max(0.0, (1.0 - stage1.default_prob) * 100.0), 1)
            return TwoStageResult(
                stage1=stage1,
                stage2=None,
                final_score=final_score,
                final_decision="否決",
                skipped_stage2=True,
            )

        stage2 = self._run_stage2(inputs, stage1)
        final_score = round(max(0.0, min(100.0, stage2.approval_prob * 100.0)), 1)

        return TwoStageResult(
            stage1=stage1,
            stage2=stage2,
            final_score=final_score,
            final_decision=stage2.decision,
            skipped_stage2=False,
        )

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _run_stage1(self, inputs: Dict[str, Any]) -> Stage1Result:
        """Stage 1: デフォルトリスクモデルで信用リスクを評価する。"""
        model = _load_default_risk_model()
        if model is None:
            return Stage1Result(
                default_prob=0.0,
                passed=True,
                threshold=self.stage1_threshold,
                model_used="none",
                reasons=["Stage 1 モデル未存在: スキップ（全件合格扱い）"],
            )

        try:
            import numpy as np

            row = _build_stage1_feature_row(inputs)
            n_expected = getattr(model, "n_features_in_", len(row))
            if len(row) != n_expected:
                return Stage1Result(
                    default_prob=0.0,
                    passed=True,
                    threshold=self.stage1_threshold,
                    model_used="lgbm_default",
                    reasons=[
                        f"特徴量数不一致（期待 {n_expected}、実際 {len(row)}）: Stage 1 スキップ"
                    ],
                )

            X = np.array([row], dtype=float)
            default_prob = float(model.predict_proba(X)[0][1])
            passed = default_prob < self.stage1_threshold

            reasons: List[str] = []
            if not passed:
                reasons.append(
                    f"デフォルト確率 {default_prob:.1%} が閾値 {self.stage1_threshold:.1%} を超過"
                )

            return Stage1Result(
                default_prob=round(default_prob, 4),
                passed=passed,
                threshold=self.stage1_threshold,
                model_used="lgbm_default",
                reasons=reasons,
            )
        except Exception as exc:
            logger.warning("Stage 1 evaluation error: %s", exc)
            return Stage1Result(
                default_prob=0.0,
                passed=True,
                threshold=self.stage1_threshold,
                model_used="error",
                reasons=[f"Stage 1 評価エラー（スキップ）: {exc}"],
            )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def _run_stage2(
        self, inputs: Dict[str, Any], stage1: Stage1Result
    ) -> Stage2Result:
        """Stage 2: 成約予測モデルで案件成約確率を評価する。"""
        # Stage 1 のデフォルト確率を Stage 2 の特徴量として渡す
        inputs_with_stage1 = dict(inputs)
        inputs_with_stage1["stage1_default_prob"] = stage1.default_prob

        result = _run_predict_one(inputs_with_stage1)
        if result is None:
            return Stage2Result(
                approval_prob=0.5,
                decision="要審議",
                model_used="none",
                top_factors=["Stage 2 モデル未存在: 中立スコア（0.5）を使用"],
            )

        ai_prob = float(result.get("ai_prob", 0.5))
        approval_prob = round(max(0.0, min(1.0, 1.0 - ai_prob)), 4)
        decision = "承認" if approval_prob >= 0.5 else "否決"

        return Stage2Result(
            approval_prob=approval_prob,
            decision=decision,
            model_used=result.get("model_used", "rf"),
            top_factors=result.get("top5_reasons", []),
        )
