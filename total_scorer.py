"""
総合スコア計算モジュール (ver 1.1)
total scorer with ratio.docx に基づく。

物件スコア × asset_w + 借手スコア × obligor_w の加重平均で総合スコアを算出。
カテゴリ別の配分比率（ASSET_WEIGHT）を適用することで、
資産特性に即した審査判定を実現する。
"""

from category_config import ASSET_WEIGHT, SCORE_GRADES, get_grade as _get_grade
from asset_scorer import calc_asset_score, get_recommendation


def calc_total_score(
    category: str,
    asset_item_scores: dict,
    obligor_score: float,
    contract: dict = None,
) -> dict:
    """
    総合スコアを算出する。

    total = asset_score * asset_w + obligor_score * obligor_w

    Parameters
    ----------
    category : str
        "IT機器" | "産業機械" | "車両" | "医療機器"
    asset_item_scores : dict
        {item_id: score (0-100)}。未入力項目は 50 で補完。
    obligor_score : float
        借手スコア（0-100）。通常は contract_prob（確率スコア %）。
    contract : dict, optional
        契約条件。asset_scorer._adjust_weights() に渡すキーと同じ。
          lease_months, tech_life_months, has_buyout_option, is_major_maker

    Returns
    -------
    dict
        total_score      (float) : 0-100 の総合スコア
        grade            (str)   : S / A / B / C / D
        grade_text       (str)   : グレードテキスト
        grade_color      (str)   : 表示色 (hex)
        asset_score      (float) : 物件スコア
        asset_grade      (str)   : 物件グレード
        asset_grade_text (str)   : 物件グレードテキスト
        asset_weight     (float) : 物件配分比率（例: 0.35）
        obligor_score    (float) : 借手スコア（入力値）
        obligor_weight   (float) : 借手配分比率（例: 0.65）
        category         (str)   : カテゴリ名
        item_scores      (dict)  : 項目別スコア詳細
        warnings         (list)  : C/D 項目警告
        recommendation   (dict)  : 推奨リース条件
        rationale        (str)   : ASSET_WEIGHT の設定根拠
        used_default_weight (bool): ASSET_WEIGHT 未定義カテゴリのデフォルト値適用フラグ
    """
    if contract is None:
        contract = {}

    weight_cfg = ASSET_WEIGHT.get(category)
    used_default_weight = weight_cfg is None
    if used_default_weight:
        # 未定義カテゴリはデフォルト配分（物件15%：借手85%）
        weight_cfg = {
            "asset_w": 0.15,
            "obligor_w": 0.85,
            "rationale": "（標準配分 — カテゴリ未定義のためデフォルト値を使用）",
        }

    asset_w = weight_cfg["asset_w"]
    obligor_w = weight_cfg["obligor_w"]

    # 物件スコア計算
    asset_result = calc_asset_score(category, asset_item_scores, contract)
    asset_score = asset_result["total_score"]

    # 総合スコア
    total = asset_score * asset_w + obligor_score * obligor_w
    total = round(min(100.0, max(0.0, total)), 1)

    grade = _get_grade(total)
    rec = get_recommendation(grade["label"])

    result = {
        "total_score": total,
        "grade": grade["label"],
        "grade_text": grade["text"],
        "grade_color": grade["color"],
        "asset_score": asset_score,
        "asset_grade": asset_result["grade"],
        "asset_grade_text": asset_result["grade_text"],
        "asset_grade_color": asset_result["grade_color"],
        "asset_weight": asset_w,
        "obligor_score": obligor_score,
        "obligor_weight": obligor_w,
        "category": category,
        "item_scores": asset_result["item_scores"],
        "warnings": asset_result["warnings"],
        "weight_adjusted": asset_result.get("weight_adjusted", False),
        "recommendation": rec,
        "rationale": weight_cfg.get("rationale", ""),
        "used_default_weight": used_default_weight,
        "completeness_ratio": asset_result.get("completeness_ratio", 1.0),
    }
    if "usage_period_fit" in asset_result:
        result["usage_period_fit"] = asset_result["usage_period_fit"]
    if "remanufacture_score" in asset_result:
        result["remanufacture_score"] = asset_result["remanufacture_score"]
    if "assessment_label" in asset_result:
        result["assessment_label"] = asset_result["assessment_label"]
    return result
