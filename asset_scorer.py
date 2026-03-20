"""
物件スコア計算エンジン (ver 1.0)
asset scoring design and code.docx に基づく。

主な機能:
- カテゴリ別スコアリング項目の加重平均によるスコア算出
- 契約条件（リース期間/買取オプション/大手メーカー）による動的重み調整
- グレード（S/A/B/C/D）と推奨リース条件の返却
"""

from category_config import CATEGORY_SCORE_ITEMS, SCORE_GRADES


def _get_grade(score: float) -> dict:
    """スコアからグレード dict を返す。"""
    for g in SCORE_GRADES:
        if score >= g["min"]:
            return g
    return SCORE_GRADES[-1]


def _adjust_weights(category: str, base_weights: dict, contract: dict) -> dict:
    """
    契約条件に基づく動的重み調整。
    - リース期間が技術寿命の80%超 → 陳腐化系ウェイトを 1.3倍
    - 買取オプションあり     → 残価リスク系ウェイトを 0.7倍
    - 大手メーカー           → 流動性・サポート系ウェイトを 1.2倍
    最後に合計が 100 になるよう正規化。
    """
    weights = dict(base_weights)
    items_map = {item["id"]: item for item in CATEGORY_SCORE_ITEMS.get(category, [])}

    # リース期間 vs 技術寿命
    lease_months = contract.get("lease_months", 0)
    tech_life_months = contract.get("tech_life_months", 60)
    if tech_life_months > 0 and lease_months > 0 and (lease_months / tech_life_months) > 0.8:
        for item_id, item in items_map.items():
            if item.get("tag") == "obsolescence_risk":
                weights[item_id] = min(weights[item_id] * 1.3, 50)

    # 買取オプションあり → 残価リスク系を軽減
    if contract.get("has_buyout_option"):
        for item_id, item in items_map.items():
            if item.get("tag") == "residual_value":
                weights[item_id] = weights[item_id] * 0.7

    # 大手メーカー → 流動性・サポート系を強化
    if contract.get("is_major_maker"):
        for item_id, item in items_map.items():
            if item.get("tag") == "liquidity_support":
                weights[item_id] = weights[item_id] * 1.2

    # 正規化（合計 100 に）
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total * 100 for k, v in weights.items()}
    return weights


def calc_asset_score(category: str, scores: dict, contract: dict = None) -> dict:
    """
    カテゴリ別物件スコアを計算する。

    Parameters
    ----------
    category : str
        "IT機器" | "産業機械" | "車両" | "医療機器"
    scores : dict
        {item_id: score (0-100)}。未入力項目は 50（中立）で補完。
    contract : dict, optional
        契約条件。キー:
          lease_months (int)    : リース期間（月）
          tech_life_months (int): 想定技術寿命（月）
          has_buyout_option (bool): 買取オプション有無
          is_major_maker (bool)   : 大手メーカー品か

    Returns
    -------
    dict
        total_score  (float)   : 0-100 の総合スコア
        grade        (str)     : S / A / B / C / D
        grade_text   (str)     : グレードのテキスト説明
        grade_color  (str)     : グレードの表示色（hex）
        item_scores  (dict)    : 項目別 {id: {label, score, weight, contribution}}
        warnings     (list)    : C/D 項目の警告リスト
        weight_adjusted (bool) : 重み調整が行われたか
    """
    if contract is None:
        contract = {}

    items = CATEGORY_SCORE_ITEMS.get(category)
    if not items:
        return {
            "total_score": 50.0,
            "grade": "C",
            "grade_text": "要慎重検討",
            "grade_color": "#f97316",
            "item_scores": {},
            "warnings": [f"カテゴリ '{category}' は未定義です"],
            "weight_adjusted": False,
        }

    base_weights = {item["id"]: float(item["weight"]) for item in items}
    adj_weights = _adjust_weights(category, base_weights, contract)
    weight_adjusted = adj_weights != base_weights

    # ── 入力完備率の算出 ────────────────────────────────────────────────────────
    n_items = len(items)
    n_provided = sum(1 for item in items if item["id"] in scores)
    completeness_ratio = n_provided / n_items if n_items > 0 else 1.0

    total_score = 0.0
    item_scores = {}
    warnings = []

    for item in items:
        item_id = item["id"]
        is_provided = item_id in scores
        raw_score = float(scores.get(item_id, 50))  # 未入力は中立 50 点
        raw_score = max(0.0, min(100.0, raw_score))
        adj_w = adj_weights.get(item_id, float(item["weight"])) / 100.0
        contribution = raw_score * adj_w
        total_score += contribution
        item_scores[item_id] = {
            "label": item["label"],
            "score": raw_score,
            "weight": round(adj_weights.get(item_id, float(item["weight"])), 1),
            "base_weight": float(item["weight"]),
            "contribution": round(contribution, 2),
            "provided": is_provided,
        }
        # C / D 項目は警告
        item_grade = _get_grade(raw_score)
        if item_grade["label"] in ("C", "D"):
            warnings.append(
                f"⚠️ **{item['label']}**（{raw_score:.0f}点 / {item_grade['label']}）"
                f" → {item_grade['text']} — 要注意項目"
            )

    # ── 情報欠如ペナルティ係数の適用 ─────────────────────────────────────────
    # total_score × completeness_ratio + 50 × (1 - completeness_ratio)
    if completeness_ratio < 1.0:
        missing = n_items - n_provided
        total_score = total_score * completeness_ratio + 50.0 * (1.0 - completeness_ratio)
        warnings.append(
            f"⚠️ 入力情報が不完全です（{n_provided}/{n_items}項目入力済み）"
            f" — 未入力{missing}項目に中立値(50点)を補完し、完備率{completeness_ratio:.0%}でペナルティ補正済み"
        )

    total_score = round(min(100.0, max(0.0, total_score)), 1)
    grade = _get_grade(total_score)

    return {
        "total_score": total_score,
        "grade": grade["label"],
        "grade_text": grade["text"],
        "grade_color": grade["color"],
        "item_scores": item_scores,
        "warnings": warnings,
        "weight_adjusted": weight_adjusted,
        "completeness_ratio": round(completeness_ratio, 3),
    }


def get_recommendation(grade: str) -> dict:
    """グレード別の推奨最長リース年数・残価率・メモを返す。"""
    recs = {
        "S": {"max_lease_years": 7, "residual_rate": 0.20,
              "note": "長期リース・高残価設定可。積極的な提案を推奨。"},
        "A": {"max_lease_years": 5, "residual_rate": 0.10,
              "note": "標準リース期間・標準残価。通常スキームで進めてください。"},
        "B": {"max_lease_years": 3, "residual_rate": 0.05,
              "note": "短期推奨・低残価設定。条件付き承認として稟議書に根拠を記載してください。"},
        "C": {"max_lease_years": 2, "residual_rate": 0.00,
              "note": "慎重設計・残価設定不推奨。審査部との事前相談を推奨します。"},
        "D": {"max_lease_years": 0, "residual_rate": 0.00,
              "note": "リース設計困難。原則取扱不推奨。物件の見直しまたは代替案を検討してください。"},
    }
    return recs.get(grade, recs["C"])
