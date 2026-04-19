"""
物件スコア計算エンジン (ver 1.3)
asset scoring design and code.docx に基づく。

主な機能:
- カテゴリ別スコアリング項目の加重平均によるスコア算出
- 契約条件（リース期間/買取オプション/大手メーカー/EV燃料種別）による動的重み調整
- **期待使用期間.json との統合** → リース期間の最適性を動的に評価
- 産業機械: カスタマイズ度×再販市場の相互作用
- グレード（S/A/B/C/D）と推奨リース条件の返却
- 満了時推定スコア計算（useful_life_equipment.json 接続）
"""

import json
import os

from category_config import CATEGORY_SCORE_ITEMS, SCORE_GRADES, get_grade as _get_grade

try:
    from expected_usage_period import calc_lease_period_fit_score
except ImportError:
    calc_lease_period_fit_score = None

# 法定耐用年数マスタ（カテゴリ別デフォルト寿命[年]）
_USEFUL_LIFE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static_data", "useful_life_equipment.json",
)
# カテゴリ → 代表的な法定耐用年数[年]
_CATEGORY_USEFUL_LIFE: dict[str, float] = {
    "IT機器":   5.0,   # PC・サーバー・複合機等の平均
    "産業機械": 8.0,   # 工作機械・産業ロボット等の平均
    "車両":     5.0,   # トラック・貨物車等
    "医療機器": 6.0,   # 診断機器・治療機器等の平均
}

def _load_useful_life_json() -> dict:
    """useful_life_equipment.json を読み込む（失敗時は空 dict）。"""
    try:
        with open(_USEFUL_LIFE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}



def _adjust_weights(category: str, base_weights: dict, contract: dict) -> dict:
    """
    契約条件に基づく動的重み調整。
    - リース期間が技術寿命の80%超 → 陳腐化系ウェイトを 1.3倍
    - 買取オプションあり     → 残価リスク系ウェイトを 0.7倍
    - 大手メーカー           → 流動性・サポート系ウェイトを 1.2倍
    - **期待使用期間との乖離** → リース期間フィット性を評価・重み調整
    最後に合計が 100 になるよう正規化。
    
    Parameters
    ----------
    category : str
        物件カテゴリ
    base_weights : dict
        基本重み
    contract : dict
        契約条件。キー:
          lease_months (int): リース期間（月）
          tech_life_months (int): 想定技術寿命（月）
          has_buyout_option (bool): 買取オプション有無
          is_major_maker (bool): 大手メーカー品か
          vehicle_fuel_type (str): 燃料種（EV等）
          asset_name (str): 物件・設備名（期待使用期間検索用）
          item_name (str): 機種名（期待使用期間検索用）
    """
    weights = dict(base_weights)
    items_map = {item["id"]: item for item in CATEGORY_SCORE_ITEMS.get(category, [])}

    lease_months = contract.get("lease_months", 0)
    
    # ────────────────────────────────────────────────────────────────────────────
    # 期待使用期間との適合度評価（再リース機会ベース）
    # ────────────────────────────────────────────────────────────────────────────
    if calc_lease_period_fit_score and lease_months > 0:
        asset_name = contract.get("asset_name") or contract.get("item_name")
        if asset_name:
            fit_result = calc_lease_period_fit_score(asset_name, lease_months)
            remanufacture_score = fit_result.get("remanufacture_score", 50)
            
            # 再リース機会スコアに基づいた重み調整
            # remanufacture_score が低いほど（残り期間が短い）、残価リスク系の重みを上げる
            if remanufacture_score < 70:
                # 再リース機会が限定的/困難な場合、残価リスクへの警戒を強化
                for item_id, item in items_map.items():
                    if item.get("tag") in ("residual_value", "obsolescence_risk"):
                        # スコア 50以下で 1.4倍、70で 1.0倍の係数を適用
                        coeff = 1.0 + (70 - remanufacture_score) / 70 * 0.4
                        weights[item_id] = min(weights[item_id] * coeff, 50)
            elif remanufacture_score >= 85:
                # 再リース機会が豊富な場合、残가リスク系の警戒を緩和
                for item_id, item in items_map.items():
                    if item.get("tag") == "residual_value":
                        weights[item_id] = weights[item_id] * 0.9

    # ────────────────────────────────────────────────────────────────────────────
    # 従来の調整ロジック（tech_life_months ベース）
    # ────────────────────────────────────────────────────────────────────────────
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

    # EV車両 × リース期間別 → ev_tech_risk ウェイトを段階的に強化
    # バッテリー技術の急速な進化により、長期リースほど残価リスクが増大する
    if contract.get("vehicle_fuel_type") == "EV" and lease_months > 36:
        if lease_months > 60:
            ev_multiplier = 1.5
        elif lease_months > 48:
            ev_multiplier = 1.4
        else:  # 37-48ヶ月
            ev_multiplier = 1.2
        for item_id, item in items_map.items():
            if item.get("tag") == "ev_risk":
                weights[item_id] = min(weights[item_id] * ev_multiplier, 50)

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
          **asset_name (str)      : 物件・設備名（期待使用期間マスタ検索用）
          **item_name (str)       : 機種名（期待使用期間マスタ検索用）
          vehicle_fuel_type (str): 燃料種（EV等）

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
        **usage_period_fit (dict) : 期待使用期間の適合度評価（asset_name/item_name が指定時）
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

    # 産業機械: カスタマイズ度が低い（専用品）場合、再販市場スコアを0.5倍補正
    # customization_level < 40（高カスタマイズ）の場合に発動
    _custom_penalty_applied = False
    _custom_score_raw = float(scores.get("customization_level", 50)) if category == "産業機械" else 50
    if (
        category == "産業機械"
        and "customization_level" in scores
        and _custom_score_raw < 40
        and "resale_market" in scores
    ):
        _custom_penalty_applied = True

    for item in items:
        item_id = item["id"]
        is_provided = item_id in scores
        raw_score = float(scores.get(item_id, 50))  # 未入力は中立 50 点
        raw_score = max(0.0, min(100.0, raw_score))

        # 産業機械: 高カスタマイズ品の再販市場スコアを半減
        if _custom_penalty_applied and item_id == "resale_market":
            raw_score = raw_score * 0.5
            warnings.append(
                "⚠️ **カスタマイズ度が高い専用品**のため、再販市場スコアを50%に補正しました"
                " — 転売市場の流動性低下リスクを反映"
            )
            _custom_penalty_applied = False  # 警告は1回だけ

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

    # ────────────────────────────────────────────────────────────────────────────
    # 期待使用期間の適合度情報を結果に追加（再リース機会评价）
    # ────────────────────────────────────────────────────────────────────────────
    result = {
        "total_score": total_score,
        "grade": grade["label"],
        "grade_text": grade["text"],
        "grade_color": grade["color"],
        "item_scores": item_scores,
        "warnings": warnings,
        "weight_adjusted": weight_adjusted,
        "completeness_ratio": round(completeness_ratio, 3),
    }
    
    # 再リース機会の評価を追加
    lease_months = contract.get("lease_months", 0)
    if calc_lease_period_fit_score and lease_months > 0:
        asset_name = contract.get("asset_name") or contract.get("item_name")
        if asset_name:
            fit_result = calc_lease_period_fit_score(asset_name, lease_months)
            result["usage_period_fit"] = fit_result
            # 外部参照用に短シリアライズ
            result["remanufacture_score"] = fit_result.get("remanufacture_score")
            result["assessment_label"] = fit_result.get("assessment_label")
    
    return result


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


def calc_end_of_lease_score(
    category: str,
    base_score: float,
    lease_months: int,
    asset_name: str = "",
) -> dict:
    """
    リース満了時点の推定物件スコアを計算する。

    useful_life_equipment.json の法定耐用年数をベースに、
    リース期間分の価値劣化を反映した「満了時推定スコア」を返す。

    Parameters
    ----------
    category : str
        物件カテゴリ（"IT機器" | "産業機械" | "車両" | "医療機器"）
    base_score : float
        現時点の物件スコア（0-100）
    lease_months : int
        リース期間（月）
    asset_name : str, optional
        物件名（useful_life_equipment.json の品目と照合に使用）

    Returns
    -------
    dict
        end_score          (float): 満了時推定スコア（0-100）
        depreciation_ratio (float): リース期間による消費率（0.0〜1.0）
        remaining_life_years (float): 満了後の推定残余寿命（年）
        useful_life_years  (float): 想定耐用年数（年）
        is_risky           (bool): 満了時残余寿命が1年未満
        note               (str): 判定メモ
    """
    useful_life_years = _CATEGORY_USEFUL_LIFE.get(category, 6.0)

    # useful_life_equipment.json から品目名マッチングで耐用年数を精緻化
    if asset_name:
        try:
            data = _load_useful_life_json()
            asset_name_lower = asset_name.lower()
            for cat in data.get("categories", []):
                for item in cat.get("items", []):
                    if any(kw in asset_name_lower for kw in item["name"].lower().split("・")):
                        useful_life_years = float(item["years"])
                        break
                else:
                    continue
                break
        except Exception:
            pass

    lease_years = lease_months / 12.0
    depreciation_ratio = min(lease_years / useful_life_years, 1.0) if useful_life_years > 0 else 1.0
    remaining_life_years = max(useful_life_years - lease_years, 0.0)

    # 満了時スコア: base_score から劣化分を差し引く
    # - 耐用年数の100%消費 → スコアが最大40%低下（最低でも20点は保つ）
    score_decay = base_score * depreciation_ratio * 0.40
    end_score = round(max(base_score - score_decay, 20.0), 1)

    is_risky = remaining_life_years < 1.0
    if depreciation_ratio >= 0.9:
        note = "⚠️ リース期間が耐用年数に対して非常に長く、満了時の残余価値はほぼゼロです"
    elif depreciation_ratio >= 0.7:
        note = "△ リース期間が耐用年数の70%超。満了後の延長・転売には注意が必要です"
    else:
        note = "○ 満了後も相応の残余寿命が見込まれます"

    return {
        "end_score": end_score,
        "depreciation_ratio": round(depreciation_ratio, 3),
        "remaining_life_years": round(remaining_life_years, 1),
        "useful_life_years": useful_life_years,
        "is_risky": is_risky,
        "note": note,
    }
