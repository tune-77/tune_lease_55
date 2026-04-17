"""
期待使用期間マスタ管理モジュール
期待使用期間.json を読み込み、スコアリングで使用可能な形式に変換する。

主な機能:
- 機種分類・コードから期待使用期間を取得
- リース期間 vs 期待使用期間の乖離度を計算
- リース期間に応じた適切性スコアを算出
"""

import json
import os
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_USAGE_PERIOD_PATH = os.path.join(_SCRIPT_DIR, "期待使用期間.json")


def _load_usage_period_data() -> dict:
    """期待使用期間.json を読み込む。失敗時は空 dict を返す。"""
    try:
        with open(_USAGE_PERIOD_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load 期待使用期間.json: {e}")
        return {}


# グローバルキャッシュ
_USAGE_PERIOD_CACHE: Optional[dict] = None


def get_usage_period_data() -> dict:
    """キャッシュ付きで期待使用期間データを取得。"""
    global _USAGE_PERIOD_CACHE
    if _USAGE_PERIOD_CACHE is None:
        _USAGE_PERIOD_CACHE = _load_usage_period_data()
    return _USAGE_PERIOD_CACHE


def find_item_by_name(item_name: str) -> Optional[dict]:
    """
    機種名（カテゴリ）から機種情報を検索。
    
    Parameters
    ----------
    item_name : str
        機種名。例："電子計算機", "複合機", "通信機器"
    
    Returns
    -------
    dict or None
        期待使用期間情報。未見つかりの場合は None。
    """
    data = get_usage_period_data()
    if not data:
        return None
    
    item_name_lower = item_name.lower()
    for item in data.get("usage_period_data", []):
        if item.get("item_name", "").lower() == item_name_lower:
            return item
        # キーワード含む検索
        for example in item.get("examples", []):
            if item_name_lower in example.lower():
                return item
    
    return None


def find_item_by_code(code: str) -> Optional[dict]:
    """
    機種コードから機種情報を検索。
    
    Parameters
    ----------
    code : str
        機種コード。例："0104", "0201"
    
    Returns
    -------
    dict or None
        期待使用期間情報。未見つかりの場合は None。
    """
    data = get_usage_period_data()
    if not data:
        return None
    
    for item in data.get("usage_period_data", []):
        if item.get("code") == code:
            return item
    
    return None


def get_expected_years(item_name: str) -> Dict[str, int]:
    """
    機種名から期待使用期間スコアの辞書を取得。
    
    Returns
    -------
    dict
        {期間キー: スコア}。例：{"3y": 46, "4y": 66, ...}
        未見つかりの場合は空 dict。
    """
    item = find_item_by_name(item_name)
    if item:
        return item.get("periods", {})
    return {}


def calc_lease_period_fit_score(
    item_name: str,
    lease_months: int,
) -> dict:
    """
    リース期間と期待使用期間の関係から「再リース収益機会」を評価。
    
    **重要なロジック**:
    再リース期間（残り期間） = 法定耐用年数 - リース期間
    - 再リース期間が大きい = 中古市場での再リース可能性大 = スコア高
    - 再リース期間がゼロ以下 = 法定耐用年数を使い切る = スコア低
    
    Parameters
    ----------
    item_name : str
        機種名
    lease_months : int
        リース期間（月）
    
    Returns
    -------
    dict
        {
            "remanufacture_score": 0-100,    # 再リース機会スコア（高いほど良い）
            "assessment_label": str,         # 評価ラベル
            "expected_years": dict,          # 期待使用期間（min_years/max_years）
            "lease_years": float,            # リース期間（年）
            "remaining_years_min": float,    # 最小残り期間（年）
            "remaining_years_max": float,    # 最大残り期間（年）
            "recommendation": str,           # リース会社向けポイント
            "lease_period_check": dict,      # リース期間妥当性チェック
        }
    
    スコアリング基準（再リース期間ベース）:
    - 100-85: 残り期間4年以上（充分な再リース機会）
    - 84-70:  残り期間2-4年（良好な再リース機会）
    - 69-50:  残り期間1-2年（限定的な再リース機会）
    - 49-1:   残り期間0-1年（最小限）
    - 0:      再リース期間ゼロ/負（法定耐用年数使い切り）
    """
    item = find_item_by_name(item_name)
    
    if not item:
        return {
            "remanufacture_score": 50,
            "assessment_label": "未判定",
            "expected_years": {},
            "lease_years": lease_months / 12.0,
            "remaining_years_min": 0,
            "remaining_years_max": 0,
            "recommendation": "期待使用期間データが見つかりません",
            "lease_period_check": {"status": "unknown", "message": "データなし"},
        }
    
    lease_years = lease_months / 12.0
    legal_useful_life = item.get("legal_useful_life", item.get("max_years", 10))
    min_years = item.get("min_years", 3)
    max_years = item.get("max_years", 10)
    
    # 法定耐用年数に基づくリース期間の法定下限
    # リース期間の最小限度額：法定耐用年数 × 70% (10年未満) または × 60% (10年超)
    # この値以上のリース期間を設定すればOK
    if legal_useful_life < 10:
        min_lease_years = legal_useful_life * 0.7
        rule = "70%"
    else:
        min_lease_years = legal_useful_life * 0.6
        rule = "60%"
    
    # リース期間チェック（この値以上ならOK）
    if lease_years < min_lease_years:
        lease_status = "under_limit"
        lease_message = f"リース期間が法定下限({min_lease_years:.1f}年)未満です。短すぎるリース期間は、借手の実質的な購入と見なされ、適格リースと判定されない可能性があります。"
    elif lease_years < min_lease_years * 1.1:
        lease_status = "near_limit"
        lease_message = f"リース期間が法定下限に接近しています。より長いリース期間を検討してください。"
    else:
        lease_status = "within_limit"
        lease_message = f"リース期間は法定下限以上で、課税上安全です。"
    
    # 残り期間を計算（法定耐用年数ベース）
    remaining_max = legal_useful_life - lease_years
    remaining_min = min_years - lease_years  # 念のため
    
    # 再リース機会スコアを計算（法定耐用年数ベース）
    remaining_avg = remaining_max  # 法定耐用年数を基準
    
    if remaining_avg >= 4:
        remanufacture_score = 100  # 充分
    elif remaining_avg >= 3:
        remanufacture_score = 90
    elif remaining_avg >= 2:
        remanufacture_score = 75   # 良好
    elif remaining_avg >= 1:
        remanufacture_score = 60   # 限定的
    elif remaining_avg >= 0:
        remanufacture_score = 40   # 最小限
    else:
        remanufacture_score = 15   # ゼロ/負（再リース不可）
    
    # 評価ラベル
    if remanufacture_score >= 85:
        assessment_label = "優秀（再リース機会豊富）"
        recommendation = f"残り期間が{remaining_avg:.1f}年あり、再リース収益機会が高い。リース返却時の売却ポテンシャルが強い"
    elif remanufacture_score >= 70:
        assessment_label = "良好（再リース機会あり）"
        recommendation = f"残り期間が{remaining_avg:.1f}年あり、中程度の再リース機会が見込める"
    elif remanufacture_score >= 50:
        assessment_label = "標準（再リース機会限定）"
        recommendation = f"残り期間が{remaining_avg:.1f}年と限定的。スタンドアローンリースか、リース終了時の処分方針を事前検討すべき"
    elif remanufacture_score >= 30:
        assessment_label = "要注意（再リース困難）"
        recommendation = f"残り期間が{remaining_avg:.1f}年以下。法定耐用年数に迫るため、リース終了時の残価回収が困難。買取オプション検討も視野に"
    else:
        assessment_label = "リスク高（再リース不可能）"
        recommendation = f"残り期間が負（{remaining_avg:.1f}年）。法定耐用年数を超過するため、再リースは実質不可能。買取終了が前提となる"
    
    periods_data = item.get("periods", {})
    
    return {
        "remanufacture_score": remanufacture_score,
        "assessment_label": assessment_label,
        "expected_years": {
            "min_years": min_years,
            "max_years": max_years,
            "legal_useful_life": legal_useful_life,
            "calculation_method": item.get("calculation_method", "—"),
        },
        "lease_years": lease_years,
        "remaining_years_min": remaining_min,
        "remaining_years_max": remaining_max,
        "remaining_years_avg": remaining_avg,
        "recommendation": recommendation,
        "lease_period_check": {
            "status": lease_status,
            "message": lease_message,
            "legal_min_years": min_lease_years,  # 法定下限（これ以上ならOK）
            "note": f"法定耐用年数 {legal_useful_life}年 × {rule} = {min_lease_years:.1f}年が最小限度額です。これ以上のリース期間なら課税上問題ありません。",
        },
        "periods_reference": periods_data,  # 参考情報
    }


def get_all_categories() -> List[Dict[str, str]]:
    """
    すべての機種カテゴリを取得。UI 用。
    
    Returns
    -------
    list
        各要素：{
            "category": カテゴリ名（例："情報関連機器"）,
            "code": コード（例："0104"）,
            "item_name": 機種名（例："電子計算機"）,
        }
    """
    data = get_usage_period_data()
    if not data:
        return []
    
    result = []
    seen = set()
    for item in data.get("usage_period_data", []):
        key = (item.get("category", ""), item.get("code", ""), item.get("item_name", ""))
        if key not in seen:
            result.append({
                "category": item.get("category", ""),
                "code": item.get("code", ""),
                "item_name": item.get("item_name", ""),
            })
            seen.add(key)
    
    return result


def get_categories_by_group() -> Dict[str, List[Dict[str, str]]]:
    """
    カテゴリグループ別に機種情報を整理。UI のセレクトボックス用。
    
    Returns
    -------
    dict
        {
            カテゴリ名: [
                {
                    "code": "0104",
                    "item_name": "電子計算機",
                },
                ...
            ],
            ...
        }
    """
    data = get_usage_period_data()
    if not data:
        return {}
    
    result: Dict[str, List[Dict]] = {}
    for item in data.get("usage_period_data", []):
        category = item.get("category", "未分類")
        code = item.get("code", "")
        item_name = item.get("item_name", "")
        
        if category not in result:
            result[category] = []
        
        result[category].append({
            "code": code,
            "item_name": item_name,
        })
    
    return result


if __name__ == "__main__":
    # 簡易テスト
    print("=== 期待使用期間マスタ読み込みテスト ===\n")
    
    data = get_usage_period_data()
    print(f"読み込みデータ件数: {len(data.get('usage_period_data', []))}\n")
    
    # キーワード検索テスト
    item = find_item_by_name("電子計算機")
    if item:
        print(f"機種: {item.get('item_name')}")
        print(f"期待使用期間: {item.get('periods')}\n")
    
    # リース期間適合度テスト
    result = calc_lease_period_fit_score("電子計算機", 60)
    print(f"リース60ヶ月での適合度スコア:")
    print(f"  スコア: {result['remanufacture_score']:.1f}")
    print(f"  評価: {result['assessment_label']}")
    print(f"  推奨: {result['recommendation']}\n")
    
    # カテゴリ一覧表示
    categories = get_categories_by_group()
    for cat_name in list(categories.keys())[:2]:
        print(f"カテゴリ: {cat_name}")
        for item_info in categories[cat_name]:
            print(f"  - {item_info['item_name']} ({item_info['code']})")
