"""
ベイジアンネットワーク（BN）による高度リース審査エンジン
依存: pgmpy >= 0.1.21  (pip install pgmpy)

ネットワーク構造:
  ルートノード（外部入力）
    ├─ Insolvent_Status        : 債務超過
    ├─ Main_Bank_Support       : メイン銀行支援
    ├─ Related_Bank_Status     : 関係者の銀行取引良好
    ├─ Related_Assets          : 関係者の個人資産あり
    ├─ Asset_Liquidity         : 物件の資産性（中古流動性）
    ├─ Core_Business_Use       : 本業に不可欠な物件
    ├─ Co_Lease                : 銀行との50%協調リース
    ├─ Parent_Guarantor        : 親会社連帯保証
    ├─ Shorter_Lease_Term      : リース期間短縮
    ├─ One_Time_Deal           : 業況改善まで本件限り
    └─ High_Network_Risk       : 産業ネットワークリスク（グラフ理論による負の相関）

  中間ノード
    ├─ Financial_Creditworthiness : 信用力
    ├─ Hedge_Condition            : リスクヘッジ条件
    └─ Asset_Value                : 物件価値

  出力ノード
    └─ Final_Decision : 最終判断
"""

import itertools
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================
# 判定・逆走提案用の閾値設定
# ==============================================================
RULES_FILE = os.path.join(os.path.dirname(__file__), "data", "business_rules.json")
try:
    with open(RULES_FILE, "r", encoding="utf-8") as _f:
        _rules = json.load(_f)
        THRESHOLD_APPROVAL = _rules.get("thresholds", {}).get("approval", 0.70)
        THRESHOLD_REVIEW = _rules.get("thresholds", {}).get("review", 0.40)
except Exception:
    THRESHOLD_APPROVAL = 0.70
    THRESHOLD_REVIEW = 0.40

try:
    # pgmpy 1.0.0 以降: BayesianNetwork → DiscreteBayesianNetwork に変更
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork  # 旧バージョン互換
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

# ==============================================================
# ネットワーク構造定義
# ==============================================================
BN_EDGES = [
    # 外部補完・関係者属性 + 財務状況 → 信用力
    ("Insolvent_Status",    "Financial_Creditworthiness"),
    ("Main_Bank_Support",   "Financial_Creditworthiness"),
    ("Related_Bank_Status", "Financial_Creditworthiness"),
    ("Related_Assets",      "Financial_Creditworthiness"),
    ("High_Network_Risk",   "Financial_Creditworthiness"),
    ("Parent_Guarantor",    "Financial_Creditworthiness"),
    # 審査ヘッジ手段 + 債務超過フラグ → ヘッジ条件
    ("Insolvent_Status",    "Hedge_Condition"),
    ("Co_Lease",            "Hedge_Condition"),
    ("Parent_Guarantor",    "Hedge_Condition"),
    # 物件・用途 → 資産価値
    ("Core_Business_Use",   "Asset_Value"),
    ("Asset_Liquidity",     "Asset_Value"),
    # 最終判断
    ("Financial_Creditworthiness", "Final_Decision"),
    ("Hedge_Condition",            "Final_Decision"),
    ("Asset_Value",                "Final_Decision"),
    ("Shorter_Lease_Term",         "Final_Decision"),
    ("One_Time_Deal",              "Final_Decision"),
]

# ==============================================================
# ノード定義
# ==============================================================
# 入力ノード（エビデンスとして設定可能）
INPUT_NODES = [
    "Insolvent_Status", "Main_Bank_Support",
    "Related_Bank_Status", "Related_Assets",
    "Asset_Liquidity", "Core_Business_Use",
    "Co_Lease", "Parent_Guarantor",
    "Shorter_Lease_Term", "One_Time_Deal",
    "High_Network_Risk",
]

# 中間・出力ノード
INTERMEDIATE_NODES = [
    "Financial_Creditworthiness", "Hedge_Condition", "Asset_Value",
]

# 否決時に変更提案する逆転候補ノード（優先度順）
REVERSAL_CANDIDATES = [
    "Main_Bank_Support",
    "Co_Lease",
    "Parent_Guarantor",
    "Related_Assets",
    "Related_Bank_Status",
    "Shorter_Lease_Term",
    "One_Time_Deal",
]

# ノード日本語ラベル
NODE_LABELS: Dict[str, str] = {
    "Insolvent_Status":             "債務超過",
    "Main_Bank_Support":            "メイン銀行支援",
    "Related_Bank_Status":          "関係者の銀行取引良好",
    "Related_Assets":               "関係者の個人資産あり",
    "Asset_Liquidity":              "物件の資産性（中古流動性）",
    "Core_Business_Use":            "本業に不可欠な物件",
    "Co_Lease":                     "銀行との50%協調リース",
    "Parent_Guarantor":             "親会社連帯保証",
    "Shorter_Lease_Term":           "リース期間短縮",
    "One_Time_Deal":                "業況改善まで本件限り",
    "High_Network_Risk":             "産業ネットワークリスク（グラフ由来）",
    "Financial_Creditworthiness":   "信用力",
    "Hedge_Condition":              "リスクヘッジ条件",
    "Asset_Value":                  "物件価値",
    "Final_Decision":               "最終判断",
}

# ノード説明（UI表示用）
NODE_DESCRIPTIONS: Dict[str, str] = {
    "Insolvent_Status":         "純資産がマイナスの債務超過状態か",
    "Main_Bank_Support":        "メイン行が審査を後押しし、伴走支援を表明しているか",
    "Related_Bank_Status":      "代表者・関係者が主要銀行と良好な取引実績があるか",
    "Related_Assets":           "代表者・関係者に不動産・現預金など実質的な担保余力があるか",
    "Asset_Liquidity":          "リース物件に中古市場や流動性があり、残価リスクが低いか",
    "Core_Business_Use":        "当該物件が企業の主要業務に直結・不可欠か（撤収リスク低）",
    "Co_Lease":                 "メイン銀行と50%ずつ協調してリース債権を分担するか",
    "Parent_Guarantor":         "親会社・グループ会社が連帯保証人となるか",
    "Shorter_Lease_Term":       "リース期間を標準より短縮してリスク期間を圧縮するか",
    "One_Time_Deal":            "業況改善まで今回限りの対応とし、次回は再審査とするか",
    "High_Network_Risk":        "グラフ理論に基づき、関連業種や供給網の不調から波及するリスクが高いか",
}

# ==============================================================
# 実務制約事例データベース
# ==============================================================
CONSTRAINT_CASE_EXAMPLES = [
    {
        "id": "case_001",
        "title": "債務超過・協調リースで可決（製造業）",
        "description": (
            "自己資本マイナス△3,000万円の製造業。売上・キャッシュフローは安定しているが"
            "財務が脆弱。単独審査では否決ライン。"
        ),
        "evidence": {
            "Insolvent_Status": True,  "Main_Bank_Support": False,
            "Related_Bank_Status": True,  "Related_Assets": True,
            "Co_Lease": True,           "Parent_Guarantor": False,
            "Core_Business_Use": True,  "Asset_Liquidity": True,
            "Shorter_Lease_Term": True, "One_Time_Deal": True,
        },
        "constraint": "Co_Lease（協調リース）が必須条件。単独では否決。",
        "result": "承認",
        "lesson": (
            "債務超過でも「協調リース＋本業必需品＋期間短縮」の三点セットで転換。"
            "関係者資産が信用補完として機能し、銀行との協調で与信リスクを折半。"
        ),
    },
    {
        "id": "case_002",
        "title": "メイン銀行支援が最強の逆転因子（建設業）",
        "description": (
            "経常赤字2期連続の建設業。しかし地銀メイン先で代表者との関係が30年超。"
            "銀行から「我々が伴走します」と一言あった案件。"
        ),
        "evidence": {
            "Insolvent_Status": True,  "Main_Bank_Support": True,
            "Related_Bank_Status": False, "Related_Assets": False,
            "Co_Lease": False,          "Parent_Guarantor": False,
            "Core_Business_Use": True,  "Asset_Liquidity": False,
            "Shorter_Lease_Term": False,"One_Time_Deal": False,
        },
        "constraint": "Main_Bank_Support が単独で最強の逆転因子として機能。",
        "result": "承認",
        "lesson": (
            "財務が債務超過でもメイン銀行の伴走・支援確約があれば信用力は高水準（P≒0.95）。"
            "物件が本業直結であることも加点。他の条件が全てゼロでもメイン支援一つで覆る。"
        ),
    },
    {
        "id": "case_003",
        "title": "関係者資産＋期間短縮で承認（サービス業）",
        "description": (
            "代表者名義の不動産（路線価5,000万円）あり。財務は債務超過直前（純資産＋200万円）。"
            "銀行取引も良好。本業利用の設備をリース希望。"
        ),
        "evidence": {
            "Insolvent_Status": False,  "Main_Bank_Support": False,
            "Related_Bank_Status": True,"Related_Assets": True,
            "Co_Lease": False,          "Parent_Guarantor": False,
            "Core_Business_Use": True,  "Asset_Liquidity": True,
            "Shorter_Lease_Term": True, "One_Time_Deal": False,
        },
        "constraint": "Related_Assets（関係者資産）が信用補完。Shorter_Lease_Termでリスク圧縮。",
        "result": "承認",
        "lesson": (
            "財務単体で不安でも、関係者の個人資産＋銀行取引良好がセットで信用力を大幅補完。"
            "期間短縮がさらにリスクを低減。「財務が弱いから即否決」にならないことを示す典型例。"
        ),
    },
    {
        "id": "case_004",
        "title": "全条件欠如→否決・逆算提案あり（飲食業）",
        "description": (
            "飲食業。債務超過△8,000万円。メイン先なし。関係者資産なし。"
            "物件は汎用厨房機器（中古流動性低）。協調リース不可（銀行関与なし）。"
        ),
        "evidence": {
            "Insolvent_Status": True,   "Main_Bank_Support": False,
            "Related_Bank_Status": False,"Related_Assets": False,
            "Co_Lease": False,          "Parent_Guarantor": False,
            "Core_Business_Use": False, "Asset_Liquidity": False,
            "Shorter_Lease_Term": False,"One_Time_Deal": False,
        },
        "constraint": "逆転因子が一つも存在しない。協調リースか親会社保証があれば検討余地あり。",
        "result": "否決",
        "lesson": (
            "単一の逆転因子（例：協調リースの設定だけでも）確保が重要。"
            "逆算提案では「Co_Leaseをオンにすれば承認確率+26%」等が表示される。"
        ),
    },
    {
        "id": "case_005",
        "title": "親会社保証で与信補完（運送業・子会社）",
        "description": (
            "運送業の100%子会社。子会社単体は営業赤字・債務超過。"
            "親会社（東証スタンダード上場）が連帯保証人として名乗りを上げた。"
        ),
        "evidence": {
            "Insolvent_Status": True,  "Main_Bank_Support": False,
            "Related_Bank_Status": False,"Related_Assets": False,
            "Co_Lease": False,         "Parent_Guarantor": True,
            "Core_Business_Use": True, "Asset_Liquidity": True,
            "Shorter_Lease_Term": False,"One_Time_Deal": True,
        },
        "constraint": "Parent_Guarantor（親会社連帯保証）がヘッジ条件として機能。",
        "result": "承認",
        "lesson": (
            "連結グループの信用力を活用。親会社保証＋本業核心設備＋本件限り対応で承認。"
            "子会社単体の財務だけで判断しないことが重要。グループ全体の与信力を評価。"
        ),
    },
    {
        "id": "case_006",
        "title": "期間短縮＋本件限りで境界線突破（医療・福祉）",
        "description": (
            "訪問看護事業者。財務は赤字だが業種特性（診療報酬収入の安定性）あり。"
            "物件は医療機器で本業直結。期間を60→36ヶ月に短縮し本件限りで条件設定。"
        ),
        "evidence": {
            "Insolvent_Status": False,  "Main_Bank_Support": False,
            "Related_Bank_Status": True,"Related_Assets": False,
            "Co_Lease": False,          "Parent_Guarantor": False,
            "Core_Business_Use": True,  "Asset_Liquidity": True,
            "Shorter_Lease_Term": True, "One_Time_Deal": True,
        },
        "constraint": "Shorter_Lease_Term＋One_Time_Dealの組み合わせが境界線案件を転換。",
        "result": "承認",
        "lesson": (
            "財務単体では要審議の案件でも、期間短縮＋本件限り＋本業利用の組み合わせが有効。"
            "審査員が「自分の任期中に完済する」構造を作ることがリスク管理上も有効。"
        ),
    },
]


# ==============================================================
# CPT（条件付き確率表）計算ロジック
# ==============================================================
def _prob_financial_creditworthiness(i: int, m: int, r: int, ra: int, nr: int, pg: int = 0) -> float:
    """
    信用力 P(True | Insolvent, MainBank, RelBank, RelAssets, NetworkRisk, ParentGuarantor)

    ビジネスロジック：
    ・Main_Bank_Support=T → 最強の救済。債務超過でも信用力高（0.95）
    ・RelBank=T かつ RelAssets=T → 強い補完。債務超過でも高（0.88）
    ・Parent_Guarantor=T → 親会社が子会社の信用を補完。債務超過でも信用力底上げ（0.55）
    ・High_Network_Risk=T → 全体的に信用確率を低減させる（既存値に 0.88 を乗じる。pg 底上げ後に適用）
    ・サポートなし → 非債務超過でも中（0.75）、債務超過は極低（0.05）
    """
    # 入力値を 0/1 に正規化（範囲外の値をサイレントスキップさせない）
    i, m, r, ra, nr, pg = (int(bool(v)) for v in (i, m, r, ra, nr, pg))

    # 基礎確率の計算
    if m == 1:
        base = 0.95 if i == 1 else 0.98
    elif r == 1 and ra == 1:
        base = 0.88 if i == 1 else 0.97
    elif r == 1 and ra == 0:
        base = 0.55 if i == 1 else 0.88
    elif r == 0 and ra == 1:
        base = 0.50 if i == 1 else 0.85
    else:
        base = 0.05 if i == 1 else 0.75
    
    # 親会社連帯保証による信用力補完
    # 子会社単体の財務が弱くても、連結グループの信用力で補完される
    if pg == 1:
        if i == 1:
            # 債務超過＋親会社保証: 連結与信力による信用底上げ
            base = max(base, 0.55)
        else:
            # 非債務超過＋親会社保証: 追加的な信用補完
            base = max(base, 0.85)

    # ネットワークリスクによる補正（負の外部性）
    # pg の底上げ後に適用することで、親保証があっても産業ネットワークリスクは残る
    if nr == 1:
        # 高いリスクネットワークにいる場合、信用力を 12% 程度割り引く
        base *= 0.88

    return base


def _prob_hedge_condition(i: int, co: int, pg: int) -> float:
    """
    ヘッジ条件 P(True | Insolvent, CoLease, ParentGuarantor)

    ビジネスロジック：
    ・非債務超過：ヘッジ条件は加点要素（基本値 0.82）
    ・債務超過＋ヘッジなし：強制否決水準（0.02）
    ・債務超過＋協調 or 保証：承認検討可能（0.65〜0.72）
    ・債務超過＋両方：高い承認可（0.90）
    """
    if i == 0:  # 非債務超過
        if co == 1 and pg == 1: return 0.98
        if co == 1:              return 0.92
        if pg == 1:              return 0.95
        return 0.82
    else:       # 債務超過
        if co == 1 and pg == 1: return 0.90
        if co == 1:              return 0.65
        if pg == 1:              return 0.72
        return 0.02  # 債務超過×ヘッジなし = 強制否決


def _prob_asset_value(core: int, liq: int) -> float:
    """
    物件価値 P(True | CoreBusinessUse, AssetLiquidity)
    """
    table = {(0, 0): 0.15, (0, 1): 0.55, (1, 0): 0.62, (1, 1): 0.95}
    return table[(core, liq)]


def _prob_final_decision(fc: int, hc: int, av: int, st: int, ot: int) -> float:
    """
    最終判断 P(True | FinCredit, HedgeCondition, AssetValue, ShorterTerm, OneTimeDeal)

    ビジネスロジック：
    ・信用力高＋ヘッジ条件整 → 承認（0.93 base）
    ・信用力高＋ヘッジ未整   → 高め（0.72 base）
    ・信用力低＋ヘッジ条件整 → 要審議（0.48 base）
    ・信用力低＋ヘッジ未整   → 否決（0.04 base）
    ・物件価値高・期間短縮・本件限りで加算
    """
    if fc == 1 and hc == 1:
        base = 0.93
    elif fc == 1 and hc == 0:
        base = 0.72
    elif fc == 0 and hc == 1:
        base = 0.48
    else:
        base = 0.04
    if av == 1: base += 0.08
    if st == 1: base += 0.06
    if ot == 1: base += 0.03
    return min(0.99, max(0.01, base))


# ==============================================================
# BNモデル構築
# ==============================================================
def build_bn_model(custom_priors: Dict[str, List[float]] | None = None):
    """
    pgmpy BayesianNetwork を構築して (model, inference_engine) を返す。

    Args:
        custom_priors: ルートノードの事前確率を上書きする dict。
                       estimate_empirical_priors() の返り値を渡すことで
                       ジェフリーズ事前分布による経験的推定が適用される。
    """
    if not PGMPY_AVAILABLE:
        raise ImportError(
            "pgmpy がインストールされていません。\n"
            "  pip install pgmpy\n"
            "を実行してください。"
        )

    model = BayesianNetwork(BN_EDGES)

    # ---- ルートノード（先験確率） ----
    # custom_priors が渡された場合はそちらを優先（経験的ジェフリーズ事前分布）
    default_root_priors = {
        "Insolvent_Status":    [0.80, 0.20],  # [P(False), P(True)]
        "Main_Bank_Support":   [0.60, 0.40],
        "Related_Bank_Status": [0.50, 0.50],
        "Related_Assets":      [0.60, 0.40],
        "Asset_Liquidity":     [0.40, 0.60],
        "Core_Business_Use":   [0.30, 0.70],
        "Co_Lease":            [0.70, 0.30],
        "Parent_Guarantor":    [0.80, 0.20],
        "Shorter_Lease_Term":  [0.60, 0.40],
        "One_Time_Deal":       [0.70, 0.30],
        "High_Network_Risk":   [0.75, 0.25],
    }
    root_priors = {**default_root_priors, **(custom_priors or {})}
    for node, priors in root_priors.items():
        model.add_cpds(
            TabularCPD(variable=node, variable_card=2,
                       values=[[priors[0]], [priors[1]]])
        )

    # ---- Financial_Creditworthiness ----
    parents_fc = ["Insolvent_Status", "Main_Bank_Support",
                  "Related_Bank_Status", "Related_Assets", "High_Network_Risk",
                  "Parent_Guarantor"]
    combos_fc = list(itertools.product(*[range(2)] * 6))
    pt_fc = [_prob_financial_creditworthiness(*c) for c in combos_fc]
    model.add_cpds(TabularCPD(
        variable="Financial_Creditworthiness", variable_card=2,
        values=[[1 - p for p in pt_fc], pt_fc],
        evidence=parents_fc, evidence_card=[2] * 6,
    ))

    # ---- Hedge_Condition ----
    parents_hc = ["Insolvent_Status", "Co_Lease", "Parent_Guarantor"]
    combos_hc = list(itertools.product(*[range(2)] * 3))
    pt_hc = [_prob_hedge_condition(*c) for c in combos_hc]
    model.add_cpds(TabularCPD(
        variable="Hedge_Condition", variable_card=2,
        values=[[1 - p for p in pt_hc], pt_hc],
        evidence=parents_hc, evidence_card=[2] * 3,
    ))

    # ---- Asset_Value ----
    parents_av = ["Core_Business_Use", "Asset_Liquidity"]
    combos_av = list(itertools.product(*[range(2)] * 2))
    pt_av = [_prob_asset_value(*c) for c in combos_av]
    model.add_cpds(TabularCPD(
        variable="Asset_Value", variable_card=2,
        values=[[1 - p for p in pt_av], pt_av],
        evidence=parents_av, evidence_card=[2] * 2,
    ))

    # ---- Final_Decision ----
    parents_fd = ["Financial_Creditworthiness", "Hedge_Condition",
                  "Asset_Value", "Shorter_Lease_Term", "One_Time_Deal"]
    combos_fd = list(itertools.product(*[range(2)] * 5))
    pt_fd = [_prob_final_decision(*c) for c in combos_fd]
    model.add_cpds(TabularCPD(
        variable="Final_Decision", variable_card=2,
        values=[[1 - p for p in pt_fd], pt_fd],
        evidence=parents_fd, evidence_card=[2] * 5,
    ))

    assert model.check_model(), "BNモデルの検証に失敗しました。CPTの設定を確認してください。"
    infer = VariableElimination(model)
    return model, infer


# ==============================================================
# ジェフリーズ事前分布による経験的事前確率の推定
# ==============================================================

def _jeffreys_posterior(successes: int, total: int) -> float:
    """
    ジェフリーズ事前分布 (alpha=0.5, beta=0.5) を使ったベータ分布の事後期待値。
    小サンプルでも極端な 0 や 1 に収束しない。

    P(True) = (successes + 0.5) / (total + 1.0)
    """
    return (successes + 0.5) / max(total + 1.0, 1.0)


def estimate_empirical_priors(db_path: str | None = None) -> Dict[str, List[float]]:
    """
    過去案件データからジェフリーズ事前分布を使って BN ルートノードの事前確率を推定する。
    データが不足している場合はハードコードのデフォルト値にフォールバックする。

    Returns:
        {ノード名: [P(False), P(True)]} 形式の dict
    """
    # デフォルト事前確率（ハードコード）
    defaults: Dict[str, List[float]] = {
        "Insolvent_Status":    [0.80, 0.20],
        "Main_Bank_Support":   [0.60, 0.40],
        "Related_Bank_Status": [0.50, 0.50],
        "Related_Assets":      [0.60, 0.40],
        "Asset_Liquidity":     [0.40, 0.60],
        "Core_Business_Use":   [0.30, 0.70],
        "Co_Lease":            [0.70, 0.30],
        "Parent_Guarantor":    [0.80, 0.20],
        "Shorter_Lease_Term":  [0.60, 0.40],
        "One_Time_Deal":       [0.70, 0.30],
        "High_Network_Risk":   [0.75, 0.25],
    }

    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "data", "lease_data.db")

    if not os.path.exists(db_path):
        return defaults

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT data FROM past_cases WHERE final_status IN ('成約', '失注') LIMIT 500"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f"経験的事前確率の推定に失敗（DBアクセスエラー）: {e}")
        return defaults

    if len(rows) < 5:
        # データが少なすぎる場合はデフォルト
        return defaults

    # Insolvent_Status だけ case data から推定可能（net_assets < 0 で判定）
    insolvent_count = 0
    valid_count = 0
    for (data_json,) in rows:
        try:
            data = json.loads(data_json)
            inputs = data.get("inputs") or data.get("result") or {}
            net_assets = inputs.get("net_assets")
            if net_assets is not None:
                valid_count += 1
                if float(net_assets) < 0:
                    insolvent_count += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    priors = dict(defaults)
    if valid_count >= 5:
        p_insolvent = _jeffreys_posterior(insolvent_count, valid_count)
        priors["Insolvent_Status"] = [round(1.0 - p_insolvent, 4), round(p_insolvent, 4)]
        logger.info(
            f"Insolvent_Status の事前確率を経験的推定: "
            f"P(True)={p_insolvent:.3f} (n={valid_count}, 債務超過={insolvent_count}件)"
        )

    return priors


# キャッシュ（起動時に1回だけ構築）
_bn_model_cache = None
_bn_infer_cache = None


def get_bn_engine(use_empirical_priors: bool = True):
    """
    モデルと推論エンジンを返す（初回のみ構築）。
    use_empirical_priors=True の場合、過去データからジェフリーズ事前分布で事前確率を推定する。
    """
    global _bn_model_cache, _bn_infer_cache
    if _bn_model_cache is None:
        custom_priors = estimate_empirical_priors() if use_empirical_priors else None
        _bn_model_cache, _bn_infer_cache = build_bn_model(custom_priors=custom_priors)
    return _bn_model_cache, _bn_infer_cache


# ==============================================================
# pgmpy不要フォールバック推論（純Python近似計算）
# ==============================================================
def _run_inference_fallback(evidence: Dict[str, int]) -> Dict:
    """
    pgmpy なしで動く近似推論。
    各中間ノードの確率を CPT 関数で直接計算し、
    Final_Decision を全組み合わせの加重和で求める。
    """
    i    = evidence.get("Insolvent_Status",    0)
    m    = evidence.get("Main_Bank_Support",   0)
    r    = evidence.get("Related_Bank_Status", 0)
    ra   = evidence.get("Related_Assets",      0)
    co   = evidence.get("Co_Lease",            0)
    pg   = evidence.get("Parent_Guarantor",    0)
    core = evidence.get("Core_Business_Use",   0)
    liq  = evidence.get("Asset_Liquidity",     0)
    st   = evidence.get("Shorter_Lease_Term",  0)
    ot   = evidence.get("One_Time_Deal",       0)
    nr   = evidence.get("High_Network_Risk",   0)

    p_fc = _prob_financial_creditworthiness(i, m, r, ra, nr, pg)
    p_hc = _prob_hedge_condition(i, co, pg)
    p_av = _prob_asset_value(core, liq)

    # P(fd=1) = Σ_{fc,hc,av} P(fd=1|fc,hc,av,st,ot) * P(fc) * P(hc) * P(av)
    approval_prob = 0.0
    for fc in range(2):
        for hc in range(2):
            for av in range(2):
                p_combo = (
                    (p_fc if fc == 1 else 1 - p_fc) *
                    (p_hc if hc == 1 else 1 - p_hc) *
                    (p_av if av == 1 else 1 - p_av)
                )
                approval_prob += _prob_final_decision(fc, hc, av, st, ot) * p_combo

    if approval_prob >= THRESHOLD_APPROVAL:
        decision = "承認"
    elif approval_prob >= THRESHOLD_REVIEW:
        decision = "要審議"
    else:
        decision = "否決"

    return {
        "approval_prob": approval_prob,
        "decision": decision,
        "intermediate": {
            "Financial_Creditworthiness": p_fc,
            "Hedge_Condition": p_hc,
            "Asset_Value": p_av,
        },
        "evidence": evidence,
    }


# ==============================================================
# 推論インタフェース
# ==============================================================
def run_inference(evidence: Dict[str, bool]) -> Dict:
    """
    BN推論を実行。pgmpy が利用不可の場合は純Python近似推論にフォールバック。
    evidence: {ノード名: True/False}  ← INPUT_NODES から任意個を指定可能
    戻り値:
        approval_prob : float  (0〜1)
        decision      : "承認" | "要審議" | "否決"
        intermediate  : {ノード名: P(True)}  中間ノード3つ
        evidence      : 入力エビデンス
    """
    if not PGMPY_AVAILABLE:
        ev_int: Dict[str, int] = {k: int(v) for k, v in evidence.items()}
        return _run_inference_fallback(ev_int)

    _, infer = get_bn_engine()

    # bool → int（pgmpy は int 値を期待）
    ev_int: Dict[str, int] = {k: int(v) for k, v in evidence.items()}

    # 最終判断
    q_fd = infer.query(["Final_Decision"], evidence=ev_int, show_progress=False)
    approval_prob = float(q_fd.values[1])

    # 中間ノード
    intermediate: Dict[str, float] = {}
    for node in INTERMEDIATE_NODES:
        q = infer.query([node], evidence=ev_int, show_progress=False)
        intermediate[node] = float(q.values[1])

    # 判定ラベル
    if approval_prob >= THRESHOLD_APPROVAL:
        decision = "承認"
    elif approval_prob >= THRESHOLD_REVIEW:
        decision = "要審議"
    else:
        decision = "否決"

    return {
        "approval_prob": approval_prob,
        "decision": decision,
        "intermediate": intermediate,
        "evidence": evidence,
    }


# ==============================================================
# 逆算提案（否決→承認への転換ロジック）
# ==============================================================
def compute_reversal_suggestions(
    evidence: Dict[str, bool], top_n: int = 5
) -> List[Dict]:
    """
    現在の判定が「否決」または「要審議」のとき、
    どのノードを True に変えると承認確率が上がるかを列挙して返す。

    戻り値のリスト（降順ソート済み）:
        node         : ノード名
        label        : 日本語ラベル
        before_prob  : 変更前の承認確率
        after_prob   : 変更後の承認確率
        delta        : 上昇幅
        after_decision: 変更後の判定ラベル
    """
    base = run_inference(evidence)
    base_prob = base["approval_prob"]
    suggestions: List[Dict] = []

    for node in REVERSAL_CANDIDATES:
        # 既に True のノードはスキップ
        if evidence.get(node) is True:
            continue
        new_ev = dict(evidence)
        new_ev[node] = True
        try:
            new_result = run_inference(new_ev)
            delta = new_result["approval_prob"] - base_prob
            if delta > 0.005:
                suggestions.append({
                    "node": node,
                    "label": NODE_LABELS.get(node, node),
                    "before_prob": base_prob,
                    "after_prob": new_result["approval_prob"],
                    "delta": delta,
                    "after_decision": new_result["decision"],
                })
        except Exception as e:
            import traceback
            print(f"[bayesian] compute_reversal_suggestions エラー: {e}\n{traceback.format_exc()}")

    suggestions.sort(key=lambda x: -x["delta"])
    return suggestions[:top_n]


def compute_combination_reversal(
    evidence: Dict[str, bool], target_prob: float = THRESHOLD_APPROVAL
) -> Optional[Dict]:
    """
    単一ノード変更では承認確率が target_prob に届かない場合、
    2ノードの組み合わせで最も効果的な候補を探す。
    """
    candidates = [n for n in REVERSAL_CANDIDATES if evidence.get(n) is not True]
    best = None
    for n1, n2 in itertools.combinations(candidates, 2):
        new_ev = dict(evidence)
        new_ev[n1] = True
        new_ev[n2] = True
        try:
            result = run_inference(new_ev)
            if result["approval_prob"] >= target_prob:
                if best is None or result["approval_prob"] > best["after_prob"]:
                    best = {
                        "nodes": [n1, n2],
                        "labels": [NODE_LABELS.get(n1, n1), NODE_LABELS.get(n2, n2)],
                        "after_prob": result["approval_prob"],
                        "after_decision": result["decision"],
                    }
        except Exception as e:
            import traceback
            print(f"[bayesian] compute_combination_reversal エラー: {e}\n{traceback.format_exc()}")
    return best


# ==============================================================
# 結果の保存・読み込み
# ==============================================================
_BN_CASES_FILE: Optional[str] = None


def set_data_dir(base_dir: str) -> None:
    """保存先ディレクトリを設定（lease_logic_sumaho11.py から呼ぶ）"""
    global _BN_CASES_FILE
    _BN_CASES_FILE = os.path.join(base_dir, "bn_cases.jsonl")


def save_bn_case(case_data: Dict) -> bool:
    """BN推論結果を JSONL に追記保存"""
    if _BN_CASES_FILE is None:
        return False
    try:
        record = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            **case_data,
        }
        with open(_BN_CASES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_bn_cases() -> List[Dict]:
    """保存済みBN推論結果を全件読み込む"""
    if _BN_CASES_FILE is None or not os.path.exists(_BN_CASES_FILE):
        return []
    cases: List[Dict] = []
    try:
        with open(_BN_CASES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    except Exception:
        pass
    return cases


def get_constraint_case_examples() -> List[Dict]:
    """制約事例データベースを返す"""
    return CONSTRAINT_CASE_EXAMPLES
