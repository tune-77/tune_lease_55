"""
物件カテゴリ別スコアリング設定 (ver 1.1)
asset scoring design and code.docx + total scorer with ratio.docx に基づく。
"""

# ── カテゴリ別 物件/借手スコア配分比率 ─────────────────────────────────────
# asset_w  : 物件スコアの割合
# obligor_w: 借手スコア（contract_prob）の割合
# rationale: 設定根拠
ASSET_WEIGHT = {
    "車両": {
        "asset_w": 0.35,
        "obligor_w": 0.65,
        "rationale": "成熟した中古車市場により換金性が高く、物件担保力が相対的に大きい",
    },
    "産業機械": {
        "asset_w": 0.25,
        "obligor_w": 0.75,
        "rationale": "汎用機械は担保価値を持つが、借手の業況依存度も大きい",
    },
    "医療機器": {
        "asset_w": 0.20,
        "obligor_w": 0.80,
        "rationale": "大型案件だが薬機法規制リスクにより担保価値が限定される",
    },
    "IT機器": {
        "asset_w": 0.10,
        "obligor_w": 0.90,
        "rationale": "技術陳腐化が速く残存価値はほぼゼロ。借手評価を最重視",
    },
}

# ── カテゴリ別スコアリング項目 ────────────────────────────────────────────────
# weight : 各項目の基本重み（合計 100）
# help   : UI 表示用ツールチップ
# obsolescence_risk / residual_value / liquidity_support: 動的重み調整対象タグ
CATEGORY_SCORE_ITEMS = {
    "IT機器": [
        {
            "id": "tech_obsolescence",
            "label": "技術陳腐化リスク低さ",
            "weight": 30,
            "help": "陳腐化速度が遅い=スコア高。最新AI対応機は陳腐化が速いため低め。",
            "tag": "obsolescence_risk",
        },
        {
            "id": "support_period",
            "label": "サポート期間",
            "weight": 25,
            "help": "メーカーサポート残存年数が長い=スコア高",
            "tag": "liquidity_support",
        },
        {
            "id": "versatility",
            "label": "汎用性",
            "weight": 20,
            "help": "業種を問わず使える=スコア高",
            "tag": None,
        },
        {
            "id": "market_liquidity",
            "label": "市場流動性",
            "weight": 15,
            "help": "中古市場での換金しやすさ",
            "tag": "liquidity_support",
        },
        {
            "id": "lease_period_fit",
            "label": "リース期間適合性",
            "weight": 10,
            "help": "技術寿命とリース期間のバランス（短期設計=スコア高）",
            "tag": None,
        },
    ],
    "産業機械": [
        {
            "id": "versatility",
            "label": "汎用性",
            "weight": 30,
            "help": "複数業種・用途に使える=スコア高",
            "tag": None,
        },
        {
            "id": "maker_brand",
            "label": "メーカーブランド",
            "weight": 25,
            "help": "大手・有名メーカー=スコア高（流動性・サポート品質を反映）",
            "tag": "liquidity_support",
        },
        {
            "id": "operating_env",
            "label": "稼働環境",
            "weight": 20,
            "help": "屋内・管理環境での使用=スコア高（耐久性・残価に影響）",
            "tag": None,
        },
        {
            "id": "physical_durability",
            "label": "物理的耐久性",
            "weight": 15,
            "help": "耐用年数・堅牢性が高い=スコア高",
            "tag": "residual_value",
        },
        {
            "id": "resale_market",
            "label": "再販市場",
            "weight": 10,
            "help": "中古・転売市場が確立されている=スコア高",
            "tag": "residual_value",
        },
    ],
    "車両": [
        {
            "id": "versatility",
            "label": "汎用性",
            "weight": 35,
            "help": "多用途・多業種に使える=スコア高（商用バン・トラック等）",
            "tag": None,
        },
        {
            "id": "mileage_risk",
            "label": "走行距離リスク低さ",
            "weight": 25,
            "help": "低走行 or 走行距離管理良好=スコア高。過走行=スコア低",
            "tag": "obsolescence_risk",
        },
        {
            "id": "market_price",
            "label": "中古市場価格",
            "weight": 20,
            "help": "リース満了時の中古相場が高い=スコア高",
            "tag": "residual_value",
        },
        {
            "id": "ev_tech_risk",
            "label": "EV技術変化リスク低さ",
            "weight": 10,
            "help": "ガソリン・ハイブリッド=中、純EV=技術変化リスクで要注意",
            "tag": "obsolescence_risk",
        },
        {
            "id": "modification",
            "label": "改造・カスタム状況",
            "weight": 10,
            "help": "無改造・標準仕様=スコア高。大改造=転売困難でスコア低",
            "tag": "residual_value",
        },
    ],
    "医療機器": [
        {
            "id": "regulatory_risk",
            "label": "規制リスク低さ",
            "weight": 30,
            "help": "薬機法規制が安定・承認取得済み=スコア高",
            "tag": "obsolescence_risk",
        },
        {
            "id": "tech_cycle",
            "label": "技術サイクル安定性",
            "weight": 25,
            "help": "技術革新が遅い分野=スコア高（MRI・CT等）",
            "tag": "obsolescence_risk",
        },
        {
            "id": "maker_support",
            "label": "メーカーサポート",
            "weight": 25,
            "help": "長期保守・部品供給の安定性が高い=スコア高",
            "tag": "liquidity_support",
        },
        {
            "id": "install_cost",
            "label": "移設コストの低さ",
            "weight": 15,
            "help": "設置・移設コストが低い=転売しやすくスコア高",
            "tag": "residual_value",
        },
        {
            "id": "facility_dependency",
            "label": "施設非依存度",
            "weight": 5,
            "help": "特定施設に依存しない=汎用性高くスコア高",
            "tag": None,
        },
    ],
}

# ── lease_assets.json の id → スコアリングカテゴリ マッピング ────────────────
ASSET_ID_TO_CATEGORY = {
    "vehicle": "車両",
    "medical": "医療機器",
    "it_equipment": "IT機器",
    "manufacturing": "産業機械",
}

# ── グレード閾値（物件スコア・総合スコア共通） ──────────────────────────────
SCORE_GRADES = [
    {"min": 90, "label": "S", "text": "積極承認",    "color": "#22c55e"},
    {"min": 80, "label": "A", "text": "通常承認",    "color": "#3b82f6"},
    {"min": 65, "label": "B", "text": "条件付き承認", "color": "#f59e0b"},
    {"min": 50, "label": "C", "text": "要慎重検討",  "color": "#f97316"},
    {"min":  0, "label": "D", "text": "原則否決",    "color": "#ef4444"},
]
