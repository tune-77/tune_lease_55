from pydantic import BaseModel, Field
from typing import Literal, Optional

class ScoringRequest(BaseModel):
    company_no: Optional[str] = Field(default="", description="企業番号")
    company_name: Optional[str] = Field(default="", description="企業名")
    nenshu: float = Field(default=0.0, description="売上高（千円）")
    gross_profit: float = Field(default=0.0, description="売上総利益（千円）")
    op_profit: float = Field(default=0.0, description="営業利益（千円）")
    ord_profit: float = Field(default=0.0, description="経常利益（千円）")
    net_income: float = Field(default=0.0, description="純利益（千円）")
    depreciation: float = Field(default=0.0, description="減価償却費（BS/千円）")
    dep_expense: float = Field(default=0.0, description="減価償却費（PL/千円）")
    rent: float = Field(default=0.0, description="賃借料（千円）")
    rent_expense: float = Field(default=0.0, description="支払リース料（千円）")
    machines: float = Field(default=0.0, description="機械装置・運搬具（千円）")
    other_assets: float = Field(default=0.0, description="その他固定資産（千円）")
    net_assets: float = Field(default=0.0, description="純資産（千円）")
    total_assets: float = Field(default=1.0, description="総資産（千円）※0不可")
    
    industry_major: str = Field(default="D 建設業", description="大分類業種")
    industry_sub: str = Field(default="06 総合工事業", description="小分類業種")
    industry_detail: str = Field(default="", description="詳細キーワード")
    
    grade: str = Field(default="②4-6 (標準)", description="社内格付")
    customer_type: str = Field(default="既存先", description="新規先 or 既存先")
    main_bank: str = Field(default="メイン先", description="メイン先 or 非メイン先")
    competitor: str = Field(default="競合なし", description="競合状況")
    competitor_rate: Optional[float] = Field(default=None, description="競合提示金利")
    
    num_competitors: str = Field(default="未入力", description="競合社数")
    deal_occurrence: str = Field(default="不明", description="発生経緯")
    deal_source: str = Field(default="銀行紹介", description="商談ソース")
    sales_dept: str = Field(default="未設定", description="営業部")
    contract_type: str = Field(default="一般", description="契約種類")
    
    bank_credit: float = Field(default=0.0, description="銀行与信残高")
    lease_credit: float = Field(default=0.0, description="リース与信残高")
    contracts: int = Field(default=0, description="契約件数")
    
    lease_term: int = Field(default=60, description="契約期間")
    acquisition_cost: float = Field(default=0, description="取得価格")
    
    asset_score: Optional[float] = Field(default=50.0, description="物件スコア（0-100）")
    selected_asset_id: Optional[str] = Field(default="", description="選択物件ID")
    asset_name: Optional[str] = Field(default="", description="対象物件名")
    asset_detail: Optional[str] = Field(default="", description="型式・メーカー・仕様など")
    asset_purpose: Optional[str] = Field(default="", description="導入目的・用途")
    asset_location: Optional[str] = Field(default="", description="設置場所・使用場所")
    asset_evidence_level: Optional[str] = Field(default="", description="物件確認資料の充足度")
    
    # 定性評価
    qual_corr_company_history: str = Field(default="未選択")
    qual_corr_customer_stability: str = Field(default="未選択")
    qual_corr_repayment_history: str = Field(default="未選択")
    qual_corr_business_future: str = Field(default="未選択")
    qual_corr_equipment_purpose: str = Field(default="未選択")
    qual_corr_main_bank: str = Field(default="未選択")
    
    passion_text: str = Field(default="")
    intuition: int = Field(default=3)

class ScoringResponse(BaseModel):
    score: float
    hantei: str
    comparison: str
    user_op_margin: float
    user_equity_ratio: float
    bench_op_margin: float
    bench_equity_ratio: float
    score_borrower: float
    score_base: Optional[float] = None
    industry_sub: str
    industry_major: str
    ai_completed_factors: Optional[list] = None
    case_id: Optional[str] = None  # DB保存後の案件ID
    company_no: Optional[str] = None
    company_name: Optional[str] = None
    asset_score: Optional[float] = None        # 経路依存: Full審査では加重合成、quickでは警告・表示用
    asset_warnings: Optional[list] = None      # 物件リスク警告フラグ（BEP・換金性・残存価値）
    asset_bonuses: Optional[list] = None       # 物件プラス評価（換金性・残存価値優位）
    default_warnings: list = []                # 高リスク財務パターン警告（実PDではない・スコア非影響）
    quantum_risk: Optional[float] = None       # 量子干渉リスクスコア 0-100（財務矛盾検出）
    financial_consistency_score: Optional[float] = None  # 旧Q_risk: 財務・入力整合性チェック 0-100
    financial_consistency_risk: Optional[dict] = None  # 旧Q_risk詳細 {score, level, patterns, pattern_details}
    credit_quantum_strong_warning: bool = False  # 信用リスク群×Q_risk の強警戒フラグ
    mahalanobis_score: Optional[float] = None  # 財務プロファイル類似度スコア 0-100
    mahalanobis_advice: Optional[list] = None  # 改善アドバイス [{feat, direction, delta}]
    umap_anomaly_score: Optional[float] = None  # 非線形異常スコア 0-100（IF+UMAP）
    umap_x: Optional[float] = None              # UMAP 2D x座標
    umap_y: Optional[float] = None              # UMAP 2D y座標
    umap_similar: Optional[list] = None         # 近傍成約案件 [{x,y,status}]
    diagnostic_recommendations: Optional[list] = None  # UMAP/Mahalanobisなど重い補助診断の人間実行推奨
    conditional_approval_actions: Optional[list] = None  # 条件付き承認時の推奨アクション
    rate_proposal: Optional[dict] = None        # 動的金利提案サマリ
    data_source_summary: Optional[dict] = None  # 入力値・判定根拠の情報源サマリ
    screening_context_notes: Optional[dict] = None  # 入力情報を審査コメント・条件・リスクへ反映したメモ
    approval_comment_draft: Optional[dict] = None  # 稟議コメント案
    estat_context: Optional[dict] = None  # e-Stat統合コンテキスト（業種・リース・景気）
    aurion_core: Optional[dict] = None  # Q_risk/異常度を減点ではなく規律・UXへ翻訳するAURION CORE所見
    bayes_reverse_strategy: Optional[dict] = None  # BN逆転: ベイズ更新後の承認確率と軍師提案

class CaseRegisterRequest(BaseModel):
    case_id: str
    status: str
    final_rate: float
    base_rate_at_time: float
    lost_reason: Optional[str] = ""
    loan_conditions: list = []
    competitor_name: Optional[str] = ""
    competitor_rate: Optional[float] = None
    note: Optional[str] = ""


class DealClosureRequest(BaseModel):
    registration_date: Optional[str] = Field(default=None, description="データ登録日(YYYY-MM-DD)")
    estimate_sent_date: Optional[str] = Field(default=None, description="見積送付日(YYYY-MM-DD)")
    customer_response_date: Optional[str] = Field(default=None, description="顧客反応日(YYYY-MM-DD)")
    final_result_date: Optional[str] = Field(default=None, description="結果日(YYYY-MM-DD)")
    delta_send: Optional[int] = Field(default=None, description="登録→見積の日数差")
    delta_response: Optional[int] = Field(default=None, description="見積→顧客反応の日数差")
    has_cash_data: bool = Field(default=True, description="現預金データ有無")


class DealClosureResponse(BaseModel):
    closure_probability: float
    closure_probability_percent: float
    delta_send: int
    delta_response: int
    model_note: str


class LeaseNewsSummarizeRequest(BaseModel):
    url: Optional[str] = Field(default="", description="ニュースのURL")
    body_text: Optional[str] = Field(default="", description="ニュース本文テキスト")


class LeaseNewsSummaryItem(BaseModel):
    date: str
    title: str
    summary_lines: list[str] = []
    usage_memo: str = ""
    summary_codes: list[str] = []
    usage_codes: list[str] = []
    key_phrases: list[str] = []
    tags: list[str] = []
    region: str = "国内"
    importance: str = "通常"
    source: str = ""
    file_path: str = ""
    week: str = ""
    month: str = ""


class ReviewImprovementRequest(BaseModel):
    key: str
    title: str
    action: Literal["approved", "rejected", "deferred"]
    reason: str = ""


class PromptRuleRegisterRequest(BaseModel):
    title: str
    rule: str
    key: str = ""
    canonical_key: str = ""
    source: str = "manual"
    surface: str = ""
    reason: str = ""
    summary: str = ""


class WorkLogRequest(BaseModel):
    title: str
    what: str
    why_hard: str = ""
    next_time: str = ""
    lesson: str = ""
    pr: str | None = None
    tags: list[str] = ["作業ログ"]


class WorkLogResponse(BaseModel):
    memory_path: str
    obsidian: dict


class BusinessPlanCheckRequest(BaseModel):
    """事業計画チェック（簡易版）のリクエスト。金額は百万円単位。"""
    company_name: str = Field(default="", description="企業名（任意）")
    industry_major: str = Field(default="", description="大分類業種")
    nenshu: float = Field(default=0.0, description="直近売上高（百万円）")
    op_margin_pct: float = Field(default=0.0, description="直近営業利益率（%）")
    plan_nenshu: float = Field(default=0.0, description="計画売上高（百万円）")
    plan_op_margin_pct: float = Field(default=0.0, description="計画営業利益率（%）")
    lease_amount: float = Field(default=0.0, description="リース金額（百万円）")
    lease_months: int = Field(default=0, description="リース期間（回）")
    has_conservative_scenario: bool = Field(default=False, description="保守シナリオの提示有無")
    plan_basis: str = Field(default="", description="計画の根拠（担当者メモ・任意）")
