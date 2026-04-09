from pydantic import BaseModel, Field
from typing import Optional

class ScoringRequest(BaseModel):
    company_no: Optional[str] = Field(default="", description="企業番号")
    company_name: Optional[str] = Field(default="", description="企業名")
    nenshu: float = Field(default=0.0, description="売上高（千円）")
    op_profit: float = Field(default=0.0, description="営業利益（千円）")
    ord_profit: float = Field(default=0.0, description="経常利益（千円）")
    net_income: float = Field(default=0.0, description="純利益（千円）")
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
    contract_type: str = Field(default="一般", description="契約種類")
    
    bank_credit: float = Field(default=0.0, description="銀行与信残高")
    lease_credit: float = Field(default=0.0, description="リース与信残高")
    contracts: int = Field(default=0, description="契約件数")
    
    lease_term: int = Field(default=60, description="契約期間")
    acquisition_cost: float = Field(default=0, description="取得価格")
    
    asset_score: Optional[float] = Field(default=50.0, description="物件スコア（0-100）")
    selected_asset_id: Optional[str] = Field(default="", description="選択物件ID")
    
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
