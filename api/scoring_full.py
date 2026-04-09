import sys
import os
import json
from unittest.mock import MagicMock

# 既存のStreamlit環境への依存を断ち切り、APIから呼び出せるようにするためのモック環境
class MockSessionState(dict):
    def pop(self, k, default=None):
        return super().pop(k, default)
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value

# streamltモックを事前注入
mock_st = MagicMock()
mock_st.session_state = MockSessionState()
sys.modules['streamlit'] = mock_st

# --- 以下、既存ロジックの呼び出し ---
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from components.score_calculation import run_scoring
from constants import REQUIRED_FIELDS, RECOMMENDED_FIELDS

# データキャッシュ
_CACHE = {}

def _load_json(filename):
    if filename in _CACHE:
        return _CACHE[filename]
    path = os.path.join(SCRIPT_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _CACHE[filename] = data
            return data
    except Exception:
        return {}

def run_full_scoring_api(inputs: dict) -> dict:
    """
    既存の `score_calculation.py` の `run_scoring` を、Streamlitを通さずに純粋な関数として実行します。
    """
    # 依存するJSONファイルを読み込み (static_data/ を優先)
    def _get_path(f):
        p1 = os.path.join(SCRIPT_DIR, "static_data", f)
        if os.path.exists(p1): return p1
        return os.path.join(SCRIPT_DIR, f)

    benchmarks_data = _load_json(_get_path("industry_benchmarks.json"))
    hints_data = _load_json(_get_path("industry_hints.json"))
    bankruptcy_data = _load_json(_get_path("bankruptcy_cases.json"))
    jsic_data = _load_json(_get_path("industry_trends_jsic.json"))
    avg_data = _load_json(_get_path("industry_averages.json"))
    rules = _load_json(_get_path("business_rules.json"))
    capex_lease_data = _load_json(_get_path("industry_capex_lease.json"))

    # 入力項目（Pydanticモデル由来）を旧Streamlit由来の変数名にマッピング
    form_result = {
        "submitted_judge": True,
        "nenshu": inputs.get("nenshu", 0),
        "rieki": inputs.get("op_profit", 0),
        "item4_ord_profit": inputs.get("ord_profit", 0),
        "item5_net_income": inputs.get("net_income", 0),
        "item9_gross": inputs.get("gross_profit", 0),
        "item10_dep": inputs.get("depreciation", 0),
        "item11_dep_exp": inputs.get("dep_expense", 0),
        "item8_rent": inputs.get("rent", 0),
        "item12_rent_exp": inputs.get("rent_expense", 0),
        "item6_machine": inputs.get("machines", 0),
        "item7_other": inputs.get("other_assets", 0),
        "net_assets": inputs.get("net_assets", 0),
        "total_assets": inputs.get("total_assets", 1),
        "bank_credit": inputs.get("bank_credit", 0),
        "lease_credit": inputs.get("lease_credit", 0),
        "contracts": inputs.get("contracts", 0),
        "customer_type": inputs.get("customer_type", "既存先"),
        "contract_type": inputs.get("contract_type", "一般"),
        "deal_source": inputs.get("deal_source", "銀行紹介"),
        "lease_term": inputs.get("lease_term", 60),
        "acceptance_year": inputs.get("acceptance_year", 2026),
        "acquisition_cost": inputs.get("acquisition_cost", 0),
        "selected_asset_id": inputs.get("selected_asset_id", ""),
        "asset_score": inputs.get("asset_score", 50.0),
        "asset_name": inputs.get("asset_name", ""),
        "selected_major": inputs.get("industry_major", "D 建設業"),
        "selected_sub": inputs.get("industry_sub", "06 総合工事業"),
        "industry_detail_keyword": inputs.get("industry_detail", ""),
        "grade": inputs.get("grade", "②4-6 (標準)"),
        "passion_text": inputs.get("passion_text", ""),
        "strength_tags": inputs.get("strength_tags", []),
        "num_competitors": inputs.get("num_competitors", "未入力"),
        "deal_occurrence": inputs.get("deal_occurrence", "不明"),
        "main_bank": inputs.get("main_bank", "メイン先"),
        "competitor": inputs.get("competitor", "競合なし"),
        "competitor_rate": inputs.get("competitor_rate"),
        "intuition": inputs.get("intuition", 3),
        # 定性評価
        "qual_corr_company_history": inputs.get("qual_corr_company_history", "未選択"),
        "qual_corr_customer_stability": inputs.get("qual_corr_customer_stability", "未選択"),
        "qual_corr_repayment_history": inputs.get("qual_corr_repayment_history", "未選択"),
        "qual_corr_business_future": inputs.get("qual_corr_business_future", "未選択"),
        "qual_corr_equipment_purpose": inputs.get("qual_corr_equipment_purpose", "未選択"),
        "qual_corr_main_bank": inputs.get("qual_corr_main_bank", "未選択"),
    }

    # セッションステートをクリア＆初期化
    mock_st.session_state.clear()
    
    # フロントエンドからの入力をStreamlitセッションステート互換の形式に流し込む
    for k, v in form_result.items():
        mock_st.session_state[k] = v
        
    mock_st.session_state["_auto_judge"] = True  # 自動判定トリガー
    
    # 既存ロジックの実行
    run_scoring(
        form_result=form_result,
        REQUIRED_FIELDS=REQUIRED_FIELDS,
        benchmarks_data=benchmarks_data,
        hints_data=hints_data,
        bankruptcy_data=bankruptcy_data,
        jsic_data=jsic_data,
        avg_data=avg_data,
        _rules=rules,
        _SCRIPT_DIR=SCRIPT_DIR,
        RECOMMENDED_FIELDS=RECOMMENDED_FIELDS,
        capex_lease_data=capex_lease_data
    )
    
    # 処理結果は st.session_state['last_result'] に格納されるのでそれを抽出
    if "last_result" in mock_st.session_state:
        return mock_st.session_state["last_result"]
    else:
        # エラーなどで結果が出なかった場合
        return {
            "score": 0,
            "hantei": "エラー",
            "comparison": "計算コアで問題が発生しました（必須項目不足など）。",
            "score_borrower": 0,
            "ai_completed_factors": [{"factor": "システムエラー", "effect_percent": 0, "detail": "サーバー内部での実行に失敗しました"}]
        }
