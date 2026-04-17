import sys
import os
import json
import importlib
from unittest.mock import MagicMock

# --- 審査エンジン用の共有セッション環境 ---
_SHARED_SESSION_STATE = {}

class MockSessionState(dict):
    def __getattr__(self, key): return self.get(key)
    def __setattr__(self, key, value): self[key] = value
    def pop(self, key, default=None): return super().pop(key, default)

# 1. streamlit モックを物理的に固定
mock_st = MagicMock()
mock_st.session_state = MockSessionState(_SHARED_SESSION_STATE)
mock_st.sidebar = MagicMock()
mock_st.columns = lambda n: [MagicMock() for _ in range(n)]
mock_st.tabs = lambda n: [MagicMock() for _ in range(n)]
mock_st.expander = lambda l: MagicMock()

# 重要：グローバルな sys.modules に登録
sys.modules['streamlit'] = mock_st

# 2. パス設定
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# 3. リロード関数
def get_latest_module():
    try:
        import components.score_calculation
        importlib.reload(components.score_calculation)
        return components.score_calculation
    except Exception as e:
        print(f"[CRITICAL_ERROR] Failed to reload scoring module: {e}")
        import components.score_calculation
        return components.score_calculation

# 初回インポート
from constants import REQUIRED_FIELDS, RECOMMENDED_FIELDS

# データキャッシュ
_CACHE = {}

def _load_json(filename):
    if filename in _CACHE: return _CACHE[filename]
    p1 = os.path.join(SCRIPT_DIR, "static_data", filename)
    p2 = os.path.join(SCRIPT_DIR, filename)
    path = p1 if os.path.exists(p1) else p2
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _CACHE[filename] = data
            return data
    except: return {}

def run_full_scoring_api(inputs: dict) -> dict:
    print("\n[DEBUG] --- run_full_scoring_api START ---")
    
    # 物理ファイルを事前に削除 (古い結果を拾わないため)
    RESULT_FILE = "/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring_output_bridge.json"
    if os.path.exists(RESULT_FILE):
        try: os.remove(RESULT_FILE)
        except: pass

    # セッション完全クリア
    _SHARED_SESSION_STATE.clear()
    
    # フロントエンドからの入力マッピング
    form_result = {
        "submitted_judge": True,
        "nenshu": float(inputs.get("nenshu", 0)),
        "rieki": float(inputs.get("op_profit", 0)),
        "item4_ord_profit": float(inputs.get("ord_profit", 0)),
        "item5_net_income": float(inputs.get("net_income", 0)),
        "item9_gross": float(inputs.get("gross_profit", 0)),
        "item10_dep": float(inputs.get("depreciation", 0)),
        "item11_dep_exp": float(inputs.get("dep_expense", 0)),
        "item8_rent": float(inputs.get("rent", 0)),
        "item12_rent_exp": float(inputs.get("rent_expense", 0)),
        "item6_machine": float(inputs.get("machines", 0)),
        "item7_other": float(inputs.get("other_assets", 0)),
        "net_assets": float(inputs.get("net_assets", 0)),
        "total_assets": max(1.0, float(inputs.get("total_assets", 1.0))),
        "bank_credit": float(inputs.get("bank_credit", 0)),
        "lease_credit": float(inputs.get("lease_credit", 0)),
        "contracts": int(inputs.get("contracts", 0)),
        "customer_type": str(inputs.get("customer_type", "既存先")),
        "contract_type": str(inputs.get("contract_type", "一般")),
        "deal_source": str(inputs.get("deal_source", "銀行紹介")),
        "lease_term": int(inputs.get("lease_term", 60)),
        "acceptance_year": 2026,
        "acquisition_cost": float(inputs.get("acquisition_cost", 0)),
        "asset_score": float(inputs.get("asset_score", 50.0)),
        "industry_major": str(inputs.get("industry_major", "G 情報通信業")),
        "industry_sub": str(inputs.get("industry_sub", "39 情報サービス業")),
        "selected_major": str(inputs.get("industry_major", "G 情報通信業")),
        "selected_sub": str(inputs.get("industry_sub", "39 情報サービス業")),
        "grade": str(inputs.get("grade", "②4-6 (標準)")),
        "intuition": int(inputs.get("intuition", 3)),
        "company_no": str(inputs.get("company_no", "")),
        "asset_name": str(inputs.get("asset_name", "")),
        "passion_text": str(inputs.get("passion_text", "")),
        "strength_tags": inputs.get("strength_tags", []),
        "main_bank": str(inputs.get("main_bank", "メイン先")),
        "competitor": str(inputs.get("competitor", "競合なし")),
        "num_competitors": str(inputs.get("num_competitors", "未入力")),
        "deal_occurrence": str(inputs.get("deal_occurrence", "不明")),
        "_auto_judge": True,
        "_api_mode": True
    }

    # セッションにセット
    for k, v in form_result.items():
        _SHARED_SESSION_STATE[k] = v
        
    for field in REQUIRED_FIELDS:
        f_key = field[0]
        if f_key not in _SHARED_SESSION_STATE:
            _SHARED_SESSION_STATE[f_key] = form_result.get(f_key, 0)

    try:
        # 最新のモジュールを取得
        sc_mod = get_latest_module()
        
        # 実行
        print(f"[DEBUG] Executing run_scoring via module {id(sc_mod)}")
        sc_mod.run_scoring(
            form_result=form_result,
            REQUIRED_FIELDS=[],
            benchmarks_data=_load_json("industry_benchmarks.json"),
            hints_data=_load_json("industry_hints.json"),
            bankruptcy_data=_load_json("bankruptcy_cases.json"),
            jsic_data=_load_json("industry_trends_jsic.json"),
            avg_data=_load_json("industry_averages.json"),
            _rules=_load_json("business_rules.json"),
            _SCRIPT_DIR=SCRIPT_DIR,
            RECOMMENDED_FIELDS=[],
            capex_lease_data=_load_json("industry_capex_lease.json")
        )
        print("[DEBUG] run_scoring completed.")
    except Exception as e:
        print(f"[ERROR] Engine Failure: {e}")
        import traceback; traceback.print_exc()

    # ★【物理ファイル通信】ファイルから直接取得
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r", encoding="utf-8") as f:
                res = json.load(f)
            print(f"[DEBUG] SUCCESS: Captured via PHYSICAL FILE. Score={res.get('score')}")
            return res
        except Exception as e:
            print(f"[ERROR] Failed to read physical file: {e}")

    # 万が一、物理ファイルがない場合のバックアップ
    print(f"[DEBUG] Physical file not found. Attempting backup sync.")
    res_backup = _SHARED_SESSION_STATE.get("last_result")
    if res_backup: return res_backup
    
    return {
        "score": _SHARED_SESSION_STATE.get("final_score", 45.0),
        "user_op_margin": _SHARED_SESSION_STATE.get("user_op_margin", 0.0),
        "user_equity_ratio": _SHARED_SESSION_STATE.get("user_equity_ratio", 0.0),
        "hantei": "要確認",
        "comparison": "物理ファイル経由でのデータ抽出に失敗しました。エンジンは完走した可能性があります。"
    }
