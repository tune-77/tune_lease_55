import sys
import os
import json
import importlib
import threading
from unittest.mock import MagicMock

# FastAPI はシンク関数をスレッドプールで実行するため、
# 複数リクエストが同時に scoring_output_bridge.json を上書きする競合を防ぐ
_scoring_lock = threading.Lock()

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
mock_st.expander = lambda *args, **kwargs: MagicMock()

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
from constants import REQUIRED_FIELDS, RECOMMENDED_FIELDS, QUALITATIVE_SCORING_CORRECTION_ITEMS
from category_config import ASSET_ID_TO_CATEGORY

# 定性評価: ラベル文字列 → セッション保存用 1-based インデックス の変換マップ
# score_calculation.py は st.session_state["qual_corr_<id>"] に
# 「options リスト内の 1-based 位置」を整数として保存することを期待している。
def _build_qual_label_idx_map() -> dict:
    """{ item_id: { label_str: 1-based-idx } } を返す。"""
    mapping: dict = {}
    for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
        item_map: dict = {}
        for i, opt in enumerate(item.get("options") or []):
            label = opt[1] if isinstance(opt, (list, tuple)) and len(opt) >= 2 else str(opt)
            item_map[label] = i + 1  # 1-based
        mapping[item["id"]] = item_map
    return mapping

_QUAL_LABEL_IDX_MAP: dict = _build_qual_label_idx_map()

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
    with _scoring_lock:
        return _run_full_scoring_api_locked(inputs)


def _run_full_scoring_api_locked(inputs: dict) -> dict:
    print("\n[DEBUG] --- run_full_scoring_api START ---")

    # 物理ファイルを事前に削除 (古い結果を拾わないため)
    RESULT_FILE = os.path.join(SCRIPT_DIR, "scoring_output_bridge.json")
    if os.path.exists(RESULT_FILE):
        try: os.remove(RESULT_FILE)
        except: pass

    # セッション完全クリア
    # _SHARED_SESSION_STATE と mock_st.session_state は別オブジェクトなので両方クリア
    _SHARED_SESSION_STATE.clear()
    mock_st.session_state.clear()

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
        "sales_dept": str(inputs.get("sales_dept", "未設定")),
        "lease_term": int(inputs.get("lease_term", 60)),
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
        "acceptance_year": int(inputs.get("acceptance_year", 2026)),
        "asset_detail": str(inputs.get("asset_detail", "")),
        "asset_purpose": str(inputs.get("asset_purpose", "")),
        "asset_location": str(inputs.get("asset_location", "")),
        "asset_evidence_level": str(inputs.get("asset_evidence_level", "")),
        "_auto_judge": True,
        "_api_mode": True,
        "selected_asset_id": str(inputs.get("selected_asset_id", "other")),
        "asset_category": ASSET_ID_TO_CATEGORY.get(str(inputs.get("selected_asset_id", "other"))),
    }

    # 定性評価: フロントが送る文字列ラベルを 1-based インデックスへ変換して
    # セッションステートにセット（score_calculation.py が期待する形式）
    for _item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
        _key = f"qual_corr_{_item['id']}"
        _label = str(inputs.get(_key, "未選択"))
        _idx = _QUAL_LABEL_IDX_MAP.get(_item["id"], {}).get(_label, 0)
        form_result[_key] = _idx

    # セッションにセット（mock_st.session_state が score_calculation.py から参照される実体）
    for k, v in form_result.items():
        _SHARED_SESSION_STATE[k] = v
        mock_st.session_state[k] = v

    for field in REQUIRED_FIELDS:
        f_key = field[0]
        if f_key not in mock_st.session_state:
            val = form_result.get(f_key, 0)
            _SHARED_SESSION_STATE[f_key] = val
            mock_st.session_state[f_key] = val

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

            # 自己資本マイナスペナルティ（LGBMはnet_assetsを特徴量に含まないため後処理で補正）
            _net = float(inputs.get("net_assets", 0))
            _total = max(1.0, float(inputs.get("total_assets", 1.0)))
            _eq = _net / _total * 100
            if _eq < 0:
                _pen = max(-30.0, _eq * 0.5)
                for _key in ("score", "hantei_score", "score_borrower"):
                    if _key in res and isinstance(res[_key], (int, float)):
                        res[_key] = max(0.0, min(100.0, round(res[_key] + _pen, 1)))
                print(f"[DEBUG] equity_penalty={_pen:.1f} applied (equity_ratio={_eq:.1f}%)")

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
