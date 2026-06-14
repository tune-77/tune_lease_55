"""
審査スコア計算コア（Streamlit 非依存）
Flask 等の Web アプリから呼び出す用。判定ロジックの簡易版（定量メイン＋業界比較）。
"""
import os
import math
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# UMAP モデルキャッシュ（run_quick_scoring 呼び出しのたびにロードしない）
_umap_model_cache = None
_umap_model_path_cache: str | None = None

# パスを追加してからインポート
import sys
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_cases import get_effective_coeffs
from app_logger import log_warning
from estat_context import build_estat_context
from useful_life_lookup import get_legal_useful_life

APPROVAL_LINE = int(os.environ.get("APPROVAL_LINE", "71"))  # 承認ライン（デフォルト71点）

# 担当者直感スコア（1-5）の最大補正幅（点）
INTUITION_MAX_ADJ = 3.0

# 営業部 one-hot: ベース=未設定（全0）
SALES_DEPT_OPTIONS = ["宇都宮営業部", "小山営業部", "足利営業部", "埼玉営業部"]

# SHAP: 各係数キーの日本語ラベル
_FEATURE_LABELS_JA = {
    "intercept":          "基本スコア（切片）",
    "ind_medical":        "業種（医療・福祉）",
    "ind_transport":      "業種（運輸・運送）",
    "ind_construction":   "業種（建設）",
    "ind_manufacturing":  "業種（製造）",
    "ind_service":        "業種（卸売・サービス）",
    "sales_log":          "売上規模",
    "bank_credit_log":    "銀行信用枠",
    "lease_credit_log":   "リース信用枠",
    "op_profit":          "営業利益",
    "ord_profit":         "経常利益",
    "net_income":         "純利益",
    "gross_profit":       "粗利益",
    "machines":           "機械装置",
    "other_assets":       "その他資産",
    "rent":               "家賃",
    "depreciation":       "減価償却費",
    "dep_expense":        "減価償却費（費用）",
    "rent_expense":       "賃借料（費用）",
    "grade_4_6":          "格付（4-6ランク）",
    "grade_watch":        "格付（要注意）",
    "grade_none":         "格付（無格付）",
    "contracts":          "過去取引件数",
    "dept_utsunomiya":    "営業部（宇都宮）",
    "dept_oyama":         "営業部（小山）",
    "dept_ashikaga":      "営業部（足利）",
    "dept_saitama":       "営業部（埼玉）",
    "new_customer_main_bank":         "新規先×メイン先",
    "new_customer_competitor_present": "新規先×競合あり",
    "new_customer_competitor_count":   "新規先×競合社数",
    "new_customer_competitor_rate":    "新規先×競合提示金利",
    "new_customer_deal_source_bank":   "新規先×銀行紹介",
    "new_customer_deal_occurrence_nomination": "新規先×指名案件",
    "new_customer_deal_occurrence_comp": "新規先×相見積もり",
    "new_customer_contract_auto":      "新規先×自動車契約",
}


def _safe_float(val, default: float = 0.0) -> float:
    """安全なfloat変換。Noneや変換不能な値はdefaultを返す。"""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """安全なint変換。Noneや変換不能な値はdefaultを返す。"""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _load_benchmarks():
    """industry_benchmarks.json を読み込む。"""
    candidates = [
        os.path.join(_SCRIPT_DIR, "industry_benchmarks.json"),
        os.path.join(_SCRIPT_DIR, "static_data", "industry_benchmarks.json"),
        os.path.join(_SCRIPT_DIR, "data", "industry_benchmarks.json"),
        os.path.join(_REPO_ROOT, "static_data", "industry_benchmarks.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_warning(f"業界目安JSON読み込み失敗: {e}", context="_load_benchmarks")
        return {}


def _load_capex_lease_data():
    """industry_capex_lease.json を読み込む。"""
    candidates = [
        os.path.join(_SCRIPT_DIR, "data", "industry_capex_lease.json"),
        os.path.join(_SCRIPT_DIR, "static_data", "industry_capex_lease.json"),
        os.path.join(_REPO_ROOT, "data", "industry_capex_lease.json"),
        os.path.join(_REPO_ROOT, "static_data", "industry_capex_lease.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_warning(f"設備投資・リースJSON読み込み失敗: {e}", context="_load_capex_lease_data")
        return {}


def _safe_sigmoid(x):
    try:
        if x > 700:
            return 1.0
        if x < -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except (OverflowError, TypeError):
        return 0.0 if (x or 0) < 0 else 1.0


def _normalize_competitor_count_value(value) -> float:
    """競合社数を 0〜3 の連続値に正規化する。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            if v <= 0:
                return 0.0
            return min(3.0, v)
        except (TypeError, ValueError):
            return 0.0
    s = str(value)
    if "3" in s:
        return 3.0
    if "2" in s:
        return 2.0
    if "1" in s:
        return 1.0
    if "0" in s or "指名" in s:
        return 0.0
    return 0.0


def _normalize_deal_occurrence_value(value) -> float:
    """発生経緯を 0=不明, 1=指名, 2=相見積もり に正規化する。"""
    if value is None:
        return 0.0
    s = str(value)
    if "相見積" in s or "競争" in s:
        return 2.0
    if "指名" in s:
        return 1.0
    return 0.0


def _normalize_competitor_rate_value(value) -> float:
    """競合提示金利を 0〜1 の圧力指標に正規化する。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    if v <= 1.0:
        v *= 100.0
    elif v > 1000:
        v /= 1000.0
    return max(0.0, min(1.0, v / 30.0))


_LGB_QUAL_MODEL_PATH = os.path.join(_SCRIPT_DIR, "data", "lgb_qual_model.joblib")
_LGBM_DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "lgbm_model.pkl")

_main_bundle_cache: dict[str, dict | None] = {}
_lgb_qual_bundle_cache: dict | None = None
_lgbm_default_model_cache = None
_lgbm_default_model_loaded: bool = False


def clear_scoring_cache() -> None:
    """主モデルのインメモリキャッシュをクリアする。
    統合再学習後に呼ぶことで、再起動なしに新モデルが反映される。
    """
    global _main_bundle_cache
    global _lgb_qual_bundle_cache
    global _lgbm_default_model_cache, _lgbm_default_model_loaded
    _main_bundle_cache = {}
    _lgb_qual_bundle_cache = None
    _lgbm_default_model_cache = None
    _lgbm_default_model_loaded = False


def _load_lgb_qual_bundle():
    global _lgb_qual_bundle_cache
    if _lgb_qual_bundle_cache is not None:
        return _lgb_qual_bundle_cache
    if not os.path.exists(_LGB_QUAL_MODEL_PATH):
        return None
    try:
        import joblib
        _lgb_qual_bundle_cache = joblib.load(_LGB_QUAL_MODEL_PATH)
        return _lgb_qual_bundle_cache
    except Exception:
        return None


def _load_lgbm_default_model():
    global _lgbm_default_model_cache, _lgbm_default_model_loaded
    if _lgbm_default_model_loaded:
        return _lgbm_default_model_cache
    _lgbm_default_model_loaded = True
    if not os.path.exists(_LGBM_DEFAULT_MODEL_PATH):
        return None
    try:
        import joblib
        _lgbm_default_model_cache = joblib.load(_LGBM_DEFAULT_MODEL_PATH)
        return _lgbm_default_model_cache
    except Exception:
        return None


def _build_default_model_feature_row(inputs: dict) -> list[float]:
    """inputs から lgbm_model.pkl 用の特徴量ベクトルを構築する。
    retraining_pipeline.py の _inputs_to_feature_row と同一変換。
    """
    major = str(inputs.get("industry_major") or "").strip()
    grade = str(inputs.get("grade") or "").strip()

    ind_medical       = 1.0 if ("医療" in major or "福祉" in major) else 0.0
    ind_transport     = 1.0 if "運輸" in major else 0.0
    ind_construction  = 1.0 if "建設" in major else 0.0
    ind_manufacturing = 1.0 if "製造" in major else 0.0
    ind_service       = 1.0 if any(x in major for x in ["卸売", "小売", "サービス"]) else 0.0

    nenshu       = max(_safe_float(inputs.get("nenshu")), 0.0)
    bank_credit  = max(_safe_float(inputs.get("bank_credit")), 0.0)
    lease_credit = max(_safe_float(inputs.get("lease_credit")), 0.0)

    grade_4_6   = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none  = 1.0 if ("無格付" in grade or grade in ("0", "④無格付")) else 0.0

    return [
        ind_medical, ind_transport, ind_construction, ind_manufacturing, ind_service,
        np.log1p(nenshu), np.log1p(bank_credit), np.log1p(lease_credit),
        _safe_float(inputs.get("op_profit"))    / 1000.0,
        _safe_float(inputs.get("ord_profit"))   / 1000.0,
        _safe_float(inputs.get("net_income"))   / 1000.0,
        _safe_float(inputs.get("machines"))     / 1000.0,
        _safe_float(inputs.get("other_assets")) / 1000.0,
        _safe_float(inputs.get("rent"))         / 1000.0,
        _safe_float(inputs.get("gross_profit")) / 1000.0,
        _safe_float(inputs.get("depreciation")) / 1000.0,
        _safe_float(inputs.get("dep_expense"))  / 1000.0,
        _safe_float(inputs.get("rent_expense")) / 1000.0,
        grade_4_6, grade_watch, grade_none,
        _safe_float(inputs.get("contracts")),
    ]


DEFAULT_PROB_THRESHOLD = 0.3


def generate_default_warnings(financial_inputs: dict) -> list[str]:
    """デフォルト率モデルで高リスク判定した場合に警告フラグを返す。
    閾値: default_prob >= 0.3 → 警告あり
    モデルが存在しない・特徴量不足・予測失敗の場合は [] を返す。
    """
    model = _load_lgbm_default_model()
    if model is None:
        return []
    try:
        row = _build_default_model_feature_row(financial_inputs)
        n_expected = getattr(model, "n_features_in_", len(row))
        if len(row) != n_expected:
            return []
        X = np.array([row], dtype=float)
        prob = float(model.predict_proba(X)[0][1])
        if prob >= DEFAULT_PROB_THRESHOLD:
            return [
                f"財務パターンが高リスク格付先（格付9相当）と類似（確率{prob:.0%}）"
            ]
        return []
    except Exception:
        return []


def _build_lgb_qual_feature_vector(inputs: dict, feature_names: list[str], asset_to_idx: dict) -> list[float]:
    """inputs から定性 LGB モデル用特徴量ベクトルを構築する。"""
    main_bank = inputs.get("main_bank") or "非メイン先"
    competitor = inputs.get("competitor") or "競合なし"
    customer_type = inputs.get("customer_type") or "既存先"
    deal_source = inputs.get("deal_source") or "その他"
    asset_id = inputs.get("lease_asset_id") or inputs.get("lease_asset_name") or "未選択"
    qsc = inputs.get("qualitative_scoring_correction") or inputs.get("qualitative_scoring") or {}
    items_data = qsc.get("items") or {}

    base_vals = {
        "取引区分_メイン先":   1.0 if main_bank == "メイン先" else 0.0,
        "競合状況_競合あり":   1.0 if competitor == "競合あり" else 0.0,
        "顧客区分_新規先":     1.0 if customer_type == "新規先" else 0.0,
        "商談ソース_銀行紹介": 1.0 if deal_source == "銀行紹介" else 0.0,
        "リース物件":          float(asset_to_idx.get(asset_id, 0)),
    }
    row = []
    from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS
    qual_label_to_id = {it["label"]: it["id"] for it in QUALITATIVE_SCORING_CORRECTION_ITEMS}
    for fn in feature_names:
        if fn in base_vals:
            row.append(base_vals[fn])
        elif fn in qual_label_to_id:
            qid = qual_label_to_id[fn]
            val = items_data.get(qid, {})
            v = val.get("value") if isinstance(val, dict) else None
            row.append(float(v) if isinstance(v, (int, float)) else -1.0)
        else:
            row.append(0.0)
    return row


def _build_lgb_feature_vector(data_scoring: dict, inputs: dict, feature_names: list[str]) -> list[float]:
    """data_scoring + inputs から LGB モデル用特徴量ベクトルを構築する。"""
    major = data_scoring.get("industry_major") or ""
    ind_medical      = 1.0 if ("医療" in major or "福祉" in major or major.startswith("P")) else 0.0
    ind_transport    = 1.0 if ("運輸" in major or major.startswith("H")) else 0.0
    ind_construction = 1.0 if ("建設" in major or major.startswith("D")) else 0.0
    ind_manufacturing= 1.0 if ("製造" in major or major.startswith("E")) else 0.0
    ind_service      = 1.0 if (any(x in major for x in ["卸売", "小売", "サービス"]) or
                                (bool(major) and major[0] in ["I", "K", "M", "R"])) else 0.0

    nenshu = data_scoring.get("nenshu") or 0
    bank_credit = data_scoring.get("bank_credit") or 0
    lease_credit = data_scoring.get("lease_credit") or 0
    grade = data_scoring.get("grade") or "1-3"
    grade_4_6   = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none  = 1.0 if "無格付" in grade else 0.0

    main_val = {
        "ind_medical": ind_medical, "ind_transport": ind_transport,
        "ind_construction": ind_construction, "ind_manufacturing": ind_manufacturing,
        "ind_service": ind_service,
        "sales_log": np.log1p(nenshu), "bank_credit_log": np.log1p(bank_credit),
        "lease_credit_log": np.log1p(lease_credit),
        "op_profit": data_scoring.get("op_profit") or 0,
        "ord_profit": data_scoring.get("ord_profit") or 0,
        "net_income": data_scoring.get("net_income") or 0,
        "machines": data_scoring.get("machines") or 0,
        "other_assets": data_scoring.get("other_assets") or 0,
        "rent": data_scoring.get("rent") or 0,
        "gross_profit": data_scoring.get("gross_profit") or 0,
        "depreciation": data_scoring.get("depreciation") or 0,
        "dep_expense": data_scoring.get("dep_expense") or 0,
        "rent_expense": data_scoring.get("rent_expense") or 0,
        "grade_4_6": grade_4_6, "grade_watch": grade_watch, "grade_none": grade_none,
        "contracts": data_scoring.get("contracts") or 0,
        "main_bank": 1.0 if inputs.get("main_bank") == "メイン先" else 0.0,
        "competitor_present": 1.0 if inputs.get("competitor") == "競合あり" else 0.0,
        "competitor_none": 1.0 if inputs.get("competitor") == "競合なし" else 0.0,
        "customer_new": 1.0 if (inputs.get("customer_type") or "既存先") == "新規先" else 0.0,
        "deal_source_bank": 1.0 if (inputs.get("deal_source") or "") == "銀行紹介" else 0.0,
        "rate_diff_z": 0.0,
        "industry_sentiment_z": 0.0,
        "qualitative_tag_score": 0.0,
        "qualitative_passion": 0.0,
        "equity_ratio": _safe_float(inputs.get("user_equity_ratio")),
        "dscr_approx": compute_dscr_approx(inputs),
        "interest_coverage": compute_interest_coverage(inputs),
        "qualitative_combined": 0.0,
        "bn_approval_prob": 0.0, "bn_fc": 0.0, "bn_hc": 0.0, "bn_av": 0.0,
        "qual_weighted": 0.0, "qual_rank_good": 0.0, "qual_repayment": 0.0,
    }
    customer_new = 1.0 if (inputs.get("customer_type") or "既存先") == "新規先" else 0.0
    competitor_count = _normalize_competitor_count_value(inputs.get("num_competitors"))
    deal_occurrence = _normalize_deal_occurrence_value(inputs.get("deal_occurrence"))
    competitor_rate_norm = _normalize_competitor_rate_value(inputs.get("competitor_rate"))
    contract_type = inputs.get("contract_type") or "一般"
    main_val.update({
        "new_customer_main_bank": customer_new if inputs.get("main_bank") == "メイン先" else 0.0,
        "new_customer_competitor_present": customer_new if inputs.get("competitor") == "競合あり" else 0.0,
        "new_customer_competitor_count": customer_new * competitor_count,
        "new_customer_competitor_rate": customer_new * competitor_rate_norm,
        "new_customer_deal_source_bank": customer_new if (inputs.get("deal_source") or "") == "銀行紹介" else 0.0,
        "new_customer_deal_occurrence_nomination": customer_new if deal_occurrence == 1.0 else 0.0,
        "new_customer_deal_occurrence_comp": customer_new if deal_occurrence == 2.0 else 0.0,
        "new_customer_contract_auto": customer_new if contract_type == "自動車" else 0.0,
    })
    qsc = inputs.get("qualitative_scoring_correction") or inputs.get("qualitative_scoring") or {}
    if qsc:
        combined = qsc.get("combined_score") or qsc.get("weighted_score")
        if combined is not None:
            main_val["qualitative_combined"] = float(combined) / 100.0
        ws = qsc.get("weighted_score")
        if ws is not None:
            main_val["qual_weighted"] = float(ws) / 100.0
        rank = qsc.get("rank") or ""
        main_val["qual_rank_good"] = 1.0 if rank in ("A", "B") else 0.0
        rh_val = ((qsc.get("items") or {}).get("repayment_history") or {}).get("value") or 0
        main_val["qual_repayment"] = float(rh_val) / 4.0

    return [main_val.get(k, 0.0) for k in feature_names]


def _get_industry_flags(industry_major: str) -> dict[str, bool]:
    """industry_major 文字列から業種フラグ dict を返す。"""
    m = industry_major or ""
    return {
        "ind_medical":       "医療" in m or "福祉" in m or m.startswith("P"),
        "ind_transport":     "運輸" in m or m.startswith("H"),
        "ind_construction":  "建設" in m or m.startswith("D"),
        "ind_manufacturing": "製造" in m or m.startswith("E"),
        "ind_service":       any(x in m for x in ["卸売", "小売", "サービス"])
                             or (bool(m) and m[0] in ["I", "K", "M", "R"]),
    }


def _calculate_z(data, coeff_set):
    """係数セットとデータから z (ロジット) を計算。"""
    z = coeff_set.get("intercept", 0)
    flags = _get_industry_flags(data.get("industry_major"))
    for feat, active in flags.items():
        if active:
            z += coeff_set.get(feat, 0)
            break  # 業種は排他的に1つ選択

    nenshu = data.get("nenshu") or 0
    if nenshu > 0:
        z += np.log1p(nenshu) * coeff_set.get("sales_log", 0)
    bc = data.get("bank_credit") or 0
    if bc > 0:
        z += np.log1p(bc) * coeff_set.get("bank_credit_log", 0)
    lc = data.get("lease_credit") or 0
    if lc > 0:
        z += np.log1p(lc) * coeff_set.get("lease_credit_log", 0)

    z += (data.get("op_profit") or 0) * coeff_set.get("op_profit", 0)
    z += (data.get("ord_profit") or 0) * coeff_set.get("ord_profit", 0)
    z += (data.get("net_income") or 0) * coeff_set.get("net_income", 0)
    z += (data.get("machines") or 0) * coeff_set.get("machines", 0)
    z += (data.get("other_assets") or 0) * coeff_set.get("other_assets", 0)
    z += (data.get("rent") or 0) * coeff_set.get("rent", 0)
    z += (data.get("gross_profit") or 0) * coeff_set.get("gross_profit", 0)
    z += (data.get("depreciation") or 0) * coeff_set.get("depreciation", 0)
    z += (data.get("dep_expense") or 0) * coeff_set.get("dep_expense", 0)
    z += (data.get("rent_expense") or 0) * coeff_set.get("rent_expense", 0)

    grade = data.get("grade") or "1-3"
    if "4-6" in grade:
        z += coeff_set.get("grade_4_6", 0)
    elif "要注意" in grade:
        z += coeff_set.get("grade_watch", 0)
    elif "無格付" in grade:
        z += coeff_set.get("grade_none", 0)

    z += (data.get("contracts") or 0) * coeff_set.get("contracts", 0)

    sales_dept = data.get("sales_dept", "未設定")
    dept_utsunomiya = 1 if sales_dept == "宇都宮営業部" else 0
    dept_oyama      = 1 if sales_dept == "小山営業部" else 0
    dept_ashikaga   = 1 if sales_dept == "足利営業部" else 0
    dept_saitama    = 1 if sales_dept == "埼玉営業部" else 0
    z += dept_utsunomiya * coeff_set.get("dept_utsunomiya", 0.0)
    z += dept_oyama      * coeff_set.get("dept_oyama", 0.0)
    z += dept_ashikaga   * coeff_set.get("dept_ashikaga", 0.0)
    z += dept_saitama    * coeff_set.get("dept_saitama", 0.0)
    return z


def compute_score_contributions(data: dict, coeff_set: dict) -> list[dict]:
    """
    線形モデルの各特徴量の寄与度（SHAP近似）を計算して返す。
    ロジスティック回帰では β_i * x_i がそのままロジット空間での寄与に対応する。

    Returns:
        [{"feature": str, "label_ja": str, "contribution": float}, ...] を
        |contribution| 降順でソートしたリスト。
    """
    contributions = []

    # 切片
    intercept_val = coeff_set.get("intercept", 0)
    contributions.append({
        "feature": "intercept",
        "label_ja": _FEATURE_LABELS_JA["intercept"],
        "contribution": intercept_val,
    })

    ind_flags = _get_industry_flags(data.get("industry_major"))
    for feat, active in ind_flags.items():
        coeff_val = coeff_set.get(feat, 0)
        if active and coeff_val != 0:
            contributions.append({
                "feature": feat,
                "label_ja": _FEATURE_LABELS_JA.get(feat, feat),
                "contribution": coeff_val,
            })

    # 連続値特徴量
    def _log_contrib(key_data, key_coeff, log_func=np.log1p):
        val = data.get(key_data) or 0
        coeff_val = coeff_set.get(key_coeff, 0)
        if val > 0 and coeff_val != 0:
            return coeff_val * log_func(val)
        return 0.0

    for data_key, coeff_key in [("nenshu", "sales_log"),
                                  ("bank_credit", "bank_credit_log"),
                                  ("lease_credit", "lease_credit_log")]:
        c = _log_contrib(data_key, coeff_key)
        if c != 0:
            contributions.append({
                "feature": coeff_key,
                "label_ja": _FEATURE_LABELS_JA.get(coeff_key, coeff_key),
                "contribution": c,
            })

    for linear_key in ["op_profit", "ord_profit", "net_income", "machines",
                        "other_assets", "rent", "gross_profit", "depreciation",
                        "dep_expense", "rent_expense"]:
        val = data.get(linear_key) or 0
        coeff_val = coeff_set.get(linear_key, 0)
        if val != 0 and coeff_val != 0:
            contributions.append({
                "feature": linear_key,
                "label_ja": _FEATURE_LABELS_JA.get(linear_key, linear_key),
                "contribution": coeff_val * val,
            })

    # 格付フラグ
    grade = data.get("grade") or "1-3"
    grade_flags = {
        "grade_4_6":    "4-6" in grade,
        "grade_watch":  "要注意" in grade,
        "grade_none":   "無格付" in grade,
    }
    for feat, active in grade_flags.items():
        coeff_val = coeff_set.get(feat, 0)
        if active and coeff_val != 0:
            contributions.append({
                "feature": feat,
                "label_ja": _FEATURE_LABELS_JA.get(feat, feat),
                "contribution": coeff_val,
            })

    # 取引件数
    contracts_val = data.get("contracts") or 0
    contracts_coeff = coeff_set.get("contracts", 0)
    if contracts_val != 0 and contracts_coeff != 0:
        contributions.append({
            "feature": "contracts",
            "label_ja": _FEATURE_LABELS_JA["contracts"],
            "contribution": contracts_coeff * contracts_val,
        })

    # |寄与度| 降順でソート
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


def compute_dscr_approx(inputs: dict) -> float:
    """
    DSCR（債務返済カバレッジ比率）の近似値を計算する。
    既存データから: DSCR ≈ 営業利益 ÷ (減価償却費 + 賃借料費用)
    返値: DSCR 近似値。分母が0の場合は中立値 1.0 を返す。
    値の解釈: 1.0 以上 = 返済能力あり、1.5 以上 = 安全圏
    """
    op_profit = _safe_float(inputs.get("op_profit") or inputs.get("rieki"))
    dep_expense = _safe_float(inputs.get("dep_expense") or inputs.get("depreciation"))
    rent_expense = _safe_float(inputs.get("rent_expense") or inputs.get("rent"))
    denominator = dep_expense + rent_expense
    if denominator <= 0:
        return 1.0
    return round(op_profit / denominator, 3)


def compute_interest_coverage(inputs: dict) -> float:
    """
    インタレスト・カバレッジ・レシオを計算する。
    ICR = 営業利益 ÷ 支払利息
    支払利息データがない場合は 10.0（高安全）を返す。
    """
    op_profit = _safe_float(inputs.get("op_profit") or inputs.get("rieki"))
    interest = _safe_float(inputs.get("interest_expense"))
    if interest <= 0:
        return 10.0  # 無借金とみなす
    return round(op_profit / interest, 3)


def generate_asset_warnings(
    asset_name: str, term_months: int, industry_sub: str = ""
) -> tuple[list[str], list[str]]:
    """物件名・リース期間・借主業種から換金性・BEP・残存価値を評価し (warnings, bonuses) を返す。

    Args:
        industry_sub: 借主の業種小分類（例: "44 道路貨物運送業"）。
                      指定すると別表第二業種別設備の耐用年数を優先適用する。
    """
    name = (asset_name or "").lower()

    # 換金性マッピング（耐用年数は get_legal_useful_life に委譲）
    LIQUIDITY_MAPPING = [
        (["トラック", "スーパーグレード", "プロフィア", "キャリイ", "elf", "エルフ",
          "キャンター", "デュトロ", "ダイナ", "レンジャー"],      "高"),
        (["フォークリフト"],                                       "高"),
        (["ブルドーザー"],                                         "中"),
        (["クレーン"],                                             "中"),
        (["ショベル", "バックホウ", "ユンボ"],                     "中"),
        (["医療", "介護", "mri", "ct", "レントゲン", "透析"],    "中"),
        (["複合機", "コピー機", "oa", "プリンタ"],               "中"),
        (["防犯カメラ", "監視カメラ", "セキュリティカメラ"],      "低"),
        (["pc", "パソコン", "ソフト", "サーバ", "ネットワーク",
          "タブレット", "スイッチ", "ルータ"],                    "極低"),
        (["工作機械", "旋盤", "マシニング", "プレス", "射出成形",
          "製造機", "溶接機"],                                    "低"),
    ]

    # 別表第二（業種別）> 別表第一 の優先順位で耐用年数を取得
    useful_life_yr = get_legal_useful_life(name, industry_sub)
    liquidity = "中"
    for keywords, liq in LIQUIDITY_MAPPING:
        if any(kw in name for kw in keywords):
            liquidity = liq
            break

    term = max(1, term_months)
    r_annual = 2.0 / useful_life_yr
    r_monthly = r_annual / 12.0

    # BEP月: 残存価値率 V(t) = (1-r_monthly)^t が残債比率 (term-t)/term を下回る最初の月
    bep_month = term  # デフォルト：期間内にBEPなし
    for t in range(1, term + 1):
        v = (1.0 - r_monthly) ** t
        balance_ratio = (term - t) / term
        if v <= balance_ratio:
            bep_month = t
            break

    residual_at_end = (1.0 - r_monthly) ** term
    bep_ratio = bep_month / term

    warnings: list[str] = []

    if bep_month < term * 0.9 and liquidity in ("低", "極低"):
        warnings.append(
            f"⚠️ BEP割れリスク: リース期間の{bep_ratio:.0%}時点で担保価値が残債を下回ります"
        )

    if residual_at_end < 0.05:
        warnings.append("⚠️ 満了時残存価値ほぼゼロ: 出口価値がありません")

    if liquidity == "極低":
        warnings.append("⚠️ 換金性低: 特注・専用物件のため回収時の売却が困難です")

    bonuses: list[str] = []

    if liquidity == "高" and residual_at_end > 0.20:
        bonuses.append("換金性高・資産価値良好")

    if liquidity == "高" and bep_month < term * 0.5:
        bonuses.append("早期回収可能な高流動性物件")

    if liquidity == "中" and residual_at_end > 0.30:
        bonuses.append("残存価値が高く担保力あり")

    return warnings, bonuses


def run_quick_scoring(inputs: dict) -> dict:
    """
    入力辞書からスコア・判定・比較文を計算して返す。
    inputs のキー例: nenshu, op_profit(営業利益千円), ord_profit, net_income, net_assets, total_assets,
                    industry_major, industry_sub, grade, customer_type, main_bank, competitor,
                    bank_credit, lease_credit, contracts, asset_score(省略時50), など。
    返却: score, hantei, comparison, user_op_margin, user_equity_ratio, bench_op_margin, bench_equity_ratio,
          score_borrower, industry_sub, industry_major
    """
    nenshu = max(0, _safe_float(inputs.get("nenshu")))
    op_profit = _safe_float(inputs.get("op_profit") or inputs.get("rieki"))
    ord_profit = _safe_float(inputs.get("ord_profit"))
    net_income = _safe_float(inputs.get("net_income"))
    net_assets = _safe_float(inputs.get("net_assets"))
    total_assets = _safe_float(inputs.get("total_assets"))
    industry_major = (inputs.get("industry_major") or "D 建設業").strip()
    industry_sub = (inputs.get("industry_sub") or "06 総合工事業").strip()
    grade = inputs.get("grade") or "1-3"
    customer_type = inputs.get("customer_type") or "既存先"
    bank_credit = _safe_float(inputs.get("bank_credit"))
    lease_credit = _safe_float(inputs.get("lease_credit"))
    contracts = _safe_int(inputs.get("contracts"))
    sales_dept = (inputs.get("sales_dept") or "未設定").strip()
    _raw_asset_score = inputs.get("asset_score")
    used_default_asset_score = _raw_asset_score is None or str(_raw_asset_score).strip() == ""
    asset_score = _safe_float(_raw_asset_score, default=50.0)

    user_op_margin = (op_profit / nenshu * 100) if nenshu > 0 else 0.0
    user_equity_ratio = (net_assets / total_assets * 100) if total_assets > 0 else 0.0
    user_lease_credit_pct = (lease_credit / nenshu * 100) if nenshu > 0 and lease_credit > 0 else None

    benchmarks = _load_benchmarks()
    bench = benchmarks.get(industry_sub, {})
    bench_op_margin = float(bench.get("op_margin") or 0)
    beq = bench.get("equity_ratio")
    if beq is not None and isinstance(beq, (int, float)):
        bench_equity_ratio = float(beq)
    else:
        bench_equity_ratio = float(bench.get("equity_ratio_display") or 0)
    bench_comment = (bench.get("comment") or "").strip()
    bench_lease_cost_ratio = float(bench.get("lease_cost_ratio") or 0)
    capex_lease_data = _load_capex_lease_data()
    rent_expense_ky = _safe_float(inputs.get("rent_expense"))  # 千円
    user_lease_cost_ratio = (rent_expense_ky / nenshu * 100) if nenshu > 0 else 0.0
    estat_context = None
    try:
        estat_context = build_estat_context(
            selected_major=industry_major,
            selected_sub=industry_sub,
            user_op_margin=user_op_margin,
            user_equity_ratio=user_equity_ratio,
            user_current_ratio=_safe_float(inputs.get("current_ratio"), default=None),
            user_debt_ratio=_safe_float(inputs.get("debt_ratio"), default=None),
            user_asset_turnover=_safe_float(inputs.get("asset_turnover"), default=None),
            user_annual_lease_pct=None,
            user_lease_credit_pct=user_lease_credit_pct,
            benchmarks_data=benchmarks,
            capex_lease_data=capex_lease_data,
        )
    except Exception as exc:
        log_warning(f"e-Stat統合文脈生成失敗: {exc}", context="run_quick_scoring")

    comp_margin = "高い" if user_op_margin >= bench_op_margin else "低い"
    comp_equity = "高い" if user_equity_ratio >= bench_equity_ratio else "低い"
    _lease_line = ""
    if bench_lease_cost_ratio > 0:
        comp_lease = "高い" if user_lease_cost_ratio > bench_lease_cost_ratio else "低い"
        _lease_line = (
            f"\n- **リース費用比率**: {user_lease_cost_ratio:.1f}% "
            f"(業界目安: {bench_lease_cost_ratio:.1f}%) → 平均より{comp_lease}"
        )
    comparison = (
        f"- **営業利益率**: {user_op_margin:.1f}% (業界目安: {bench_op_margin:.1f}%) → 平均より{comp_margin}\n"
        f"- **自己資本比率**: {user_equity_ratio:.1f}% (業界目安: {bench_equity_ratio:.1f}%) → 平均より{comp_equity}\n"
        f"- **業界**: {bench_comment or '—'}"
        f"{_lease_line}"
    )

    # ── 単位変換メモ ───────────────────────────────────────────────────────────
    # inputs の財務数値はすべて「千円」単位で入力される（UI・Slack Bot 共通）。
    # COEFFS の係数は以下のスケールで推定されている:
    #   nenshu / bank_credit / lease_credit : 千円のまま log1p に渡す（係数で吸収）
    #   op_profit / ord_profit 等の P/L 項目 : 千円 → 百万円（÷1000）に変換して渡す
    # 変換基準を変更した場合は coeff_definitions.py の係数を必ず再推定すること。
    data_scoring = {
        "nenshu": nenshu,           # 千円単位のまま（log1p 変換して使用）
        "bank_credit": bank_credit,  # 千円単位のまま（log1p 変換して使用）
        "lease_credit": lease_credit,  # 千円単位のまま（log1p 変換して使用）
        "op_profit": op_profit / 1000,     # 千円 → 百万円
        "ord_profit": ord_profit / 1000,   # 千円 → 百万円
        "net_income": net_income / 1000,   # 千円 → 百万円
        "gross_profit": _safe_float(inputs.get("gross_profit")) / 1000,   # 千円 → 百万円
        "machines": _safe_float(inputs.get("machines")) / 1000,           # 千円 → 百万円
        "other_assets": _safe_float(inputs.get("other_assets")) / 1000,   # 千円 → 百万円
        "rent": _safe_float(inputs.get("rent")) / 1000,                   # 千円 → 百万円
        "depreciation": _safe_float(inputs.get("depreciation")) / 1000,   # 千円 → 百万円
        "dep_expense": _safe_float(inputs.get("dep_expense")) / 1000,     # 千円 → 百万円
        "rent_expense": _safe_float(inputs.get("rent_expense")) / 1000,   # 千円 → 百万円
        "contracts": contracts,          # 件数（無次元）
        "grade": grade,
        "industry_major": industry_major,
        "sales_dept": sales_dept,
    }

    coeff_key = "全体_既存先"
    if customer_type == "新規先":
        coeff_key = "全体_新規先"
    coeffs = get_effective_coeffs(coeff_key)
    z_main = _calculate_z(data_scoring, coeffs)
    score_prob = _safe_sigmoid(z_main)

    # 定量主モデル（RF優先）。失敗時のみ従来式を使う。
    try:
        from scoring.predict_one import predict_one, map_industry_major_to_scoring

        rf_context = {
            "gross_profit": _safe_float(inputs.get("gross_profit")) * 1000.0,
            "op_profit": op_profit * 1000.0,
            "ord_profit": ord_profit * 1000.0,
            "net_income": net_income * 1000.0,
            "dep_expense": _safe_float(inputs.get("dep_expense")) * 1000.0,
            "depreciation": _safe_float(inputs.get("depreciation")) * 1000.0,
            "nenshu": nenshu * 1000.0,
            "machines": _safe_float(inputs.get("machines")) * 1000.0,
            "other_assets": _safe_float(inputs.get("other_assets")) * 1000.0,
            "rent": _safe_float(inputs.get("rent")) * 1000.0,
            "rent_expense": _safe_float(inputs.get("rent_expense")) * 1000.0,
            "bank_credit": bank_credit * 1000.0,
            "lease_credit": lease_credit * 1000.0,
            "acquisition_cost": _safe_float(inputs.get("acquisition_cost")) * 1000.0,
            "lease_term": _safe_float(inputs.get("lease_term")),
            "contracts": contracts,
            "lease_asset_score": asset_score,
            "industry": map_industry_major_to_scoring(industry_major),
            "customer_type": customer_type,
            "main_bank": inputs.get("main_bank") or "非メイン先",
            "competitor": inputs.get("competitor") or "競合なし",
            "competitor_rate": _safe_float(inputs.get("competitor_rate")),
            "grade": grade,
            "contract_type": inputs.get("contract_type") or "一般",
            "deal_source": inputs.get("deal_source") or "銀行紹介",
            "sales_dept": sales_dept,
            "base_rate": _safe_float(inputs.get("base_rate")),
            "q_history": _safe_float(inputs.get("q_history")),
            "q_stability": _safe_float(inputs.get("q_stability")),
            "q_repayment": _safe_float(inputs.get("q_repayment")),
            "q_future": _safe_float(inputs.get("q_future")),
            "q_equip": _safe_float(inputs.get("q_equip")),
            "q_mainbk": _safe_float(inputs.get("q_mainbk")),
            "q_weighted": _safe_float(inputs.get("q_weighted")),
            "sys_score": _safe_float(inputs.get("sys_score")),
            "sys_score_b": _safe_float(inputs.get("sys_score_b")),
            "sys_dscr": _safe_float(inputs.get("sys_dscr")),
            "sys_op_margin": _safe_float(inputs.get("sys_op_margin")),
            "sys_icr": _safe_float(inputs.get("sys_icr")),
            "sys_approval": _safe_float(inputs.get("sys_approval")),
            "sys_ind_score": _safe_float(inputs.get("sys_ind_score")),
            "sys_bench": _safe_float(inputs.get("sys_bench")),
            "winning_spread": _safe_float(inputs.get("winning_spread")),
        }
        rf_result = predict_one(
            revenue=nenshu * 1000.0,
            total_assets=max(total_assets, 1.0) * 1000.0,
            equity=net_assets * 1000.0,
            operating_profit=op_profit * 1000.0,
            net_income=net_income * 1000.0,
            machinery_equipment=_safe_float(inputs.get("machines")) * 1000.0,
            other_fixed_assets=_safe_float(inputs.get("other_assets")) * 1000.0,
            depreciation=_safe_float(inputs.get("depreciation")) * 1000.0,
            rent_expense=_safe_float(inputs.get("rent_expense")) * 1000.0,
            industry=rf_context["industry"],
            context=rf_context,
        )
        if rf_result and rf_result.get("ai_prob") is not None:
            score_prob = max(0.0, min(1.0, 1.0 - float(rf_result["ai_prob"])))
    except Exception:
        pass

    # 定性 LGB は単体の補助確率として参照する。ブレンドはしない。
    try:
        _qual_bundle = _load_lgb_qual_bundle()
        if _qual_bundle is not None:
            _qual_feat = _qual_bundle["feature_names"]
            _qual_ati = _qual_bundle.get("asset_to_idx", {})
            _Xq = _build_lgb_qual_feature_vector(inputs, _qual_feat, _qual_ati)
            _ = float(_qual_bundle["model"].predict_proba([_Xq])[0][1])
    except Exception:
        pass

    score_borrower = score_prob * 100

    # 物件スコアは警告フラグのみ（final_score への寄与ゼロ）
    base_score = max(0, min(100, round(score_borrower, 1)))

    # ── 自己資本マイナスペナルティ ────────────────────────────────────────────
    # RandomForestモデルはnet_assetsを特徴量に含まないため後処理で補正する
    equity_penalty = 0.0
    if user_equity_ratio < 0:
        # -1%ごとに約0.5点減点、最大-30点
        equity_penalty = max(-30.0, user_equity_ratio * 0.5)
    base_score = max(0, min(100, round(base_score + equity_penalty, 1)))

    # ── リース費用比率ペナルティ ────────────────────────────────────────────────
    # 検証結果(n=1106)に基づく設計:
    #   D(1.5-2倍): 低・高ベンチ両グループで否決率73-77% → -1.5点は有効
    #   E(2倍超) × 高ベンチ(≥2%): 飲食・不動産はリース積極活用=優良傾向 → ペナルティ無効
    #   E(2倍超) × 低ベンチ(<2%): 卸売0.6%等は比率が爆発しやすいため絶対lcr>3%を条件に追加
    lease_ratio_adj = 0.0
    if bench_lease_cost_ratio > 0 and user_lease_cost_ratio > 0:
        ratio_vs_bench = user_lease_cost_ratio / bench_lease_cost_ratio
        if ratio_vs_bench > 2.0:
            if bench_lease_cost_ratio < 2.0 and user_lease_cost_ratio > 3.0:
                lease_ratio_adj = -3.0   # 低ベンチ業種 かつ 絶対lcr>3%：実態として過大
            # 高ベンチ業種(≥2%)の2倍超はペナルティなし
        elif ratio_vs_bench > 1.5:
            lease_ratio_adj = -1.5   # 1.5-2倍：両グループで否決率高い
    base_score = max(0, min(100, round(base_score + lease_ratio_adj, 1)))

    # ── 担当者直感スコア補正（1-5スケール、中立=3、±INTUITION_MAX_ADJ 点まで）──
    # 画面や経路によって直感スコアのキーが `intuition_score` / `intuition` に分かれていたため、
    # どちらでも拾えるようにする。優先は `intuition_score`。
    intuition_raw = inputs.get("intuition_score", inputs.get("intuition"))
    intuition_score = _safe_float(intuition_raw, default=0)
    intuition_adj = 0.0
    if 1.0 <= intuition_score <= 5.0:
        # 3を中立として ±INTUITION_MAX_ADJ 点に線形マッピング
        intuition_adj = round((intuition_score - 3.0) / 2.0 * INTUITION_MAX_ADJ, 2)

    final_score = max(0, min(100, round(base_score + intuition_adj, 1)))

    # マクロリスク補正（macro_context.json が存在する場合のみ適用）
    macro_adj = 0.0
    try:
        _macro_path = os.path.join(_SCRIPT_DIR, "static_data", "macro_context.json")
        if os.path.exists(_macro_path):
            with open(_macro_path, encoding="utf-8") as _mf:
                _mc = json.load(_mf)
            _boj_rate = _mc.get("boj_policy_rate", {}).get("value")
            _mrisk = _mc.get("assessment", {}).get("macro_risk_score")
            if isinstance(_boj_rate, (int, float)) and _boj_rate >= 1.0:
                macro_adj -= 3.0
            if isinstance(_mrisk, (int, float)):
                if _mrisk >= 50:
                    macro_adj -= 2.0
                elif _mrisk <= 20:
                    macro_adj += 1.0
    except Exception:
        pass

    final_score = max(0, min(100, round(final_score + macro_adj, 1)))
    hantei = "承認圏内" if final_score >= APPROVAL_LINE else "要審議"

    # 直感スコアが高いのに要審議 → 上長確認フラグ
    manager_review_flag = (intuition_score >= 4.0) and (final_score < APPROVAL_LINE)

    # ── SHAP近似: 各特徴量の寄与度を計算 ──
    score_contributions = compute_score_contributions(data_scoring, coeffs)

    # 物件スコアのデフォルト使用フラグ
    asset_score_warnings = []
    if used_default_asset_score:
        asset_score_warnings.append("物件スコア未入力のためデフォルト値(50)を使用")

    # 物件リスク警告フラグ（BEP・換金性・残存価値）
    _asset_name = (
        inputs.get("asset_name")
        or inputs.get("lease_asset_name")
        or inputs.get("selected_asset_id")
        or ""
    )
    _term_months = _safe_int(inputs.get("lease_term"), default=60)
    _industry_sub = str(inputs.get("industry_sub") or "")
    asset_warnings, asset_bonuses = generate_asset_warnings(_asset_name, _term_months, _industry_sub)

    # デフォルト率モデルによる高リスク警告フラグ（スコアには影響しない）
    default_warnings = generate_default_warnings(inputs)

    credit_risk_group = {
        "score": 0.0,
        "level": "unavailable",
        "flag": False,
        "reasons": [],
        "available": False,
    }
    try:
        from credit_risk_detector import detect_credit_risk_group

        credit_risk_group = detect_credit_risk_group(inputs)
    except Exception as exc:
        credit_risk_group = {
            "score": 0.0,
            "level": "unavailable",
            "flag": False,
            "reasons": [f"信用リスク群検知を利用できません: {exc}"],
            "available": False,
        }
    credit_risk_warnings = credit_risk_group.get("reasons", []) if credit_risk_group.get("flag") else []
    try:
        from quantum_analysis_module import compute_simple_q_risk
        quantum_risk_score = float(compute_simple_q_risk(inputs).get("quantum_risk", 0.0))
    except Exception:
        quantum_risk_score = None

    mahalanobis_score: float | None = None
    mahalanobis_advice: list | None = None
    try:
        import pandas as _pd
        from mahalanobis_engine import MahalanobisScorer
        from train_mahalanobis import FEATURES as _MAHA_FEATURES, _extract_val as _maha_extract
        _maha_path = os.path.join(_SCRIPT_DIR, "data", "mahalanobis_model.joblib")
        if os.path.exists(_maha_path):
            _maha = MahalanobisScorer.load(_maha_path)
            _maha_row = {f: _maha_extract({"inputs": inputs}, f) for f in _MAHA_FEATURES}
            _maha_df = _pd.DataFrame([_maha_row])
            _ms, _, _, _ = _maha.get_analysis(_maha_df)
            mahalanobis_score = round(float(_ms), 1)
            mahalanobis_advice = [
                {"feat": a["feat"], "direction": a["direction"], "delta": round(float(a["delta"]), 0)}
                for a in _maha.advise_improvement(_maha_df, top_k=3)
            ]
    except Exception:
        pass

    umap_anomaly_score: float | None = None
    umap_x: float | None = None
    umap_y: float | None = None
    umap_similar: list | None = None
    try:
        import pandas as _pd
        import threading as _umap_threading
        from umap_anomaly_engine import UMAPAnomalyScorer
        from train_mahalanobis import FEATURES as _UMAP_FEATURES, _extract_val as _umap_extract
        _umap_path = os.path.join(_SCRIPT_DIR, "data", "umap_anomaly_model.joblib")
        if os.path.exists(_umap_path):
            # モジュールレベルでキャッシュ済みモデルを再利用してロードコストを削減
            global _umap_model_cache, _umap_model_path_cache
            if _umap_model_cache is None or _umap_model_path_cache != _umap_path:
                _umap_model_cache = UMAPAnomalyScorer.load(_umap_path)
                _umap_model_path_cache = _umap_path
            _umap = _umap_model_cache
            _umap_row = {f: _umap_extract({"inputs": inputs}, f) for f in _UMAP_FEATURES}
            _umap_df = _pd.DataFrame([_umap_row])
            # UMAP.transform() はデータポイントごとに勾配降下法を実行するため非常に遅い。
            # 5秒タイムアウト付きスレッドで実行し、超過した場合はスキップする。
            _umap_result: list = []
            def _run_umap():
                try:
                    _s, _x, _y = _umap.score(_umap_df)
                    _sim = _umap.find_similar(_umap_df, top_k=3)
                    _umap_result.extend([_s, _x, _y, _sim])
                except Exception:
                    pass
            _umap_thread = _umap_threading.Thread(target=_run_umap, daemon=True)
            _umap_thread.start()
            _umap_thread.join(timeout=5.0)
            if len(_umap_result) == 4:
                umap_anomaly_score, umap_x, umap_y, umap_similar = _umap_result
    except Exception:
        pass

    credit_quantum_strong_warning = (
        credit_risk_group.get("score", 0.0) >= 70.0
        and quantum_risk_score is not None
        and quantum_risk_score >= 60.0
    )
    if credit_quantum_strong_warning:
        credit_risk_warnings.append(
            f"強警戒: 信用リスク群スコア {credit_risk_group.get('score', 0.0):.1f} かつ Q_risk {quantum_risk_score:.1f}"
        )

    # ── DSCR / インタレスト・カバレッジ（次回再学習の特徴量候補）──
    dscr_approx = compute_dscr_approx(inputs)
    interest_coverage = compute_interest_coverage(inputs)

    return {
        "score": final_score,
        "score_base": base_score,
        "hantei": hantei,
        "comparison": comparison,
        "user_op_margin": user_op_margin,
        "user_equity_ratio": user_equity_ratio,
        "bench_op_margin": bench_op_margin,
        "bench_equity_ratio": bench_equity_ratio,
        "user_lease_cost_ratio": round(user_lease_cost_ratio, 2),
        "bench_lease_cost_ratio": bench_lease_cost_ratio,
        "lease_ratio_adj": lease_ratio_adj,
        "score_borrower": round(score_borrower, 1),
        "industry_sub": industry_sub,
        "industry_major": industry_major,
        "approval_line": APPROVAL_LINE,
        "used_default_asset_score": used_default_asset_score,
        "asset_score_warnings": asset_score_warnings,
        "credit_risk_group_score": credit_risk_group.get("score", 0.0),
        "credit_risk_group_level": credit_risk_group.get("level", "unavailable"),
        "credit_risk_group_flag": bool(credit_risk_group.get("flag")),
        "credit_risk_group_reasons": credit_risk_group.get("reasons", []),
        "credit_risk_warnings": credit_risk_warnings,
        "quantum_risk": quantum_risk_score,
        "credit_quantum_strong_warning": credit_quantum_strong_warning,
        "umap_anomaly_score": umap_anomaly_score,
        "umap_x": umap_x,
        "umap_y": umap_y,
        "umap_similar": umap_similar,
        # 直感スコア関連
        "intuition_score": intuition_score,
        "intuition_adj": intuition_adj,
        "macro_adj": macro_adj,
        "manager_review_flag": manager_review_flag,
        # SHAP近似: 各特徴量の寄与度（上位5件はUI表示に活用）
        "score_contributions": score_contributions,
        # CF系指標（表示・次回ML再学習用）
        "dscr_approx": dscr_approx,
        "interest_coverage": interest_coverage,
        # 物件リスク警告フラグ（BEP・換金性・残存価値）— スコアには影響しない
        "asset_warnings": asset_warnings,
        # 物件プラス評価（換金性・残存価値優位）— 軍師AIのコメント生成に使用
        "asset_bonuses": asset_bonuses,
        # 物件スコア（表示用のみ）
        "asset_score": asset_score,
        # デフォルト率モデルによる高リスク警告フラグ— スコアには影響しない
        "default_warnings": default_warnings,
        "estat_context": estat_context,
    }


def compute_optimal_approval_line(past_cases: list) -> dict:
    """
    過去の成約/失注データから ROC 最適閾値を算出する。
    APPROVAL_LINE = 71 の根拠を過去データから正当化するために使用。

    Args:
        past_cases: [{"final_score": float, "final_status": str}, ...] のリスト

    Returns:
        {
            "optimal_threshold": float,   # Youden's J で最適化した閾値
            "current_threshold": int,     # 現在の APPROVAL_LINE
            "auc": float,                 # ROC-AUC
            "n_cases": int,               # 使用ケース数
            "n_contracts": int,           # 成約件数
            "note": str,                  # 説明
        }
    """
    CONTRACT_STATUSES = {"成約", "検収", "検収完了"}

    valid = [
        c for c in past_cases
        if c.get("final_score") is not None and c.get("final_status") is not None
    ]
    if len(valid) < 10:
        return {
            "optimal_threshold": APPROVAL_LINE,
            "current_threshold": APPROVAL_LINE,
            "auc": None,
            "n_cases": len(valid),
            "n_contracts": 0,
            "note": f"データ不足（{len(valid)}件）。10件以上の成約/失注データが必要です。",
        }

    scores = [float(c["final_score"]) for c in valid]
    labels = [1 if c["final_status"] in CONTRACT_STATUSES else 0 for c in valid]
    n_contracts = sum(labels)

    if n_contracts == 0 or n_contracts == len(valid):
        return {
            "optimal_threshold": APPROVAL_LINE,
            "current_threshold": APPROVAL_LINE,
            "auc": None,
            "n_cases": len(valid),
            "n_contracts": n_contracts,
            "note": "成約または失注のどちらかのみのデータです。両方が必要です。",
        }

    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        youden_idx = int(np.argmax(tpr - fpr))
        optimal = float(thresholds[youden_idx])
    except ImportError:
        # sklearn がない場合は単純な中央値フォールバック
        contract_scores = [scores[i] for i, l in enumerate(labels) if l == 1]
        non_scores = [scores[i] for i, l in enumerate(labels) if l == 0]
        optimal = (np.mean(contract_scores) + np.mean(non_scores)) / 2
        auc = None

    return {
        "optimal_threshold": round(optimal, 1),
        "current_threshold": APPROVAL_LINE,
        "auc": round(auc, 3) if auc is not None else None,
        "n_cases": len(valid),
        "n_contracts": n_contracts,
        "note": f"Youden's J 統計量による最適閾値（成約={n_contracts}件 / 全{len(valid)}件）",
    }
