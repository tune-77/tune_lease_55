"""
Google Colab 用 LightGBM 学習スクリプト
=========================================
【使い方】
1. ローカルで export_cases_for_colab.py を実行 → data/cases_for_colab.json を生成
2. このファイル (train_lgb_colab.py) と cases_for_colab.json を Colab にアップロード
3. Colab で下記のセルを順番に実行
4. ダウンロードされた 4 つのファイルをローカルの data/ フォルダに入れる

ダウンロードされるファイル:
  - lgb_main_model.joblib      ← 定量モデル
  - lgb_qual_model.joblib      ← 定性モデル
  - ensemble_config.json       ← 定量アンサンブル alpha
  - ensemble_config_qual.json  ← 定性アンサンブル alpha
"""

# ============================================================
# セル 1: ライブラリインストール
# ============================================================
# !pip install lightgbm scikit-learn joblib numpy --quiet

# ============================================================
# セル 2: データアップロード（Colab のファイルアップロード UI）
# ============================================================
# from google.colab import files
# uploaded = files.upload()   # cases_for_colab.json を選択してアップロード

# ============================================================
# セル 3: 学習実行（このブロックをまるごとコピーして実行）
# ============================================================

import json
import math
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb_lib

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, "data")

# ---------- データ読み込み ----------
_cases_path = os.path.join(_DATA, "cases_for_colab.json")
with open(_cases_path, encoding="utf-8") as f:
    cases = json.load(f)

print(f"読み込み件数: {len(cases)} 件")

# ---------- 定数 ----------
COEFF_MAIN_KEYS = [
    "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing", "ind_service",
    "sales_log", "bank_credit_log", "lease_credit_log",
    "op_profit", "ord_profit", "net_income", "machines", "other_assets", "rent",
    "gross_profit", "depreciation", "dep_expense", "rent_expense",
    "grade_4_6", "grade_watch", "grade_none", "contracts",
]
COEFF_EXTRA_KEYS = [
    "main_bank", "competitor_present", "competitor_none",
    "rate_diff_z", "industry_sentiment_z", "qualitative_tag_score", "qualitative_passion",
    "equity_ratio", "qualitative_combined",
    "bn_approval_prob", "bn_fc", "bn_hc", "bn_av",
    "qual_weighted", "qual_rank_good", "qual_repayment",
    # CF 系指標（次世代モデル用: DSCR・インタレスト・カバレッジ）
    "dscr_approx", "interest_coverage",
]
QUAL_SCORING_IDS = [
    "company_history", "customer_stability", "repayment_history",
    "business_future", "equipment_purpose", "main_bank",
]
QUAL_SCORING_LABELS = [
    "設立・経営年数", "顧客安定性", "返済履歴",
    "事業将来性", "設備目的", "メイン取引銀行",
]
LGBM_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                   num_leaves=15, min_child_samples=5, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, verbose=-1)

# ---------- ヘルパー ----------
def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def ind_flags(major):
    m = major or ""
    return {
        "ind_medical":       1.0 if ("医療" in m or "福祉" in m or m.startswith("P")) else 0.0,
        "ind_transport":     1.0 if ("運輸" in m or m.startswith("H")) else 0.0,
        "ind_construction":  1.0 if ("建設" in m or m.startswith("D")) else 0.0,
        "ind_manufacturing": 1.0 if ("製造" in m or m.startswith("E")) else 0.0,
        "ind_service":       1.0 if (any(x in m for x in ["卸売", "小売", "サービス"])
                                     or (bool(m) and m[0] in ["I","K","M","R"])) else 0.0,
    }

# ---------- 定量モデル用行構築 ----------
def build_quant_row(c):
    inp = c.get("inputs") or {}
    res = c.get("result") or {}
    major = (inp.get("industry_major") or res.get("industry_major") or "").strip()
    grade = inp.get("grade") or res.get("grade") or "1-3"
    nenshu      = safe_float(inp.get("nenshu"))
    bank_credit = safe_float(inp.get("bank_credit"))
    lease_credit= safe_float(inp.get("lease_credit"))
    flags = ind_flags(major)
    row = [
        flags["ind_medical"], flags["ind_transport"], flags["ind_construction"],
        flags["ind_manufacturing"], flags["ind_service"],
        np.log1p(nenshu), np.log1p(bank_credit), np.log1p(lease_credit),
        safe_float(inp.get("op_profit")) / 1000,
        safe_float(inp.get("ord_profit")) / 1000,
        safe_float(inp.get("net_income")) / 1000,
        safe_float(inp.get("machines")) / 1000,
        safe_float(inp.get("other_assets")) / 1000,
        safe_float(inp.get("rent")) / 1000,
        safe_float(inp.get("gross_profit")) / 1000,
        safe_float(inp.get("depreciation")) / 1000,
        safe_float(inp.get("dep_expense")) / 1000,
        safe_float(inp.get("rent_expense")) / 1000,
        1.0 if "4-6" in grade else 0.0,
        1.0 if "要注意" in grade else 0.0,
        1.0 if "無格付" in grade else 0.0,
        safe_float(inp.get("contracts")),
        1.0 if c.get("main_bank") == "メイン先" else 0.0,
        1.0 if c.get("competitor") == "競合あり" else 0.0,
        1.0 if c.get("competitor") == "競合なし" else 0.0,
        0.0,  # rate_diff_z
        safe_float(res.get("industry_sentiment_z")),
        0.0,  # qualitative_tag_score
        0.0,  # qualitative_passion
        safe_float(res.get("user_eq")),  # equity_ratio
    ]
    qsc = (res.get("qualitative_scoring_correction")
           or inp.get("qualitative_scoring") or {})
    combined = qsc.get("combined_score") or qsc.get("weighted_score")
    row.append(float(combined) / 100.0 if combined is not None else 0.0)
    bn = c.get("bn_engine") or {}
    bni = bn.get("intermediate") or {}
    row += [
        safe_float(bn.get("approval_prob")),
        safe_float(bni.get("Financial_Creditworthiness")),
        safe_float(bni.get("Hedge_Condition")),
        safe_float(bni.get("Asset_Value")),
    ]
    ws = qsc.get("weighted_score")
    rank = qsc.get("rank") or ""
    rh = ((qsc.get("items") or {}).get("repayment_history") or {}).get("value") or 0
    row += [
        float(ws) / 100.0 if ws is not None else 0.0,
        1.0 if rank in ("A", "B") else 0.0,
        float(rh) / 4.0,
    ]
    # CF 系指標
    dep_e = safe_float(inp.get("dep_expense") or inp.get("depreciation"))
    rent_e = safe_float(inp.get("rent_expense") or inp.get("rent"))
    op_p = safe_float(inp.get("op_profit"))
    denom = dep_e + rent_e
    dscr = round(op_p / denom, 3) if denom > 0 else 1.0
    interest_e = safe_float(c.get("interest_expense") or inp.get("interest_expense"))
    icr = round(op_p / interest_e, 3) if interest_e > 0 else 10.0
    row += [dscr, icr]
    return row

# ---------- 定性モデル用行構築 ----------
def build_qual_row(c, asset_to_idx):
    inp = c.get("inputs") or {}
    res = c.get("result") or {}
    main_bank    = c.get("main_bank") or inp.get("main_bank") or "非メイン先"
    competitor   = c.get("competitor") or inp.get("competitor") or "競合なし"
    customer_type= c.get("customer_type") or inp.get("customer_type") or "既存先"
    deal_source  = inp.get("deal_source") or "その他"
    asset_id     = inp.get("lease_asset_id") or inp.get("lease_asset_name") or "未選択"
    row = [
        1.0 if main_bank    == "メイン先"  else 0.0,
        1.0 if competitor   == "競合あり"  else 0.0,
        1.0 if customer_type == "新規先"  else 0.0,
        1.0 if deal_source  == "銀行紹介" else 0.0,
        float(asset_to_idx.get(asset_id, 0)),
    ]
    qsc = (res.get("qualitative_scoring_correction")
           or inp.get("qualitative_scoring") or {})
    items_data = qsc.get("items") or {}
    for qid in QUAL_SCORING_IDS:
        val = items_data.get(qid, {})
        v = val.get("value") if isinstance(val, dict) else None
        row.append(float(v) if isinstance(v, (int, float)) else -1.0)
    return row

# ---------- アンサンブル alpha 最適化 ----------
def optimize_alpha(p_lr, p_lgb, y):
    best_a, best_auc = 0.5, 0.0
    for a in np.linspace(0, 1, 11):
        p = a * p_lr + (1 - a) * p_lgb
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, p)
        if auc > best_auc:
            best_auc, best_a = auc, a
    return float(best_a), float(best_auc)

# ============================================================
# 定量モデル学習
# ============================================================
print("\n--- 定量モデル学習 ---")
quant_feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
quant_rows, quant_y = [], []
for c in cases:
    try:
        quant_rows.append(build_quant_row(c))
        quant_y.append(1 if c["final_status"] == "成約" else 0)
    except Exception as e:
        pass

X_q = np.array(quant_rows, dtype=float)
y_q = np.array(quant_y, dtype=int)
print(f"  サンプル数: {len(y_q)}  成約: {y_q.sum()}  失注: {len(y_q)-y_q.sum()}")

X_tr, X_te, y_tr, y_te = train_test_split(X_q, y_q, test_size=0.25, random_state=42, stratify=y_q)
lgb_main = lgb_lib.LGBMClassifier(**LGBM_PARAMS)
lgb_main.fit(X_tr, y_tr)
p_lgb = lgb_main.predict_proba(X_te)[:, 1]

lr_main = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_main.fit(X_tr, y_tr)
p_lr = lr_main.predict_proba(X_te)[:, 1]

best_alpha_q, best_auc_q = optimize_alpha(p_lr, p_lgb, y_te)
print(f"  AUC_LGB={roc_auc_score(y_te, p_lgb):.3f}  AUC_LR={roc_auc_score(y_te, p_lr):.3f}")
print(f"  最適 alpha={best_alpha_q:.2f}  アンサンブル AUC={best_auc_q:.3f}")

joblib.dump({"model": lgb_main, "feature_names": quant_feature_names},
            os.path.join(_DATA, "lgb_main_model.joblib"))
with open(os.path.join(_DATA, "ensemble_config.json"), "w", encoding="utf-8") as f:
    json.dump({"ensemble_alpha": best_alpha_q, "auc_ensemble": best_auc_q}, f)
print("  → data/lgb_main_model.joblib / data/ensemble_config.json 保存済み")

# ============================================================
# 定性モデル学習
# ============================================================
print("\n--- 定性モデル学習 ---")
asset_ids = list({
    (c.get("inputs") or {}).get("lease_asset_id") or
    (c.get("inputs") or {}).get("lease_asset_name") or "未選択"
    for c in cases
})
asset_to_idx = {a: i for i, a in enumerate(asset_ids)}
qual_feature_names = [
    "取引区分_メイン先", "競合状況_競合あり", "顧客区分_新規先",
    "商談ソース_銀行紹介", "リース物件",
] + QUAL_SCORING_LABELS

qual_rows, qual_y = [], []
for c in cases:
    try:
        qual_rows.append(build_qual_row(c, asset_to_idx))
        qual_y.append(1 if c["final_status"] == "成約" else 0)
    except Exception:
        pass

X_qal = np.array(qual_rows, dtype=float)
y_qal = np.array(qual_y, dtype=int)
print(f"  サンプル数: {len(y_qal)}  成約: {y_qal.sum()}  失注: {len(y_qal)-y_qal.sum()}")

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_qal, y_qal, test_size=0.25, random_state=42, stratify=y_qal)
lgb_qual = lgb_lib.LGBMClassifier(**LGBM_PARAMS)
lgb_qual.fit(X_tr2, y_tr2)
p_lgb2 = lgb_qual.predict_proba(X_te2)[:, 1]

lr_qual = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_qual.fit(X_tr2, y_tr2)
p_lr2 = lr_qual.predict_proba(X_te2)[:, 1]

best_alpha_qal, best_auc_qal = optimize_alpha(p_lr2, p_lgb2, y_te2)
print(f"  AUC_LGB={roc_auc_score(y_te2, p_lgb2):.3f}  AUC_LR={roc_auc_score(y_te2, p_lr2):.3f}")
print(f"  最適 alpha={best_alpha_qal:.2f}  アンサンブル AUC={best_auc_qal:.3f}")

joblib.dump({"model": lgb_qual, "feature_names": qual_feature_names, "asset_to_idx": asset_to_idx},
            os.path.join(_DATA, "lgb_qual_model.joblib"))
with open(os.path.join(_DATA, "ensemble_config_qual.json"), "w", encoding="utf-8") as f:
    json.dump({"ensemble_alpha": best_alpha_qal, "auc_ensemble": best_auc_qal}, f)
print("  → data/lgb_qual_model.joblib / data/ensemble_config_qual.json 保存済み")

print(f"\n✅ 完了！ファイルは {_DATA}/ に保存されました。")
print("   - data/lgb_main_model.joblib")
print("   - data/lgb_qual_model.joblib")
print("   - data/ensemble_config.json")
print("   - data/ensemble_config_qual.json")
