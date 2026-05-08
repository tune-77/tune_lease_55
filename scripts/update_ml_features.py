"""
ml_features テーブルを増分更新するスクリプト。

## パイプライン構成（2段アンサンブル）
  Step1: scoring_core.py の run_quick_scoring() が
         LGBM借手スコアモデル（data/lgb_main_model.joblib）で
         sys_score_b（借手スコア 0-100）を計算し、result["score_borrower"] に格納する。
  Step2: data/ml_rf_v3.pkl（RF 52特徴量モデル）が
         sys_score_b を含む52特徴量から成約確率を予測し、
         pred_proba_v3 / pred_score_v3 として ml_features テーブルに保存する。

  このスクリプトは past_cases の data_json から Step1 の結果（sys_score_b）を取り出し、
  Step2 の特徴量ベクトルに組み込んで ml_features を算出・保存する。
  新規登録ケースは data_cases.save_case_log() から自動的にトリガーされる。

単体実行:
    python scripts/update_ml_features.py          # 未登録ケースのみ追加
    python scripts/update_ml_features.py --all    # 全件再計算して上書き

関数としての利用:
    from scripts.update_ml_features import update_ml_features
    update_ml_features(case_ids=["id1", "id2"])   # 指定IDのみ更新
    update_ml_features()                           # 未登録ケースのみ追加
"""
import os
import sys
import json
import logging
from datetime import datetime
from contextlib import closing

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DB_PATH        = os.path.join(_PROJECT_ROOT, "data", "lease_data.db")
MODEL_PATH     = os.path.join(_PROJECT_ROOT, "data", "ml_rf_v3.pkl")
ENCODER_PATH   = os.path.join(_PROJECT_ROOT, "data", "ml_rf_industry_encoder.pkl")

logger = logging.getLogger(__name__)

# ── モデルをモジュールレベルでキャッシュ ─────────────────────────────
_bundle  = None
_model   = None
_FEAT    = None
_enc     = None
_imp     = None
_medians = None

def _load_model():
    global _bundle, _model, _FEAT, _enc, _imp, _medians
    if _model is not None:
        return
    import joblib
    _bundle  = joblib.load(MODEL_PATH)
    _model   = _bundle["model"]
    _FEAT    = _bundle["feature_names"]
    _enc     = _bundle["encoders"]
    _imp     = _bundle["imputer"]
    _medians = dict(zip(_FEAT, _imp.statistics_))


# ── 特徴量計算 ────────────────────────────────────────────────────────
_GRADE_MAP = {
    "①1-3 (優良)": 1, "①1-3（優良）": 1,
    "②4-6 (標準)": 2, "②4-6（標準）": 2,
    "③7-9 (注意)": 3, "③7-9（注意）": 3,
    "④無格付": 4,
}
_DEFAULT_ENC = {
    "industry":      "R サービス業(他に分類されないもの)",
    "contract_type": "一般",
    "deal_source":   "銀行紹介",
    "sales_dept":    "足利営業部",
}


def _safe_div(a, b):
    return float(a / b) if b and b != 0 else 0.0


def _enc_safe(key, val):
    le = _enc[key]
    v  = val if val in le.classes_ else _DEFAULT_ENC[key]
    return int(le.transform([v])[0])


def _build_record(case_id, data_json, final_status):
    """JSON文字列 → (case_id, feature_dict, pred_proba, pred_score, judgment)"""
    d   = json.loads(data_json)
    inp = d.get("inputs", {})
    res = d.get("result", {})
    qs  = inp.get("qualitative_scoring") or res.get("qualitative_scoring_correction") or {}
    qs_items = qs.get("items", {}) if isinstance(qs, dict) else {}

    def qs_val(k):
        item = qs_items.get(k, {})
        return item.get("value") if isinstance(item, dict) else None

    # 財務（千円→百万円）
    gp   = (inp.get("gross_profit") or 0) / 1000
    op   = (inp.get("op_profit")    or 0) / 1000
    ep   = (inp.get("ord_profit")   or 0) / 1000
    ni   = (inp.get("net_income")   or 0) / 1000
    dep  = (inp.get("dep_expense")  or 0) / 1000
    depr = (inp.get("depreciation") or 0) / 1000
    ns   = (inp.get("nenshu")       or 0) / 1000
    mach = (inp.get("machines")     or 0) / 1000
    oa   = (inp.get("other_assets") or 0) / 1000
    rent = (inp.get("rent")         or 0) / 1000
    rexp = (inp.get("rent_expense") or 0) / 1000
    bk   = (inp.get("bank_credit")  or 0) / 1000
    lc   = (inp.get("lease_credit") or 0) / 1000
    acq  = (inp.get("acquisition_cost") or 0) / 1000
    lt   = float(inp.get("lease_term") or 60)
    ct   = float(inp.get("contracts") or 0)
    las  = float(inp.get("lease_asset_score") or 0)

    # 派生比率
    gpm         = _safe_div(gp, ns)
    ord_margin  = _safe_div(ep, gp)
    net_margin  = _safe_div(ni, gp)
    dep_ratio   = _safe_div(dep, gp)
    bank_to_ns  = _safe_div(bk, ns)
    lease_to_ns = _safe_div(lc, ns)
    mach_to_ns  = _safe_div(mach, ns)
    acq_to_ns   = _safe_div(acq, ns)
    op_margin   = _safe_div(op, ns)
    dep_to_loan = _safe_div(dep, bk + lc) if (bk + lc) > 0 else 0.0

    # カテゴリ
    industry_raw = d.get("industry_major") or "不明"
    ind_code  = _enc_safe("industry",      industry_raw)
    ct_code   = _enc_safe("contract_type", inp.get("contract_type") or "不明")
    ds_code   = _enc_safe("deal_source",   inp.get("deal_source")   or "不明")
    sd_code   = _enc_safe("sales_dept",    d.get("sales_dept")      or "不明")
    cust_type = 1 if d.get("customer_type") == "既存先" else 0
    main_bk   = 1 if (d.get("main_bank") or "").startswith("メイン先") else 0
    competitor= 1 if d.get("competitor") == "競合あり" else 0
    comp_rate = float(d.get("competitor_rate") or 0)
    grade     = _GRADE_MAP.get(inp.get("grade", ""), 4)
    base_rate = d.get("base_rate_at_time") or None

    # 定性スコア
    q_history   = qs_val("company_history")
    q_stability = qs_val("customer_stability")
    q_repayment = qs_val("repayment_history")
    q_future    = qs_val("business_future")
    q_equip     = qs_val("equipment_purpose")
    q_mainbk    = qs_val("main_bank")
    q_weighted  = qs.get("weighted_score") if isinstance(qs, dict) else None

    # システムスコア（Step1: scoring_core の run_quick_scoring() 出力から取得）
    # sys_score_b は scoring_core 内の LGBM借手スコアモデルが計算した借手スコア(0-100)。
    # このスクリプトは Step2 の RF 特徴量として sys_score_b をそのまま利用する。
    sys_score   = res.get("score")
    sys_score_b = res.get("score_borrower")
    sys_dscr    = res.get("dscr_approx")
    sys_op_m    = res.get("user_op_margin")
    sys_icr     = res.get("interest_coverage")
    sys_appr    = res.get("approval_line")
    sys_ind     = res.get("ind_score")
    sys_bench   = res.get("bench_score")

    feat_map = {
        "gross_profit": gp,    "op_profit": op,    "ord_profit": ep,
        "net_income": ni,      "dep_expense": dep, "depreciation": depr,
        "nenshu": ns,          "machines": mach,   "other_assets": oa,
        "rent": rent,          "rent_expense": rexp,
        "bank_credit": bk,     "lease_credit": lc,
        "gpm": gpm,            "ord_margin": ord_margin,
        "net_margin": net_margin, "dep_ratio": dep_ratio,
        "bank_to_ns": bank_to_ns, "lease_to_ns": lease_to_ns,
        "mach_to_ns": mach_to_ns, "acq_to_ns": acq_to_ns,
        "op_margin": op_margin, "dep_to_loan": dep_to_loan,
        "acquisition_cost": acq, "lease_term": lt,
        "contracts": ct,       "lease_asset_score": las,
        "industry": ind_code,  "customer_type": cust_type,
        "main_bank": main_bk,  "competitor": competitor,
        "competitor_rate": comp_rate, "grade": grade,
        "contract_type": ct_code, "deal_source": ds_code,
        "sales_dept": sd_code, "base_rate": base_rate,
        "q_history": q_history, "q_stability": q_stability,
        "q_repayment": q_repayment, "q_future": q_future,
        "q_equip": q_equip,    "q_mainbk": q_mainbk,
        "q_weighted": q_weighted,
        "sys_score": sys_score, "sys_score_b": sys_score_b,
        "sys_dscr": sys_dscr,  "sys_op_margin": sys_op_m,
        "sys_icr": sys_icr,    "sys_approval": sys_appr,
        "sys_ind_score": sys_ind, "sys_bench": sys_bench,
    }

    # Step2: RF 52特徴量モデルで成約確率を予測（sys_score_b が特徴量に含まれる）
    X_raw = np.array([[feat_map.get(f) for f in _FEAT]], dtype=float)
    X     = _imp.transform(X_raw)
    proba = float(_model.predict_proba(X)[0, 1])
    score = int(round(proba * 100))
    judgment = "承認" if score >= 65 else ("条件付" if score >= 45 else "否認")

    row = (
        case_id, final_status,
        gp, op, ep, ni, dep, depr, ns, mach, oa, rent, rexp, bk, lc, acq,
        gpm, ord_margin, net_margin, dep_ratio,
        bank_to_ns, lease_to_ns, mach_to_ns, acq_to_ns, op_margin, dep_to_loan,
        lt, ct, las,
        industry_raw, cust_type, main_bk, competitor, comp_rate,
        grade, ind_code, ct_code, ds_code, sd_code, base_rate,
        q_history, q_stability, q_repayment, q_future,
        q_equip, q_mainbk, q_weighted,
        sys_score, sys_score_b, sys_dscr, sys_op_m,
        sys_icr, sys_appr, sys_ind, sys_bench,
        proba, score, judgment,
        "v3_fullfeature_52feat", datetime.now().isoformat(),
    )
    return row


_UPSERT_COLS = [
    "case_id", "final_status",
    "gross_profit_m", "op_profit_m", "ord_profit_m", "net_income_m",
    "dep_expense_m", "depreciation_m", "nenshu_m", "machines_m",
    "other_assets_m", "rent_m", "rent_expense_m",
    "bank_credit_m", "lease_credit_m", "acquisition_cost_m",
    "gpm", "ord_margin", "net_margin", "dep_ratio",
    "bank_to_ns", "lease_to_ns", "mach_to_ns", "acq_to_ns", "op_margin", "dep_to_loan",
    "lease_term", "contracts", "lease_asset_score",
    "industry_raw", "customer_type_bin", "main_bank_bin", "competitor_bin",
    "competitor_rate", "grade_ordinal", "industry_code", "contract_type_code",
    "deal_source_code", "sales_dept_code", "base_rate",
    "q_history", "q_stability", "q_repayment", "q_future",
    "q_equip", "q_mainbk", "q_weighted",
    "sys_score", "sys_score_b", "sys_dscr", "sys_op_margin",
    "sys_icr", "sys_approval", "sys_ind_score", "sys_bench",
    "pred_proba_v3", "pred_score_v3", "pred_judgment_v3",
    "model_version", "computed_at",
]
_UPSERT_SQL = (
    f"INSERT OR REPLACE INTO ml_features ({','.join(_UPSERT_COLS)}) "
    f"VALUES ({','.join(['?'] * len(_UPSERT_COLS))})"
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_features (
    case_id              TEXT PRIMARY KEY,
    final_status         TEXT,
    gross_profit_m       REAL, op_profit_m    REAL, ord_profit_m   REAL,
    net_income_m         REAL, dep_expense_m  REAL, depreciation_m REAL,
    nenshu_m             REAL, machines_m     REAL, other_assets_m REAL,
    rent_m               REAL, rent_expense_m REAL,
    bank_credit_m        REAL, lease_credit_m REAL, acquisition_cost_m REAL,
    gpm                  REAL, ord_margin     REAL, net_margin      REAL,
    dep_ratio            REAL, bank_to_ns     REAL, lease_to_ns     REAL,
    mach_to_ns           REAL, acq_to_ns      REAL, op_margin       REAL,
    dep_to_loan          REAL,
    lease_term           REAL, contracts      REAL, lease_asset_score REAL,
    industry_raw         TEXT, customer_type_bin INTEGER,
    main_bank_bin        INTEGER, competitor_bin   INTEGER,
    competitor_rate      REAL,  grade_ordinal    INTEGER,
    industry_code        INTEGER, contract_type_code INTEGER,
    deal_source_code     INTEGER, sales_dept_code   INTEGER,
    base_rate            REAL,
    q_history            REAL, q_stability    REAL, q_repayment     REAL,
    q_future             REAL, q_equip        REAL, q_mainbk        REAL,
    q_weighted           REAL,
    sys_score            REAL, sys_score_b    REAL, sys_dscr        REAL,
    sys_op_margin        REAL, sys_icr        REAL, sys_approval    REAL,
    sys_ind_score        REAL, sys_bench      REAL,
    pred_proba_v3        REAL, pred_score_v3  INTEGER, pred_judgment_v3 TEXT,
    model_version        TEXT, computed_at    TEXT
)
"""


def update_ml_features(case_ids: list | None = None) -> int:
    """
    ml_features を更新する。

    Parameters
    ----------
    case_ids : list | None
        Noneの場合: past_cases にあって ml_features にない未登録ケースを追加。
        リストの場合: 指定した case_id を INSERT OR REPLACE で上書き。

    Returns
    -------
    int : 更新した件数（0 も正常）
    """
    import sqlite3

    if not os.path.exists(DB_PATH):
        logger.warning("[ml_features] DB が見つかりません: %s", DB_PATH)
        return 0

    _load_model()

    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(_CREATE_SQL)

            if case_ids is None:
                # 未登録ケースを取得
                rows = conn.execute("""
                    SELECT p.id, p.data, p.final_status
                    FROM past_cases p
                    LEFT JOIN ml_features m ON p.id = m.case_id
                    WHERE m.case_id IS NULL
                """).fetchall()
            else:
                if not case_ids:
                    return 0
                placeholders = ",".join("?" * len(case_ids))
                rows = conn.execute(
                    f"SELECT id, data, final_status FROM past_cases WHERE id IN ({placeholders})",
                    case_ids,
                ).fetchall()

            if not rows:
                logger.info("[ml_features] 追加対象なし（0件）")
                return 0

            records, skipped = [], 0
            for case_id, data_json, final_status in rows:
                try:
                    records.append(_build_record(case_id, data_json, final_status))
                except Exception as e:
                    skipped += 1
                    logger.warning("[ml_features] SKIP %s: %s", case_id, e)

            if records:
                conn.executemany(_UPSERT_SQL, records)
                conn.commit()

            n = len(records)
            logger.info("[ml_features] %d件追加/更新（スキップ %d件）", n, skipped)
            return n

    except Exception as e:
        logger.error("[ml_features] 更新失敗: %s", e)
        return 0


# ── CLI エントリポイント ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sqlite3

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ml_features テーブルを更新する")
    parser.add_argument("--all", action="store_true",
                        help="全件再計算して上書き（デフォルト: 未登録のみ追加）")
    args = parser.parse_args()

    if args.all:
        # 全件の case_id を取得して渡す
        with closing(sqlite3.connect(DB_PATH)) as conn:
            all_ids = [r[0] for r in conn.execute("SELECT id FROM past_cases").fetchall()]
        logger.info("全件再計算モード: %d件対象", len(all_ids))
        n = update_ml_features(case_ids=all_ids)
    else:
        n = update_ml_features()

    # 最終件数を表示
    with closing(sqlite3.connect(DB_PATH)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM ml_features").fetchone()[0]

    print(f"[完了] {n}件追加/更新 | ml_features 合計: {total}件")
