"""
SHAP説明可能AI コンポーネント
================================
過去案件データでXGBoostモデルを学習し、SHAPによる
判定根拠の可視化をStreamlitで表示する。

使い方（analysis_results.py などから呼び出し）:
    from components.shap_explanation import render_shap_explanation
    render_shap_explanation(current_inputs, current_result)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import io

import streamlit as st

# ==============================
# 日本語フォント設定
# ==============================
_FONT_PATH = Path(__file__).parent.parent / "NotoSansCJKjp-Regular.otf"
if _FONT_PATH.exists():
    fm.fontManager.addfont(str(_FONT_PATH))
    _FONT_NAME = fm.FontProperties(fname=str(_FONT_PATH)).get_name()
else:
    _FONT_NAME = None

def _set_font():
    if _FONT_NAME:
        plt.rcParams["font.family"] = _FONT_NAME
    else:
        plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "DejaVu Sans"]

# ==============================
# 定数
# ==============================
_DATA_PATH = Path(__file__).parent.parent.parent / "past_cases.jsonl"
if not _DATA_PATH.exists():
    _DATA_PATH = Path(__file__).parent.parent / "past_cases.jsonl"

CAT_COLS = ["industry_major", "customer_type", "main_bank",
            "competitor", "grade", "deal_source", "lease_asset"]
NUM_COLS = ["nenshu", "op_profit", "ord_profit", "bank_credit", "lease_credit",
            "acquisition_cost", "contracts", "lease_asset_score",
            "score", "score_borrower", "pd_percent", "qual_score"]
FEATURE_COLS = CAT_COLS + NUM_COLS

LABEL_JA = {
    "industry_major":   "業種",
    "customer_type":    "顧客区分",
    "main_bank":        "メイン銀行",
    "competitor":       "競合有無",
    "grade":            "等級",
    "deal_source":      "案件ルート",
    "lease_asset":      "リース物件",
    "nenshu":           "年商",
    "op_profit":        "営業利益",
    "ord_profit":       "経常利益",
    "bank_credit":      "銀行与信",
    "lease_credit":     "リース与信",
    "acquisition_cost": "取得価額",
    "contracts":        "契約件数",
    "lease_asset_score":"物件スコア",
    "score":            "総合スコア",
    "score_borrower":   "借手スコア",
    "pd_percent":       "デフォルト確率(%)",
    "qual_score":       "定性スコア",
}
FEATURE_NAMES_JA = [LABEL_JA.get(c, c) for c in FEATURE_COLS]


# ==============================
# データ→DataFrame
# ==============================
def _case_to_row(c, target=None):
    inp = c.get("inputs") or {}
    res = c.get("result") or {}
    qs  = inp.get("qualitative_scoring") or {}
    row = {
        "industry_major":   c.get("industry_major", "")[:3],
        "customer_type":    c.get("customer_type", ""),
        "main_bank":        c.get("main_bank", ""),
        "competitor":       c.get("competitor", ""),
        "grade":            inp.get("grade", ""),
        "deal_source":      inp.get("deal_source", ""),
        "lease_asset":      inp.get("lease_asset_name", ""),
        "nenshu":           float(inp.get("nenshu", 0)),
        "op_profit":        float(inp.get("op_profit", 0)),
        "ord_profit":       float(inp.get("ord_profit", 0)),
        "bank_credit":      float(inp.get("bank_credit", 0)),
        "lease_credit":     float(inp.get("lease_credit", 0)),
        "acquisition_cost": float(inp.get("acquisition_cost", 0)),
        "contracts":        float(inp.get("contracts", 0)),
        "lease_asset_score":float(inp.get("lease_asset_score", 0)),
        "score":            float(res.get("score", 0)),
        "score_borrower":   float(res.get("score_borrower", 0)),
        "pd_percent":       float(res.get("pd_percent", 0)),
        "qual_score":       float(qs.get("weighted_score", 0)),
    }
    if target is not None:
        row["target"] = target
    return row


# ==============================
# モデル学習（キャッシュ）
# ==============================
@st.cache_resource(show_spinner=False)
def _load_model_and_explainer():
    """XGBoostモデルとSHAP Explainerを構築してキャッシュ"""
    try:
        import shap
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        return None, None, None, None, str(e)

    if not _DATA_PATH.exists():
        return None, None, None, None, f"データファイルが見つかりません: {_DATA_PATH}"

    cases = [json.loads(l) for l in open(_DATA_PATH, encoding="utf-8")]
    labeled = [c for c in cases if c.get("final_status") in ("成約", "失注")]

    if len(labeled) < 5:
        return None, None, None, None, f"学習に必要な件数が不足（{len(labeled)}件）。成約/失注登録後に再試行してください。"

    rows = [_case_to_row(c, target=1 if c["final_status"] == "成約" else 0) for c in labeled]
    df = pd.DataFrame(rows)

    encoders = {}
    df_enc = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURE_COLS].values.astype(float)
    y = df_enc["target"].values

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer(pd.DataFrame(X, columns=FEATURE_NAMES_JA))

    return model, explainer, encoders, shap_values_all, None


def _encode_row(row_dict, encoders):
    """1件分の辞書をエンコードしてndarrayで返す"""
    enc = dict(row_dict)
    for col in CAT_COLS:
        le = encoders.get(col)
        val = str(enc.get(col, ""))
        if le is not None:
            if val in le.classes_:
                enc[col] = float(le.transform([val])[0])
            else:
                enc[col] = 0.0
        else:
            enc[col] = 0.0
    return np.array([float(enc.get(c, 0)) for c in FEATURE_COLS], dtype=float)


# ==============================
# グラフ描画ユーティリティ
# ==============================
def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_summary_bar(shap_values_all):
    """全変数の平均|SHAP|バーグラフ"""
    import shap
    _set_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_values_all, max_display=12, show=False, ax=ax)
    ax.set_title("変数重要度（平均|SHAP値|）\n成約/失注への影響が大きい変数ランキング", fontsize=11, pad=10)
    ax.set_xlabel("平均 |SHAP値|（成約確率への影響度）", fontsize=9)
    plt.tight_layout()
    return _fig_to_bytes(fig)


def _plot_waterfall(explainer, x_vec, pred_prob, label=""):
    """個別案件のwaterfallグラフ"""
    import shap
    _set_font()
    sv = explainer(pd.DataFrame([x_vec], columns=FEATURE_NAMES_JA))
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    direction = "▲ 成約方向" if pred_prob >= 0.5 else "▼ 失注方向"
    title = f"判定根拠（SHAP値）{label}\n成約確率: {pred_prob*100:.1f}%  {direction}"
    plt.title(title, fontsize=11, pad=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def _plot_beeswarm(shap_values_all):
    """全案件のビースウォームグラフ"""
    import shap
    _set_font()
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values_all, max_display=12, show=False)
    plt.title("各変数の影響方向と大きさ\n赤=高値・青=低値  右=成約方向・左=失注方向", fontsize=11, pad=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


# ==============================
# メインレンダリング関数
# ==============================
def render_shap_explanation(current_case: dict | None = None):
    """
    SHAP説明パネルをStreamlitに描画する。

    Parameters
    ----------
    current_case : dict | None
        現在審査中の案件辞書（load_all_cases()の1件分）。
        None の場合は全体分析のみ表示。
    """
    st.markdown("### 🔍 SHAP 判定根拠の可視化")
    st.caption(
        "過去の成約/失注データでXGBoostモデルを学習し、"
        "SHAPにより「何がどれだけ判定に影響したか」を可視化します。"
    )

    # モデル読み込み
    with st.spinner("SHAPモデルを準備中..."):
        model, explainer, encoders, shap_values_all, err = _load_model_and_explainer()

    if err:
        st.warning(f"⚠ SHAPモデルを構築できません: {err}")
        return

    n_cases = len(shap_values_all)
    st.info(f"学習データ: {n_cases}件（成約/失注登録済み）")

    tab1, tab2, tab3 = st.tabs(["📊 変数重要度", "🌡 影響方向", "🔎 この案件の根拠"])

    # --- Tab1: Summary Bar ---
    with tab1:
        st.markdown("**成約/失注の予測に最も影響する変数ランキング**")
        buf = _plot_summary_bar(shap_values_all)
        st.image(buf, use_container_width=True)
        # テキストサマリ
        mean_abs = np.abs(shap_values_all.values).mean(axis=0)
        ranking  = np.argsort(mean_abs)[::-1]
        rows_tbl = []
        for rank, i in enumerate(ranking[:8]):
            rows_tbl.append({
                "順位": rank + 1,
                "変数": FEATURE_NAMES_JA[i],
                "影響度": round(float(mean_abs[i]), 4),
            })
        st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

    # --- Tab2: Beeswarm ---
    with tab2:
        st.markdown("**各変数の値（赤=高/青=低）と成約への影響方向**")
        buf = _plot_beeswarm(shap_values_all)
        st.image(buf, use_container_width=True)
        st.caption("右向き（プラス）= 成約確率UP  /  左向き（マイナス）= 成約確率DOWN")

    # --- Tab3: 個別案件 Waterfall ---
    with tab3:
        if current_case is None:
            st.info("審査フォームで案件を読み込むと、この案件の判定根拠が表示されます。")
            return

        try:
            import shap
            row = _case_to_row(current_case)
            x_vec = _encode_row(row, encoders)
            pred_prob = float(model.predict_proba([x_vec])[0][1])

            # 判定バッジ
            col_a, col_b = st.columns(2)
            col_a.metric("SHAP成約確率", f"{pred_prob*100:.1f}%")
            actual = current_case.get("final_status", "未登録")
            col_b.metric("実績", actual)

            buf = _plot_waterfall(
                explainer, x_vec, pred_prob,
                label=f"| {current_case.get('industry_major','')}"
            )
            st.image(buf, use_container_width=True)
            st.caption(
                "縦軸左の数値=その案件の実際の値  "
                "赤バー=成約確率を上げた要因  青バー=成約確率を下げた要因"
            )

            # 上位寄与要因をテキストで表示
            sv_one = explainer(pd.DataFrame([x_vec], columns=FEATURE_NAMES_JA))
            sv_arr = sv_one.values[0]
            top_pos = np.argsort(sv_arr)[::-1][:3]
            top_neg = np.argsort(sv_arr)[:3]

            st.markdown("**▲ 成約を後押しした要因（上位3）**")
            for i in top_pos:
                if sv_arr[i] > 0:
                    st.markdown(f"- {FEATURE_NAMES_JA[i]}: **+{sv_arr[i]:.3f}**")

            st.markdown("**▼ 成約を抑制した要因（上位3）**")
            for i in top_neg:
                if sv_arr[i] < 0:
                    st.markdown(f"- {FEATURE_NAMES_JA[i]}: **{sv_arr[i]:.3f}**")

        except Exception as e:
            st.error(f"SHAP計算エラー: {e}")
