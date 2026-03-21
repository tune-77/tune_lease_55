"""
バッチ審査コンポーネント。
CSV アップロード → 一括スコアリング → 結果テーブル表示・ダウンロード。
"""
import io
import streamlit as st
import pandas as pd
import numpy as np

from data_cases import get_effective_coeffs, get_score_weights
from constants import APPROVAL_LINE, REVIEW_LINE


# ─────────────────────────────────────────────────────────────────────────────
# CSVテンプレート
# ─────────────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "業種小分類",
    "売上高(万円)",
    "経常利益(万円)",
    "純利益(万円)",
    "純資産(万円)",
    "総資産(万円)",
    "取得価格(万円)",
    "リース期間(月)",
    "契約件数",
    "格付(1-10)",
    "取引区分",       # "既存先" or "新規先"
    "物件ID（任意）", # vehicle / it_equipment / medical / manufacturing 等。空欄可
]

_CSV_SAMPLE = pd.DataFrame([
    ["06 総合工事業",         50000,  2000, 1500,  8000,  30000, 3000, 60, 3, 5, "既存先",  "construction_machine"],
    ["13 輸送用機械器具製造業", 120000, 5000, 3000, 20000, 80000, 8000, 48, 1, 4, "新規先",  "vehicle"],
    ["75 情報サービス業",       80000,  3000, 2000, 15000, 40000, 2000, 36, 5, 3, "既存先",  "it_equipment"],
    ["83 医療・福祉",           60000,  1000,  500, 10000, 35000, 5000, 60, 2, 6, "新規先",  "medical"],
    ["21 食料品製造業",         40000,  1500, 1000,  5000, 20000, 1500, 48, 1, 5, "既存先",  ""],  # 物件ID空欄→フォールバック
], columns=_CSV_COLUMNS)


def _get_csv_template() -> bytes:
    """テンプレートCSVをバイト列で返す。"""
    return _CSV_SAMPLE.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# 1件分のバッチスコア計算
# ─────────────────────────────────────────────────────────────────────────────

def _score_one(row: dict) -> dict:
    """
    1案件分のスコアを計算して返す。

    物件ID（任意）が入力されている場合:
        category_config.ASSET_ID_TO_CATEGORY でカテゴリを解決し、
        asset_scorer.calc_asset_score() + ASSET_WEIGHT で個別審査と同一ロジックを使用する。

    物件IDが空欄の場合:
        従来の簡易ロジック (term_ok + cost_ok) / 2 * 100 にフォールバックする。
    """
    try:
        gross        = float(row.get("売上高(万円)") or 0)
        ord_profit   = float(row.get("経常利益(万円)") or 0)
        net_income   = float(row.get("純利益(万円)") or 0)
        net_assets   = float(row.get("純資産(万円)") or 0)
        total_assets = float(row.get("総資産(万円)") or 0)
        acq_cost     = float(row.get("取得価格(万円)") or 0)
        lease_term   = int(row.get("リース期間(月)") or 60)
        contracts    = int(row.get("契約件数") or 0)
        grade        = int(row.get("格付(1-10)") or 5)
        customer_type = str(row.get("取引区分") or "既存先")
        asset_id_raw  = str(row.get("物件ID（任意）") or "").strip()

        # ── 借手スコア（財務指標 × 係数） ────────────────────────────────────
        coeff_key = "全体_既存先" if customer_type == "既存先" else "全体_新規先"
        coeffs = get_effective_coeffs(coeff_key)

        rieki_rate   = ord_profit / gross        if gross > 0        else 0.0
        equity_ratio = net_assets / total_assets if total_assets > 0 else 0.0
        roe          = net_income / net_assets   if net_assets > 0   else 0.0

        w_rr  = coeffs.get("rieki_rate", 0.20)
        w_eq  = coeffs.get("equity_ratio", 0.25)
        w_roe = coeffs.get("roe", 0.10)
        w_grd = coeffs.get("grade", 0.30)
        w_cnt = coeffs.get("contracts", 0.05)

        grade_score  = max(0.0, (11 - grade) / 10.0)
        repeat_score = min(1.0, contracts * 0.1)
        rr_score     = max(0.0, min(1.0, rieki_rate  + 0.5))
        eq_score     = max(0.0, min(1.0, equity_ratio + 0.3))
        roe_score    = max(0.0, min(1.0, roe + 0.5))

        borrower_raw = (
            w_grd * grade_score  +
            w_rr  * rr_score     +
            w_eq  * eq_score     +
            w_roe * roe_score    +
            w_cnt * repeat_score
        )
        weight_sum     = w_grd + w_rr + w_eq + w_roe + w_cnt
        borrower_score = (borrower_raw / weight_sum * 100) if weight_sum > 0 else 50.0

        # ── 物件スコア & 総合スコア ───────────────────────────────────────────
        asset_category = None
        asset_grade_label = "—"
        scoring_method = "簡易"

        # 物件IDが入力されている場合 → calc_asset_score() を使用
        if asset_id_raw and asset_id_raw.lower() not in ("", "nan", "none"):
            try:
                from category_config import ASSET_ID_TO_CATEGORY, ASSET_WEIGHT
                from asset_scorer import calc_asset_score, get_recommendation

                asset_category = ASSET_ID_TO_CATEGORY.get(asset_id_raw)
                if asset_category:
                    # 詳細スコア未入力（空dict）→ completeness_ratio=0 でペナルティ適用
                    contract = {"lease_months": lease_term}
                    asset_result = calc_asset_score(asset_category, {}, contract)
                    asset_score  = asset_result["total_score"]
                    asset_grade_label = asset_result["grade"]

                    # ASSET_WEIGHT でカテゴリ別の物件/借手配分を使用
                    wt      = ASSET_WEIGHT.get(asset_category, {})
                    w_a     = wt.get("asset_w", 0.15)
                    w_b_cat = wt.get("obligor_w", 0.85)
                    total_score   = borrower_score * w_b_cat + asset_score * w_a
                    scoring_method = "標準"
                else:
                    # IDは入力されたが ASSET_ID_TO_CATEGORY に未登録 → フォールバック
                    raise ValueError(f"未登録のasset_id: {asset_id_raw}")
            except Exception:
                # フォールバック
                asset_category = None
                asset_id_raw   = ""

        # 物件IDなし or 解決失敗 → 従来の簡易ロジック
        if not asset_category:
            term_ok     = 1.0 if 36 <= lease_term <= 72 else 0.6
            cost_ok     = 1.0 if 500 < acq_cost < 50000 else 0.7
            asset_score = (term_ok + cost_ok) / 2.0 * 100
            w_b_cat, w_a = get_score_weights()[:2]
            total_score  = borrower_score * w_b_cat + asset_score * w_a

        # ── PD 概算 / 判定 ─────────────────────────────────────────────────
        pd_map = {1: 0.3, 2: 0.5, 3: 1.0, 4: 2.0, 5: 3.5,
                  6: 6.0, 7: 10.0, 8: 15.0, 9: 25.0, 10: 40.0}
        pd_pct    = pd_map.get(grade, 5.0)
        score_int = int(round(total_score))
        if score_int >= APPROVAL_LINE:
            hantei = "良決"
        elif score_int >= REVIEW_LINE:
            hantei = "ボーダー"
        else:
            hantei = "否決"

        return {
            "借手スコア":     round(borrower_score, 1),
            "物件スコア":     round(asset_score, 1),
            "物件グレード":   asset_grade_label,
            "物件カテゴリ":   asset_category or "—",
            "スコアリング":   scoring_method,
            "総合スコア":     score_int,
            "PD概算(%)":      pd_pct,
            "自己資本比率(%)": round(equity_ratio * 100, 1),
            "経常利益率(%)":   round(rieki_rate * 100, 1),
            "判定":           hantei,
            "エラー":         "",
        }
    except Exception as e:
        return {
            "借手スコア": None, "物件スコア": None, "物件グレード": None,
            "物件カテゴリ": None, "スコアリング": None,
            "総合スコア": None, "PD概算(%)": None,
            "自己資本比率(%)": None, "経常利益率(%)": None,
            "判定": "エラー", "エラー": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_scoring():
    """バッチ審査 UI を描画する。"""
    st.title("⚡ バッチ審査（CSV一括判定）")
    st.info(
        "複数案件を CSV でアップロードして一括スコアリングします。\n\n"
        "**「物件ID（任意）」列を入力すると個別審査と同じ `calc_asset_score()` ロジックが適用されます**（標準モード）。"
        "空欄の場合は取得価格・リース期間ベースの簡易スコアを使用します（簡易モード）。\n\n"
        "対応物件ID: `vehicle` / `it_equipment` / `medical` / `manufacturing` / "
        "`construction_machine` / `office_furniture` / `restaurant` / `renewable` / `other` 等"
    )

    # テンプレートダウンロード
    st.download_button(
        "📄 CSV テンプレートをダウンロード",
        data=_get_csv_template(),
        file_name="batch_shinsa_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("CSV をアップロード", type=["csv"], key="batch_csv_upload")
    if uploaded is None:
        st.caption("上記テンプレートを参考に CSV を作成してアップロードしてください。")
        return

    try:
        df_in = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception:
        try:
            uploaded.seek(0)
            df_in = pd.read_csv(uploaded, encoding="shift_jis")
        except Exception as e:
            st.error(f"CSV 読み込みエラー: {e}")
            return

    # 必須列チェック
    missing_cols = [c for c in ["売上高(万円)", "総資産(万円)"] if c not in df_in.columns]
    if missing_cols:
        st.error(f"必須列が不足しています: {missing_cols}")
        return

    st.success(f"{len(df_in)} 件を読み込みました。")
    st.dataframe(df_in.head(5), use_container_width=True)

    if st.button("🚀 一括スコアリング実行", type="primary"):
        results = []
        prog = st.progress(0, text="スコアリング中...")
        for i, (_, row) in enumerate(df_in.iterrows()):
            results.append(_score_one(row.to_dict()))
            prog.progress((i + 1) / len(df_in), text=f"{i+1}/{len(df_in)} 件処理中...")
        prog.empty()

        df_out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        # サマリー表示
        col1, col2, col3 = st.columns(3)
        total = len(df_out)
        col1.metric("良決", f"{(df_out['判定'] == '良決').sum()}件 / {total}件")
        col2.metric("ボーダー", f"{(df_out['判定'] == 'ボーダー').sum()}件 / {total}件")
        col3.metric("否決", f"{(df_out['判定'] == '否決').sum()}件 / {total}件")

        # 結果テーブル
        st.subheader("判定結果")

        # スコアリングモード別件数を表示
        n_standard = (df_out.get("スコアリング") == "標準").sum() if "スコアリング" in df_out else 0
        n_simple   = len(df_out) - n_standard
        st.caption(
            f"🟢 標準モード（calc_asset_score 使用）: {n_standard}件　"
            f"⬜ 簡易モード（フォールバック）: {n_simple}件"
        )

        def _color_hantei(val):
            if val == "良決":
                return "background-color: #d4edda; color: #155724"
            elif val == "ボーダー":
                return "background-color: #fff3cd; color: #856404"
            elif val == "否決":
                return "background-color: #f8d7da; color: #721c24"
            return ""

        def _color_scoring(val):
            if val == "標準":
                return "background-color: #e8f5e9; color: #2e7d32"
            return "color: #9e9e9e"

        styled = (
            df_out.style
            .applymap(_color_hantei, subset=["判定"])
            .applymap(_color_scoring, subset=["スコアリング"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ダウンロード
        csv_out = df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 結果 CSV をダウンロード",
            data=csv_out,
            file_name="batch_shinsa_result.csv",
            mime="text/csv",
        )
