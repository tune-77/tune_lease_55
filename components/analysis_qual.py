import streamlit as st
import pandas as pd
from data_cases import load_all_cases
from analysis_regression import run_qualitative_contract_analysis
from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS

QUALITATIVE_ANALYSIS_MIN_CASES = 50

def render_qualitative_analysis():
    """📉 定性要因分析 (50件〜) タブのUIとロジックを描画する"""
    st.title("📉 定性要因で成約予測")
    st.caption("取引区分・競合状況・顧客区分・商談ソース・リース物件・定性スコアリング6項目（設立・経営年数、顧客安定性、返済履歴、事業将来性、設置目的、メイン取引銀行）のみを使って、ロジスティック回帰とLightGBMで成約/不成約を分析します。")
    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    n_reg = len(registered)
    if n_reg < QUALITATIVE_ANALYSIS_MIN_CASES:
        st.warning(f"成約・失注の登録が **{QUALITATIVE_ANALYSIS_MIN_CASES}件** 以上で利用できます。（現在: **{n_reg}件**）")
    else:
        st.success(f"登録件数: **{n_reg}件**（成約+失注）。分析を実行できます。")
        if st.button("🚀 ロジスティック回帰とLightGBMを実行", key="run_qual_analysis"):
            with st.spinner("分析中..."):
                result = run_qualitative_contract_analysis(QUALITATIVE_SCORING_CORRECTION_ITEMS)
            if result is None:
                st.error("件数不足で分析できませんでした。")
            else:
                st.session_state["qualitative_analysis_result"] = result
            st.rerun()
        result = st.session_state.get("qualitative_analysis_result")
        if result and result.get("n_cases") == n_reg:
            st.subheader("結果サマリ")
            st.metric("分析件数", f"{result['n_cases']}件（成約{result['n_positive']} / 失注{result['n_negative']}）")
            c1, c2, c3 = st.columns(3)
            with c1:
                if "accuracy_lr" in result:
                    st.metric("ロジスティック回帰 正解率", f"{result['accuracy_lr']*100:.1f}%")
                if "auc_lr" in result and result.get("auc_lr") is not None:
                    st.metric("ロジスティック回帰 AUC", f"{result['auc_lr']:.3f}")
                if "lr_error" in result:
                    st.error(result["lr_error"])
            with c2:
                if "accuracy_lgb" in result:
                    st.metric("LightGBM 正解率", f"{result['accuracy_lgb']*100:.1f}%")
                if "auc_lgb" in result and result.get("auc_lgb") is not None:
                    st.metric("LightGBM AUC", f"{result['auc_lgb']:.3f}")
                if "lgb_error" in result:
                    st.error(result["lgb_error"])
            with c3:
                if "auc_ensemble" in result:
                    st.metric("アンサンブル 正解率", f"{result['accuracy_ensemble']*100:.1f}%")
                    st.metric("アンサンブル AUC", f"{result['auc_ensemble']:.3f}")
                    alpha = result.get("ensemble_alpha", 0.5)
                    st.caption(f"最適割合: LR {alpha:.0%} + LGB {1-alpha:.0%}")
            st.divider()
            st.subheader("ロジスティック回帰 係数（成約に効く方向: 正で成約にプラス）")
            if "lr_coef" in result:
                lr_df = pd.DataFrame(result["lr_coef"], columns=["項目", "係数"])
                lr_df = lr_df.sort_values("係数", key=abs, ascending=False)
                st.dataframe(lr_df, width='stretch', hide_index=True)
                if "lr_intercept" in result:
                    st.caption(f"切片: {result['lr_intercept']:.4f}")
            st.divider()
            st.subheader("LightGBM 特徴量重要度")
            if "lgb_importance" in result:
                imp_df = pd.DataFrame(result["lgb_importance"], columns=["項目", "重要度"])
                imp_df = imp_df.sort_values("重要度", ascending=False)
                st.dataframe(imp_df, width='stretch', hide_index=True)
            if "shap_importance" in result:
                st.divider()
                st.subheader("SHAP 特徴量重要度（成約への影響）")
                shap_df = pd.DataFrame(result["shap_importance"], columns=["項目", "SHAP重要度"])
                shap_df = shap_df.sort_values("SHAP重要度", ascending=False)
                st.bar_chart(shap_df.set_index("項目")["SHAP重要度"])
                st.caption("各項目の平均|SHAP値|。値が大きいほど成約判定への影響が大きい。")
        else:
            result = None
        if result is None and n_reg >= QUALITATIVE_ANALYSIS_MIN_CASES:
            st.info("上の「ロジスティック回帰とLightGBMを実行」ボタンで分析を開始してください。")
