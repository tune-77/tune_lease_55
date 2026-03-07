import streamlit as st
import pandas as pd
from data_cases import load_all_cases
from analysis_regression import (
    run_quantitative_contract_analysis,
    run_quantitative_by_industry,
    run_quantitative_by_indicator,
    optimize_score_weights_from_regression
)
from analysis_regression import (
    QUALITATIVE_ANALYSIS_MIN_CASES,
    COEFF_LABELS,
    INDUSTRY_BASES,
    BENCH_BASES,
)
from data_cases import (
    DEFAULT_WEIGHT_QUANT,
    DEFAULT_WEIGHT_QUAL,
    get_score_weights,
    load_coeff_overrides,
    save_coeff_overrides,
)


def render_quantitative_analysis():
    """📈 定量要因分析 (50件〜) タブのUIとロジックを描画する"""
    st.title("📈 定量要因で成約予測")
    st.caption("業種モデルと同様の定量項目（売上・与信・利益・資産・格付・取引・競合・金利差・業界景気・定性タグ・自己資本比率・定性スコア合計など）のみで、ロジスティック回帰とLightGBMにより成約/不成約を分析します。アンサンブル割合はテストデータでAUC最大化により最適化します。")
    
    all_logs = load_all_cases()
    registered_quant = [c for c in all_logs if c.get("final_status") in ["成約", "失注"]]
    n_reg_q = len(registered_quant)
    
    if n_reg_q < QUALITATIVE_ANALYSIS_MIN_CASES:
        st.warning(f"成約・失注の登録が **{QUALITATIVE_ANALYSIS_MIN_CASES}件** 以上で利用できます。（現在: **{n_reg_q}件**）")
    else:
        st.success(f"登録件数: **{n_reg_q}件**（成約+失注）。分析を実行できます。")
        if st.button("🚀 ロジスティック回帰とLightGBMを実行", key="run_quant_analysis"):
            with st.spinner("分析中..."):
                result_q = run_quantitative_contract_analysis()
            if result_q is None:
                st.error("件数不足またはデータ不備で分析できませんでした。")
            else:
                st.session_state["quantitative_analysis_result"] = result_q
            st.rerun()
            
        result_q = st.session_state.get("quantitative_analysis_result")
        
        if result_q and result_q.get("n_cases") == n_reg_q:
            st.subheader("結果サマリ")
            st.metric("分析件数", f"{result_q['n_cases']}件（成約{result_q['n_positive']} / 失注{result_q['n_negative']}）")
            c1, c2, c3 = st.columns(3)
            with c1:
                if "accuracy_lr" in result_q:
                    st.metric("ロジスティック回帰 正解率", f"{result_q['accuracy_lr']*100:.1f}%")
                if "auc_lr" in result_q and result_q.get("auc_lr") is not None:
                    st.metric("ロジスティック回帰 AUC", f"{result_q['auc_lr']:.3f}")
                if "lr_error" in result_q:
                    st.error(result_q["lr_error"])
            with c2:
                if "accuracy_lgb" in result_q:
                    st.metric("LightGBM 正解率", f"{result_q['accuracy_lgb']*100:.1f}%")
                if "auc_lgb" in result_q and result_q.get("auc_lgb") is not None:
                    st.metric("LightGBM AUC", f"{result_q['auc_lgb']:.3f}")
                if "lgb_error" in result_q:
                    st.error(result_q["lgb_error"])
            with c3:
                if "auc_ensemble" in result_q:
                    st.metric("アンサンブル 正解率", f"{result_q['accuracy_ensemble']*100:.1f}%")
                    st.metric("アンサンブル AUC", f"{result_q['auc_ensemble']:.3f}")
                    alpha_q = result_q.get("ensemble_alpha", 0.5)
                    st.caption(f"最適割合: LR {alpha_q:.0%} + LGB {1-alpha_q:.0%}")
                    
            st.divider()
            st.subheader("ロジスティック回帰 係数（成約に効く方向: 正で成約にプラス）")
            if "lr_coef" in result_q:
                labels = [COEFF_LABELS.get(k, k) for k in result_q["feature_names"]]
                lr_df_q = pd.DataFrame([(labels[i], c) for i, (_, c) in enumerate(result_q["lr_coef"])], columns=["項目", "係数"])
                lr_df_q = lr_df_q.sort_values("係数", key=abs, ascending=False)
                st.dataframe(lr_df_q, use_container_width=True, hide_index=True)
                if "lr_intercept" in result_q:
                    st.caption(f"切片: {result_q['lr_intercept']:.4f}")
                    
            st.divider()
            st.subheader("LightGBM 特徴量重要度")
            if "lgb_importance" in result_q:
                labels_q = [COEFF_LABELS.get(k, k) for k in result_q["feature_names"]]
                imp_df_q = pd.DataFrame([(labels_q[i], imp) for i, (_, imp) in enumerate(result_q["lgb_importance"])], columns=["項目", "重要度"])
                imp_df_q = imp_df_q.sort_values("重要度", ascending=False)
                st.dataframe(imp_df_q, use_container_width=True, hide_index=True)
            if "shap_importance" in result_q:
                st.divider()
                st.subheader("SHAP 特徴量重要度（成約への影響）")
                labels_q2 = [COEFF_LABELS.get(k, k) for k in result_q["feature_names"]]
                shap_df_q = pd.DataFrame([(labels_q2[i], v) for i, (_, v) in enumerate(result_q["shap_importance"])], columns=["項目", "SHAP重要度"])
                shap_df_q = shap_df_q.sort_values("SHAP重要度", ascending=False)
                st.bar_chart(shap_df_q.set_index("項目")["SHAP重要度"])
                st.caption("各項目の平均|SHAP値|。値が大きいほど成約判定への影響が大きい。")
        else:
            result_q = None
            
        if result_q is None and n_reg_q >= QUALITATIVE_ANALYSIS_MIN_CASES:
            st.info("上の「ロジスティック回帰とLightGBMを実行」ボタンで分析を開始してください。")

        st.divider()
        st.subheader("業種ごと定量分析")
        st.caption("業種（全体・医療・運送業・サービス業・製造業）ごとにLR+LGB+アンサンブルを実行。データが50件未満の業種は50件にブートストラップして学習します。")
        if st.button("🚀 業種ごと分析を実行", key="run_quant_by_industry"):
            with st.spinner("業種ごとに分析中..."):
                by_ind = run_quantitative_by_industry()
            if by_ind is not None:
                st.session_state["quant_by_industry"] = by_ind
            st.rerun()
            
        by_industry = st.session_state.get("quant_by_industry")
        if by_industry:
            for base in INDUSTRY_BASES:
                res = by_industry.get(base, {})
                if res.get("skip"):
                    with st.expander(f"**{base}**", expanded=False):
                        st.caption(res.get("reason", "スキップ"))
                else:
                    with st.expander(f"**{base}** — 元データ{res.get('n_cases_orig', res['n_cases'])}件" + ("（50件にリサンプル済）" if res.get("bootstrapped") else ""), expanded=False):
                        st.metric("分析件数", f"{res['n_cases']}件（成約{res['n_positive']}/失注{res['n_negative']}）")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if "accuracy_lr" in res: st.metric("LR 正解率", f"{res['accuracy_lr']*100:.1f}%")
                            if "auc_lr" in res and res.get("auc_lr"): st.metric("LR AUC", f"{res['auc_lr']:.3f}")
                        with c2:
                            if "accuracy_lgb" in res: st.metric("LGB 正解率", f"{res['accuracy_lgb']*100:.1f}%")
                            if "auc_lgb" in res and res.get("auc_lgb"): st.metric("LGB AUC", f"{res['auc_lgb']:.3f}")
                        with c3:
                            if "auc_ensemble" in res:
                                st.metric("アンサンブル AUC", f"{res['auc_ensemble']:.3f}")
                                st.caption(f"最適: LR {res.get('ensemble_alpha', 0.5):.0%} + LGB {1-res.get('ensemble_alpha', 0.5):.0%}")
                        if "lgb_importance" in res:
                            names = [COEFF_LABELS.get(k, k) for k in res["feature_names"]]
                            imp = pd.DataFrame([(names[i], v) for i, (_, v) in enumerate(res["lgb_importance"])], columns=["項目", "重要度"]).sort_values("重要度", ascending=False)
                            st.dataframe(imp.head(10), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("指標ごと定量分析")
        st.caption("指標モデル（全体_指標・医療_指標・運送業_指標・サービス業_指標・製造業_指標）ごとにLR+LGB+アンサンブルを実行。データ不足時は50件にブートストラップ。")
        if st.button("🚀 指標ごと分析を実行", key="run_quant_by_indicator"):
            with st.spinner("指標ごとに分析中..."):
                by_ind = run_quantitative_by_indicator()
            if by_ind is not None:
                st.session_state["quant_by_indicator"] = by_ind
            st.rerun()
            
        by_indicator = st.session_state.get("quant_by_indicator")
        if by_indicator:
            for bench in BENCH_BASES:
                res = by_indicator.get(bench, {})
                if res.get("skip"):
                    with st.expander(f"**{bench}**", expanded=False):
                        st.caption(res.get("reason", "スキップ"))
                else:
                    with st.expander(f"**{bench}** — 元データ{res.get('n_cases_orig', res['n_cases'])}件" + ("（50件にリサンプル済）" if res.get("bootstrapped") else ""), expanded=False):
                        st.metric("分析件数", f"{res['n_cases']}件（成約{res['n_positive']}/失注{res['n_negative']}）")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if "accuracy_lr" in res: st.metric("LR 正解率", f"{res['accuracy_lr']*100:.1f}%")
                            if "auc_lr" in res and res.get("auc_lr"): st.metric("LR AUC", f"{res['auc_lr']:.3f}")
                        with c2:
                            if "accuracy_lgb" in res: st.metric("LGB 正解率", f"{res['accuracy_lgb']*100:.1f}%")
                            if "auc_lgb" in res and res.get("auc_lgb"): st.metric("LGB AUC", f"{res['auc_lgb']:.3f}")
                        with c3:
                            if "auc_ensemble" in res:
                                st.metric("アンサンブル AUC", f"{res['auc_ensemble']:.3f}")
                                st.caption(f"最適: LR {res.get('ensemble_alpha', 0.5):.0%} + LGB {1-res.get('ensemble_alpha', 0.5):.0%}")
                        if "lgb_importance" in res:
                            fnames = res["feature_names"]
                            imp = pd.DataFrame([(fnames[i], v) for i, (_, v) in enumerate(res["lgb_importance"])], columns=["項目", "重要度"]).sort_values("重要度", ascending=False)
                            st.dataframe(imp.head(10), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("重み最適化（回帰）")
        st.caption("成約/失注データでロジスティック回帰を行い、借手スコア・物件スコアの推奨割合と、総合スコア・定性スコアの推奨割合を算出します。参考値として表示します。")
        if st.button("🔄 回帰で重みを最適化", key="run_weight_optimize"):
            with st.spinner("回帰で重みを算出中..."):
                opt = optimize_score_weights_from_regression()
            if opt is not None:
                st.session_state["weight_optimize_result"] = opt
            else:
                st.session_state["weight_optimize_result"] = None
            st.rerun()
            
        wopt = st.session_state.get("weight_optimize_result")
        if wopt:
            w_b_cur, w_a_cur, w_q_cur, w_ql_cur = get_score_weights()
            st.success(f"分析件数: **{wopt['n_cases']}件**。回帰AUC: **{wopt.get('auc_borrower_asset', 0):.3f}**")
            st.markdown("**推奨: 借手** " + f"**{wopt['recommended_borrower_pct']*100:.0f}%** / **物件** **{wopt['recommended_asset_pct']*100:.0f}%**（現在 {w_b_cur*100:.0f}% / {w_a_cur*100:.0f}%）")
            if "recommended_quant_pct" in wopt and "recommended_qual_pct" in wopt:
                st.markdown("**推奨: 総合** " + f"**{wopt['recommended_quant_pct']*100:.0f}%** / **定性** **{wopt['recommended_qual_pct']*100:.0f}%**（現在 {w_q_cur*100:.0f}% / {w_ql_cur*100:.0f}%）")
                if wopt.get("n_cases_with_qual"):
                    st.caption(f"定性あり {wopt['n_cases_with_qual']}件・AUC {wopt.get('auc_quant_qual', 0):.3f}")
            else:
                st.caption("定性データ不足のため総合/定性は 60%/40% のまま")
                
            if st.button("💾 推奨を保存してスコア計算に反映", key="save_weight_overrides"):
                overrides = load_coeff_overrides() or {}
                overrides["score_weights"] = {
                    "borrower": wopt["recommended_borrower_pct"],
                    "asset": wopt["recommended_asset_pct"],
                    "quant": wopt.get("recommended_quant_pct", DEFAULT_WEIGHT_QUANT),
                    "qual": wopt.get("recommended_qual_pct", DEFAULT_WEIGHT_QUAL),
                }
                if save_coeff_overrides(overrides):
                    st.success("保存しました。今後の審査でこの重みを使います。")
                    st.rerun()
                else:
                    st.error("保存に失敗しました。")
        elif n_reg_q >= QUALITATIVE_ANALYSIS_MIN_CASES:
            st.info("「回帰で重みを最適化」ボタンで、データに基づく推奨割合を算出できます。")
