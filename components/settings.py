import streamlit as st
import pandas as pd
from data_cases import load_all_cases
from analysis_regression import (
    build_design_matrix_from_logs,
    run_regression_and_get_coeffs,
    build_design_matrix_indicator_from_logs,
    run_regression_indicator_and_get_coeffs
)
from data_cases import get_effective_coeffs, load_coeff_overrides, save_coeff_overrides
from analysis_regression import (
    COEFF_MAIN_KEYS,
    COEFF_EXTRA_KEYS,
    INDUSTRY_MODEL_KEYS,
    INDICATOR_MODEL_KEYS,
    PRIOR_COEFF_MODEL_KEYS,
)

def render_coeff_analysis():
    """🔧 係数分析・更新 (β) タブのUI表示とロジック"""
    st.title("🔧 係数分析・更新（成約/失注で係数を更新）")
    st.info("結果登録した「成約・失注」を目的変数に、審査モデルと同一仕様のロジスティック回帰で係数を推定し、審査スコアに反映できます。")
    
    all_logs = load_all_cases()
    if not all_logs:
        st.warning("分析するためのデータがまだありません。審査を実行し、結果登録で成約/失注を登録してください。")
    else:
        X_reg, y_reg = build_design_matrix_from_logs(all_logs)
        n_ok = int((y_reg == 1).sum()) if y_reg is not None else 0
        n_ng = int((y_reg == 0).sum()) if y_reg is not None else 0
        n_total = n_ok + n_ng
        
        if X_reg is None or n_total < 5:
            st.error(f"回帰分析には成約/失注が登録されたデータが少なくとも5件必要です。（現在: 成約 {n_ok} 件・失注 {n_ng} 件）")
        else:
            st.write(f"**目的変数**: 成約=1, 失注=0")
            st.write(f"分析対象: **{n_total}件**（成約: {n_ok}件, 失注: {n_ng}件）")
            
            if st.button("🚀 回帰分析を実行して係数を算出", key="btn_run_regression"):
                try:
                    coeff_dict, model = run_regression_and_get_coeffs(X_reg, y_reg)
                    acc = model.score(X_reg, y_reg)
                    st.session_state["regression_coeffs"] = coeff_dict
                    st.session_state["regression_accuracy"] = acc
                    st.success("回帰完了。下記の係数を「係数を更新して保存」で審査スコアに反映できます。")
                except Exception as e:
                    st.error(f"回帰エラー: {e}")
                    import traceback
                    with st.expander("詳細", expanded=False):
                        st.code(traceback.format_exc())
            
            if "regression_coeffs" in st.session_state:
                coeff_dict = st.session_state["regression_coeffs"]
                acc = st.session_state.get("regression_accuracy", 0)
                st.subheader("算出された係数（既存項目＋追加項目）")
                res_rows = [{"変数": "intercept", "算出係数": coeff_dict.get("intercept", 0)}]
                for k in COEFF_MAIN_KEYS:
                    res_rows.append({"変数": k, "算出係数": coeff_dict.get(k, 0)})
                for k in COEFF_EXTRA_KEYS:
                    res_rows.append({"変数": k, "算出係数": coeff_dict.get(k, 0)})
                st.dataframe(pd.DataFrame(res_rows).style.format({"算出係数": "{:.6f}"}), use_container_width=True)
                st.metric("モデル予測精度 (Accuracy)", f"{acc:.1%}")
                
                if st.button("💾 係数を更新して保存", key="btn_save_coeffs"):
                    overrides = load_coeff_overrides() or {}
                    overrides["全体_既存先"] = coeff_dict
                    if save_coeff_overrides(overrides):
                        st.success("係数を保存しました。以降の審査スコアはこの係数で計算されます。")
                    else:
                        st.error("保存に失敗しました。")
            
            st.divider()
            st.divider()
            st.subheader("業種・指標ごとのベイズ回帰（既存項目＋追加項目）")
            st.caption("業種モデル（全体/運送業/サービス業/製造業×既存先/新規先）と指標モデル（全体/運送業/サービス業/製造業 指標×既存先/新規先）を、それぞれデータが5件以上ある組だけ回帰し、係数を更新して保存します。")
            if st.button("🔄 業種・指標ごとにベイズ回帰を実行して保存", key="btn_bayesian_all"):
                overrides = load_coeff_overrides() or {}
                min_n = 5
                results = []
                for model_key in INDUSTRY_MODEL_KEYS:
                    X_k, y_k = build_design_matrix_from_logs(all_logs, model_key=model_key)
                    n_k = len(y_k) if y_k is not None else 0
                    if n_k >= min_n:
                        try:
                            coeff_k, mod_k = run_regression_and_get_coeffs(X_k, y_k)
                            overrides[model_key] = coeff_k
                            acc_k = mod_k.score(X_k, y_k)
                            results.append(f"{model_key}: {n_k}件, Accuracy={acc_k:.1%}")
                        except Exception as e:
                            results.append(f"{model_key}: エラー {e}")
                    else:
                        results.append(f"{model_key}: データ不足 ({n_k}件)")
                for ind_key in INDICATOR_MODEL_KEYS:
                    X_i, y_i = build_design_matrix_indicator_from_logs(all_logs, ind_key)
                    n_i = len(y_i) if y_i is not None else 0
                    if n_i >= min_n:
                        try:
                            coeff_i, mod_i = run_regression_indicator_and_get_coeffs(X_i, y_i)
                            overrides[ind_key] = coeff_i
                            acc_i = mod_i.score(X_i, y_i)
                            results.append(f"{ind_key}: {n_i}件, Accuracy={acc_i:.1%}")
                        except Exception as e:
                            results.append(f"{ind_key}: エラー {e}")
                    else:
                        results.append(f"{ind_key}: データ不足 ({n_i}件)")
                if save_coeff_overrides(overrides):
                    st.success("業種・指標ごとの係数を保存しました。")
                else:
                    st.error("保存に失敗しました。")
                for r in results:
                    st.caption(r)

            st.subheader("参考: 現在の審査で使っている係数（全体_既存先）")
            current = get_effective_coeffs("全体_既存先")
            overrides = load_coeff_overrides()
            if overrides and "全体_既存先" in overrides:
                st.caption("※ 成約/失注で更新した係数（既存＋追加項目）が適用されています。")
            ref_rows = [{"変数": k, "現在の係数": current.get(k, 0)} for k in ["intercept"] + COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS]
            st.dataframe(pd.DataFrame(ref_rows).style.format({"現在の係数": "{:.6f}"}), use_container_width=True)

def render_prior_coeff_input():
    """📐 係数入力（事前係数）タブのUI表示とロジック"""
    st.title("📐 事前係数入力")
    st.info("運送業・医療など、業種ごとの基本事前係数を後から入力・編集できます。保存すると審査スコアに反映されます。")
    overrides = load_coeff_overrides() or {}
    selected_key = st.selectbox(
        "編集するモデルを選択",
        options=PRIOR_COEFF_MODEL_KEYS,
        format_func=lambda k: k + (" （オーバーライド済み）" if k in overrides else " （初期値）"),
        key="prior_coeff_model_select",
    )
    if selected_key:
        current = get_effective_coeffs(selected_key)
        keys_sorted = ["intercept"] + [k for k in sorted(current.keys()) if k != "intercept"]
        edited = {}
        st.subheader(f"係数: {selected_key}")
        n_cols = 3
        for i in range(0, len(keys_sorted), n_cols):
            cols = st.columns(n_cols)
            for j, k in enumerate(keys_sorted[i:i + n_cols]):
                with cols[j]:
                    val = current.get(k, 0)
                    if isinstance(val, (int, float)):
                        new_val = st.number_input(
                            k,
                            value=float(val),
                            step=0.0001,
                            format="%.6f",
                            key=f"prior_{selected_key}_{k}",
                        )
                        edited[k] = new_val
        if edited and st.button("💾 このモデルの係数を保存", key="btn_save_prior_coeffs"):
            overrides = load_coeff_overrides() or {}
            overrides[selected_key] = edited
            if save_coeff_overrides(overrides):
                st.success(f"{selected_key} の係数を保存しました。")
            else:
                st.error("保存に失敗しました。")
        st.caption("※ 運送業・医療は個別に事前係数を入力できます。指標モデル（全体_指標など）を編集すると、既存先・新規先の両方の基準に反映されます。")
