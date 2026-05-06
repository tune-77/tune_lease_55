import streamlit as st
import pandas as pd
from data_cases import load_all_cases
from analysis_regression import run_qualitative_contract_analysis
from analysis_regression_gemini import run_qualitative_contract_analysis_gemini
from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS
from tunnel_optimizer import load_tunnel_model
from soul_factor_miner import mine_soul_factors, mine_reverse_bayes_bonus, mine_fp0_patch_candidates

QUALITATIVE_ANALYSIS_MIN_CASES = 50

def render_qualitative_analysis():
    """📉 定性要因分析タブのUIとロジックを描画する"""
    st.title("📉 定性要因で成約予測")
    st.caption("取引区分・競合状況・顧客区分・商談ソース・リース物件・定性スコアリング6項目（設立・経営年数、顧客安定性、返済履歴、事業将来性、設置目的、メイン取引銀行）のみを使って、ロジスティック回帰とLightGBMで成約/不成約を分析します。")

    # ── エンジン選択 ──────────────────────────────────────────────────────────
    engine = st.radio(
        "分析エンジン",
        ["🤖 Gemini（推奨）", "💻 ローカル (LR/LGB)"],
        horizontal=True,
        key="qual_engine",
    )
    use_gemini = engine.startswith("🤖")

    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    n_reg = len(registered)

    min_cases = 3 if use_gemini else QUALITATIVE_ANALYSIS_MIN_CASES
    if n_reg < min_cases:
        st.warning(f"成約・失注の登録が **{min_cases}件** 以上で利用できます。（現在: **{n_reg}件**）")
    else:
        st.success(f"登録件数: **{n_reg}件**（成約+失注）。分析を実行できます。")
        if st.button("🚀 ロジスティック回帰とLightGBMを実行", key="run_qual_analysis"):
            with st.spinner("分析中..." if not use_gemini else "Gemini に分析を依頼中…"):
                if use_gemini:
                    result = run_qualitative_contract_analysis_gemini(QUALITATIVE_SCORING_CORRECTION_ITEMS)
                else:
                    result = run_qualitative_contract_analysis(QUALITATIVE_SCORING_CORRECTION_ITEMS)
            if result is None:
                st.error("分析できませんでした。")
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
                st.caption("アンサンブルは採用していません。LR と LightGBM を個別に比較します。")
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
        if result is None and n_reg >= min_cases:
            st.info("上の「ロジスティック回帰とLightGBMを実行」ボタンで分析を開始してください。")

    st.divider()
    st.subheader("🧪 代理定性因子（トンネル最適化）")
    proxy_model = load_tunnel_model()
    if proxy_model is None:
        st.caption("成約/失注データが 1,500 件未満、または総当たり検証で既存モデルを上回らないため、採用していません。")
        st.caption("欠損は 0 埋めせず、30項目の障壁として扱う設計です。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("学習件数", f"{proxy_model.n_cases}件")
        with c2:
            st.metric("成功件数", f"{proxy_model.n_success}件")
        with c3:
            st.metric("AUC", f"{proxy_model.metrics.get('auc', 0):.3f}")
        with c4:
            st.metric("F2", f"{proxy_model.metrics.get('f2', 0):.3f}")
        st.caption(f"学習時刻: {proxy_model.trained_at} / 選択特徴量: {', '.join(proxy_model.selected_features[:6])}")

    st.divider()
    st.subheader("🔎 直近DATAの soul 因子総当たり")
    st.caption("定性スコアは使わず、昨日入力した直近DATAの項目を主軸に、参考変数として Q_risk とマハラノビス距離も加えて 2〜3 項目の組み合わせを総当たりします。")

    if st.button("🔍 総当たりを実行", key="run_soul_factor_mining"):
        with st.spinner("総当たり探索中..."):
            result = mine_soul_factors(force_recompute=True)
        st.session_state["soul_factor_mining_result"] = result

    result = st.session_state.get("soul_factor_mining_result")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("対象件数", f"{result['n_cases']}件")
        with c2:
            st.metric("成約", f"{result['n_success']}件")
        with c3:
            st.metric("失注", f"{result['n_loss']}件")
        with c4:
            st.metric("候補数", f"{result['n_atoms']}個")

        st.caption(
            f"探索空間: {result['search_space']['pairs']} 組合せ + {result['search_space']['triplets']} 組合せ"
        )
        derived_vars = result.get("derived_variables") or []
        if derived_vars:
            st.caption("派生変数: " + " / ".join(derived_vars[:10]))
        derived_triplets = result.get("derived_triplet_variables") or []
        if derived_triplets:
            st.caption("3項目派生: " + " / ".join(derived_triplets[:8]))

        single_patterns = result.get("single_patterns") or []
        if single_patterns:
            st.subheader("単項の差分が大きい項目")
            single_df = pd.DataFrame(single_patterns[:20])
            single_df = single_df[[
                "pattern",
                "success_coverage",
                "loss_coverage",
                "support_diff",
                "success_count",
                "loss_count",
            ]]
            single_df["success_coverage"] = (single_df["success_coverage"] * 100).round(1)
            single_df["loss_coverage"] = (single_df["loss_coverage"] * 100).round(1)
            single_df["support_diff"] = (single_df["support_diff"] * 100).round(1)
            single_df = single_df.rename(columns={
                "pattern": "項目",
                "success_coverage": "成約出現率(%)",
                "loss_coverage": "失注出現率(%)",
                "support_diff": "差分(%)",
                "success_count": "成約件数",
                "loss_count": "失注件数",
            })
            st.dataframe(single_df, width="stretch", hide_index=True)

        fn_patterns = result.get("fn_only_patterns") or []
        success_patterns = result.get("success_overall_patterns") or []
        if fn_patterns:
            st.subheader("FNにだけ多い項目")
            fn_df = pd.DataFrame(fn_patterns[:20])
            fn_df = fn_df[[
                "pattern",
                "fn_rate",
                "success_rate",
                "diff",
                "fn_count",
                "success_count",
            ]]
            fn_df["fn_rate"] = (fn_df["fn_rate"] * 100).round(1)
            fn_df["success_rate"] = (fn_df["success_rate"] * 100).round(1)
            fn_df["diff"] = (fn_df["diff"] * 100).round(1)
            fn_df = fn_df.rename(columns={
                "pattern": "項目",
                "fn_rate": "FN出現率(%)",
                "success_rate": "成約出現率(%)",
                "diff": "差分(%)",
                "fn_count": "FN件数",
                "success_count": "成約件数",
            })
            st.dataframe(fn_df, width="stretch", hide_index=True)

        if success_patterns:
            st.subheader("成約全体に多い項目")
            sc_df = pd.DataFrame(success_patterns[:20])
            sc_df = sc_df[[
                "pattern",
                "success_rate",
                "fn_rate",
                "diff",
                "success_count",
                "fn_count",
            ]]
            sc_df["success_rate"] = (sc_df["success_rate"] * 100).round(1)
            sc_df["fn_rate"] = (sc_df["fn_rate"] * 100).round(1)
            sc_df["diff"] = (sc_df["diff"] * 100).round(1)
            sc_df = sc_df.rename(columns={
                "pattern": "項目",
                "success_rate": "成約出現率(%)",
                "fn_rate": "FN出現率(%)",
                "diff": "差分(%)",
                "success_count": "成約件数",
                "fn_count": "FN件数",
            })
            st.dataframe(sc_df, width="stretch", hide_index=True)

        exact = result.get("exact_patterns", [])
        if exact:
            st.success("80%以上の成約に現れ、失注には出ない組み合わせが見つかりました。")
            exact_df = pd.DataFrame(exact)
            exact_df = exact_df[[
                "pattern",
                "success_count",
                "success_coverage",
                "loss_count",
                "support",
                "precision",
                "lift",
            ]]
            exact_df["success_coverage"] = (exact_df["success_coverage"] * 100).round(1)
            exact_df["precision"] = (exact_df["precision"] * 100).round(1)
            exact_df["lift"] = exact_df["lift"].round(2)
            exact_df = exact_df.rename(columns={
                "pattern": "組み合わせ",
                "success_count": "成約件数",
                "success_coverage": "成約内割合(%)",
                "loss_count": "失注件数",
                "support": "総支持件数",
                "precision": "純度(%)",
                "lift": "Lift",
            })
            st.dataframe(exact_df, width="stretch", hide_index=True)
        else:
            st.warning("80%以上の成約に現れ、かつ失注には出ない 2〜3 項目の組み合わせは見つかりませんでした。")

        best_df = pd.DataFrame(result.get("best_patterns", [])[:20])
        if not best_df.empty:
            st.subheader("近い候補")
            best_df = best_df[[
                "pattern",
                "success_count",
                "success_coverage",
                "loss_count",
                "support",
                "precision",
                "lift",
            ]]
            best_df["success_coverage"] = (best_df["success_coverage"] * 100).round(1)
            best_df["precision"] = (best_df["precision"] * 100).round(1)
            best_df["lift"] = best_df["lift"].round(2)
            best_df = best_df.rename(columns={
                "pattern": "組み合わせ",
                "success_count": "成約件数",
                "success_coverage": "成約内割合(%)",
                "loss_count": "失注件数",
                "support": "総支持件数",
                "precision": "純度(%)",
                "lift": "Lift",
            })
            st.dataframe(best_df, width="stretch", hide_index=True)
            st.caption("ここに出るのは近い候補です。条件は強いが、失注にも出るものは除外していません。")

        st.divider()
        st.subheader("🧩 魂のパッチ探索（FP=0）")
        st.caption("失注への誤爆を 0 件に固定したまま、成約を 1 件でも救える gate を総当たりで探します。")
        fp0_last_scan_count = st.session_state.get("fp0_patch_last_scan_count")
        fp0_auto_triggered = False
        if fp0_last_scan_count is not None and (n_reg - fp0_last_scan_count) >= 100:
            with st.spinner("100件追加を検知したため、FP=0 パッチを自動再探索中..."):
                patch_result = mine_fp0_patch_candidates(force_recompute=True)
            st.session_state["fp0_patch_result"] = patch_result
            st.session_state["fp0_patch_last_scan_count"] = n_reg
            fp0_auto_triggered = True
        elif fp0_last_scan_count is None and "fp0_patch_result" in st.session_state:
            st.session_state["fp0_patch_last_scan_count"] = n_reg
        if st.button("🧵 FP=0 パッチを再計算", key="run_fp0_patch_candidates"):
            with st.spinner("FP=0 パッチを探索中..."):
                patch_result = mine_fp0_patch_candidates(force_recompute=True)
            st.session_state["fp0_patch_result"] = patch_result
            st.session_state["fp0_patch_last_scan_count"] = n_reg

        patch_result = st.session_state.get("fp0_patch_result")
        if patch_result:
            pc1, pc2 = st.columns(2)
            with pc1:
                st.metric("対象件数", f"{patch_result['n_cases']}件")
            with pc2:
                st.caption("条件: loss_count = 0 / success_count >= 1")
            if fp0_auto_triggered:
                st.info("成約/失注の登録件数が 100 件増えたため、FP=0 パッチを自動更新しました。")

            adoption_cands = patch_result.get("adoption_candidates") or []
            if adoption_cands:
                st.subheader("採用候補")
                st.caption("全体で効き、失注への誤爆が 0 件で、かつ十分な成約件数を拾う gate です。")
                adf = pd.DataFrame(adoption_cands)[:20]
                adf = adf[[
                    "pattern",
                    "success_count",
                    "loss_count",
                    "success_coverage",
                    "precision",
                    "support_diff",
                    "title",
                    "explanation",
                    "action",
                    "description",
                ]]
                adf["success_coverage"] = (adf["success_coverage"] * 100).round(1)
                adf["precision"] = (adf["precision"] * 100).round(1)
                adf["support_diff"] = (adf["support_diff"] * 100).round(1)
                adf = adf.rename(columns={
                    "pattern": "組み合わせ",
                    "success_count": "成約件数",
                    "loss_count": "失注件数",
                    "success_coverage": "成約内割合(%)",
                    "precision": "純度(%)",
                    "support_diff": "差分(%)",
                    "title": "タイトル",
                    "explanation": "解説",
                    "action": "アクション",
                    "description": "説明",
                })
                st.dataframe(adf, width="stretch", hide_index=True)

            aux_rules = patch_result.get("auxiliary_rules") or []
            if aux_rules:
                st.subheader("説明つき補助ルール")
                st.caption("全体採用まではしないが、局所的に誤判定を抑えるための gate です。")
                a_df = pd.DataFrame(aux_rules)[:20]
                a_df = a_df[[
                    "pattern",
                    "success_count",
                    "loss_count",
                    "success_coverage",
                    "precision",
                    "support_diff",
                    "title",
                    "explanation",
                    "action",
                    "description",
                ]]
                a_df["success_coverage"] = (a_df["success_coverage"] * 100).round(1)
                a_df["precision"] = (a_df["precision"] * 100).round(1)
                a_df["support_diff"] = (a_df["support_diff"] * 100).round(1)
                a_df = a_df.rename(columns={
                    "pattern": "組み合わせ",
                    "success_count": "成約件数",
                    "loss_count": "失注件数",
                    "success_coverage": "成約内割合(%)",
                    "precision": "純度(%)",
                    "support_diff": "差分(%)",
                    "title": "タイトル",
                    "explanation": "解説",
                    "action": "アクション",
                    "description": "説明",
                })
                st.dataframe(a_df, width="stretch", hide_index=True)

            for scope_kind, title in (("dept_results", "FN率の高い営業部での局所候補"), ("industry_results", "FN率の高い業種での局所候補")):
                scope_results = patch_result.get(scope_kind) or []
                if not scope_results:
                    continue
                st.subheader(title)
                for scope_item in scope_results:
                    scope_name = scope_item.get("scope", "未設定")
                    scope_cands = scope_item.get("candidates") or []
                    if not scope_cands:
                        continue
                    with st.expander(f"{scope_name} / 対象 {scope_item.get('n_cases', 0)}件", expanded=False):
                        sdf = pd.DataFrame(scope_cands)[:10]
                        sdf = sdf[[
                            "pattern",
                            "success_count",
                            "loss_count",
                            "success_coverage",
                            "precision",
                            "support_diff",
                            "title",
                            "explanation",
                            "action",
                            "description",
                        ]]
                        sdf["success_coverage"] = (sdf["success_coverage"] * 100).round(1)
                        sdf["precision"] = (sdf["precision"] * 100).round(1)
                        sdf["support_diff"] = (sdf["support_diff"] * 100).round(1)
                        sdf = sdf.rename(columns={
                            "pattern": "組み合わせ",
                            "success_count": "成約件数",
                            "loss_count": "失注件数",
                            "success_coverage": "成約内割合(%)",
                            "precision": "純度(%)",
                            "support_diff": "差分(%)",
                            "title": "タイトル",
                            "explanation": "解説",
                            "action": "アクション",
                            "description": "説明",
                        })
                        st.dataframe(sdf, width="stretch", hide_index=True)

        st.divider()
        st.subheader("🎯 逆転のベイズ加点候補")
        st.caption("deal_source=その他 かつ customer_type=新規先 の案件だけに発動する、FN偏りの強い組み合わせです。")
        if st.button("🔮 逆転ベイズを再計算", key="run_reverse_bayes_bonus"):
            with st.spinner("逆転候補を探索中..."):
                reverse_result = mine_reverse_bayes_bonus(force_recompute=True)
            st.session_state["reverse_bayes_bonus_result"] = reverse_result

        reverse_result = st.session_state.get("reverse_bayes_bonus_result")
        if reverse_result:
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("対象件数", f"{reverse_result['n_cases']}件")
            with rc2:
                st.metric("FN件数", f"{reverse_result['n_fn']}件")
            with rc3:
                st.metric("他件数", f"{reverse_result['n_other']}件")
            with rc4:
                st.metric("発動閾値", f"{reverse_result['activation_threshold']:.2f}")

            if reverse_result.get("activation_description"):
                st.info(reverse_result["activation_description"])

            reverse_df = pd.DataFrame(reverse_result.get("rules") or [])
            if not reverse_df.empty:
                reverse_df = reverse_df[[
                    "pattern",
                    "fn_count",
                    "other_count",
                    "fn_rate",
                    "other_rate",
                    "posterior",
                    "bonus_points",
                    "activation_conditions_text",
                    "activation_description",
                ]]
                reverse_df["fn_rate"] = (reverse_df["fn_rate"] * 100).round(1)
                reverse_df["other_rate"] = (reverse_df["other_rate"] * 100).round(1)
                reverse_df["posterior"] = (reverse_df["posterior"] * 100).round(1)
                reverse_df = reverse_df.rename(columns={
                    "pattern": "ルール",
                    "fn_count": "FN件数",
                    "other_count": "他件数",
                    "fn_rate": "FN出現率(%)",
                    "other_rate": "他出現率(%)",
                    "posterior": "FN後方確率(%)",
                    "bonus_points": "加点",
                    "activation_conditions_text": "発動条件",
                    "activation_description": "説明文",
                })
                st.dataframe(reverse_df, width="stretch", hide_index=True)
