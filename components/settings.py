import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

def render_math_proposals():
    """Dr.Algo Optimization Proposals UI"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prop_path = os.path.join(base_dir, "data", "math_proposals.json")
    if not os.path.exists(prop_path): return
    try:
        with open(prop_path, "r", encoding="utf-8") as f:
            props = json.load(f)
    except: return
    pending = [p for p in props if p.get("status") == "pending"]
    if not pending: return
    st.markdown("### 🔬 Dr.Algoからの最適化提案")
    for p in pending:
        with st.expander(f"💡 {p['method_name']} ({p['ts'][:10]})", expanded=True):
            st.write(f"**根拠:** {p['reason']}")
            st.json(p["changes"])
            c1, c2 = st.columns(2)
            if c1.button("✅ 承認", key=f"app_{p['method_name']}", type="primary"):
                from data_cases import load_coeff_overrides, save_coeff_overrides
                ovr = load_coeff_overrides()
                ovr.update(p["changes"])
                save_coeff_overrides(ovr, comment=f"Dr.Algo({p['method_name']})承認")
                p["status"] = "approved"
                with open(prop_path, "w", encoding="utf-8") as f: json.dump(props, f, ensure_ascii=False, indent=2)
                st.success("反映完了！")
                st.rerun()
            if c2.button("❌ 却下", key=f"rej_{p['method_name']}"):
                p["status"] = "rejected"
                with open(prop_path, "w", encoding="utf-8") as f: json.dump(props, f, ensure_ascii=False, indent=2)
                st.rerun()
    st.divider()

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

    # ── AI からの最適化提案 (Phase 3) ─────────────────────────────────────────
    render_math_proposals()

    # ── 自動学習ステータスパネル ────────────────────────────────────────────
    try:
        from auto_optimizer import get_training_status, run_auto_optimization, MIN_START, RETRAIN_INTERVAL
        _s = get_training_status()
        with st.container(border=True):
            st.markdown("#### 🧠 係数自動学習ステータス")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("登録件数", f"{_s['count']}件")
            c2.metric("前回学習", f"{_s['last_trained_count']}件" if _s['last_trained_count'] else "未実施")
            c3.metric("累計学習回数", f"{_s['total_runs']}回")
            c4.metric("前回AUC", f"{_s['last_auc']:.3f}" if _s['last_auc'] else "—")

            if _s["phase"] == "waiting":
                st.progress(_s["count"] / MIN_START,
                            text=f"初回学習まであと {_s['next_trigger']}件（目標: {MIN_START}件）")
            elif _s["phase"] == "active":
                gap = _s["count"] - _s["last_trained_count"]
                st.progress(gap / RETRAIN_INTERVAL,
                            text=f"次回更新まであと {_s['next_trigger']}件（+{gap}/{RETRAIN_INTERVAL}件）")
            else:
                st.success(f"✅ 学習トリガー条件を満たしています（{_s['count']}件）")

            if _s["last_trained_at"]:
                st.caption(f"前回学習日時: {_s['last_trained_at']}")

            _force_btn = st.button(
                "🚀 今すぐ自動最適化を実行",
                key="btn_auto_optimize",
                disabled=(_s["count"] < MIN_START),
                help=f"成約/失注が{MIN_START}件以上の場合に実行できます。",
                type="primary" if _s["should_retrain"] else "secondary",
            )
            if _force_btn:
                with st.spinner("係数を最適化中..."):
                    _result = run_auto_optimization(force=True)
                if _result:
                    _auc = _result.get("auc_borrower_asset")
                    st.success(
                        f"✅ 最適化完了（{_result['n_cases']}件）　"
                        f"借手重み: {_result['recommended_borrower_pct']:.1%} / "
                        f"物件重み: {_result['recommended_asset_pct']:.1%}"
                        + (f"　AUC: {_auc:.3f}" if _auc else "")
                    )
                    st.rerun()
                else:
                    st.warning("最適化できませんでした。成約/失注データが不足している可能性があります。")
    except Exception as _ae:
        st.caption(f"⚠️ 自動学習ステータス取得エラー: {_ae}")

    st.divider()

    # --- 新規追加: LLMによる定性的PDCAリフレクション ---
    st.markdown("#### 📝 月次AIリフレクション (定性PDCA)")
    st.caption("直近の審査結果（成約・失注等）をAIに読み込ませ、現在の審査傾向を分析し、翌日からの審査アシスタント（軍師AI等）のプロンプトに注意事項として自動追加・フィードバックします。データが少なくても安全に審査目線を補正できます。")
    try:
        from llm_pdca_reflection import load_pdca_rules, run_monthly_pdca_reflection
        
        rules = load_pdca_rules()
        if rules:
            st.success(f"✅ 前回の分析日時: {rules.get('last_run', '—')} (分析件数: {rules.get('analyzed_count', 0)}件)")
            with st.expander("現在のAI審査 反映ルール", expanded=True):
                st.write("**【直近の傾向分析】**")
                st.write(rules.get("reflection_summary", ""))
                st.write("**【AIプロンプトへの追加指示】**")
                for r in rules.get("ai_prompt_addons", []):
                    st.markdown(f"- {r}")
        else:
            st.info("まだAIリフレクションは実行されていません。")

        c1, c2 = st.columns([1, 2])
        _force_pdca = c1.button("🧠 今すぐAIリフレクションを実行", type="primary", key="btn_run_pdca")
        _pdca_num_cases = c2.number_input("分析対象の直近案件数", min_value=5, max_value=50, value=20, step=5, key="pdca_num_cases")
        
        if _force_pdca:
            with st.spinner("過去の案件を読み込み、AIが定性的な傾向を分析中です...（最大1分ほどかかります）"):
                res = run_monthly_pdca_reflection(force=True, max_cases=_pdca_num_cases)
            if res and res.get("status") == "success":
                st.success("✅ 分析が完了し、新しい審査ルールがAIへ設定されました！")
                st.rerun()
            elif res and res.get("status") == "skipped":
                st.warning("分析対象となる成約/失注データが少なすぎます。（最低5件）")
            else:
                err_msg = res.get("reason", "不明なエラー") if res else "不明なエラー"
                st.error(f"分析に失敗しました。詳細: {err_msg}")
    except Exception as e:
        st.error(f"AIリフレクション機能エラー: {e}")

    st.divider()
    
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


def render_coeff_history():
    """係数変更履歴を表示する。"""
    from data_cases import load_coeff_history
    st.subheader("📋 係数変更履歴")
    records = load_coeff_history()  # 新しい順で返る
    if not records:
        st.info("変更履歴がありません。係数を更新すると自動で記録されます。")
        return
    st.caption(f"全 {len(records)} 件（新しい順）")
    for i, rec in enumerate(records):
        ts = rec.get("timestamp", "—")
        change_type = rec.get("change_type", "—")
        comment = rec.get("comment", "")
        changed_keys = rec.get("changed_keys") or {}
        snapshot = rec.get("snapshot_after") or {}

        type_label = {"manual": "手動", "auto": "自動"}.get(change_type, change_type)
        header = f"{ts}　[{type_label}]　{comment or '（コメントなし）'}"
        with st.expander(header, expanded=(i == 0)):
            if changed_keys:
                rows = [
                    {"変数": k, "変更前": v.get("before"), "変更後": v.get("after")}
                    for k, v in changed_keys.items()
                ]
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(rows).style.format(
                        {"変更前": lambda x: f"{x:.6f}" if isinstance(x, float) else ("—" if x is None else x),
                         "変更後": lambda x: f"{x:.6f}" if isinstance(x, float) else ("—" if x is None else x)}
                    ),
                    use_container_width=True,
                )
            else:
                st.caption("変更キーの記録なし")
            if snapshot:
                with st.expander("全係数スナップショット（保存後）", expanded=False):
                    snap_rows = [{"変数": k, "値": v} for k, v in snapshot.items()]
                    st.dataframe(
                        pd.DataFrame(snap_rows).style.format(
                            {"値": lambda x: f"{x:.6f}" if isinstance(x, float) else x}
                        ),
                        use_container_width=True,
                    )


def render_app_logs():
    """アプリログを表示する。"""
    import os
    st.subheader("🪵 アプリログ")
    log_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit.log"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "app.log"),
    ]
    log_path = next((p for p in log_candidates if os.path.exists(p)), None)
    if not log_path:
        st.info("ログファイルが見つかりません。")
        return
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        n = st.slider("表示行数（末尾から）", min_value=50, max_value=1000, value=200, step=50)
        tail = lines[-n:]
        st.caption(f"{log_path}  （全 {len(lines)} 行 / 末尾 {n} 行表示）")
        st.code("".join(tail), language="text")
    except Exception as e:
        st.error(f"ログ読み込みエラー: {e}")
