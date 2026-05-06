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
    except (json.JSONDecodeError, OSError): return
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
from data_cases import get_effective_coeffs, load_coeff_overrides, save_coeff_overrides
from analysis_regression import (
    COEFF_MAIN_KEYS,
    COEFF_EXTRA_KEYS,
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

    # ── 再学習ボタン（主モデル + 係数更新） ─────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 🚀 ロジスティック回帰 + 主モデル 再学習")
        st.caption(
            "前回保存済みの係数を事前分布（warm-start 初期値）として使い、"
            "ロジスティック回帰（全モデルキー）と、既存先/新規先で分けた主モデルを同時に再学習・保存します。"
        )
        if st.button(
            "🚀 主モデル再学習（既存/新規分割）",
            key="btn_unified_retrain",
            type="primary",
        ):
            from analysis_regression import (
                run_bayesian_warm_start_all_keys,
                train_lgbm_from_cases,
                backup_coeff_overrides,
                backup_lgbm_model,
            )
            from data_cases import save_coeff_overrides
            from scoring_core import clear_scoring_cache

            _all_logs = load_all_cases()
            _n_labeled = sum(1 for c in _all_logs if c.get("final_status") in ["成約", "失注"])
            if _n_labeled < 5:
                st.warning(f"成約/失注データが不足しています（現在 {_n_labeled} 件、最低 5 件必要）。")
            else:
                _backup_msgs = []

                # バックアップ: LR 係数
                try:
                    _coeff_bak = backup_coeff_overrides()
                    if _coeff_bak:
                        _backup_msgs.append(f"LR 係数: `{os.path.basename(_coeff_bak)}`")
                except Exception as _be:
                    _backup_msgs.append(f"LR 係数バックアップ失敗: {_be}")

                # バックアップ: 主モデル pkl
                try:
                    _lgbm_bak = backup_lgbm_model()
                    if _lgbm_bak:
                        _backup_msgs.append(f"主モデル: `{os.path.basename(_lgbm_bak)}`")
                    else:
                        _backup_msgs.append("主モデル: 既存モデルなし（初回学習）")
                except Exception as _be:
                    _backup_msgs.append(f"主モデルバックアップ失敗: {_be}")

                # Step 1: ロジスティック回帰（ベイズ warm-start）
                _lr_prog = st.progress(0, text="ロジスティック回帰 学習中...")
                try:
                    _overrides, _lr_results = run_bayesian_warm_start_all_keys(_all_logs)
                    _lr_prog.progress(50, text="ロジスティック回帰 完了 → 係数保存中...")
                    if save_coeff_overrides(_overrides, comment="統合再学習(LR warm-start)"):
                        _lr_prog.progress(100, text="ロジスティック回帰 保存完了 ✅")
                        _lr_ok = True
                    else:
                        _lr_prog.progress(100, text="⚠️ 係数保存に失敗しました")
                        _lr_ok = False
                except Exception as _e:
                    _lr_prog.progress(100, text=f"❌ LR エラー: {_e}")
                    _lr_results = [f"エラー: {_e}"]
                    _lr_ok = False

                # Step 2: 主モデル
                _lgbm_prog = st.progress(0, text="主モデル 学習中...")
                try:
                    _acc, _auc, _path, _n_pos, _n_neg = train_lgbm_from_cases(_all_logs)
                    _auc_str = f"  AUC: {_auc:.3f}" if _auc else ""
                    _lgbm_prog.progress(
                        100,
                        text=f"主モデル 完了 ✅  Accuracy: {_acc:.1%}{_auc_str}  "
                             f"(成約{_n_pos}件 / 失注{_n_neg}件)",
                    )
                    _lgbm_ok = True
                except Exception as _e:
                    _lgbm_prog.progress(100, text=f"❌ 主モデル エラー: {_e}")
                    _lgbm_ok = False

                # キャッシュクリア（新モデルを即時反映）
                clear_scoring_cache()

                # 結果サマリー
                if _lr_ok and _lgbm_ok:
                    st.success("✅ 主モデル（既存先/新規先分割）の再学習が完了しました。")
                else:
                    st.warning("一部の学習でエラーが発生しました。詳細を確認してください。")

                with st.expander("バックアップ / LR 各モデルキーの結果", expanded=False):
                    if _backup_msgs:
                        st.caption("**保存済みバックアップ (data/backups/)**")
                        for _m in _backup_msgs:
                            st.caption(f"  • {_m}")
                        st.divider()
                    for _r in _lr_results:
                        st.caption(_r)

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

    # ── モデル見直しフック ─────────────────────────────────────────────────
    try:
        from model_review_hooks import render_model_review_hook_panel
        render_model_review_hook_panel()
    except Exception as e:
        st.caption(f"⚠️ モデル見直しフック読み込みエラー: {e}")

    st.divider()

    st.subheader("参考: 現在の審査で使っている係数（全体_既存先）")
    current = get_effective_coeffs("全体_既存先")
    overrides = load_coeff_overrides()
    if overrides and "全体_既存先" in overrides:
        st.caption("※ 成約/失注で更新した係数（既存＋追加項目）が適用されています。")
    ref_rows = [{"変数": k, "現在の係数": current.get(k, 0)} for k in ["intercept"] + COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS]
    st.dataframe(pd.DataFrame(ref_rows).style.format({"現在の係数": "{:.6f}"}), width='stretch')

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
                def _safe_val(val):
                    if val is None: return "—"
                    if isinstance(val, (int, float)): return f"{val:.6f}"
                    return str(val)
                rows = [
                    {"変数": str(k), "変更前": _safe_val(v.get("before")), "変更後": _safe_val(v.get("after"))}
                    for k, v in changed_keys.items()
                ]
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), width='stretch')
            else:
                st.caption("変更キーの記録なし")
            if snapshot:
                with st.expander("全係数スナップショット（保存後）", expanded=False):
                    snap_rows = [{"変数": str(k), "値": _safe_val(v)} for k, v in snapshot.items()]
                    st.dataframe(pd.DataFrame(snap_rows), width='stretch')


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
