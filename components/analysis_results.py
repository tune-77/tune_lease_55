import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time
import datetime

from data_cases import load_all_cases, load_case_news, update_case_field
from ai_chat import (
    is_ai_available, get_ollama_model, chat_with_retry, 
    get_ai_3d_comment, get_ai_industry_advice,
    get_ai_byoki_with_industry, generate_battle_special_move,
    save_debate_log
)
from knowledge import build_knowledge_context
from data_cases import append_consultation_memory
from indicators import compute_financial_indicators, analyze_indicators_vs_bench
from web_services import (
    fetch_industry_benchmarks_from_web, fetch_industry_trend_extended,
    get_trend_extended, get_advice_context_extras, search_subsidies_by_industry,
    get_stats, _WEB_BENCH_KEYS
)
from charts import LOWER_IS_BETTER_NAMES
from ai_chat import _get_gemini_key_from_secrets
from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS
from config import GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT

from charts import (
    plot_3d_analysis, plot_3d_profit_position, plot_3d_repayment, plot_3d_safety_score,
    plot_radar_chart_plotly, plot_gauge_plotly, plot_waterfall_plotly, plot_financial_bullet_plotly,
    plot_score_boxplot_plotly, plot_scoring_top5_factors_plotly, plot_contract_prob_factors_plotly,
    plot_indicators_gap_analysis_plotly, plot_past_scores_histogram_plotly, plot_score_models_comparison_plotly,
    plot_balance_sheet_plotly, plot_cash_flow_bridge_plotly, plot_ebitda_coverage_plotly, plot_break_even_point_plotly
)
from ai_chat import get_ai_quick_comment, get_ai_comprehensive_evaluation
from constants import get_review_alert, QUALITATIVE_SCORING_LEVEL_LABELS, get_dashboard_image_path
from screening_report import build_screening_report_pdf
from montecarlo import (
    map_industry_from_major, INDUSTRY_VOLATILITY, AdvancedMonteCarloEngine, CompanyData, generate_pdf_bytes,
    make_company_chart, make_portfolio_chart
)
from future_simulation import render_future_simulation_ui
from bayesian_engine import THRESHOLD_APPROVAL
from credit_limit import render_credit_limit_ui
from components.form_apply import render_quick_edit_panel

def render_analysis_results(
    nav_mode,
    res,
    jsic_data,
    avg_data,
    knowhow_data,
    benchmarks_data,
    bankruptcy_data,
    trend_info,
    past_cases_log,
    current_case_data,
    current_case_id
):
    if nav_mode == "📊 分析結果":
        # ── クイック再入力パネル（全項目） ───────────────────────────
        with st.expander("✏️ 全項目編集して再判定", expanded=False):
            from components.form_apply import render_quick_edit_panel
            import os, json
            from constants import BASE_DIR
            _assets_path = os.path.join(BASE_DIR, "lease_assets.json")
            _assets_list = []
            if os.path.isfile(_assets_path):
                try:
                    with open(_assets_path, "r", encoding="utf-8") as f:
                        _assets_list = json.load(f).get("items", [])
                except Exception:
                    pass
            quick_res = render_quick_edit_panel(jsic_data, _assets_list)
            
            if quick_res["rejudge_clicked"]:
                # 業種
                st.session_state["select_major"] = quick_res["q_major"]
                st.session_state["select_sub"] = quick_res["q_sub"]
                # P/L
                st.session_state["nenshu"] = quick_res["q_nenshu"]
                st.session_state["item9_gross"] = quick_res["q_gross"]
                st.session_state["rieki"] = quick_res["q_rieki"]
                st.session_state["item4_ord_profit"] = quick_res["q_ord"]
                st.session_state["item5_net_income"] = quick_res["q_net_income"]
                # 資産・経費
                st.session_state["item10_dep"] = quick_res["q_dep"]
                st.session_state["item11_dep_exp"] = quick_res["q_dep_exp"]
                st.session_state["item8_rent"] = quick_res["q_rent"]
                st.session_state["item12_rent_exp"] = quick_res["q_rent_exp"]
                st.session_state["item6_machine"] = quick_res["q_machine"]
                st.session_state["item7_other"] = quick_res["q_other"]
                st.session_state["net_assets"] = quick_res["q_net"]
                st.session_state["total_assets"] = quick_res["q_total"]
                # 信用情報
                st.session_state["grade"] = quick_res["q_grade"]
                st.session_state["bank_credit"] = quick_res["q_bank"]
                st.session_state["lease_credit"] = quick_res["q_lease"]
                st.session_state["contracts"] = quick_res["q_contracts"]
                # 契約条件
                st.session_state["customer_type"] = quick_res["q_ctype"]
                st.session_state["contract_type"] = quick_res["q_contract_type"]
                st.session_state["deal_source"] = quick_res["q_deal_source"]
                st.session_state["lease_term"] = quick_res["q_lease_term"]
                st.session_state["acceptance_year"] = quick_res["q_acceptance_year"]
                st.session_state["acquisition_cost"] = quick_res["q_acq"]
                if quick_res["q_asset_sel"] is not None:
                    st.session_state["selected_asset_index"] = quick_res["q_asset_sel"]
                # 車両タイプをメインフォームへ同期（_quick_asset_detail → asset_vtype_select）
                _q_vtype = quick_res.get("q_asset_detail", "")
                _VT_OPTS = [
                    "", "ハイエース バン（商用バン）", "キャラバン（商用バン）",
                    "軽商用バン（エブリイ / ハイゼット等）", "営業用トラック", "自家用トラック",
                    "一般乗用車（ヤリス / カローラ等）", "レクサス・外車等（役員車）",
                ]
                st.session_state["asset_vtype_select"] = _q_vtype if _q_vtype in _VT_OPTS else ""
                # 定性スコアリング
                for _qid, _qval in quick_res["q_qual"].items():
                    st.session_state[f"qual_corr_{_qid}"] = _qval
                # チャット履歴をリセット（新しい判定なので前の会話を引き継がない）
                st.session_state["messages"] = []
                st.session_state["debate_history"] = []
                # 判定トリガー
                st.session_state["_auto_judge"] = True
                st.session_state["_nav_pending"] = "📝 審査入力"
                st.rerun()
        # ──────────────────────────────────────────────────────────────

        # --- GLOBAL VARIABLE RECOVERY (Must be first) ---
        selected_major = "D 建設業" # Default
        selected_sub = "06 総合工事業" # Default
        score_percent = 0
        user_equity_ratio = 0
        user_op_margin = 0
        if "last_result" in st.session_state:
            res_g = st.session_state["last_result"]
            selected_major = res_g.get("industry_major", "D 建設業")
            selected_sub = res_g.get("industry_sub", "06 総合工事業")
            score_percent = res_g.get("score", 0)
            user_equity_ratio = res_g.get("user_eq", 0)
            user_op_margin = res_g.get("user_op", 0)
        # ------------------------------------------------
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            # --- 変数完全復元 (画面分割対策) ---
            score_percent = res.get("score", 0)
            selected_major = res.get("industry_major", "D 建設業")
            user_equity_ratio = res.get("user_eq", 0)
            user_op_margin = res.get("user_op", 0)
            # --------------------------------
            selected_major = res.get("industry_major", "D 建設業")
            selected_sub = res.get("industry_sub", "06 総合工事業")
            hantei = res.get("hantei", "")
            industry_major = res.get("industry_major", "")
            asset_name = res.get("asset_name", "") or ""
            comparison_text = res.get("comparison", "")
            if jsic_data and selected_major in jsic_data:
                trend_info = jsic_data[selected_major]["sub"].get(selected_sub, "")
            # 業界トレンド拡充（ネット取得済みキャッシュがあれば追加）
            trend_extended = get_trend_extended(selected_sub)
            if trend_extended:
                trend_info = (trend_info or "") + "\n\n【ネットで補足】\n" + trend_extended[:1500]
            # --------------------------------------
            # 現在の案件IDを取得（審査直後ならセッションに入っている想定）
            current_case_id = st.session_state.get("current_case_id")

            # ── 審査実績DB 自動保存（新規審査ごとに1回だけ） ──────────────────
            try:
                from customer_db import save_record as _db_save_auto, init_db as _db_init_auto
                _db_auto_key = st.session_state.get("current_case_id") or (
                    f"{res.get('pd_percent', 0):.4f}"
                    f"_{res.get('score', 0):.4f}"
                    f"_{res.get('industry_major', '')}"
                )
                if st.session_state.get("_db_auto_saved_for") != str(_db_auto_key):
                    _db_init_auto()
                    _db_inp_auto = st.session_state.get("last_submitted_inputs") or {}
                    _new_db_id = _db_save_auto(res, _db_inp_auto, "")
                    st.session_state["_db_auto_saved_for"] = str(_db_auto_key)
                    if _new_db_id:
                        st.session_state["db_last_saved_id"] = _new_db_id
            except Exception:
                pass  # DB が利用不可の環境でも続行

            # ── モンテカルロ 自動実行（新規審査ごとに1回だけ実行） ──────────────
            try:
                from montecarlo import (
                    AdvancedMonteCarloEngine, CompanyData,
                    map_industry_from_major as _mc_map_ind,
                )
                _mc_auto_key = (
                    f"{res.get('pd_percent', 0):.4f}"
                    f"_{res.get('score', 0):.4f}"
                    f"_{res.get('industry_major', '')}"
                )
                if st.session_state.get("mc_auto_run_done_for") != _mc_auto_key:
                    _fin_ao   = res.get("financials") or {}
                    _inp_ao   = st.session_state.get("last_submitted_inputs") or {}
                    _ao_rev_m = max(1, int((_fin_ao.get("nenshu", 0) or 0) / 1000))
                    _ao_op    = max(-30.0, min(50.0, float(res.get("user_op", 5.0) or 5.0)))
                    _ao_eq    = max(1.0, min(99.0, float(res.get("user_eq", 30.0) or 30.0)))
                    _ao_net   = float(_fin_ao.get("net_assets", 0) or 0)
                    _ao_ast   = float(_fin_ao.get("assets", 0) or 0)
                    _ao_debt_m = max(0, int((_ao_ast - _ao_net) / 1000))
                    _ao_lease_m = max(1, int(
                        (_inp_ao.get("lease_credit", _fin_ao.get("lease_credit", 5000)) or 5000) / 10
                    ))
                    _ao_months  = max(6, min(120, int(_inp_ao.get("lease_term", 36) or 36)))
                    _ao_ind     = _mc_map_ind(res.get("industry_major", ""))
                    _ao_name    = st.session_state.get("rep_company") or "審査対象"
                    _ao_co = CompanyData(
                        name=_ao_name,
                        industry=_ao_ind,
                        revenue=_ao_rev_m * 1_000_000,
                        operating_margin=_ao_op / 100,
                        equity_ratio=max(_ao_eq / 100, 0.01),
                        total_debt=_ao_debt_m * 1_000_000,
                        lease_amount=_ao_lease_m * 10_000,
                        lease_months=_ao_months,
                    )
                    with st.spinner("モンテカルロ シミュレーション自動実行中…"):
                        _ao_engine = AdvancedMonteCarloEngine(n_simulations=3000)
                        _ao_pf = _ao_engine.analyze_portfolio([_ao_co])
                    st.session_state["mc_portfolio_result"] = _ao_pf
                    st.session_state["mc_companies"] = [{
                        "name": _ao_name,
                        "industry": _ao_ind,
                        "revenue_m": _ao_rev_m,
                        "op_margin": _ao_op,
                        "equity_ratio": _ao_eq,
                        "debt_m": _ao_debt_m,
                        "lease_amt_man": _ao_lease_m,
                        "lease_months": _ao_months,
                    }]
                    st.session_state["mc_auto_run_done_for"] = _mc_auto_key
            except Exception:
                pass  # montecarlo が利用不可の環境でも続行

            # ==================== ダッシュボードレイアウト（プロ仕様） ====================
            st.markdown("---")
            # ----- タイトル + 画像 -----
            img_path, img_caption = get_dashboard_image_path(hantei, industry_major, selected_sub, asset_name)
            col_title, col_img = st.columns([3, 1])
            with col_title:
                st.markdown(f"### 📊 分析ダッシュボード — {selected_sub}")
            with col_img:
                if img_path and os.path.isfile(img_path):
                    st.image(img_path, caption=img_caption, use_container_width=True)
                else:
                    st.caption("画像: dashboard_images に画像を配置するか、環境変数 DASHBOARD_IMAGES_ASSETS を指定してください。")

            st.divider()
            # ----- 判定サマリーカード（最重要項目 + スコアゲージ） -----
            _hantei_color = "#0d9488" if "承認" in res.get("hantei", "") else "#b91c1c"
            _yield_str = f"{res['yield_pred']:.2f}%" if "yield_pred" in res else "—"
            _sum_col, _gauge_col = st.columns([3, 2])
            with _sum_col:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1e3a5f 0%,#334155 100%);
                            color:#fff;padding:1.2rem 1.5rem;border-radius:12px;height:100%;box-sizing:border-box;">
                  <div style="font-size:0.85rem;opacity:0.75;margin-bottom:0.75rem;">📋 審査結果サマリー — {selected_sub}</div>
                  <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start;">
                    <div>
                      <div style="font-size:0.8rem;opacity:0.7;">判定</div>
                      <div style="font-size:2rem;font-weight:bold;color:{_hantei_color};">{res.get("hantei","—")}</div>
                    </div>
                    <div>
                      <div style="font-size:0.8rem;opacity:0.7;">総合スコア<span style="font-size:0.65rem;opacity:0.8;margin-left:4px;">（信用力参考値）</span></div>
                      <div style="font-size:2rem;font-weight:bold;">{res['score']:.1f}%</div>
                    </div>
                    <div>
                      <div style="font-size:0.8rem;opacity:0.7;">契約期待度<span style="font-size:0.65rem;opacity:0.8;margin-left:4px;">（暫定値・データ収集中）</span></div>
                      <div style="font-size:2rem;font-weight:bold;">{res.get('contract_prob',0):.1f}%</div>
                    </div>
                    <div>
                      <div style="font-size:0.8rem;opacity:0.7;">予測利回り</div>
                      <div style="font-size:2rem;font-weight:bold;">{_yield_str}</div>
                    </div>
                    <div>
                      <div style="font-size:0.8rem;opacity:0.7;">デフォルト率</div>
                      <div style="font-size:2rem;font-weight:bold;">{res.get('pd_percent',0):.1f}%</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with _gauge_col:
                st.plotly_chart(plot_gauge_plotly(res['score'], "総合スコア（信用力参考値）"), use_container_width=True, key="gauge_score")

            st.divider()

            # ── 📊 スコア配分内訳（ASSET_WEIGHT） ────────────────────────────────
            _ts = res.get("total_scorer_result") or st.session_state.get("_ts_result")
            if _ts:
                _ts_color  = _ts.get("grade_color", "#6b7280")
                _ts_label  = _ts.get("grade", "—")
                _ts_text   = _ts.get("grade_text", "—")
                _ts_total  = _ts.get("total_score", 0)
                _as_score  = _ts.get("asset_score", 0)
                _ob_score  = _ts.get("obligor_score", 0)
                _asset_w   = int(_ts.get("asset_weight", 0) * 100)
                _ob_w      = int(_ts.get("obligor_weight", 0) * 100)
                _category  = _ts.get("category", "—")
                _rationale = _ts.get("rationale", "")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1e3a5f 0%,#334155 100%);
                            color:#fff;padding:1rem 1.25rem;border-radius:10px;margin-bottom:0.75rem;">
                  <div style="font-size:0.8rem;opacity:0.7;margin-bottom:0.5rem;">
                    📊 スコア配分【{_category}】— 物件 {_asset_w}% ／ 借手 {_ob_w}%
                  </div>
                  <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end;">
                    <div>
                      <div style="font-size:0.75rem;opacity:0.7;">物件スコア × {_asset_w}%</div>
                      <div style="font-size:1.5rem;font-weight:bold;">{_as_score:.1f}点</div>
                    </div>
                    <div style="font-size:1.3rem;opacity:0.6;padding-bottom:0.2rem;">＋</div>
                    <div>
                      <div style="font-size:0.75rem;opacity:0.7;">借手スコア × {_ob_w}%</div>
                      <div style="font-size:1.5rem;font-weight:bold;">{_ob_score:.1f}点</div>
                    </div>
                    <div style="font-size:1.3rem;opacity:0.6;padding-bottom:0.2rem;">＝</div>
                    <div>
                      <div style="font-size:0.75rem;opacity:0.7;">総合スコア</div>
                      <div style="font-size:2rem;font-weight:bold;color:{_ts_color};">{_ts_total:.1f}点 [{_ts_label}] {_ts_text}</div>
                    </div>
                  </div>
                  <div style="font-size:0.72rem;opacity:0.6;margin-top:0.5rem;">{_rationale}</div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()

            # ── ⚔️ 軍師AIコメント（審査結果に直接表示）──────────────────────
            try:
                from components.shinsa_gunshi import render_gunshi_ai_comment
                render_gunshi_ai_comment(
                    res=res,
                    submitted_inputs=st.session_state.get("last_submitted_inputs"),
                    model_name=st.session_state.get("ollama_model", "llama3") or "llama3",
                    trend_info=trend_info or "",
                    bn_evidence=st.session_state.get("_bn_s_evidence"),
                )
            except Exception as _gac_err:
                st.caption(f"⚠️ 軍師AIコメント読み込みエラー: {_gac_err}")

            st.divider()

            # ── 報告書PDF ──────────────────────────────────────────────────────
            with st.expander("📄 審査報告書 PDF出力", expanded=False):
                if "last_result" not in st.session_state:
                    st.info("👈「新規審査」で審査を実行すると報告書を出力できます。")
                else:
                    from screening_report import build_screening_report_pdf
                    _rep_res = st.session_state["last_result"]
                    st.caption(f"業種：{_rep_res.get('industry_sub','')}　スコア：{_rep_res.get('score',0):.1f}")

                    st.divider()
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        _company_name = st.text_input("企業名（任意）", key="rep_company", placeholder="例：株式会社〇〇")
                        _screener = st.text_input("担当者名（任意）", key="rep_screener", placeholder="例：鈴木 一郎")
                    with col_r2:
                        _note = st.text_area("担当者メモ（任意）", key="rep_note", placeholder="特記事項・条件等", height=90)
                    if st.button("📥 PDF を生成してダウンロード", type="primary", key="rep_gen"):
                        with st.spinner("PDF 生成中..."):
                            try:
                                # BN証拠をチェックボックスの widget キーから直接再構築
                                # （PDF生成はBNエクスパンダーより上で実行されるため
                                #   session_state の "_bn_s_evidence" は1描画前の値になる。
                                #   widget キー自体は前描画で確定済みなので直接読む）
                                _bn_ev_for_pdf = {
                                    "Insolvent_Status":    1 if st.session_state.get("bn_s_insolvent",  False) else 0,
                                    "Main_Bank_Support":   1 if st.session_state.get("bn_s_main_bank",  False) else 0,
                                    "Related_Bank_Status": 1 if st.session_state.get("bn_s_rel_bank",   False) else 0,
                                    "Related_Assets":      1 if st.session_state.get("bn_s_rel_assets", False) else 0,
                                    "Co_Lease":            1 if st.session_state.get("bn_s_co_lease",   False) else 0,
                                    "Parent_Guarantor":    1 if st.session_state.get("bn_s_parent",     False) else 0,
                                    "Core_Business_Use":   1 if st.session_state.get("bn_s_core",       False) else 0,
                                    "Asset_Liquidity":     1 if st.session_state.get("bn_s_liquidity",  False) else 0,
                                    "Shorter_Lease_Term":  1 if st.session_state.get("bn_s_shorter",    False) else 0,
                                    "One_Time_Deal":       1 if st.session_state.get("bn_s_one_time",   False) else 0,
                                }
                                _bn_result_for_pdf   = None
                                _bn_evidence_for_pdf = _bn_ev_for_pdf
                                _bn_reversal_for_pdf = []
                                try:
                                    from bayesian_engine import (
                                        run_inference as _bn_infer_pdf,
                                        compute_reversal_suggestions as _bn_rev_fn,
                                    )
                                    _bn_result_for_pdf   = _bn_infer_pdf(_bn_ev_for_pdf)
                                    _bn_reversal_for_pdf = _bn_rev_fn(_bn_ev_for_pdf, top_n=5)
                                except Exception:
                                    pass
                                # モンテカルロ結果をサマリ化してPDFに渡す
                                _mc_pf_raw  = st.session_state.get("mc_portfolio_result")
                                _mc_summary = None
                                if _mc_pf_raw:
                                    try:
                                        try:
                                            from montecarlo import _generate_comment as _mc_gen_cmt
                                        except Exception:
                                            _mc_gen_cmt = None
                                        _mc_summary = {
                                            "weighted_default_prob": float(getattr(_mc_pf_raw, "weighted_default_prob", 0)),
                                            "concentration_risk":    float(getattr(_mc_pf_raw, "concentration_risk",    0)),
                                            "expected_loss":         float(getattr(_mc_pf_raw, "expected_loss",         0)),
                                            "portfolio_var_95":      float(getattr(_mc_pf_raw, "portfolio_var_95",      0)),
                                            "results": [
                                                {
                                                    "name":         getattr(getattr(_r, "company", None), "name", ""),
                                                    "default_prob": float(getattr(_r, "default_prob", 0)),
                                                    "score_median": float(getattr(_r, "score_median", 0)),
                                                    "var_95":       float(getattr(_r, "var_95",       0)),
                                                    "risk_level":   str(getattr(_r,  "risk_level",   "")),
                                                    "comment":      (_mc_gen_cmt(_r) if _mc_gen_cmt else ""),
                                                }
                                                for _r in (getattr(_mc_pf_raw, "results", []) or [])[:5]
                                            ],
                                        }
                                    except Exception:
                                        _mc_summary = None
                                # 審査結果レポートテキスト生成（PDF埋め込み用）
                                try:
                                    _report_text_for_pdf = generate_full_report_from_res(
                                        _rep_res, st.session_state
                                    )
                                except Exception:
                                    _report_text_for_pdf = None
                                # 軍師データをPDF用に変換
                                _gunshi_pdf = None
                                try:
                                    _g_raw = st.session_state.get("_gunshi_pdf_data")
                                    if _g_raw:
                                        from components.shinsa_gunshi import build_gunshi_pdf_data
                                        _gunshi_pdf = build_gunshi_pdf_data(_g_raw)
                                    else:
                                        # まだ開いていない場合はその場で計算
                                        from components.shinsa_gunshi import compute_gunshi_from_res, build_gunshi_pdf_data
                                        _g_raw = compute_gunshi_from_res(
                                            _rep_res,
                                            st.session_state.get("last_submitted_inputs"),
                                            bn_evidence=_bn_ev_for_pdf,  # チェックボックスの現在値を使用
                                        )
                                        _gunshi_pdf = build_gunshi_pdf_data(_g_raw)
                                except Exception:
                                    _gunshi_pdf = None
                                _pdf_bytes = build_screening_report_pdf(
                                    _rep_res,
                                    st.session_state.get("last_submitted_inputs"),
                                    {
                                        "company_name": _company_name,
                                        "screener":     _screener,
                                        "note":         _note,
                                        "bn_result":    _bn_result_for_pdf,
                                        "bn_evidence":  _bn_evidence_for_pdf,
                                        "bn_reversal":  _bn_reversal_for_pdf,
                                        "mc_summary":   _mc_summary,
                                        "report_text":  _report_text_for_pdf,
                                        "gunshi":       _gunshi_pdf,
                                        "ai_industry_advice": getattr(_rep_res, "ai_industry_advice", ""),
                                        "ai_byoki":           getattr(_rep_res, "ai_byoki", ""),
                                        "mc_chart_bytes": (
                                            __import__("montecarlo").make_portfolio_chart(
                                                st.session_state["mc_portfolio_result"]
                                            )
                                            if st.session_state.get("mc_portfolio_result")
                                            else None
                                        ),
                                    },
                                )
                                import datetime as _dt
                                _fname = f"審査報告書_{_company_name or '案件'}_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                st.download_button(
                                    "⬇️ PDF をダウンロード",
                                    data=_pdf_bytes,
                                    file_name=_fname,
                                    mime="application/pdf",
                                    key="rep_dl",
                                )
                                st.success("PDF を生成しました。上のボタンからダウンロードしてください。")
                            except Exception as _e:
                                st.error(f"PDF 生成エラー: {_e}")

            # ── ⚔️ 軍師モード（承認奪取）────────────────────────────────────────
            # スコアが承認ライン未満のときは自動展開（BNシミュレータをすぐ見せる）
            from bayesian_engine import THRESHOLD_APPROVAL
            _gu_approval_line = THRESHOLD_APPROVAL * 100
            _gunshi_auto_expand = (
                st.session_state.get("last_result", {}).get("score", 100) < _gu_approval_line
            )
            with st.expander(
                "⚔️ 軍師モード（承認奪取）— ベイズ推薦 × 100選 × LLM",
                expanded=_gunshi_auto_expand,
            ):
                if "last_result" not in st.session_state:
                    st.info("👈「新規審査」で審査を実行すると軍師分析が有効になります。")
                else:
                    _gu_res = st.session_state["last_result"]

                    # ── STEP 1: BN逆転条件シミュレーション（否決・ボーダー時のみ） ──
                    if _gu_res.get("score", 100) <= _gu_approval_line:
                        st.markdown("#### 🔄 STEP 1 — 逆転条件シミュレーション")
                        st.caption("条件をチェックすると承認確率がリアルタイムで変化し、下の軍師フレーズにも反映されます。")
                        try:
                            from bayesian_engine import (
                                run_inference as _bn_infer,
                                compute_reversal_suggestions as _bn_reversal,
                            )
                            _bn_c1, _bn_c2, _bn_c3 = st.columns(3)
                            with _bn_c1:
                                st.markdown("**財務・信用**")
                                _bn_insolvent  = st.checkbox("債務超過",              key="bn_s_insolvent",  value=False)
                                _bn_main_bank  = st.checkbox("メイン銀行支援あり",    key="bn_s_main_bank",  value=False)
                                _bn_rel_bank   = st.checkbox("関係者の銀行取引良好",  key="bn_s_rel_bank",   value=False)
                                _bn_rel_assets = st.checkbox("関係者の個人資産あり",  key="bn_s_rel_assets", value=False)
                            with _bn_c2:
                                st.markdown("**ヘッジ手段**")
                                _bn_co_lease = st.checkbox("銀行との協調リース",   key="bn_s_co_lease", value=False)
                                _bn_parent   = st.checkbox("親会社連帯保証",       key="bn_s_parent",   value=False)
                            with _bn_c3:
                                st.markdown("**物件・取引条件**")
                                _bn_core      = st.checkbox("本業に不可欠な物件",   key="bn_s_core",      value=False)
                                _bn_liquidity = st.checkbox("物件の中古流動性あり",  key="bn_s_liquidity", value=False)
                                _bn_shorter   = st.checkbox("リース期間を短縮",      key="bn_s_shorter",   value=False)
                                _bn_one_time  = st.checkbox("業況改善まで本件限り",  key="bn_s_one_time",  value=False)
                            _bn_ev = {
                                "Insolvent_Status":    1 if _bn_insolvent  else 0,
                                "Main_Bank_Support":   1 if _bn_main_bank  else 0,
                                "Related_Bank_Status": 1 if _bn_rel_bank   else 0,
                                "Related_Assets":      1 if _bn_rel_assets else 0,
                                "Co_Lease":            1 if _bn_co_lease   else 0,
                                "Parent_Guarantor":    1 if _bn_parent     else 0,
                                "Core_Business_Use":   1 if _bn_core       else 0,
                                "Asset_Liquidity":     1 if _bn_liquidity  else 0,
                                "Shorter_Lease_Term":  1 if _bn_shorter    else 0,
                                "One_Time_Deal":       1 if _bn_one_time   else 0,
                            }
                            st.session_state["_bn_s_result"]   = _bn_infer(_bn_ev)
                            st.session_state["_bn_s_evidence"] = _bn_ev
                            _bnr = st.session_state.get("_bn_s_result")
                            _bne = st.session_state.get("_bn_s_evidence", {})
                            if _bnr:
                                _bn_prob = _bnr["approval_prob"]
                                _bn_dec  = _bnr["decision"]
                                _bn_col  = "#0d9488" if _bn_dec == "承認" else ("#f59e0b" if _bn_dec == "要審議" else "#b91c1c")
                                _im      = _bnr.get("intermediate", {})
                                _r1, _r2, _r3, _r4 = st.columns(4)
                                _r1.metric("🎯 承認確率",  f"{_bn_prob:.1%}")
                                _r2.metric("財務信用度",   f"{_im.get('Financial_Creditworthiness', 0):.1%}")
                                _r3.metric("ヘッジ条件",   f"{_im.get('Hedge_Condition', 0):.1%}")
                                # 物件価値 — ASSET_WEIGHT と合わせて表示
                                _ts_for_bn = _gu_res.get("total_scorer_result") or st.session_state.get("_ts_result")
                                if _ts_for_bn:
                                    _bn_asset_score = _ts_for_bn.get("asset_score", 0)
                                    _bn_asset_w     = _ts_for_bn.get("asset_weight", 0)
                                    _bn_category    = _ts_for_bn.get("category", "")
                                    _r4.metric(
                                        f"物件価値【{_bn_category}】",
                                        f"{_bn_asset_score:.0f}点",
                                        delta=f"ウェイト {int(_bn_asset_w * 100)}%",
                                        delta_color="off",
                                    )
                                else:
                                    _r4.metric("物件価値", f"{_im.get('Asset_Value', 0):.1%}")
                                st.markdown(f"**判定: <span style='color:{_bn_col}'>{_bn_dec}</span>**", unsafe_allow_html=True)
                                _rev = _bn_reversal(_bne, top_n=5)
                                if _rev:
                                    st.markdown("**💡 逆転提案（PDFにも記載）**")
                                    for _r in _rev:
                                        _gain_pct = int(_r["delta"] * 100)
                                        st.markdown(
                                            f"- **{_r['label']}** → "
                                            f"{_r['before_prob']:.0%} → **{_r['after_prob']:.0%}**"
                                            f"（+{_gain_pct}%pt）　{_r['after_decision']}"
                                        )
                            _checked_count = sum(1 for v in _bn_ev.values() if v == 1)
                            if _checked_count > 0:
                                st.info(f"✅ {_checked_count}件の条件が選択されています。下の軍師フレーズに反映中...")
                        except Exception as _bn_err:
                            st.caption(f"🧠 BNエンジン利用不可: {type(_bn_err).__name__}")
                        st.divider()
                        st.markdown("#### ⚔️ STEP 2 — 承認奪取フレーズ & AI推薦")

                    # ── STEP 2（良決は直接）: 軍師フレーズ・LLM ──────────────────
                    try:
                        from components.shinsa_gunshi import render_gunshi_in_results
                        _gu_model = st.session_state.get("ollama_model", "llama3") or "llama3"
                        _gunshi_g = render_gunshi_in_results(
                            res=_gu_res,
                            submitted_inputs=st.session_state.get("last_submitted_inputs"),
                            model_name=_gu_model,
                            bn_evidence=st.session_state.get("_bn_s_evidence"),
                        )
                        if _gunshi_g:
                            st.session_state["_gunshi_pdf_data"] = _gunshi_g
                    except Exception as _ge:
                        st.error(f"軍師モードエラー: {_ge}")

            # ── 与信枠 ──────────────────────────────────────────────────────────
            with st.expander("💳 与信枠提案", expanded=False):
                if "last_result" not in st.session_state:
                    st.info("👈「新規審査」で審査を実行すると与信枠の試算ができます。")
                else:
                    from credit_limit import render_credit_limit_ui
                    render_credit_limit_ui(st.session_state["last_result"])

            # ── モンテカルロ ────────────────────────────────────────────────────
            with st.expander("📊 モンテカルロ リース審査シミュレーション", expanded=False):
                st.caption("業種別ボラティリティを用いたGBMで、リース期間中の財務悪化確率と審査スコア分布を10,000回シミュレーション。")
                try:
                    from montecarlo import (
                        AdvancedMonteCarloEngine, CompanyData, res_to_company_data,
                        INDUSTRY_VOLATILITY, make_company_chart, make_portfolio_chart,
                        generate_pdf_bytes,
                    )
                    _mc_available = True
                except ImportError as _mc_err:
                    st.error(f"montecarlo.py の読み込みに失敗しました: {_mc_err}")
                    _mc_available = False

                if _mc_available:
                    st.divider()

                    if "mc_companies" not in st.session_state:
                        st.session_state["mc_companies"] = []

                    from montecarlo import map_industry_from_major

                    _last_res  = st.session_state.get("last_result")
                    _last_inp  = st.session_state.get("last_submitted_inputs") or {}

                    if _last_res:
                        _fin = _last_res.get("financials") or {}
                        _auto_revenue_m = max(1, int((_fin.get("nenshu", 0) or 0) / 1000))
                        _auto_op        = max(-30.0, min(50.0, float(_last_res.get("user_op", 5.0) or 5.0)))
                        _auto_eq        = max(1.0, min(99.0, float(_last_res.get("user_eq", 30.0) or 30.0)))
                        _auto_assets    = (_fin.get("assets",     0) or 0)
                        _auto_net       = (_fin.get("net_assets", 0) or 0)
                        _auto_debt_m    = max(0, int((_auto_assets - _auto_net) / 1000))
                        _auto_lease_man = max(1, int((_last_inp.get("lease_credit",
                                                       _fin.get("lease_credit", 5000)) or 5000) / 10))
                        _auto_months    = max(6, min(120, int(_last_inp.get("lease_term", 36) or 36)))
                        _auto_industry  = map_industry_from_major(_last_res.get("industry_major", ""))
                        _auto_score     = _last_res.get("score", 0)

                        st.subheader("🔄 審査結果から取り込み")
                        with st.container(border=True):
                            _ia1, _ia2, _ia3, _ia4 = st.columns(4)
                            _ia1.metric("業種", _auto_industry)
                            _ia2.metric("年商", f"{_auto_revenue_m:,}百万円")
                            _ia3.metric("営業利益率", f"{_auto_op:.1f}%")
                            _ia4.metric("自己資本比率", f"{_auto_eq:.1f}%")
                            _ib1, _ib2, _ib3, _ib4 = st.columns(4)
                            _ib1.metric("負債合計（総資産－純資産）", f"{_auto_debt_m:,}百万円")
                            _ib2.metric("リース希望額（当社残高）", f"{_auto_lease_man:,}万円")
                            _ib3.metric("リース期間", f"{_auto_months}ヶ月")
                            _ib4.metric("審査スコア", f"{_auto_score:.1f}%")

                            _auto_name = st.text_input(
                                "企業名（任意）",
                                placeholder="例：株式会社〇〇",
                                key="mc_auto_name_input",
                            )
                            if st.button("✅ この案件をリストに追加", type="primary",
                                         use_container_width=True, key="mc_auto_add"):
                                st.session_state["mc_companies"].append({
                                    "name": _auto_name or "審査対象",
                                    "industry": _auto_industry,
                                    "revenue_m": _auto_revenue_m,
                                    "op_margin": _auto_op,
                                    "equity_ratio": _auto_eq,
                                    "debt_m": _auto_debt_m,
                                    "lease_amt_man": _auto_lease_man,
                                    "lease_months": _auto_months,
                                })
                                st.success(f"✅ {_auto_name or '審査対象'} を追加しました。")
                                st.rerun()
                    else:
                        st.info("💡 審査タブで審査を実行すると、結果がここに自動表示されます。")

                    st.divider()
                    with st.expander("➕ 手動で企業を追加（比較用）", expanded=not bool(_last_res)):
                        with st.form("mc_add_company_form"):
                            _fc1, _fc2 = st.columns(2)
                            with _fc1:
                                _mc_name = st.text_input("企業名", value="比較企業A社", key="mc_name")
                                _mc_industry = st.selectbox(
                                    "業種", options=list(INDUSTRY_VOLATILITY.keys()),
                                    index=0, key="mc_industry"
                                )
                                _mc_revenue = st.number_input("年商（百万円）",
                                    value=500, min_value=1, step=10, key="mc_revenue")
                                _mc_op_margin = st.number_input("営業利益率（%）",
                                    value=5.0, min_value=-30.0, max_value=50.0, step=0.1, key="mc_op_margin")
                            with _fc2:
                                _mc_eq = st.number_input("自己資本比率（%）",
                                    value=30.0, min_value=1.0, max_value=99.0, step=0.5, key="mc_eq")
                                _mc_debt = st.number_input("負債合計（総資産－純資産、百万円）",
                                    value=100, min_value=0, step=10, key="mc_debt")
                                _mc_lease_amt = st.number_input("リース希望額（万円）",
                                    value=500, min_value=1, step=100, key="mc_lease_amt")
                                _mc_lease_mo  = st.number_input("リース期間（月）",
                                    value=36, min_value=6, max_value=120, step=6, key="mc_lease_mo")
                            _mc_submitted = st.form_submit_button("➕ リストに追加", use_container_width=True)

                        if _mc_submitted:
                            st.session_state["mc_companies"].append({
                                "name": _mc_name,
                                "industry": _mc_industry,
                                "revenue_m": _mc_revenue,
                                "op_margin": _mc_op_margin,
                                "equity_ratio": _mc_eq,
                                "debt_m": _mc_debt,
                                "lease_amt_man": _mc_lease_amt,
                                "lease_months": int(_mc_lease_mo),
                            })
                            st.success(f"✅ {_mc_name} を追加しました。")
                            st.rerun()

                    _mc_list = st.session_state.get("mc_companies", [])
                    if _mc_list:
                        st.subheader(f"📋 分析対象 {len(_mc_list)}社")
                        for _i, _co in enumerate(_mc_list):
                            _cx1, _cx2 = st.columns([5, 1])
                            with _cx1:
                                st.caption(
                                    f"**{_co['name']}** | {_co['industry']} | "
                                    f"年商{_co['revenue_m']}M | 利益率{_co['op_margin']:.1f}% | "
                                    f"自己資本{_co['equity_ratio']:.1f}% | リース{_co['lease_amt_man']}万円/{_co['lease_months']}ヶ月"
                                )
                            with _cx2:
                                if st.button("🗑", key=f"mc_del_{_i}"):
                                    st.session_state["mc_companies"].pop(_i)
                                    st.rerun()

                        _mc_n_sim = st.select_slider("シミュレーション回数", options=[1000, 3000, 5000, 10000], value=5000)

                        st.divider()
                        _mc_run_col, _mc_clear_col = st.columns([3, 1])
                        with _mc_run_col:
                            _mc_run = st.button("▶ シミュレーション実行", type="primary", use_container_width=True, key="mc_run_btn")
                        with _mc_clear_col:
                            if st.button("🗑️ リストをクリア", use_container_width=True, key="mc_clear_btn"):
                                st.session_state["mc_companies"] = []
                                st.session_state.pop("mc_portfolio_result", None)
                                st.rerun()

                        if _mc_run:
                            with st.spinner(f"モンテカルロシミュレーション実行中… ({_mc_n_sim:,}回 × {len(_mc_list)}社)"):
                                _engine = AdvancedMonteCarloEngine(n_simulations=_mc_n_sim)
                                _companies = [
                                    CompanyData(
                                        name=co["name"],
                                        industry=co["industry"],
                                        revenue=co["revenue_m"] * 1_000_000,
                                        operating_margin=co["op_margin"] / 100,
                                        equity_ratio=max(co["equity_ratio"] / 100, 0.01),
                                        total_debt=co["debt_m"] * 1_000_000,
                                        lease_amount=co["lease_amt_man"] * 10_000,
                                        lease_months=co["lease_months"],
                                    )
                                    for co in _mc_list
                                ]
                                _portfolio = _engine.analyze_portfolio(_companies)
                            st.session_state["mc_portfolio_result"] = _portfolio
                            st.success("シミュレーション完了！")
                            st.rerun()

                    _mc_pf = st.session_state.get("mc_portfolio_result")
                    if _mc_pf:
                        st.divider()
                        st.subheader("📈 ポートフォリオ分析結果")
                        _pf_c1, _pf_c2, _pf_c3, _pf_c4 = st.columns(4)
                        _pf_c1.metric("加重平均デフォルト確率", f"{_mc_pf.weighted_default_prob:.1%}")
                        _pf_c2.metric("集中リスク（上位3社）", f"{_mc_pf.concentration_risk:.1%}")
                        _pf_c3.metric("期待損失額", f"{_mc_pf.expected_loss/1e4:,.0f}万円")
                        _pf_c4.metric("ポートフォリオVaR(95%)", f"{_mc_pf.portfolio_var_95:.1f}pt")

                        _pf_chart = make_portfolio_chart(_mc_pf)
                        st.image(_pf_chart, use_container_width=True)

                        st.divider()
                        st.subheader("🏢 個社別 詳細結果")
                        for _r in _mc_pf.results:
                            _risk_emoji = {"低リスク": "🟢", "中リスク": "🟡", "高リスク": "🔴", "極高リスク": "🟣"}.get(_r.risk_level, "⚪")
                            with st.expander(f"{_risk_emoji} **{_r.company.name}** — {_r.risk_level}  |  デフォルト確率 {_r.default_prob:.1%}  |  スコア {_r.score_median:.1f}", expanded=False):
                                _dc1, _dc2, _dc3 = st.columns(3)
                                _dc1.metric("デフォルト確率", f"{_r.default_prob:.1%}")
                                _dc2.metric("スコア中央値", f"{_r.score_median:.1f}")
                                _dc3.metric("VaR (95%)", f"{_r.var_95:.1f}pt")
                                _comp_chart = make_company_chart(_r)
                                st.image(_comp_chart, use_container_width=True)

                        st.divider()
                        with st.spinner("PDFレポート生成中…"):
                            _pdf_bytes = generate_pdf_bytes(_mc_pf)
                        _pdf_name = f"montecarlo_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="📥 PDFレポートをダウンロード",
                            data=_pdf_bytes,
                            file_name=_pdf_name,
                            mime="application/pdf",
                            use_container_width=True,
                            key="mc_pdf_download",
                        )
                    else:
                        if not _mc_list:
                            st.info("👆 企業を追加してシミュレーションを実行してください。")


            # ── 審査実績DB 閲覧・分析 ──────────────────────────────────────────
            with st.expander("📂 審査実績DB", expanded=False):
                try:
                    from customer_db import (
                        get_stats as _db_stats, get_records as _db_records,
                        get_industry_list as _db_ind_list,
                        get_total_count as _db_count, delete_record as _db_del,
                        get_db_path as _db_path_fn,
                    )
                    import pandas as _pd_db

                    _db_total = _db_count()
                    if _db_total == 0:
                        st.info("まだデータがありません。審査を実行すると自動的に蓄積されます。")
                    else:
                        _dbst = _db_stats()

                        # ── KPI ──
                        st.markdown(f"**📊 蓄積件数: {_db_total:,} 件**")
                        _kd1, _kd2, _kd3, _kd4 = st.columns(4)
                        _kd1.metric("平均スコア", f"{_dbst['score_avg']}")
                        _kd2.metric("最高スコア", f"{_dbst['score_max']}")
                        _kd3.metric("最低スコア", f"{_dbst['score_min']}")
                        _appr = _dbst["judgment_counts"].get("承認圏内", 0)
                        _appr_r = _appr / _db_total * 100 if _db_total else 0
                        _kd4.metric("承認率", f"{_appr_r:.1f}%")

                        # ── 判定分布 ──
                        st.divider()
                        _dbc1, _dbc2 = st.columns(2)
                        with _dbc1:
                            st.markdown("**判定分布**")
                            for _jt, _jc in _dbst["judgment_counts"].items():
                                _jr = _jc / _db_total * 100
                                _col = "🟢" if "承認" in _jt else "🟡"
                                st.write(f"{_col} {_jt}: **{_jc}件** ({_jr:.1f}%)")

                        with _dbc2:
                            st.markdown("**スコア帯分布**")
                            for _band, _cnt in sorted(_dbst.get("score_dist", {}).items()):
                                _br = _cnt / _db_total * 100
                                st.write(f"スコア {_band}: **{_cnt}件** ({_br:.1f}%)")

                        # ── 業種別集計 ──
                        if _dbst.get("by_industry"):
                            st.divider()
                            st.markdown("**業種別 件数・平均スコア**")
                            _ind_df = _pd_db.DataFrame(_dbst["by_industry"])
                            _ind_df.columns = ["業種", "件数", "平均スコア"]
                            st.dataframe(
                                _ind_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                        # ── 平均財務指標 ──
                        st.divider()
                        st.markdown("**📈 平均財務指標（匿名化・全件）**")
                        _fa1, _fa2, _fa3, _fa4 = st.columns(4)
                        _fa1.metric("平均年商", f"{int(_dbst.get('avg_revenue_m') or 0):,}百万円")
                        _fa2.metric("平均総資産", f"{int(_dbst.get('avg_assets_m') or 0):,}百万円")
                        _fa3.metric("平均自己資本比率", f"{_dbst.get('avg_equity') or '—'}%")
                        _fa4.metric("平均リース額", f"{int(_dbst.get('avg_lease_m') or 0):,}百万円")

                        # ── レコード一覧 ──
                        st.divider()
                        st.markdown("**🗂 レコード一覧**")
                        _fil1, _fil2, _fil3, _fil4 = st.columns(4)
                        with _fil1:
                            _f_ind = st.selectbox(
                                "業種フィルタ", ["（全て）"] + _db_ind_list(),
                                key="db_filter_ind"
                            )
                        with _fil2:
                            _f_jdg = st.selectbox(
                                "判定フィルタ", ["（全て）", "承認圏内", "要審議"],
                                key="db_filter_jdg"
                            )
                        with _fil3:
                            _f_sc_min = st.number_input("スコア下限", value=0, min_value=0,
                                                         max_value=100, key="db_sc_min")
                        with _fil4:
                            _f_sc_max = st.number_input("スコア上限", value=100, min_value=0,
                                                         max_value=100, key="db_sc_max")

                        _recs = _db_records(
                            industry_major=("" if _f_ind == "（全て）" else _f_ind),
                            judgment=("" if _f_jdg == "（全て）" else _f_jdg),
                            score_min=_f_sc_min,
                            score_max=_f_sc_max,
                            limit=100,
                        )
                        if _recs:
                            _rec_df = _pd_db.DataFrame(_recs)
                            _disp_cols = {
                                "id": "ID", "created_at": "審査日時",
                                "industry_sub": "業種（中）", "customer_type": "区分",
                                "revenue_m": "年商(百万)", "equity_ratio": "自己資本比率%",
                                "lease_amount_m": "リース額(百万)", "lease_term": "期間(月)",
                                "score": "スコア", "judgment": "判定",
                                "contract_prob": "成約確率", "memo": "メモ",
                            }
                            _rec_df = _rec_df[[c for c in _disp_cols if c in _rec_df.columns]]
                            _rec_df = _rec_df.rename(columns=_disp_cols)
                            st.dataframe(_rec_df, use_container_width=True, hide_index=True)
                            st.caption(f"表示: {len(_recs)}件 / 全{_db_total}件")

                            # 削除
                            with st.expander("🗑 レコード削除", expanded=False):
                                _del_id = st.number_input(
                                    "削除するID", min_value=1, step=1, key="db_del_id"
                                )
                                if st.button("削除実行", key="db_del_btn", type="secondary"):
                                    _db_del(int(_del_id))
                                    st.success(f"ID {_del_id} を削除しました。")
                                    st.rerun()
                        else:
                            st.info("条件に一致するレコードがありません。")

                        # DB ファイルパス
                        st.caption(f"📁 DB: `{_db_path_fn()}`")

                except Exception as _db_view_err:
                    st.error(f"DB表示エラー: {_db_view_err}")

            # ----- 🤖 AIひとこと評価（自動生成） -----
            _quick_key = "ai_quick_comment_result"
            _quick_result_id = f"ai_quick_{res.get('score', 0):.1f}_{res.get('industry_sub', '')}"
            # スコア+業種が変わったときだけ再生成
            if st.session_state.get("ai_quick_comment_id") != _quick_result_id:
                st.session_state[_quick_key] = None
                st.session_state["ai_quick_comment_id"] = _quick_result_id
            if is_ai_available() and st.session_state.get(_quick_key) is None:
                with st.spinner("AIコメント生成中…"):
                    _qc = get_ai_quick_comment(res)
                st.session_state[_quick_key] = _qc if _qc else ""
            _qc_text = st.session_state.get(_quick_key) or ""
            if _qc_text:
                st.info(f"🤖 **AIコメント** {_qc_text}")
            elif not is_ai_available():
                st.caption("💬 AIコメント: サイドバーでAIエンジンを設定すると自動評価が表示されます。")

            # ----- 🤖 AI総合評価 -----
            with st.expander("🤖 AI総合評価（5項目）", expanded=False):
                st.caption("ローカルLLM（またはGemini）が財務データ・スコアを総合的に判断して評価します。")
                _ai_eval_key = "ai_comprehensive_eval_result"
                _ai_eval_loading_key = "ai_comprehensive_eval_loading"

                if st.button("▶ AI評価を生成", key="btn_ai_comprehensive_eval"):
                    st.session_state[_ai_eval_loading_key] = True
                    st.session_state[_ai_eval_key] = None

                if st.session_state.get(_ai_eval_loading_key):
                    with st.spinner("AI評価を生成中… （ローカルLLMは30〜90秒かかる場合があります）"):
                        _eval_result = get_ai_comprehensive_evaluation(res)
                    st.session_state[_ai_eval_key] = _eval_result
                    st.session_state[_ai_eval_loading_key] = False

                _eval_text = st.session_state.get(_ai_eval_key)
                if _eval_text:
                    # ①〜⑤ を色付きで表示
                    _eval_lines = _eval_text.splitlines()
                    _formatted = []
                    for _line in _eval_lines:
                        _line = _line.strip()
                        if not _line:
                            continue
                        if _line.startswith("①") or _line.startswith("②") or _line.startswith("③") or _line.startswith("④"):
                            _formatted.append(f"**{_line}**")
                        elif _line.startswith("⑤"):
                            _formatted.append(f"\n**{_line}**")
                        else:
                            _formatted.append(_line)
                    st.markdown("\n\n".join(_formatted))
                elif _eval_text is not None:
                    st.warning("AI評価を取得できませんでした。AIエンジンの設定（サイドバー）を確認してから再試行してください。")

            # ----- 主要KPI（業界実績）-----
            past_stats = get_stats(selected_sub)
            with st.expander("📊 業界実績KPI", expanded=False):
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.metric("業界 成約率", f"{past_stats.get('close_rate', 0) * 100:.1f}%" if past_stats.get("count") else "—", help="同業種の成約率")
                with kpi2:
                    st.metric("業界 成約件数", f"{past_stats.get('closed_count', 0)}件" if past_stats.get("count") else "—", help="同業種の成約件数")
                with kpi3:
                    avg_r = past_stats.get("avg_winning_rate")
                    st.metric("業界 平均金利", f"{avg_r:.2f}%" if avg_r is not None and avg_r > 0 else "—", help="同業種の平均成約金利")

            # ----- 要確認アラート（承認ライン直下・本社と学習モデルの判定差） -----
            review_need, review_reasons = get_review_alert(res)
            if review_need and review_reasons:
                st.warning("⚠️ **要確認**: " + " / ".join(review_reasons))

            # ----- AIが補完した判定要因（進化するダッシュボード） -----
            ai_factors = res.get("ai_completed_factors") or []
            if ai_factors:
                with st.expander("🤖 AIが補完した判定要因", expanded=True):
                    st.caption("あなたの設定した財務指標に加え、以下の要因を成約率（契約期待度）に反映しました。")
                    for f in ai_factors:
                        sign = "+" if f.get("effect_percent", 0) >= 0 else ""
                        st.markdown(f"- **{f.get('factor', '')}** … {sign}{f.get('effect_percent', 0)}% （{f.get('detail', '')}）")

            # ── 将来事業計画シミュレーション（モンテカルロ法ベース） ─────────────────────
            with st.expander("🔮 将来事業計画（売上・利益）シミュレーション", expanded=False):
                try:
                    from future_simulation import render_future_simulation_ui
                    render_future_simulation_ui(res)
                except ImportError as e:
                    st.error(f"シミュレーション機能の読み込みに失敗しました: {e}")
                    
            # ----- 定性スコアリング（総合×60%＋定性×40%でランクA〜E） -----
            qcorr = res.get("qualitative_scoring_correction")
            with st.expander("📋 定性スコアリング", expanded=bool(qcorr)):
                if qcorr:
                    r = qcorr
                    st.caption("**ランク（A〜E）は 総合×重み＋定性×重み（デフォルト60%/40%）に基づきます。**")
                    total_score = res.get("score", 0)  # 総合スコア（借手+物件）
                    qual_score = r.get("weighted_score", 0)
                    combined = r.get("combined_score", 0)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("総合スコア（信用力参考値）", f"{total_score:.1f}", help="借手スコア85%＋物件スコア15%。業種内相対評価です。デフォルト予測値ではありません。")
                    with c2:
                        st.metric("定性スコア", f"{qual_score} / 100", help="項目別5段階の加重平均")
                    with c3:
                        st.metric("合計（総合×重み＋定性×重み）", f"{combined}", help="ランク算出の元")
                    with c4:
                        st.metric("ランク", f"{r.get('rank', '—')} {r.get('rank_text', '')}", help=r.get("rank_desc", ""))
                    st.caption(r.get("rank_desc", ""))
                    st.markdown("**項目別**")
                    for item_id, data in (r.get("items") or {}).items():
                        val = data.get("value")
                        if val is not None:
                            label_short = data.get("level_label") or QUALITATIVE_SCORING_LEVEL_LABELS.get(val, f"{int((val or 0)/4*100)}点")
                            st.markdown(f"- **{data.get('label', item_id)}**（重み{data.get('weight', 0)}%）: {label_short}")
                else:
                    st.info("審査入力の「定性スコアリング」で項目を選択すると、ここに集計結果が表示されます。ランクは総合×重み＋定性×重みで算出。定性を1件も選んでいない場合は総合スコアのみで判定します。")

            # ----- 学習モデル（業種別ハイブリッド）の予測結果（融合機能）・常に表示 -----
            scoring_result = res.get("scoring_result")
            with st.expander("📈 学習モデル（業種別ハイブリッド）デフォルト確率", expanded=False):
                if scoring_result:
                    st.caption("**いずれも「デフォルト確率」（高い＝リスク大）です。** 上記の本システム「契約期待度」（成約率）とは尺度が逆です。成約率に換算するなら 約 100% − デフォルト確率。ハイブリッドは「業種別回帰のデフォルト確率」と「AIのデフォルト確率」の加重平均なので、同じ尺度同士の組み合わせです。")
                    sr1, sr2, sr3, sr4 = st.columns(4)
                    with sr1:
                        st.metric("既存（業種別回帰）デフォルト確率", f"{scoring_result.get('legacy_prob', 0)*100:.2f}%", help="学習モデル側の業種別回帰")
                    with sr2:
                        st.metric("AI（LightGBM）デフォルト確率", f"{scoring_result.get('ai_prob', 0)*100:.2f}%", help="LightGBM統合")
                    with sr3:
                        st.metric("ハイブリッド デフォルト確率", f"{scoring_result.get('hybrid_prob', 0)*100:.2f}%", help="0.3×既存+0.7×AI（同尺度）")
                    with sr4:
                        dec = scoring_result.get("decision", "—")
                        st.metric("学習モデル判定", dec, help="デフォルト確率50%未満で承認")
                    # グラフ表示（Top5要因のみ）
                    st.divider()
                    st.subheader("📊 学習モデル分析グラフ")
                        
                    # Top5要因グラフ
                    top5 = scoring_result.get("top5_reasons") or []
                    if top5:
                        st.caption("**判定に効いている指標 Top5**")
                        fig_top5 = plot_scoring_top5_factors_plotly(scoring_result)
                        if fig_top5:
                            st.plotly_chart(fig_top5, use_container_width=True, key="plotly_scoring_top5")
                        else:
                            # グラフが描けない場合はテキスト表示
                            _feat_ja = {
                                "ROA": "総資産利益率（ROA）", "ROE": "自己資本利益率（ROE）",
                                "operating_margin": "売上高営業利益率", "net_margin": "売上高純利益率",
                                "equity_ratio": "自己資本比率", "debt_ratio": "負債比率", "debt_equity_ratio": "負債対自己資本比率",
                                "machinery_ratio": "機械設備比率", "fixed_asset_ratio": "固定資産比率",
                                "fixed_to_equity": "固定資産対純資産比率", "machinery_equity_coverage": "機械設備の自己資本カバー率",
                                "rent_to_revenue": "リース料負担率（対売上高）", "operating_profit_to_rent": "営業利益のリース料カバー率",
                                "rent_to_equity": "リース料の純資産負担率", "lease_dependency": "リース依存度",
                                "total_fixed_cost_ratio": "総固定費負担率", "depreciation_to_revenue": "減価償却費率（対売上高）",
                                "EBITDA_margin": "EBITDAマージン", "depreciation_rate": "設備償却進行度",
                                "asset_turnover": "総資産回転率", "fixed_asset_turnover": "固定資産回転率",
                                "log_revenue": "売上高（対数）", "log_assets": "総資産（対数）",
                                "is_loss": "赤字フラグ", "is_operating_loss": "営業赤字フラグ",
                                "low_equity_ratio": "自己資本比率20%未満", "low_ROA": "ROA2%未満",
                                "high_rent_burden": "リース負担大", "rent_exceeds_profit": "リース料＞営業利益",
                                "industry_encoded": "業種（コード）",
                            }
                            for idx, r in enumerate(top5, 1):
                                if ":" in r:
                                    _name, _val = r.split(":", 1)
                                    _label = _feat_ja.get(_name.strip(), _name.strip())
                                else:
                                    _label, _val = r, ""
                                st.caption(f"#{idx} {_label}: {_val.strip()}")
                else:
                    st.info(
                        "**デフォルト確率を出すには、次の2つが必要です。**\n\n"
                        "1. **総資産**と**純資産**を入力してから「判定開始」を押す\n\n"
                        "2. **学習済みモデル（5個のpklファイル）**を用意する：\n"
                        "   - 別ツール（リース与信スコアリング）で「業種別ハイブリッド」を学習すると、`models/industry_specific/` フォルダに pkl ができます\n"
                        "   - その中身（industry_coefficients.pkl など5ファイル）を、このアプリのフォルダ内にある\n"
                        "     `lease_logic_sumaho10/scoring/models/industry_specific/` にコピーしてください\n\n"
                        "※ モデルがなくても、本システムのスコア（成約率）だけで審査はできます。"
                    )

            st.divider()
            # ----- カード: 本件スコア内訳・利回り -----
            pd_val = res.get("pd_percent")
            if pd_val is None:
                fin = res.get("financials", {})
                total_assets = fin.get("assets") or 0
                net_assets = fin.get("net_assets") or 0
                machines = fin.get("machines") or 0
                other_assets = fin.get("other_assets") or 0
                user_eq = res.get("user_eq", 0)
                user_op = res.get("user_op", 0)
                liability_total = total_assets - net_assets if total_assets and net_assets is not None else 0
                current_approx = max(0, total_assets - machines - other_assets)
                current_ratio = (current_approx / liability_total * 100) if liability_total > 0 else 100.0
                pd_val = calculate_pd(user_eq, current_ratio, user_op)

            with st.expander("📐 スコア内訳・利回り詳細", expanded=False):
                k2, k3, k4 = st.columns(3)
                with k2:
                    st.metric("判定", res.get("hantei", "—"), help="承認圏内 or 要審議")
                with k3:
                    st.metric("契約期待度（暫定）", f"{res.get('contract_prob', 0):.1f}%", help="現在は定性補正ベースの暫定値。競合社数・発生経緯などのデータ収集後に専用モデルへ移行予定。")
                with k4:
                    if "yield_pred" in res:
                        st.metric("予測利回り", f"{res['yield_pred']:.2f}%", delta=f"{res.get('rate_diff', 0):+.2f}%", help="AI予測利回り")
                    else:
                        st.metric("予測利回り", "—", help="利回りモデル未適用")
                # ----- スコア内訳（借手・物件説明 + 3モデル） -----
                if "score_borrower" in res and "asset_score" in res:
                    st.caption(f"📌 借手 {res['score_borrower']:.1f}% × 0.85 ＋ 物件「{res.get('asset_name', '')}」{res['asset_score']}点 × 0.15 → 総合 {res['score']:.1f}%")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("① 全体モデル", f"{res['score']:.1f}%", help="全業種共通係数")
                with cols[1]:
                    ind_label = res.get("ind_name", "全体_既存先")
                    second_label = "② 業種モデル" if (ind_label.split("_")[0] != "全体") else "② 業種(全体)"
                    st.metric(second_label, f"{res['ind_score']:.1f}%", delta=f"{res['ind_score']-res['score']:+.1f}%")
                with cols[2]:
                    st.metric("③ 指標ベンチマーク", f"{res['bench_score']:.1f}%", delta=f"{res['bench_score']-res['score']:+.1f}%", delta_color="inverse")

            # ----- 業界比較テキスト（サマリー直下に表示） -----
            industry_key = res["industry_major"]
            if industry_key in avg_data:
                avg = avg_data[industry_key]
                u_sales = res["financials"]["nenshu"]
                a_sales = avg["nenshu"]
                u_op_r = res['user_op']
                a_op_r = (avg["op_profit"]/avg["nenshu"]*100) if avg["nenshu"] > 0 else 0
                sales_ratio = u_sales / a_sales
                if sales_ratio >= 1.2: sales_msg = f"平均の{sales_ratio:.1f}倍規模"
                elif sales_ratio <= 0.8: sales_msg = f"平均より小規模({sales_ratio:.1f}倍)"
                else: sales_msg = "業界平均並み"
                if u_op_r >= a_op_r + 2.0: prof_msg = f"高収益({u_op_r:.1f}%)"
                elif u_op_r < a_op_r: prof_msg = f"平均以下({u_op_r:.1f}%)"
                else: prof_msg = f"標準({u_op_r:.1f}%)"
                st.caption(f"業界比較 — 規模: {sales_msg} / 収益: {prof_msg}")
                if comparison_text:
                    with st.expander("📊 財務指標 vs 業界目安", expanded=True):
                        st.markdown(comparison_text)

            # ----- リース負担率 vs 業種平均（e-Stat 年度版）-----
            _lbd = res.get("lease_burden_data", {})
            if _lbd and _lbd.get("bench_lease_burden") is not None:
                _ualp  = _lbd.get("user_annual_lease_pct")
                _ulcp  = _lbd.get("user_lease_credit_pct")
                _blb   = _lbd["bench_lease_burden"]
                _bcs   = _lbd.get("bench_capex_to_sales")
                _blc   = _lbd.get("bench_lease_to_capex")
                _cv    = _ualp if _ualp is not None else _ulcp
                _tlbl  = "年換算推定" if _ualp is not None else "与信/売上比（参考）"

                if _cv is not None and _blb > 0:
                    _lbr = _cv / _blb
                    if   _lbr >= 3.0: _icon, _msg = "🔴", f"業種平均の{_lbr:.1f}倍（過大）"
                    elif _lbr >= 2.0: _icon, _msg = "🟠", f"業種平均の{_lbr:.1f}倍（高め）"
                    elif _lbr >= 1.5: _icon, _msg = "🟡", f"業種平均の{_lbr:.1f}倍（やや高め）"
                    else:             _icon, _msg = "🟢", f"業種平均以下（良好）"
                    _capex_txt = f" | 業種設備投資率: {_bcs:.2f}%" if _bcs else ""
                    _lc_txt    = f" | 業種リース/設備投資: {_blc:.1f}%" if _blc else ""
                    st.caption(
                        f"{_icon} **リース負担率（{_tlbl}）**: "
                        f"{_cv:.2f}% vs 業種平均 {_blb:.2f}% → **{_msg}**"
                        f"{_capex_txt}{_lc_txt}"
                    )

            # ----- DSCR ＋ 追加財務指標 vs 業種平均 -----
            _u_dscr      = res.get("user_dscr")
            _dscr_source = res.get("dscr_source")
            _u_roa    = res.get("user_roa")
            _u_curr   = res.get("user_current_ratio")
            _u_debt   = res.get("user_debt_ratio")
            _u_turn   = res.get("user_asset_turnover")
            _b_roa    = res.get("bench_roa")
            _b_curr   = res.get("bench_current_ratio")
            _b_debt   = res.get("bench_debt_ratio")
            _b_turn   = res.get("bench_asset_turnover")

            _has_extra = any(v is not None for v in [_u_dscr, _b_roa, _b_curr, _b_debt])
            if _has_extra:
                with st.expander("📐 財務指標 詳細比較（DSCR・ROA・流動比率・負債比率）", expanded=False):

                    # DSCR（最重要 → 先頭に大きく表示）
                    if _u_dscr is not None:
                        _dscr_color = "normal" if _u_dscr >= 1.5 else ("off" if _u_dscr >= 1.0 else "inverse")
                        _dscr_label = "良好" if _u_dscr >= 1.5 else ("注意" if _u_dscr >= 1.0 else "要警戒🔴")
                        _dscr_help = (
                            "(営業利益＋減価償却費) ÷ 年間賃借料。1.5倍以上が目安。\n"
                            f"分母: {_dscr_source}" if _dscr_source else
                            "(営業利益＋減価償却費) ÷ 年間賃借料。1.5倍以上が目安。"
                        )
                        st.metric(
                            "DSCR（債務返済余力）",
                            f"{_u_dscr:.2f} 倍",
                            delta=_dscr_label,
                            delta_color=_dscr_color,
                            help=_dscr_help
                        )
                        # 分母の出所を明示
                        if _dscr_source == "決算書賃借料":
                            st.caption(
                                "DSCR = (営業利益 ＋ 減価償却費) ÷ **決算書・賃借料費用**"
                                "（他社リース・賃貸料を含む全社実績値）"
                            )
                        elif _dscr_source == "当社与信推計（参考）":
                            st.caption(
                                "⚠️ DSCR = (営業利益 ＋ 減価償却費) ÷ **当社与信から推計した年間リース料**"
                                "（賃借料費用が未入力のため参考値。決算書の賃借料費用を入力すると精度が上がります）"
                            )
                        st.divider()

                    # 追加財務指標 3列比較
                    _cols = st.columns(3)
                    def _metric_vs(col, label, user_v, bench_v, unit="%", higher_good=True, fmt=".1f"):
                        if user_v is None:
                            return
                        delta_str = None
                        delta_col = "normal"
                        if bench_v is not None:
                            diff = user_v - bench_v
                            delta_str = f"業種比 {diff:+.1f}{unit}"
                            if higher_good:
                                delta_col = "normal" if diff >= 0 else "inverse"
                            else:
                                delta_col = "normal" if diff <= 0 else "inverse"
                        col.metric(label, f"{user_v:{fmt}}{unit}",
                                   delta=delta_str, delta_color=delta_col,
                                   help=f"業種平均: {bench_v:{fmt}}{unit}" if bench_v is not None else "業種データなし")

                    _metric_vs(_cols[0], "ROA", _u_roa, _b_roa, unit="%", higher_good=True)
                    _metric_vs(_cols[1], "流動比率", _u_curr, _b_curr, unit="%", higher_good=True, fmt=".0f")
                    _metric_vs(_cols[2], "負債比率", _u_debt, _b_debt, unit="%", higher_good=False)

                    if _b_roa is None and _b_curr is None and _b_debt is None:
                        st.caption("業種ベンチマークデータなし。industry_benchmarks.json を確認してください。")

            # ----- SHAP 判定根拠の可視化 -----
            st.divider()
            with st.expander("🔍 SHAP 判定根拠の可視化（説明可能AI）", expanded=False):
                try:
                    from components.shap_explanation import render_shap_explanation
                    _shap_cases = load_all_cases()
                    _shap_case = next(
                        (c for c in _shap_cases if c.get("id") == current_case_id),
                        None
                    )
                    render_shap_explanation(current_case=_shap_case)
                except Exception as _shap_err:
                    st.warning(f"SHAP表示エラー: {_shap_err}")

            # ----- 審査に有用な Plotly グラフ（4種） -----
            st.divider()
            with st.expander("📊 審査に有用なグラフ", expanded=False):
                st.caption("スコア内訳・契約期待度の要因・過去分布・バランスシート内訳をインタラクティブに表示します。")
                row1_a, row1_b = st.columns(2)
                with row1_a:
                    st.plotly_chart(plot_score_models_comparison_plotly(res), use_container_width=True, key="plotly_score_models")
                with row1_b:
                    factors_fig = plot_contract_prob_factors_plotly(res.get("ai_completed_factors") or [])
                    if factors_fig:
                        st.plotly_chart(factors_fig, use_container_width=True, key="plotly_contract_factors")
                    else:
                        st.caption("契約期待度の要因は判定実行後に表示されます。")
                row2_a, row2_b = st.columns(2)
                with row2_a:
                    hist_fig = plot_past_scores_histogram_plotly(res.get("score"), load_all_cases())
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True, key="plotly_past_hist")
                    else:
                        st.caption("過去案件データがあるとスコア分布を表示します。")
                with row2_b:
                    bal_fig = plot_balance_sheet_plotly(res.get("financials"))
                    if bal_fig:
                        st.plotly_chart(bal_fig, use_container_width=True, key="plotly_balance_sheet")
                    else:
                        st.caption("審査入力で資産・負債を入力すると内訳を表示します。")
                # ----- 追加グラフ（4種）-----
                st.divider()
                st.caption("📌 追加分析グラフ（返済余力・財務比率・スコア分布・CF構造）")
                row3_a, row3_b = st.columns(2)
                with row3_a:
                    ebitda_fig = plot_ebitda_coverage_plotly(res.get("financials"))
                    if ebitda_fig:
                        st.plotly_chart(ebitda_fig, use_container_width=True, key="plotly_ebitda_cov")
                    else:
                        st.caption("財務データを入力するとEBITDAカバレッジを表示します。")
                with row3_b:
                    bullet_fig = plot_financial_bullet_plotly(res, avg_data)
                    if bullet_fig:
                        st.plotly_chart(bullet_fig, use_container_width=True, key="plotly_fin_bullet")
                    else:
                        st.caption("業界データがあると財務指標比較を表示します。")
                row4_a, row4_b = st.columns(2)
                with row4_a:
                    box_fig = plot_score_boxplot_plotly(res.get("score"), selected_sub, load_all_cases())
                    if box_fig:
                        st.plotly_chart(box_fig, use_container_width=True, key="plotly_score_box")
                    else:
                        st.caption("過去案件データが蓄積されるとスコアボックスプロットを表示します。")
                with row4_b:
                    cf_fig = plot_cash_flow_bridge_plotly(res.get("financials"))
                    if cf_fig:
                        st.plotly_chart(cf_fig, use_container_width=True, key="plotly_cf_bridge")
                    else:
                        st.caption("財務データを入力するとCFブリッジを表示します。")

            st.divider()
            with st.container():
                st.subheader(":round_pushpin: 3D多角分析（回転・拡大可能）")
                st.caption("過去事例と今回案件を3軸で比較。★今回の案件の位置を確認してください。")
                _fin3d = res.get("financials", {})
                current_case_data = {
                    "sales": _fin3d.get("nenshu", 0) or 0,
                    "op_margin": res.get("user_op", 0) or 0,
                    "equity_ratio": res.get("user_eq", 0) or 0,
                    "op_profit": _fin3d.get("op_profit") or _fin3d.get("rieki", 0) or 0,
                    "depreciation": _fin3d.get("depreciation", 0) or 0,
                    "lease_credit": _fin3d.get("lease_credit", 0) or 0,
                    "bank_credit": _fin3d.get("bank_credit", 0) or 0,
                    "score": res.get("score", 0) or 0,
                }
                past_cases_log = load_all_cases()
                _3d_col1, _3d_col2, _3d_col3 = st.columns(3)
                with _3d_col1:
                    fig_3d_1 = plot_3d_profit_position(current_case_data, past_cases_log)
                    if fig_3d_1:
                        st.plotly_chart(fig_3d_1, use_container_width=True, key="plotly_3d_v1")
                        st.caption("① 売上 × 利益率 × 自己資本比率")
                    else:
                        st.caption("①過去データ不足")
                with _3d_col2:
                    fig_3d_2 = plot_3d_repayment(current_case_data, past_cases_log)
                    if fig_3d_2:
                        st.plotly_chart(fig_3d_2, use_container_width=True, key="plotly_3d_v2")
                        st.caption("② 売上 × EBITDAカバレッジ × スコア")
                    else:
                        st.caption("②過去データ不足")
                with _3d_col3:
                    fig_3d_3 = plot_3d_safety_score(current_case_data, past_cases_log)
                    if fig_3d_3:
                        st.plotly_chart(fig_3d_3, use_container_width=True, key="plotly_3d_v3")
                        st.caption("③ 自己資本比率 × 利益率 × スコア")
                    else:
                        st.caption("③過去データ不足")

                # ----- 3D AIポジショニングコメント（チャート下・全幅） -----
                _3d_comment_key = "ai_3d_comment_result"
                _3d_comment_id = f"3d_{current_case_data.get('score', 0):.0f}_{current_case_data.get('op_margin', 0):.1f}"
                if st.session_state.get("ai_3d_comment_id") != _3d_comment_id:
                    st.session_state[_3d_comment_key] = None
                    st.session_state["ai_3d_comment_id"] = _3d_comment_id
                if is_ai_available() and st.session_state.get(_3d_comment_key) is None:
                    with st.spinner("3D分析コメント生成中…"):
                        _3d_c = get_ai_3d_comment(current_case_data, past_cases_log)
                    st.session_state[_3d_comment_key] = _3d_c if _3d_c else ""
                _3d_c_text = st.session_state.get(_3d_comment_key) or ""
                if _3d_c_text:
                    st.info(f"🤖 **ポジショニング分析** {_3d_c_text}")
                elif not is_ai_available():
                    st.caption("💬 ポジショニングコメント: サイドバーでAIを設定すると自動表示されます。")

            st.divider()
            with st.container():
                st.subheader("🔮 審査突破のためのAIアドバイス")
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    st.subheader("📋 類似案件の「勝ちパターン」")
                    # -----------------------------------------------------
                    # [SAFETY] Ensure variables are defined for list comprehension
                    if "res" in locals():
                        selected_major = res.get("industry_major", "D 建設業")
                        score_percent = res.get("score", 0)
                    else:
                        if "last_result" in st.session_state:
                            res_safety = st.session_state["last_result"]
                            selected_major = res_safety.get("industry_major", "D 建設業")
                            score_percent = res_safety.get("score", 0)
                        else:
                            selected_major = "D 建設業"
                            score_percent = 0
                    # -----------------------------------------------------
                    similar_success_cases = []
                    if load_all_cases():
                        cases = load_all_cases()
                        # -----------------------------------------------------
                        # [SAFETY] Ensure variables are defined for list comprehension
                        if "res" in locals():
                            selected_major = res.get("industry_major", "D 建設業")
                            score_percent = res.get("score", 0)
                        else:
                            if "last_result" in st.session_state:
                                res_safety = st.session_state["last_result"]
                                selected_major = res_safety.get("industry_major", "D 建設業")
                                score_percent = res_safety.get("score", 0)
                            else:
                                selected_major = "D 建設業"
                                score_percent = 0
                        try:
                            from bayesian_engine import THRESHOLD_APPROVAL
                            approval_line = THRESHOLD_APPROVAL * 100
                        except ImportError:
                            approval_line = 70.0

                        similar_success_cases = [
                            c for c in cases 
                            if c.get("industry_major") == selected_major
                            and abs(c.get("result", {}).get("score", 0) - score_percent) < 15
                            and c.get("result", {}).get("score", 0) >= approval_line
                        ]

                    if similar_success_cases:
                        st.info(f"スコアや業種が似ている承認事例が {len(similar_success_cases)} 件見つかりました。")
                        for i, c in enumerate(similar_success_cases[:3]): 
                            with st.expander(f"事例{i+1}: {c.get('industry_sub')} (スコア {c['result']['score']:.0f})"):
                                summary = c.get("chat_summary", "詳細なし")
                                st.write(f"**承認の決め手**: {summary}")
                    else:
                        st.warning("条件の近い成功事例はまだありません。")
                        # ノウハウデータからの代替提案
                        if "qualitative_appeal" in knowhow_data:
                            st.markdown("**💡 一般的な定性アピールのヒント:**")
                            for k in knowhow_data["qualitative_appeal"]:
                                st.caption(f"- **{k['title']}**: {k['content']}")

                with col_adv2:
                    st.subheader("🔧 決算書・スキーム調整のヒント")
                    advice_list = []
                    # ノウハウデータからの引用ロジック
                    if knowhow_data:
                        # 財務改善
                        if user_equity_ratio < 20 and "financial_improvement" in knowhow_data:
                            k = knowhow_data["financial_improvement"][0] # 役員借入金
                            advice_list.append(f"💡 **{k['title']}**: {k['content']}")
                        if user_op_margin < 0 and "financial_improvement" in knowhow_data:
                            k = knowhow_data["financial_improvement"][1] # 赤字除外
                            advice_list.append(f"💡 **{k['title']}**: {k['content']}")
                        # スキーム
                        if score_percent < 60 and "scheme_strategy" in knowhow_data:
                            k = knowhow_data["scheme_strategy"][1] # 連帯保証
                            advice_list.append(f"🛡️ **{k['title']}**: {k['content']}")
                    # 業種別ノウハウ
                    ind_key = res["industry_major"].split(" ")[1] if " " in res["industry_major"] else res["industry_major"]
                    if "industry_specific" in knowhow_data and ind_key in knowhow_data["industry_specific"]:
                        advice_list.append(f"🏭 **{ind_key}の鉄則**: {knowhow_data['industry_specific'][ind_key]}")
                    if not advice_list:
                        advice_list.append("特段の懸念点はありません。定性面（導入効果）の強化に集中してください。")
                    for advice in advice_list:
                        st.success(advice)
                    # 該当業種の補助金（URLで公式サイトにすぐ飛べる）
                    subs_adv = search_subsidies_by_industry(res.get("industry_sub", ""))
                    if subs_adv:
                        with st.expander("📎 該当業種の補助金（クリックで公式サイトへ）", expanded=False):
                            for s in subs_adv:
                                name = s.get("name") or ""
                                url = (s.get("url") or "").strip()
                                if url:
                                    st.markdown(f"**{name}**")
                                    try:
                                        st.link_button("🔗 公式サイトを開く", url, type="secondary")
                                    except Exception:
                                        safe_url = url.replace('"', "%22").replace("'", "%27")
                                        st.markdown(f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">🔗 公式サイトを開く</a>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f"**{name}**")
                                st.caption((s.get("summary") or "")[:100] + "…")
                                st.caption(f"申請目安: {s.get('application_period')}")

                # ======================================================================
                # 📚 この案件に紐づくニュース（詳細はエキスパンダー）
                # ======================================================================
                with st.expander("📚 この案件に紐づくニュース", expanded=False):
                    if current_case_id:
                        case_news_list = load_case_news(current_case_id)
                        if case_news_list:
                            for idx, news in enumerate(case_news_list):
                                with st.expander(f"{idx+1}. {news.get('title', 'タイトル不明')}"):
                                    st.caption(f"保存日時: {news.get('saved_at', 'N/A')}")
                                    if news.get("url"):
                                        st.markdown(f"[記事URLを開く]({news['url']})")
                                    content_preview = (news.get("content") or "")[:300]
                                    if content_preview:
                                        st.write(content_preview + ("..." if len(news.get("content", "")) > 300 else ""))
                                    if st.button("このニュースをAIに反映する", key=f"use_news_{idx}"):
                                        st.session_state.selected_news_content = {"title": news.get("title", ""), "content": news.get("content", "")}
                                        st.success("このニュースを、以降のAIアドバイス・ディベートで参照するように設定しました。")
                        else:
                            st.caption("この案件には、まだ紐づけられたニュースがありません。")
                    else:
                        st.caption("案件IDが未取得のため、紐づくニュースを特定できません。")

            st.divider()
            st.markdown("### 📊 財務ベンチマーク分析")
            # 1. 財務レーダーチャートの準備
            # 簡易偏差値ロジック (平均=50, 標準偏差=適当に仮定)
            def calc_hensachi(val, mean, is_higher_better=True):
                if mean == 0: return 50
                diff = (val - mean) / abs(mean) * 10 * (1 if is_higher_better else -1)
                return max(20, min(80, 50 + diff))

            radar_metrics = {
                "収益性": calc_hensachi(res['user_op'], res['bench_op']),
                "安全性": calc_hensachi(res['user_eq'], res['bench_eq']),
                "効率性": 50, # 仮
                "成長性": 50, # 仮
                "返済力": 50  # 仮
            }
            radar_bench = {k: 50 for k in radar_metrics.keys()}

            # 2. 過去案件データ取得
            past_cases = load_all_cases()

            # 3. グラフ描画エリア（PCで大きくなりすぎないよう幅を制限）
            col_graphs, _ = st.columns([0.65, 0.35])
            with col_graphs:
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(plot_radar_chart_plotly(radar_metrics, radar_bench), use_container_width=True, key="radar_analysis")
                with g2:
                    # 損益分岐点グラフ
                    sales_k = res["financials"]["nenshu"]
                    gross_k = res["financials"]["gross_profit"] * 1000
                    op_k = res["financials"]["rieki"] * 1000
                    vc = sales_k - gross_k
                    fc = gross_k - op_k
                    bep_fig = plot_break_even_point_plotly(sales_k, vc, fc)
                    if bep_fig:
                        st.plotly_chart(bep_fig, use_container_width=True, key="bep_analysis")
                    else:
                        fallback = plot_break_even_point(sales_k, vc, fc)
                        if fallback:
                            st.pyplot(fallback)

            # ========== 中分類ごとにネットで業界目安を取得して比較 ==========
            selected_sub = res.get("industry_sub", "")
            bench = dict(benchmarks_data.get(selected_sub, {}))
            try:
                web_bench = fetch_industry_benchmarks_from_web(selected_sub)
                for k in _WEB_BENCH_KEYS:
                    if web_bench.get(k) is not None:
                        bench[k] = web_bench[k]
            except Exception:
                web_bench = {"snippets": [], "op_margin": None, "equity_ratio": None}

            with st.expander("🌐 中分類ごとにネットで調べた業界目安", expanded=False):
                st.caption(f"業種「{selected_sub}」の業界目安です。結果は web_industry_benchmarks.json に保存され、毎年4月1日を境に1年ごとに再検索します。営業利益率・自己資本比率・売上高総利益率・ROA・流動比率など抽出できた指標は、下の「算出可能指標」の業界目安に反映します。")
                if web_bench.get("snippets"):
                    for i, s in enumerate(web_bench["snippets"]):
                        st.markdown(f"**[{s['title']}]({s['href']})**")
                        st.caption(s["body"][:200] + ("..." if len(s["body"]) > 200 else ""))
                        st.divider()
                    extracted = [(k, web_bench[k]) for k in _WEB_BENCH_KEYS if web_bench.get(k) is not None]
                    if extracted:
                        u = lambda k: "回" if k in ("asset_turnover", "fixed_asset_turnover") else "%"
                        parts = [f"{k}: {v:.1f}{u(k)}" for k, v in extracted]
                        st.success("抽出した業界目安: " + ", ".join(parts[:8]) + (" …" if len(parts) > 8 else ""))
                else:
                    st.caption("検索結果がありません。ネットワークまたは検索キーワードを確認してください。")

            with st.expander("📈 業界トレンド（拡充）", expanded=False):
                st.markdown(trend_info or "業界トレンドのデータがありません。")
                if st.button("📡 この業種のトレンドをネットで検索して拡充", key="btn_extend_trend"):
                    with st.spinner("検索中…"):
                        try:
                            fetch_industry_trend_extended(selected_sub, force_refresh=True)
                            st.success("拡充しました。表示を更新します。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"検索エラー: {e}")

            # ========== 算出可能指標（入力から計算した有効指標） ==========
            st.markdown("### 📈 算出可能指標")
            with st.expander("ℹ️ 業界目安の出典", expanded=False):
                st.caption("業界目安は、ネット検索で保存した値（web_industry_benchmarks.json）を優先し、不足分を大分類の業界平均（industry_averages.json）で補っています。サイドバー「今のデータを検索して保存」で指標の業界目安も検索・保存できます。")
            fin = res.get("financials", {})
            # 業界目安を業界平均（大分類）で補強（取れるだけ追加）
            bench_ext = dict(bench) if bench else {}
            major = res.get("industry_major")
            if major and avg_data and major in avg_data:
                avg = avg_data[major]
                an = avg.get("nenshu") or 0
                if an > 0:
                    if bench_ext.get("gross_margin") is None:
                        bench_ext["gross_margin"] = (avg.get("gross_profit") or 0) / an * 100
                    if bench_ext.get("ord_margin") is None:
                        bench_ext["ord_margin"] = (avg.get("ord_profit") or 0) / an * 100
                    if bench_ext.get("net_margin") is None:
                        bench_ext["net_margin"] = (avg.get("net_income") or 0) / an * 100
                    if bench_ext.get("dep_ratio") is None:
                        bench_ext["dep_ratio"] = (avg.get("depreciation") or 0) / an * 100
                total_avg = (avg.get("machines") or 0) + (avg.get("other_assets") or 0) + (avg.get("bank_credit") or 0) + (avg.get("lease_credit") or 0)
                if total_avg > 0:
                    if bench_ext.get("roa") is None:
                        bench_ext["roa"] = (avg.get("net_income") or 0) / total_avg * 100
                    if bench_ext.get("asset_turnover") is None:
                        bench_ext["asset_turnover"] = an / total_avg
                    if bench_ext.get("fixed_ratio") is None:
                        bench_ext["fixed_ratio"] = ((avg.get("machines") or 0) + (avg.get("other_assets") or 0)) / total_avg * 100
                    if bench_ext.get("debt_ratio") is None:
                        bench_ext["debt_ratio"] = ((avg.get("bank_credit") or 0) + (avg.get("lease_credit") or 0)) / total_avg * 100
            indicators = compute_financial_indicators(fin, bench_ext)
            if indicators:
                # 業界目安より良い＝緑、悪い＝赤（LOWER_IS_BETTER_NAMES は低い方が良い）
                cell_style = "text-align:center; vertical-align:middle; padding:4px 6px;"
                rows_html = []
                for ind in indicators:
                    name = ind["name"]
                    value = ind["value"]
                    unit = ind.get("unit", "%")
                    bench = ind.get("bench")
                    bench_ok = bench is not None and (not isinstance(bench, float) or bench == bench)
                    if bench_ok:
                        diff = value - bench
                        is_good = (diff > 0 and name not in LOWER_IS_BETTER_NAMES) or (diff < 0 and name in LOWER_IS_BETTER_NAMES)
                        color = "#22c55e" if is_good else "#ef4444"
                        row_bg = "background-color:rgba(34,197,94,0.18);" if is_good else "background-color:rgba(239,68,68,0.12);"
                        name_cell = f'<span style="color:{color}; font-weight:600;">{name.replace("&", "&amp;").replace("<", "&lt;")}</span>'
                    else:
                        row_bg = ""
                        name_cell = name.replace("&", "&amp;").replace("<", "&lt;")
                    bench_str = f"{bench:.1f}{unit}" if bench_ok else "—"
                    rows_html.append(f"<tr style='{row_bg}'><td style='{cell_style}'>{name_cell}</td><td style='{cell_style}'>{value:.1f}{unit}</td><td style='{cell_style}'>{bench_str}</td></tr>")
                table_html = (
                    "<table style='border-collapse:collapse; font-size:0.8rem; line-height:1.2; table-layout:fixed; width:100%;'>"
                    "<colgroup><col style='width:52%'><col style='width:24%'><col style='width:24%'></colgroup>"
                    "<thead><tr>"
                    f"<th style='{cell_style} font-weight:600;'>指標</th><th style='{cell_style} font-weight:600;'>貴社</th><th style='{cell_style} font-weight:600;'>業界目安</th>"
                    "</tr></thead><tbody>"
                    + "".join(rows_html) + "</tbody></table>"
                )
                st.markdown(
                    "<div style='max-width:400px; margin:0.25rem 0; overflow-x:auto;'>" + table_html + "</div>",
                    unsafe_allow_html=True,
                )
                st.caption("緑＝業界より良い / 赤＝要確認")
                # 指標と業界目安の差の分析（図＋文章＋AIによる指標の分析）
                summary, detail = analyze_indicators_vs_bench(indicators)
                st.markdown("#### 📊 差の分析")
                col_sum, col_fig = st.columns([1, 1])
                with col_sum:
                    st.info(summary)
                fig_gap = plot_indicators_gap_analysis_plotly(indicators)
                with col_fig:
                    if fig_gap:
                        st.plotly_chart(fig_gap, use_container_width=True, key="indicators_gap")
                # 指標の分析（AI）：同一案件のキャッシュがあれば表示、なければボタンで生成
                _case_id = st.session_state.get("current_case_id")
                _cached = st.session_state.get("indicator_ai_analysis")
                _cached_case = st.session_state.get("indicator_ai_analysis_case_id")
                if _cached and _cached_case == _case_id:
                    st.markdown("##### 指標の分析（AI）")
                    st.markdown(_cached)
                else:
                    st.markdown("##### 指標の分析（AI）")
                    if st.button("AIに指標の分析を生成", key="gen_indicator_ai"):
                        if not is_ai_available():
                            if st.session_state.get("ai_engine") == "gemini":
                                st.error("Gemini APIキーを設定してください。")
                            else:
                                st.error("Ollama が起動していないか、Gemini に切り替えてください。")
                        else:
                            ind_list = "\n".join([f"- {x['name']}: 貴社 {x['value']:.1f}{x.get('unit','%')} / 業界目安 {x['bench']:.1f}{x.get('unit','%')}" if x.get("bench") is not None else f"- {x['name']}: 貴社 {x['value']:.1f}{x.get('unit','%')}" for x in indicators])
                            prompt = f"""あなたはリース審査のプロです。以下の「指標と業界目安の差の分析」を踏まえ、この企業の財務指標について2〜4文で簡潔に分析してください。
・強み（業界目安を上回っている点）があれば触れる。
・業界目安を下回っている指標があれば、なぜそうなっている可能性があるか・改善の方向性を1〜2文で述べる。
・借入金等依存度・固定比率など「低い方が良い」指標の解釈も含める。
数値は既にまとめにあるので、重複せず要点だけ書いてください。

【要約】
{summary}

【差の内訳】
{detail}

【指標一覧】
{ind_list}
"""
                            with st.spinner("AIが指標を分析しています..."):
                                try:
                                    ans = chat_with_retry(model=get_ollama_model(), messages=[{"role": "user", "content": prompt}], timeout_seconds=90)
                                    content = (ans.get("message") or {}).get("content", "")
                                    if content and "APIキーが" not in content and "エラー" not in content[:50]:
                                        st.session_state["indicator_ai_analysis"] = content
                                        st.session_state["indicator_ai_analysis_case_id"] = _case_id
                                        st.rerun()
                                    else:
                                        st.error(content or "AIの応答を取得できませんでした。")
                                except Exception as e:
                                    st.error(f"分析の生成に失敗しました: {e}")
                    else:
                        st.caption("上の「AIに指標の分析を生成」を押すと、業界目安との差を踏まえた分析文をAIが生成します。")
                    st.caption("左＝要確認 / 右＝良い。借入金等依存度・減価償却費/売上は低いと緑。")
                    with st.expander("差の内訳（数値）", expanded=False):
                        st.markdown(detail)
                    # 利益構造（ウォーターフォール）
                    nenshu_k = fin.get("nenshu") or 0
                    gross_k = fin.get("gross_profit") or 0
                    op_k = fin.get("rieki") or fin.get("op_profit") or 0
                    ord_k = fin.get("ord_profit") or 0
                    net_k = fin.get("net_income") or 0
                    if nenshu_k > 0:
                        st.markdown("#### 利益構造")
                        col_wf, _ = st.columns([0.65, 0.35])
                        with col_wf:
                            st.plotly_chart(plot_waterfall_plotly(nenshu_k, gross_k, op_k, ord_k, net_k), use_container_width=True, key="waterfall_result")
            else:
                st.caption("指標を算出するには、審査入力で売上高・損益・資産などを入力してください。")

            # 新機能: 業界情報の自動収集と分析アドバイス
            st.divider()
            st.subheader("📈 AI業界分析アドバイス")
            st.caption("自動収集したWeb上の最新業界情報や財務目安をもとに、本件特有の着眼点をAIがアドバイスします。")
            
            advice_case_id = st.session_state.get("ai_industry_advice_case_id")
            advice_text = st.session_state.get("ai_industry_advice_text")
            selected_sub_res = res.get("industry_sub", "")
            comp_text = res.get("comparison", "")

            if advice_text and advice_case_id == current_case_id:
                st.success("✨ **AI業界分析アドバイス**\n\n" + advice_text)
                if st.button("アドバイスを再生成", key="btn_advice_regenerate"):
                    st.session_state["ai_industry_advice_text"] = None
                    st.session_state["ai_industry_advice_case_id"] = None
                    st.rerun()
            else:
                if st.button("▶️ 業界情報の自動収集・分析を実行", key="btn_advice_generate"):
                    with st.spinner("Webから最新の業界情報を収集・分析中... ⏳"):
                        # 新規作成したAIチャット関数をインポートして呼び出す
                        from ai_chat import get_ai_industry_advice
                        from data_cases import update_case_field
                        text = get_ai_industry_advice(selected_sub_res, comp_text)
                        if text:
                            st.session_state["ai_industry_advice_text"] = text
                            st.session_state["ai_industry_advice_case_id"] = current_case_id
                            update_case_field(current_case_id, "ai_industry_advice", text)
                            st.rerun()
                        else:
                            st.error("アドバイスの生成に失敗しました。AIサーバーやAPIキーの設定をご確認ください。")

            # AIのぼやき（ネット検索した業界情報を使いAIが自分で生成・アップデート）+ 定例の愚痴
            st.divider()
            st.subheader("🤖 AIのぼやき")
            u_eq = res.get("user_eq", 0)
            u_op = res.get("user_op", 0)
            comp_text = res.get("comparison", "")
            net_risk = res.get("network_risk_summary", "") or ""
            selected_sub_res = res.get("industry_sub", "")
            byoki_case_id = st.session_state.get("ai_byoki_case_id")
            byoki_text = st.session_state.get("ai_byoki_text")
            if byoki_text and byoki_case_id == current_case_id:
                st.info("🐟 " + byoki_text)
                if st.button("ぼやきを再生成（業界情報を再取得）", key="btn_byoki_regenerate"):
                    st.session_state["ai_byoki_text"] = None
                    st.session_state["ai_byoki_case_id"] = None
                    st.rerun()
            else:
                if st.button("AIにぼやきを言わせる（業界情報を参照）", key="btn_byoki_generate"):
                    with st.spinner("業界情報を取得して、AIがぼやきを考えています…"):
                        from data_cases import update_case_field
                        text = get_ai_byoki_with_industry(selected_sub_res, u_eq, u_op, comp_text, net_risk)
                        if text:
                            st.session_state["ai_byoki_text"] = text
                            st.session_state["ai_byoki_case_id"] = current_case_id
                            update_case_field(current_case_id, "ai_byoki", text)
                            st.rerun()
                        else:
                            st.error("生成できませんでした。APIキー・Ollamaを確認してください。")
                if not byoki_text:
                    st.caption("上のボタンで、ネット検索した業界情報をもとにAIが愚痴を1つ生成します。")

            # ----- カードバトル（別枠・開発中） -----
            with st.expander("⚔️ 審査委員会カードバトル（開発中）", expanded=False):
                st.caption("判定結果をカードバトル風に振り返ります。仕様は変更される可能性があります。")
                if "battle_data" in st.session_state and res:
                    bd = st.session_state["battle_data"]
                    if bd.get("special_move_name") is None:
                        strength_tags = res.get("strength_tags") or []
                        passion_text = res.get("passion_text") or ""
                        name, effect = generate_battle_special_move(strength_tags, passion_text)
                        bd["special_move_name"] = name
                        bd["special_effect"] = effect
                        score = bd.get("score", 0)
                        log_lines = [
                            "【実況】審査委員会、開廷。",
                            "慎重派「数値だけ見ると厳しいが、業界相対で見るべきだ。」",
                            f"推進派「スコア{score:.0f}%。逆転材料があれば十分戦える。」" if score < 75 else "推進派「スコアは十分圏内。定性面を確認しよう。」",
                            "【議事】定性エビデンスを検討中…",
                        ]
                        similar_prompt = res.get("similar_past_cases_prompt", "")
                        if similar_prompt and "過去の類似案件" in similar_prompt:
                            log_lines.append("慎重派「過去の類似案件を参照した。同様のケースでは成約例あり。」")
                        log_lines.append("【判定】採決に入ります。")
                        bd["battle_log"] = log_lines
                        bd["dice"] = random.randint(1, 6)
                        st.session_state["battle_data"] = bd
                    bd = st.session_state["battle_data"]
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#334155 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;">
                        <div style="font-size:0.85rem;opacity:0.9;">HP</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['hp']}</div>
                        <div style="font-size:0.75rem;">自己資本</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#b45309 0%,#c2410c 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;">
                        <div style="font-size:0.85rem;opacity:0.9;">ATK</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['atk']}</div>
                        <div style="font-size:0.75rem;">利益率</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#0d9488 0%,#0f766e 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;">
                        <div style="font-size:0.85rem;opacity:0.9;">SPD</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['spd']}</div>
                        <div style="font-size:0.75rem;">流動性</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("**🎴 必殺技**")
                    st.markdown(f"""
                    <div style="background:#f8fafc;border:2px solid #b45309;border-radius:10px;padding:1rem;">
                    <span style="font-weight:bold;color:#1e3a5f;">{bd.get('special_move_name', '逆転の意気')}</span>
                    <span style="color:#64748b;"> … </span>
                    <span>{bd.get('special_effect', 'スコア+5%')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    for eff in (bd.get("environment_effects") or []):
                        st.caption(f"• {eff}")
                    st.markdown("**📜 バトル実況**")
                    for line in bd.get("battle_log", []):
                        st.caption(line)
                    dice = bd.get("dice") or 1
                    st.caption(f"🎲 運命のダイス: **{dice}** → {'やや有利' if dice >= 4 else 'やや不利'}")
                    if bd.get("is_approved"):
                        st.success("🏆 WIN — 承認圏内")
                    else:
                        st.info("📋 LOSE — 要審議")
                else:
                    st.caption("判定を実行すると、ここにカードバトルが表示されます。")

        else:
            st.info('👈 左側の「審査入力」タブでデータを入力し、審査を実行してください。')
