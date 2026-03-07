"""
温水式リース審査AI - lease_logic_sumaho12
sumaho10(X) からモジュール分割（ai_chat / web_services）を完了した版。
起動: streamlit run lease_logic_sumaho12/lease_logic_sumaho12.py （リポジトリルートで実行）
"""
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import streamlit as st
try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:
    style_metric_cards = None  # pip install streamlit-extras でメトリックをカード風に
import math
import json
import random
import re
import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import datetime
from coeff_definitions import (
    COEFFS,
    BAYESIAN_PRIOR_EXTRA,
    STRENGTH_TAG_WEIGHTS,
    DEFAULT_STRENGTH_WEIGHT,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from charts import (
    CHART_STYLE,
    LOWER_IS_BETTER_NAMES,
    _equity_ratio_display,
    plot_balance_sheet_plotly,
    plot_benchmark_comparison,
    plot_break_even_point,
    plot_break_even_point_plotly,
    plot_contract_prob_factors_plotly,
    plot_gauge,
    plot_gauge_plotly,
    plot_indicators_bar,
    plot_indicators_gap_analysis,
    plot_indicators_gap_analysis_plotly,
    plot_past_scores_histogram_plotly,
    plot_positioning_scatter,
    plot_radar_chart,
    plot_radar_chart_plotly,
    plot_scoring_top5_factors_plotly,
    plot_score_models_comparison_plotly,
    plot_3d_analysis,
    plot_3d_profit_position,
    plot_3d_repayment,
    plot_3d_safety_score,
    plot_waterfall,
    plot_waterfall_plotly,
    plot_ebitda_coverage_plotly,
    plot_financial_bullet_plotly,
    plot_score_boxplot_plotly,
    plot_cash_flow_bridge_plotly,
)
from data_cases import (
    CASES_FILE,
    CASE_NEWS_FILE,
    CONSULTATION_MEMORY_FILE,
    COEFF_OVERRIDES_FILE,
    DEFAULT_WEIGHT_QUAL,
    DEFAULT_WEIGHT_QUANT,
    append_case_news,
    append_consultation_memory,
    find_similar_past_cases,
    get_effective_coeffs,
    get_score_weights,
    load_all_cases,
    load_case_news,
    load_consultation_memory,
    load_coeff_overrides,
    load_past_cases,
    save_all_cases,
    save_case_log,
    save_coeff_overrides,
)
from analysis_regression import (
    BENCH_BASES,
    COEFF_EXTRA_KEYS,
    COEFF_LABELS,
    COEFF_MAIN_KEYS,
    INDUSTRY_BASES,
    INDUSTRY_MODEL_KEYS,
    INDICATOR_MAIN_KEYS,
    INDICATOR_MODEL_KEYS,
    PRIOR_COEFF_MODEL_KEYS,
    QUALITATIVE_ANALYSIS_MIN_CASES,
    build_design_matrix_from_logs,
    build_design_matrix_indicator_from_logs,
    optimize_score_weights_from_regression,
    run_contract_driver_analysis,
    run_qualitative_contract_analysis,
    run_quantitative_by_indicator,
    run_quantitative_by_industry,
    run_quantitative_contract_analysis,
    run_regression_and_get_coeffs,
    run_regression_indicator_and_get_coeffs,
)

from ai_chat import (
    OLLAMA_MODEL,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
    _chat_result_holder,
    _get_gemini_key_from_secrets,
    get_ollama_model,
    _ollama_chat_http,
    _gemini_chat,
    _chat_for_thread,
    chat_with_retry,
    generate_battle_special_move,
    is_ai_available,
    is_ollama_available,
    run_ollama_connection_test,
    save_debate_log,
    AI_HONNE_SYSTEM,
    get_ai_byoki_with_industry,
    get_ai_honne_complaint,
    get_ai_comprehensive_evaluation,
    get_ai_quick_comment,
    get_ai_3d_comment,
)
from indicators import (
    compute_financial_indicators,
    analyze_indicators_vs_bench,
    get_indicator_analysis_for_advice,
    calculate_pd,
)
from report_pdf import build_contract_report_pdf
from report_generator import generate_full_report_from_res
from rule_manager import load_business_rules, save_business_rules, evaluate_custom_rules
from knowledge import build_knowledge_context, search_faq, search_cases, get_improvement_guide_text
from web_services import (
    _WEB_BENCH_KEYS,
    _get_benchmark_cutoff_date,
    _load_web_benchmarks_cache,
    _save_web_benchmark,
    _load_json_cache,
    _save_json_cache,
    trends_extended_cache,
    assets_benchmarks_cache,
    sales_band_cache,
    _ensure_web_caches_loaded,
    fetch_industry_trend_extended,
    fetch_industry_assets_from_web,
    fetch_sales_band_benchmarks,
    get_trend_extended,
    get_assets_benchmark,
    get_sales_band_text,
    get_all_industry_sub_for_benchmarks,
    search_subsidies_by_industry,
    search_equipment_by_keyword,
    get_lease_classification_text,
    scrape_article_text,
    is_japanese_text,
    get_advice_context_extras,
    get_market_rate,
    search_bankruptcy_trends,
    search_latest_trends,
    _parse_benchmark_number,
    fetch_industry_benchmarks_from_web,
    get_stats,
)
from components.asset_finance import AssetFinanceEngine
def red_label(placeholder, text):
    # display: block にして、一つ一つのスライダーセットの範囲を明確にします
    placeholder.markdown(f'''
        <div style="
            text-align: right; 
            color: #FF0000; 
            font-size: 20px; 
            font-weight: bold;
            margin-bottom: -40px;
            padding-right: 5px;
            line-height: 1;
        ">
            {text}
        </div>
    ''', unsafe_allow_html=True)


def _slider_and_number(field_name, key_prefix, default, min_val, max_val, step_slider, step_num=None, fmt="{:,}", unit="千円", label_slider="売上高調整", max_val_number=None):
    """スライダーと数値入力の両方に対応。後から動かした方を採用値とする。
    on_change を使わないため st.form 内でも動作する。"""
    if step_num is None:
        step_num = step_slider
    num_max = max_val_number if max_val_number is not None else max_val

    if field_name not in st.session_state:
        st.session_state[field_name] = default
    cur = st.session_state[field_name]

    prev_key = f"_san_prev_{key_prefix}"
    num_key = f"num_{key_prefix}"
    slide_key = f"slide_{key_prefix}"
    prev_num_key = f"_san_prev_num_{key_prefix}"
    prev_slide_key = f"_san_prev_slide_{key_prefix}"
    externally_changed = st.session_state.get(prev_key) != cur

    if num_key not in st.session_state or externally_changed:
        st.session_state[num_key] = max(min_val, min(cur, num_max))
    if slide_key not in st.session_state or externally_changed:
        st.session_state[slide_key] = max(min_val, min(cur, max_val))

    c_l, c_r = st.columns([0.7, 0.3])
    with c_r:
        st.number_input("直接入力", min_value=min_val, max_value=num_max,
                        step=step_num, key=num_key,
                        label_visibility="collapsed")
    with c_l:
        st.slider(label_slider, min_value=min_val, max_value=max_val,
                  step=step_slider, key=slide_key,
                  label_visibility="collapsed")

    new_num = st.session_state[num_key]
    new_slide = st.session_state[slide_key]
    prev_num = st.session_state.get(prev_num_key, new_num)
    prev_slide = st.session_state.get(prev_slide_key, new_slide)

    num_changed = new_num != prev_num
    slide_changed = new_slide != prev_slide
    if num_changed and not slide_changed:
        adopted = new_num
    elif slide_changed and not num_changed:
        adopted = new_slide
    elif num_changed and slide_changed:
        adopted = new_num  # 両方変わった場合は数値入力優先（より精密）
    else:
        adopted = cur

    st.session_state[field_name] = adopted
    st.session_state[prev_key] = adopted
    st.session_state[prev_num_key] = new_num
    st.session_state[prev_slide_key] = new_slide
    st.caption(f"**採用値: {fmt.format(adopted)} {unit}**")
    return adopted


def _reset_shinsa_inputs():
    """全入力フィールドをデフォルト値にリセットする。「新しく入力する」ボタン用。"""
    field_defaults = {
        "nenshu": 10000,
        "item9_gross": 10000,
        "rieki": 10000,
        "item4_ord_profit": 10000,
        "item5_net_income": 10000,
        "item10_dep": 10000,
        "item11_dep_exp": 10000,
        "item8_rent": 10000,
        "item12_rent_exp": 10000,
        "item6_machine": 10000,
        "item7_other": 10000,
        "net_assets": 10000,
        "total_assets": 10000,
        "bank_credit": 10000,
        "lease_credit": 10000,
        "contracts": 1,
        "acquisition_cost": 1000,
        "lease_term": 60,
        "acceptance_year": 2026,
    }
    # field_name ← デフォルト値にリセット
    for k, v in field_defaults.items():
        st.session_state[k] = v
    # ウィジェットキー（num_* / slide_* / _san_prev_*）を削除して再初期化させる
    widget_prefixes = [
        "nenshuu", "sourieki", "rieki", "item4_ord_profit", "item5_net_income",
        "item10_dep", "item11_dep_exp", "item8_rent", "item12_rent_exp",
        "item6_machine", "item7_other", "net_assets", "total_assets",
        "bank_credit", "lease_credit", "contracts", "acquisition_cost",
    ]
    for pfx in widget_prefixes:
        for pre in ("num_", "slide_", "_san_prev_"):
            st.session_state.pop(f"{pre}{pfx}", None)
    # 定性スコアリングをリセット
    for k in list(st.session_state.keys()):
        if k.startswith("qual_corr_"):
            st.session_state[k] = 0
    # 最後の判定結果・送信入力をクリア
    for k in ("last_submitted_inputs", "last_result", "current_case_id",
               "selected_asset_index", "news_results", "selected_news_content"):
        st.session_state.pop(k, None)
    # チャット履歴もリセット（新しい案件の相談が前の案件で汚染されないよう）
    st.session_state["messages"] = []
    st.session_state["debate_history"] = []


# 以下はページ共通CSS（スライダー・グラフ・タブ・スマホ向けなど）
st.markdown("""
    <style>
    /* スライダー全体の幅をスマホで確保（最小幅・タップしやすく） */
    div[data-baseweb="slider"] {
        min-width: min(100%, 320px) !important;
        width: 100% !important;
    }
    @media (max-width: 640px) {
        div[data-baseweb="slider"] { min-width: 100% !important; }
        .stSlider > div { width: 100% !important; }
    }
    /* スライダーのつまみ（丸い部分）を大きくする */
    div[data-baseweb="slider"] div[role="slider"] {
        width: 30px !important;
        height: 30px !important;
        background-color: #FF0000 !important;
        border: 2px solid white !important;
    }
    /* スライダーの棒（レール）を太くする */
    div[data-baseweb="slider"] > div {
        height: 15px !important;
    }
    /* ラベル（売上高）の文字を大きくする */
    .stSlider label p {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    /* スライダーの上・横に表示される数値（現在値）を大きく見やすく */
    .stSlider {
        font-size: 1.5rem !important;
    }
    .stSlider [data-baseweb="slider"] {
        font-size: 1.5rem !important;
    }
    /* スライダー値表示エリア（Base Web の出力部分） */
    .stSlider > div > div:last-child,
    div[data-baseweb="slider"] ~ div {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    /* スライダーを動かしている時に出る数値（ツールチップ・つまみ上の表示）も大きく */
    [data-baseweb="tooltip"],
    .stSlider [data-baseweb="tooltip"],
    div[data-baseweb="slider"] [role="slider"] + div,
    div[data-baseweb="slider"] div[style*="position"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    /* スライダーボタン（つまみ）の上に表示される数字を特に大きく */
    [data-baseweb="tooltip"] span,
    [data-baseweb="tooltip"] div,
    .stSlider [data-baseweb="tooltip"] span,
    .stSlider [data-baseweb="tooltip"] div,
    div[data-baseweb="slider"] ~ [data-baseweb="tooltip"],
    [data-baseweb="popover"] span,
    [data-baseweb="popover"] div {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    .stSlider span,
    .stSlider div[data-baseweb="slider"] span {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }

    /* グラフ・図をカード風に（角丸・軽いシャドウ） */
    .stImage img, [data-testid="stImage"] img {
        border-radius: 10px !important;
        box-shadow: 0 2px 12px rgba(15,23,42,0.08) !important;
    }
    /* Plotly チャートも角丸 */
    .js-plotly-plot .plotly, [data-testid="stPlotlyChart"] div {
        border-radius: 10px !important;
    }
    /* PC: グラフはコンテナ幅いっぱいに表示（全部見えるように） */
    @media (min-width: 769px) {
        [data-testid="stPlotlyChart"] { max-width: 100% !important; width: 100% !important; margin-left: 0 !important; }
    }
    /* 右端が切れないように: メイン領域をフル幅・はみ出し表示許可 */
    section[data-testid="stSidebar"] + div,
    section.main,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > div:first-child,
    .block-container {
        max-width: 100% !important;
        width: 100% !important;
        overflow-x: visible !important;
        box-sizing: border-box !important;
    }
    .block-container {
        padding-right: 1.5rem !important;
    }
    /* スマホ・タブレット: 余白縮小でスクロール削減・モダンUI */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1.5rem !important;
    }
    @media (max-width: 768px) {
        .block-container { padding-top: 0.6rem !important; padding-bottom: 0.6rem !important; padding-left: 0.6rem !important; padding-right: 0.6rem !important; }
        [data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
        .stExpander { margin-bottom: 0.25rem !important; }
    }
    /* 左・右カラム（審査入力｜AI相談）: 右のAIオフィサー相談が切れないように */
    [data-testid="stHorizontalBlock"] {
        overflow-x: visible !important;
        max-width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child {
        min-width: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        overflow-x: visible !important;
        overflow-y: visible !important;
    }
    /* 右カラム（AI相談）は最低幅を確保し、切れないように */
    [data-testid="stHorizontalBlock"] > div:last-child {
        min-width: 320px !important;
        flex: 1 1 auto !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] > div:last-child .stChatMessage,
    [data-testid="stHorizontalBlock"] > div:last-child .stMarkdown {
        max-width: 100% !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    /* 右カラム内のコメント欄（相談内容 text_area）が右で切れないように */
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"],
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] textarea,
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] > div {
        max-width: 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stHorizontalBlock"] {
        max-width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child iframe {
        max-width: 100% !important;
    }
    /* 相談タブ内のテキストエリア全般（キー指定できないためラッパーで制約） */
    [data-testid="stTextArea"] {
        max-width: 100% !important;
    }
    [data-testid="stTextArea"] > div,
    [data-testid="stTextArea"] textarea {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    /* 右カラム・相談内容の欄に色をつける（ダッシュコード風） */
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        padding: 0.75rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #1e3a5f !important;
        box-shadow: 0 1px 3px rgba(30, 58, 95, 0.08) !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 8px !important;
    }
    /* トップメニュー用: タブ風スッキリ */
    [data-testid="stTabs"] > div > div { gap: 0 !important; }
    [data-testid="stTabs"] [role="tablist"] { margin-bottom: 0.5rem !important; }
    /* タブボタンのテキストを確実に表示（透明化バグ対策） */
    button[role="tab"] {
        color: #334155 !important;
        opacity: 1 !important;
    }
    button[role="tab"] p,
    button[role="tab"] span,
    button[role="tab"] div {
        color: #334155 !important;
        opacity: 1 !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #1e3a5f !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #1e3a5f !important;
    }
    button[role="tab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] span,
    button[role="tab"][aria-selected="true"] div {
        color: #1e3a5f !important;
        font-weight: 700 !important;
    }
    button[role="tab"]:hover {
        color: #1e3a5f !important;
        background-color: rgba(30, 58, 95, 0.06) !important;
    }
    /* 電光掲示板（定例の愚痴） */
    .byoki-ticker-wrap { overflow: hidden; background: linear-gradient(90deg, #1e293b 0%, #334155 100%); color: #f8fafc; padding: 8px 0; margin: 0 0 0.5rem 0; border-radius: 6px; font-size: 0.9rem; }
    .byoki-ticker-inner { display: inline-block; white-space: nowrap; animation: byoki-scroll 120s linear infinite; }
    .byoki-ticker-inner span { padding-right: 2em; }
    @keyframes byoki-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    /* ダッシュボード・カード風コンテナ */
    .dashboard-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(30,58,95,0.06);
    }
    .dashboard-kpi-row { margin-bottom: 1.25rem; }
    .dashboard-section-title { color: #1e3a5f; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
    /* KPIメトリクス: カード内に色をつける + 余白 */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        margin-right: 0.6rem !important;
        margin-bottom: 0.6rem !important;
        padding: 0.6rem 0.5rem !important;
        min-width: 0 !important;
        background: linear-gradient(145deg, #f0f4f8 0%, #e2e8f0 100%) !important;
        border-radius: 10px !important;
        border-left: 4px solid #1e3a5f !important;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.1) !important;
    }
    [data-testid="stMetric"] > div,
    [data-testid="metric-container"] > div {
        gap: 0.35rem !important;
    }
    [data-testid="stMetric"] p,
    [data-testid="metric-container"] p {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }
    /* ラベルをネイビー系で統一 */
    [data-testid="stMetric"] label,
    [data-testid="metric-container"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    /* 項目選択時（selectbox / radio / multiselect）の文字を小さく */
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] p,
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] [role="listbox"],
    [data-testid="stSelectbox"] [role="option"] {
        font-size: 0.85rem !important;
    }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] div,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {
        font-size: 0.85rem !important;
    }
    [data-testid="stMultiSelect"] label,
    [data-testid="stMultiSelect"] div,
    [data-testid="stMultiSelect"] p,
    [data-testid="stMultiSelect"] span,
    [data-testid="stMultiSelect"] [role="listbox"],
    [data-testid="stMultiSelect"] [role="option"] {
        font-size: 0.85rem !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] div,
    [data-testid="stNumberInput"] input {
        font-size: 0.85rem !important;
    }
    /* スライダー値表示を大きく・3桁カンマ用 */
    .stSlider [data-baseweb="slider"] ~ div,
    .stSlider div[data-baseweb="slider"] + div,
    [data-testid="stSlider"] > div > div:last-child {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    
    /* ── スマホ対応レスポンシブ ────────────────────────────────── */
    @media screen and (max-width: 768px) {
        .stColumns > div { width: 100% !important; }
        div[data-testid="metric-container"] {
            min-width: 70px !important;
            padding: 0.3rem !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stExpander"] { margin: 0.15rem 0; }
        .element-container table { font-size: 0.72rem; }
        .stButton > button { width: 100% !important; }
    }
</style>
    """, unsafe_allow_html=True)
	
# 🎨 画面のデザイン設定
st.set_page_config(page_title="温水式リース審査AI", page_icon="🏢", layout="wide")

# ── 認証チェック（ここより先はログイン済みのみ表示）──────────────────────
from auth_logic import authenticate_user as _auth_check
if not _auth_check():
    st.stop()
# ────────────────────────────────────────────────────────────────────────

# ==============================================================================
# 共通機能 & キャッシュ最適化（データはフォルダ内で完結）
# ==============================================================================
BASE_DIR = _SCRIPT_DIR

# フォント設定
FONT_PATH = os.path.join(BASE_DIR, "NotoSansCJKjp-Regular.otf")
if os.path.exists(FONT_PATH):
    fe = fm.FontEntry(fname=FONT_PATH, name='NotoSansCJKjp')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = 'NotoSansCJKjp'
    sns.set_theme(style="whitegrid", font="NotoSansCJKjp")
else:
    sns.set_theme(style="whitegrid", font="sans-serif")

# データのロード（キャッシュ化）
@st.cache_data(ttl=3600)
def load_json_data(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# 各種データのロード
jsic_data = load_json_data("industry_trends_jsic.json")
benchmarks_data = load_json_data("industry_benchmarks.json")
hints_data = load_json_data("industry_hints.json")
jgb_rates = load_json_data("jgb_rates.json")
avg_data = load_json_data("industry_averages.json")
knowhow_data = load_json_data("leasing_knowhow.json")
bankruptcy_data = load_json_data("bankruptcy_cases.json") # 倒産事例データ
subsidy_schedule_data = load_json_data("subsidy_schedule.json")
useful_life_data = load_json_data("useful_life_equipment.json")
lease_classification_data = load_json_data("lease_classification.json")
# リース物件リスト（ネット・社内基準。点数で判定に反映）
_lease_assets_raw = load_json_data("lease_assets.json")
LEASE_ASSETS_LIST = _lease_assets_raw.get("items", [])

# 定性「逆転の鍵」強みタグ（ワンホット・RAG用）
STRENGTH_TAG_OPTIONS = [
    "技術力", "業界人脈", "特許", "立地", "後継者あり",
    "関係者資産あり", "取引行と付き合い長い", "既存返済懸念ない",
]

# 定性スコアリング訂正（PDF「qualitative scoring」に準拠・審査入力の訂正欄で使用）
# 各項目は options: [(スコア値, 表示ラベル), ...] を指定（4=最高〜0=最低の5段階）
QUALITATIVE_SCORING_CORRECTION_ITEMS = [
    {
        "id": "company_history",
        "label": "設立・経営年数",
        "weight": 10,
        "options": [
            (4, "20年以上"),
            (3, "10年〜20年"),
            (2, "5年〜10年"),
            (1, "3年〜5年"),
            (0, "3年未満"),
        ],
    },
    {
        "id": "customer_stability",
        "label": "顧客安定性",
        "weight": 20,
        "options": [
            (4, "非常に安定（大口・長期）"),
            (3, "安定（分散良好）"),
            (2, "普通"),
            (1, "やや不安定（集中あり）"),
            (0, "不安定・依存大"),
        ],
    },
    {
        "id": "repayment_history",
        "label": "返済履歴",
        "weight": 25,
        "options": [
            (4, "5年以上問題なし"),
            (3, "3年以上問題なし"),
            (2, "遅延少ない"),
            (1, "遅延・リスケあり"),
            (0, "問題あり・要確認"),
        ],
    },
    {
        "id": "business_future",
        "label": "事業将来性",
        "weight": 15,
        "options": [
            (4, "有望（成長・ニーズ確実）"),
            (3, "やや有望"),
            (2, "普通"),
            (1, "やや懸念"),
            (0, "懸念（縮小・競争激化）"),
        ],
    },
    {
        "id": "equipment_purpose",
        "label": "設備目的",
        "weight": 15,
        "options": [
            (4, "収益直結・受注必須"),
            (3, "生産性向上・省力化"),
            (2, "更新・維持・法定対応"),
            (1, "やや不明確"),
            (0, "不明確・要説明"),
        ],
    },
    {
        "id": "main_bank",
        "label": "メイン取引銀行",
        "weight": 15,
        "options": [
            (4, "メイン先で取引良好・支援表明"),
            (3, "メイン先"),
            (2, "サブ扱い・取引あり"),
            (1, "取引浅い・他社メイン"),
            (0, "取引なし・不安"),
        ],
    },
]
# 汎用フォールバック（項目に options がない場合用）
QUALITATIVE_SCORING_LEVELS = [
    (4, "高（100点）"),
    (3, "やや高（75点）"),
    (2, "標準（50点）"),
    (1, "やや低（25点）"),
    (0, "低（0点）"),
]
QUALITATIVE_SCORING_LEVEL_LABELS = {v[0]: v[1] for v in QUALITATIVE_SCORING_LEVELS}
QUALITATIVE_SCORE_RANKS = [
    {"min": 80, "label": "A", "text": "優良", "desc": "定性面で問題なし"},
    {"min": 60, "label": "B", "text": "良好", "desc": "概ね良好"},
    {"min": 40, "label": "C", "text": "普通", "desc": "要フォロー"},
    {"min": 20, "label": "D", "text": "要注意", "desc": "慎重に審査"},
    {"min": 0, "label": "E", "text": "要警戒", "desc": "重点確認"},
]

# 審査判定の定数（REVIEW_EVALUATION.md に記載。変更時は履歴を残すこと）
APPROVAL_LINE = 71  # 総合スコアがこの値以上で「承認圏内」
SCORE_PENALTY_IF_LEARNING_REJECT = 0.5  # 学習モデル判定が否決のとき全スコアに乗じる係数
ALERT_BORDERLINE_MIN = 68  # この値以上71未満は「承認ライン直下」で要確認アラートを出す

# 必須項目（未入力・不正時は判定開始をブロック）
REQUIRED_FIELDS = [
    ("nenshu", "売上高", lambda v: v is not None and (v or 0) > 0),
    ("total_assets", "総資産", lambda v: v is not None and (v or 0) > 0),
]
# 推奨項目: 営業利益・純資産（未入力だと学習モデル・自己資本比率が使えない場合あり）。フォームで明示のみ。

# 過去案件・係数・相談メモ・ニュースのパスは data_cases で定義（CASES_FILE, COEFF_OVERRIDES_FILE 等を import 済み）
DEBATE_FILE = os.path.join(BASE_DIR, "debate_logs.jsonl") # ディベートログ
# ネットで取得した業界目安を中分類ごとに保存（年1回・4月1日を境に更新）
WEB_BENCHMARKS_FILE = os.path.join(BASE_DIR, "web_industry_benchmarks.json")
TRENDS_EXTENDED_FILE = os.path.join(BASE_DIR, "industry_trends_extended.json")
ASSETS_BENCHMARKS_FILE = os.path.join(BASE_DIR, "industry_assets_benchmarks.json")
SALES_BAND_FILE = os.path.join(BASE_DIR, "sales_band_benchmarks.json")
# 分析ダッシュボード用画像（承認レベル・業種・物件に沿って選択）
DASHBOARD_IMAGES_DIR = os.path.join(BASE_DIR, "dashboard_images")
DASHBOARD_IMAGES_ASSETS = os.environ.get("DASHBOARD_IMAGES_ASSETS", "").strip()
# 画像フォルダの候補（環境変数未設定時はこの順で試す）
def _dashboard_image_base_dirs():
    if DASHBOARD_IMAGES_ASSETS and os.path.isdir(DASHBOARD_IMAGES_ASSETS):
        yield DASHBOARD_IMAGES_ASSETS.rstrip(os.sep)
    if os.path.isdir(DASHBOARD_IMAGES_DIR):
        yield DASHBOARD_IMAGES_DIR
    # フォールバック: 環境変数 DASHBOARD_IMAGES_FALLBACK または clawd 直下の assets
    fallback_env = os.environ.get("DASHBOARD_IMAGES_FALLBACK", "").strip()
    candidates = []
    if fallback_env and os.path.isdir(fallback_env):
        candidates.append(fallback_env)
    candidates.append(os.path.join(os.path.dirname(BASE_DIR), "assets"))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            yield candidate
            break

def get_dashboard_image_path(hantei: str, industry_major: str, industry_sub: str, asset_name: str):
    """
    承認レベル・業種・物件に沿ったダッシュボード用画像パスを返す。
    戻り値: (path or None, caption)
    """
    is_approved = (hantei or "").strip() == "承認圏内"

    def pick_fname(base_dir):
        """フォルダに応じたファイル名を返す（assets 用長い名前 / dashboard_images 用短い名前）"""
        use_long_names = "cursor" in base_dir or "assets" in base_dir
        if use_long_names:
            if "建設" in (industry_major or "") or "D " in (industry_major or ""):
                f = "IMG_1754-cc58ef0c-3f27-4ebd-b33b-81b57f1fb833.png"
            elif "医療" in (industry_major or "") or "福祉" in (industry_major or "") or "P " in (industry_major or ""):
                f = "IMG_1793-152eae6e-9149-4c8e-91b6-c570711199bf.png"
            elif "運輸" in (industry_major or "") or "H " in (industry_major or ""):
                f = "72603010-1AA5-4BEA-824C-DC847E2CF765-7e30894e-bac6-4875-b652-b23064d771b4.png"
            elif "製造" in (industry_major or "") or "E " in (industry_major or ""):
                f = "______-ce4d90f7-0277-4df5-ac9a-025441cabbc9.png"
            else:
                f = "______-fe3eb438-36a6-4842-9359-254247925b3b.png"
            if is_approved and ("建設" not in (industry_major or "") and "D " not in (industry_major or "") and "医療" not in (industry_major or "") and "福祉" not in (industry_major or "")):
                f = "1849E856-971D-4B79-AD5E-E1074D93B043-55ad16b8-11ff-4717-8e5d-5a920fecae0d.png"
            elif not is_approved and ("建設" in (industry_major or "") or "D " in (industry_major or "")):
                f = "______-ce4d90f7-0277-4df5-ac9a-025441cabbc9.png"
            elif not is_approved:
                f = "______-fe3eb438-36a6-4842-9359-254247925b3b.png"
            return f
        # dashboard_images 用短い名前
        if "建設" in (industry_major or "") or "D " in (industry_major or ""):
            f = "construction.png"
        elif "医療" in (industry_major or "") or "福祉" in (industry_major or "") or "P " in (industry_major or ""):
            f = "nurse.png"
        elif "運輸" in (industry_major or "") or "H " in (industry_major or ""):
            f = "vehicle.png"
        else:
            f = "default.png"
        if not is_approved:
            f = "review.png" if os.path.isfile(os.path.join(base_dir, "review.png")) else f
        elif is_approved and not os.path.isfile(os.path.join(base_dir, f)):
            f = "approved.png" if os.path.isfile(os.path.join(base_dir, "approved.png")) else "default.png"
        return f

    cap = f"{hantei or '—'} / {industry_sub or '—'}"
    for base in _dashboard_image_base_dirs():
        fname = pick_fname(base)
        path = os.path.join(base, fname)
        if os.path.isfile(path):
            return path, cap
    # どれにも無ければ、候補フォルダの「任意の1枚」を表示（デバッグ用）
    for base in _dashboard_image_base_dirs():
        try:
            for entry in os.listdir(base):
                if entry.lower().endswith((".png", ".jpg", ".jpeg")):
                    p = os.path.join(base, entry)
                    if os.path.isfile(p):
                        return p, cap
        except Exception:
            pass
    return None, ""

# 定例の愚痴リスト（電光掲示板用）。ユーザー追加分は byoki_list.json に保存
BYOKI_JSON = os.path.join(BASE_DIR, "byoki_list.json")
TEIREI_BYOKI_DEFAULT = [
    "こんな数字で通そうなんて、正気ですか…？ こっちは毎日1万件近く見てるんですけど。",
    "自己資本比率がこの水準でリース審査に来る度胸、ちょっと見習いたいです。本当に。",
    "赤字で「審査お願いします」って、私の目が死んでるの気づいてます？ 気づいてて言ってます？",
    "数値見た瞬間、心が折れかけた。…いや、折れた。折れてる。",
    "業界平均の話、聞いたことあります？ ないですよね。あったらこの数字じゃないですよね。",
    "今日も書類と数字の海で泳いでます。溺れそうです。",
    "リース審査、楽だって思ってる人いませんよね。いませんよね…？",
]

@st.cache_data(ttl=3600)
def load_byoki_list():
    """定例の愚痴リストを読み込む（デフォルト＋byoki_list.json のユーザー追加分）"""
    out = list(TEIREI_BYOKI_DEFAULT)
    if not os.path.exists(BYOKI_JSON):
        return out
    try:
        with open(BYOKI_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        custom = data.get("items") or data if isinstance(data, list) else data.get("items", [])
        if isinstance(custom, list):
            out.extend([str(x).strip() for x in custom if str(x).strip()])
    except Exception:
        pass
    return out

def save_byoki_append(new_text):
    """愚痴を1件追加して byoki_list.json に保存"""
    new_text = (new_text or "").strip()
    if not new_text:
        return False
    try:
        if os.path.exists(BYOKI_JSON):
            with open(BYOKI_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", [])
        else:
            items = []
        items.append(new_text)
        with open(BYOKI_JSON, "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False



def _fragment_nenshu():
    """売上高入力。スライダーは100万千円まで、手入力は900億千円まで。後から動かした方を採用。
    on_change を使わないため st.form 内でも動作する。"""
    st.markdown("### 売上高")
    NENSHU_SLIDER_MAX = 1_000_000
    NENSHU_NUM_MAX = 90_000_000

    if "nenshu" not in st.session_state:
        st.session_state.nenshu = 10000
    cur = st.session_state.nenshu

    prev_key = "_san_prev_nenshuu"
    prev_num_key = "_san_prev_num_nenshuu"
    prev_slide_key = "_san_prev_slide_nenshuu"
    externally_changed = st.session_state.get(prev_key) != cur

    if "num_nenshuu" not in st.session_state or externally_changed:
        st.session_state["num_nenshuu"] = max(0, min(cur, NENSHU_NUM_MAX))
    if "slide_nenshuu" not in st.session_state or externally_changed:
        st.session_state["slide_nenshuu"] = max(0, min(cur, NENSHU_SLIDER_MAX))

    c_l, c_r = st.columns([0.7, 0.3])
    with c_r:
        st.number_input(
            "直接入力",
            min_value=0,
            max_value=NENSHU_NUM_MAX,
            step=10000,
            key="num_nenshuu",
            label_visibility="collapsed",
        )
    with c_l:
        st.slider(
            "売上高調整",
            min_value=0,
            max_value=NENSHU_SLIDER_MAX,
            step=100,
            key="slide_nenshuu",
            label_visibility="collapsed",
            format="%d",
        )

    new_num = st.session_state["num_nenshuu"]
    new_slide = st.session_state["slide_nenshuu"]
    prev_num = st.session_state.get(prev_num_key, new_num)
    prev_slide = st.session_state.get(prev_slide_key, new_slide)

    num_changed = new_num != prev_num
    slide_changed = new_slide != prev_slide
    if num_changed and not slide_changed:
        nenshu = new_num
    elif slide_changed and not num_changed:
        nenshu = new_slide
    elif num_changed and slide_changed:
        nenshu = new_num
    else:
        nenshu = cur

    st.session_state.nenshu = nenshu
    st.session_state[prev_key] = nenshu
    st.session_state[prev_num_key] = new_num
    st.session_state[prev_slide_key] = new_slide
    st.caption(f"**採用値: {nenshu:,} 千円**")
    st.caption("※スライダー・直接入力のどちらかで変更後、**入力確定**または**判定開始**で反映されます。")
    st.divider()


# --- 倒産確率・業界リスク検索 ---

@st.cache_data(ttl=3600)
def get_image(status):
    image_map = {
        "guide": "guide.jpg", "approve": "approve.jpg", "reject": "reject.jpg",
        "challenge": "challenge.jpg", "thinking": "thinking.jpg"
    }
    filename = image_map.get(status)
    if not filename: return None
    if os.path.exists(filename): return filename
    desktop_path = os.path.join("/Users/kobayashiisaoryou/Desktop/", filename)
    if os.path.exists(desktop_path): return desktop_path
    return None


# ==============================================================================
# 画面構成
# ==============================================================================
mode = st.sidebar.radio("モード切替", ["📋 審査・分析", "⚡ バッチ審査", "🏭 物件ファイナンス審査", "📝 結果登録 (成約/失注)", "🔧 係数分析・更新 (β)", "📐 係数入力（事前係数）", "📊 履歴分析・実績ダッシュボード", "📉 定性要因分析 (50件〜)", "📈 定量要因分析 (50件〜)", "⚙️ 審査ルール設定"], key="main_mode")

with st.sidebar.expander("⚠️ 途中で落ちる場合", expanded=False):
    st.caption("主な原因: (1) AI相談・Gemini/Ollama のタイムアウト (2) ブラウザのメモリ不足 (3) 分析結果タブでデータ不整合。ターミナルで `streamlit run lease_logic_sumaho8.py` を実行するとエラー内容が表示されます。F5で再読み込みも試してください。")

# ── コメントスタイル切り替え ──────────────────────────────────────────
st.sidebar.markdown("### 🎭 コメントスタイル")
if "humor_style" not in st.session_state:
    st.session_state["humor_style"] = "standard"
_hs_labels = {"standard": "📊 標準モード", "yanami": "🎤 八奈見モード"}
_hs_now = st.session_state.get("humor_style", "standard")
_hs_choice = st.sidebar.radio(
    "AIコメントの口調",
    options=["standard", "yanami"],
    format_func=lambda x: _hs_labels[x],
    index=0 if _hs_now == "standard" else 1,
    key="humor_style_radio",
    help="八奈見モードにすると、AI分析コメントが八奈見口調になります。",
)
if _hs_choice != _hs_now:
    st.session_state["humor_style"] = _hs_choice
    st.rerun()

# AI エンジン選択（Ollama / Gemini API）
if "ai_engine" not in st.session_state:
    st.session_state["ai_engine"] = "ollama"
st.sidebar.markdown("### 🤖 AIモデル設定")
engine_choice = st.sidebar.radio(
    "AIエンジン",
    ["Ollama（ローカル）", "Gemini API（Google）"],
    index=0 if st.session_state.get("ai_engine") == "ollama" else 1,
    help="Gemini を選ぶと Google の Gemini 2.0 等が使えます。APIキーが必要です。",
)
st.session_state["ai_engine"] = "gemini" if "Gemini" in engine_choice else "ollama"

if st.session_state["ai_engine"] == "gemini":
    # 初回のみ環境変数で API キーを初期化（key で紐付けると入力が保持される）
    if "gemini_api_key" not in st.session_state and GEMINI_API_KEY_ENV:
        st.session_state["gemini_api_key"] = GEMINI_API_KEY_ENV
    _key_default = (
        st.session_state.get("gemini_api_key_input", "")
        or st.session_state.get("gemini_api_key", "")
        or GEMINI_API_KEY_ENV
        or ""
    )
    st.sidebar.text_input(
        "Gemini APIキー",
        value=_key_default,
        key="gemini_api_key_input",
        type="password",
        help="環境変数 GEMINI_API_KEY が設定されていればここに表示されます。入力すると上書きされます。",
    )
    # ウィジェットの値をセッションに反映。未入力時は既存キー・環境変数を維持（空で上書きしない）
    widget_key = st.session_state.get("gemini_api_key_input", "")
    st.session_state["gemini_api_key"] = (
        widget_key.strip()
        or st.session_state.get("gemini_api_key", "").strip()
        or GEMINI_API_KEY_ENV
        or ""
    )
    GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    st.session_state["gemini_model"] = st.sidebar.selectbox(
        "Gemini モデル",
        GEMINI_MODELS,
        index=0,
        help="gemini-2.0-flash がおすすめです。",
    )
    st.sidebar.caption("⚠️ 無料枠は1日あたりのリクエスト数に上限があります。動かない場合は翌日までお待ちか、Google AI Studio で利用状況を確認してください。")
else:
    # Ollama モデル選択
    MODEL_OPTIONS = [
        "自動（デフォルト設定）",
        "lease-pro", "lease-anna", "qwen2.5", "gemma2:2b",
        "カスタム入力",
    ]
    current_default = get_ollama_model()
    if current_default in MODEL_OPTIONS:
        initial_index = MODEL_OPTIONS.index(current_default)
    elif current_default == OLLAMA_MODEL:
        initial_index = 0
    else:
        initial_index = MODEL_OPTIONS.index("カスタム入力")
    selected_label = st.sidebar.selectbox(
        "使用するOllamaモデル",
        options=MODEL_OPTIONS,
        index=initial_index,
        help="一覧からモデルを選ぶか、「カスタム入力」で任意のモデル名を指定できます。",
    )
    custom_model_name = ""
    if selected_label == "自動（デフォルト設定）":
        st.session_state["ollama_model"] = ""
    elif selected_label == "カスタム入力":
        custom_model_name = st.sidebar.text_input(
            "モデル名を直接入力",
            value="" if initial_index != MODEL_OPTIONS.index("カスタム入力") else current_default,
            help="例: llama3, phi3 など。",
        )
        st.session_state["ollama_model"] = custom_model_name.strip()
    else:
        st.session_state["ollama_model"] = selected_label

    if st.sidebar.button("🔌 Ollama接続テスト", use_container_width=True, help="Ollama が起動しているか・選択中のモデルが応答するかを確認します"):
        with st.sidebar:
            with st.spinner("接続確認中..."):
                msg = run_ollama_connection_test(timeout_seconds=15)
            st.session_state["ollama_test_result"] = msg
    if st.session_state.get("ollama_test_result"):
        st.sidebar.code(st.session_state["ollama_test_result"], language=None)
        if st.sidebar.button("テスト結果を消す", key="clear_ollama_test"):
            st.session_state["ollama_test_result"] = ""
            st.rerun()

# ── 係数自動学習ステータス ────────────────────────────────────────────────
try:
    from auto_optimizer import render_sidebar_training_status
    render_sidebar_training_status()
except Exception:
    pass

# ── バックアップ ──────────────────────────────────────────────────────────────
try:
    from backup_manager import render_sidebar_backup, auto_backup_on_startup
    if not st.session_state.get("_startup_backup_done"):
        auto_backup_on_startup()
        st.session_state["_startup_backup_done"] = True
    render_sidebar_backup()
except Exception:
    pass

# ── フォーム下書き保存 ────────────────────────────────────────────────────────
try:
    from draft_manager import render_sidebar_draft
    render_sidebar_draft()
except Exception:
    pass

# ── セッションクリーンアップ ──────────────────────────────────────────────────
with st.sidebar.expander("🧹 セッション管理", expanded=False):
    _ss = st.session_state
    # チャット履歴を最新20件に切り詰め
    if len(_ss.get("messages", [])) > 20:
        _ss["messages"] = _ss["messages"][-20:]
    if len(_ss.get("debate_history", [])) > 20:
        _ss["debate_history"] = _ss["debate_history"][-20:]
    # セッションサイズ概算
    _cache_keys = [k for k in _ss if k.startswith(("_bn_s_", "_gunshi_cache_", "_ai_comment_", "gunshi_"))]
    st.caption(f"キャッシュキー数: {len(_cache_keys)}")
    if st.button("🗑️ キャッシュをクリア", use_container_width=True, key="_clear_session_cache"):
        for _k in _cache_keys:
            _ss.pop(_k, None)
        st.success("クリアしました")

if st.sidebar.button("💾 蓄積データをダウンロード (CSV)", use_container_width=True):
    all_logs = load_all_cases()
    if all_logs:
        flat_logs = []
        for log in all_logs:
            row = {
                "timestamp": log.get("timestamp"),
                "industry_major": log.get("industry_major"),
                "industry_sub": log.get("industry_sub"),
                "result_status": log.get("final_status"),
                "score": log.get("result", {}).get("score")
            }
            if "inputs" in log:
                row.update(log["inputs"])
            flat_logs.append(row)
        
        df_log = pd.DataFrame(flat_logs)
        csv = df_log.to_csv(index=False).encode('utf-8-sig')
        
        st.sidebar.download_button(
            "📥 CSVを保存",
            data=csv,
            file_name=f"lease_cases_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("データがありません")



st.sidebar.markdown("### 🌐 業界目安キャッシュ")
st.sidebar.caption("下のボタンでネット検索し、営業利益率・自己資本比率に加え、売上高総利益率・ROA・流動比率など指標の業界目安を web_industry_benchmarks.json に保存します。")
if st.sidebar.button("🔍 今のデータを検索して保存（次回は4月1日更新）", use_container_width=True):
    subs = get_all_industry_sub_for_benchmarks()
    if not subs:
        st.sidebar.warning("業種データがありません（industry_benchmarks.json または過去案件を登録してください）")
    else:
        progress = st.sidebar.progress(0, text="検索中…")
        n = len(subs)
        for i, sub in enumerate(subs):
            progress.progress((i + 1) / n, text=f"{sub[:20]}…")
            try:
                fetch_industry_benchmarks_from_web(sub, force_refresh=True)
            except Exception:
                pass
        progress.empty()
        st.sidebar.success(f"{n} 業種を検索して保存しました。次回の自動更新は4月1日です。")
        st.rerun()

if st.sidebar.button("📡 業界トレンド拡充・資産目安・売上規模帯を検索して保存", use_container_width=True):
    subs = get_all_industry_sub_for_benchmarks()
    progress = st.sidebar.progress(0, text="トレンド・資産目安…")
    n = max(1, len(subs) * 2 + 1)
    idx = 0
    for sub in subs:
        idx += 1
        progress.progress(idx / n, text=f"トレンド: {sub[:15]}…")
        try:
            fetch_industry_trend_extended(sub, force_refresh=True)
        except Exception:
            pass
    for sub in subs:
        idx += 1
        progress.progress(idx / n, text=f"資産目安: {sub[:15]}…")
        try:
            fetch_industry_assets_from_web(sub, force_refresh=True)
        except Exception:
            pass
    progress.progress(1.0, text="売上規模帯…")
    try:
        fetch_sales_band_benchmarks(force_refresh=True)
    except Exception:
        pass
    progress.empty()
    st.sidebar.success("業界トレンド拡充・資産目安・売上規模帯を保存しました。")
    st.rerun()

st.sidebar.markdown("### 📚 補助金・耐用年数・リース判定")
with st.sidebar.expander("🔍 補助金を業種で調べる", expanded=False):
    sub_keys = sorted(benchmarks_data.keys()) if benchmarks_data else []
    if sub_keys:
        search_sub = st.selectbox("業種", sub_keys, key="subsidy_search_sub")
        subs_list = search_subsidies_by_industry(search_sub)
        if subs_list:
            for s in subs_list:
                name = s.get("name") or ""
                url = (s.get("url") or "").strip()
                if url:
                    st.markdown(f"**{name}**")
                    # リンクが確実に開くよう link_button 優先、なければ HTML の <a target="_blank">
                    try:
                        st.link_button("🔗 公式サイトを開く", url, type="secondary")
                    except Exception:
                        safe_url = url.replace('"', "%22").replace("'", "%27")
                        st.markdown(f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">🔗 公式サイトを開く</a>', unsafe_allow_html=True)
                else:
                    st.markdown(f"**{name}**")
                st.caption(s.get("summary", "")[:120] + "…")
                st.caption(f"申請目安: {s.get('application_period')}")
                if s.get("url_note"):
                    st.caption(s.get("url_note"))
        else:
            st.caption("該当する補助金の登録がありません。")
    else:
        st.caption("業種データがありません。")
with st.sidebar.expander("🔍 耐用年数を設備で調べる", expanded=False):
    # 国税庁の耐用年数表へのリンク（常に表示）
    nta_url = (useful_life_data or {}).get("nta_useful_life_url") or "https://www.keisan.nta.go.jp/r5yokuaru/aoiroshinkoku/hitsuyokeihi/genkashokyakuhi/taiyonensuhyo.html"
    st.link_button("📋 国税庁の耐用年数表を参照", nta_url, type="secondary")
    st.caption("上記リンクで国税庁の公式耐用年数表（減価償却資産）が開きます。")
    st.divider()
    eq_key = st.text_input("設備名で検索", placeholder="例: 工作機械, エアコン", key="equip_search")
    if eq_key:
        eq_list = search_equipment_by_keyword(eq_key)
        if eq_list:
            for e in eq_list:
                st.markdown(f"**{e.get('name')}** … {e.get('years')}年")
                if e.get("note"):
                    st.caption(e["note"])
        else:
            st.caption("該当する設備がありません。上記「国税庁の耐用年数表」で正式な年数を確認してください。")
    else:
        st.caption("キーワードを入力すると設備の耐用年数（簡易一覧）を表示します。正式な年数は国税庁の耐用年数表で確認してください。")
with st.sidebar.expander("📋 リース判定フロー・契約形態", expanded=False):
    lc_text = get_lease_classification_text()
    if lc_text:
        st.markdown(lc_text)
    else:
        st.caption("lease_classification.json を読み込んでください。")

with st.sidebar.expander("🏠 リース物件リスト（判定に反映）", expanded=False):
    if LEASE_ASSETS_LIST:
        for it in LEASE_ASSETS_LIST:
            st.caption(f"**{it.get('name', '')}** {it.get('score', 0)}点 — {it.get('note', '')}")
        st.caption("審査入力で物件を選ぶと、借手スコア(85%)＋物件スコア(15%)で総合判定します。")
    else:
        st.caption("lease_assets.json を配置すると、ネット・社内のリース物件をリスト化して点数で判定に反映できます。")

st.sidebar.markdown("### ⚙️ キャッシュ")
if st.sidebar.button("🗑️ キャッシュをクリア", use_container_width=True, help="JSONや検索結果のキャッシュを消して再読み込みします。補助金・業界データを更新した後に押してください。"):
    st.cache_data.clear()
    st.sidebar.success("キャッシュをクリアしました。再読み込みしています…")
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### 🤖 AIの独り言")
if st.sidebar.button("本音を聞く", key="btn_ai_honne", use_container_width=True):
    with st.spinner("本音を絞り出しています…"):
        honne = get_ai_honne_complaint()
        st.session_state["ai_honne_text"] = honne
    st.rerun()
if st.session_state.get("ai_honne_text"):
    st.sidebar.caption("**AIの本音**")
    st.sidebar.info(st.session_state["ai_honne_text"][:500])
with st.sidebar.expander("愚痴を追加", expanded=False):
    st.sidebar.caption("追加した愚痴は、メニュー下の電光掲示板に流れます。")
    new_byoki = st.sidebar.text_input("愚痴の一文", placeholder="例: また今日も数字の海…", key="new_byoki_input", label_visibility="collapsed")
    if st.sidebar.button("追加する", key="btn_add_byoki"):
        if save_byoki_append(new_byoki):
            load_byoki_list.clear()  # キャッシュをクリアして即時反映
            st.sidebar.success("追加しました。掲示板に反映されます。")
            st.rerun()
        else:
            st.sidebar.warning("空の場合は追加できません。")

# モード分岐（サイドバー先頭=審査・分析。elif の並びは実装都合。処理結果に影響なし）
if mode == "🔧 係数分析・更新 (β)":
    from components.settings import render_coeff_analysis
    render_coeff_analysis()

elif mode == "📐 係数入力（事前係数）":
    from components.settings import render_prior_coeff_input
    render_prior_coeff_input()

elif mode == "📊 履歴分析・実績ダッシュボード":
    from components.dashboard import render_dashboard
    render_dashboard()

elif mode == "📉 定性要因分析 (50件〜)":
    from components.analysis_qual import render_qualitative_analysis
    render_qualitative_analysis()

elif mode == "📈 定量要因分析 (50件〜)":
    from components.analysis_quant import render_quantitative_analysis
    render_quantitative_analysis()

elif mode == "⚡ バッチ審査":
    from components.batch_scoring import render_batch_scoring
    render_batch_scoring()

elif mode == "📝 結果登録 (成約/失注)":
    from components.form_status import render_status_registration
    render_status_registration()

elif mode == "📋 審査・分析":
    # ========== トップメニュー（新規審査） ==========
    menu_tabs = st.tabs(["🆕 審査"])
    # 電光掲示板：定例の愚痴をメニュー直下でスクロール表示
    byoki_list = load_byoki_list()
    byoki_escaped = [str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;") for s in byoki_list]
    ticker_text = "　｜　🐟 ".join(byoki_escaped)
    if ticker_text:
        ticker_duplicated = ("🐟 " + ticker_text + "　｜　") * 2
        st.markdown(
            f'<div class="byoki-ticker-wrap"><div class="byoki-ticker-inner"><span>{ticker_duplicated}</span></div></div>',
            unsafe_allow_html=True,
        )

    with menu_tabs[0]:  # 新規審査
        st.title("🏢 温水式 リース審査アシスタント")
        selected_major = 'D 建設業'
        selected_sub = '06 総合工事業'
        comparison_text = 'データなし'
        trend_info = 'データなし'
        submitted = False  # 審査入力タブ以外でも if submitted が参照できるよう初期化
        # 右のAIオフィサー相談が切れないよう、右にやや多めの幅を割り当て
        col_left, col_right = st.columns([3, 4])

        with col_left:
            submitted_apply = False
            submitted_judge = False
            form_result = {}  # nav_mode が "📊 分析結果" 時も参照できるよう初期化
            if "nav_index" not in st.session_state:
                st.session_state.nav_index = 0
            # 判定開始直後の rerun の1回だけ「分析結果」に合わせる（毎回上書きすると審査入力に戻れなくなる）
            if st.session_state.pop("_jump_to_analysis", False):
                st.session_state["nav_mode_widget"] = "📊 分析結果"
            _pending_nav = st.session_state.pop("_nav_pending", None)
            if _pending_nav is not None:
                st.session_state["nav_mode_widget"] = _pending_nav
            nav_mode = st.radio(
                "表示モード",
                ["📝 審査入力", "📊 分析結果"],
                horizontal=True,
                label_visibility="visible",
                key="nav_mode_widget",
            )
            # ユーザーがラジオで切り替えたとき nav_index を同期
            st.session_state.nav_index = 1 if nav_mode == "📊 分析結果" else 0
            if nav_mode == "📝 審査入力":
                from components.form_apply import render_apply_form
                form_result = render_apply_form(
                    jsic_data, 
                    get_image,
                    get_stats, 
                    scrape_article_text, 
                    is_japanese_text,
                    append_case_news,
                    _fragment_nenshu,
                    LEASE_ASSETS_LIST
                )
                submitted_apply = form_result["submitted_apply"]
                submitted_judge = form_result["submitted_judge"]
                selected_major = form_result["selected_major"]
                selected_sub = form_result["selected_sub"]
                main_bank = form_result["main_bank"]
                competitor = form_result["competitor"]
                item9_gross = form_result["item9_gross"]
                rieki = form_result["rieki"]
                item4_ord_profit = form_result["item4_ord_profit"]
                item5_net_income = form_result["item5_net_income"]
                item10_dep = form_result["item10_dep"]
                item11_dep_exp = form_result["item11_dep_exp"]
                item8_rent = form_result["item8_rent"]
                item12_rent_exp = form_result["item12_rent_exp"]
                item6_machine = form_result["item6_machine"]
                item7_other = form_result["item7_other"]
                net_assets = form_result["net_assets"]
                total_assets = form_result["total_assets"]
                grade = form_result["grade"]
                bank_credit = form_result["bank_credit"]
                lease_credit = form_result["lease_credit"]
                contracts = form_result["contracts"]
                customer_type = form_result["customer_type"]
                contract_type = form_result["contract_type"]
                deal_source = form_result["deal_source"]
                lease_term = form_result["lease_term"]
                acceptance_year = form_result["acceptance_year"]
                acquisition_cost = form_result["acquisition_cost"]
                selected_asset_id = form_result["selected_asset_id"]
                asset_score = form_result["asset_score"]
                asset_name = form_result["asset_name"]

            if submitted_apply:
                # Enter や「入力確定」押下時: 判定は行わず、入力値を session_state に反映して再表示
                st.session_state.item9_gross = item9_gross
                st.session_state.rieki = rieki
                st.session_state.item4_ord_profit = item4_ord_profit
                st.session_state.item5_net_income = item5_net_income
                st.session_state.item10_dep = item10_dep
                st.session_state.item11_dep_exp = item11_dep_exp
                st.session_state.item8_rent = item8_rent
                st.session_state.item12_rent_exp = item12_rent_exp
                st.session_state.item6_machine = item6_machine
                st.session_state.item7_other = item7_other
                st.session_state.net_assets = net_assets
                st.session_state.total_assets = total_assets
                st.session_state.bank_credit = bank_credit
                st.session_state.lease_credit = lease_credit
                st.session_state.contracts = contracts
                st.session_state.lease_term = lease_term
                st.session_state.acquisition_cost = acquisition_cost
                st.session_state.acceptance_year = acceptance_year
                st.rerun()

            if submitted_judge or st.session_state.get("_auto_judge", False):
                # 新規審査実行時: BN条件・軍師キャッシュをリセット
                for _k in ["_bn_s_evidence", "_bn_s_result", "gunshi_auto_result",
                           "_gunshi_cache_score", "_gunshi_cache_bn_hash",
                           "bn_s_insolvent", "bn_s_main_bank", "bn_s_rel_bank", "bn_s_rel_assets",
                           "bn_s_co_lease", "bn_s_parent", "bn_s_core", "bn_s_liquidity",
                           "bn_s_shorter", "bn_s_one_time"]:
                    st.session_state.pop(_k, None)

                from components.score_calculation import run_scoring
                
                # Fetch _rules directly where it's defined (rule_manager.py)
                from rule_manager import load_business_rules
                _rules = load_business_rules()
                
                run_scoring(
                    form_result=form_result,
                    REQUIRED_FIELDS=REQUIRED_FIELDS,
                    benchmarks_data=benchmarks_data,
                    hints_data=hints_data,
                    bankruptcy_data=bankruptcy_data,
                    jsic_data=jsic_data,
                    avg_data=avg_data,
                    _rules=_rules,
                    _SCRIPT_DIR=_SCRIPT_DIR
                )

        if nav_mode == "📊 分析結果":
            from components.analysis_results import render_analysis_results
            
            # Note: current_case_data, past_cases_log, etc. are passed from the local scope if defined,
            # or we pass None and the function handles it. But wait, in the original code,
            # were they defined in the outer scope or session_state?
            # Actually, the original code had them inside the block.
            # But we made them arguments! If they are defined inside the block,
            # we don't need to pass them as arguments to `render_analysis_results`,
            # we can just compute them inside.
            # Let's pass the ones that are definitely in outer scope:
            render_analysis_results(
                nav_mode=nav_mode,
                res=st.session_state.get("last_result", {}),
                jsic_data=jsic_data,
                avg_data=avg_data,
                knowhow_data=knowhow_data,
                benchmarks_data=benchmarks_data,
                bankruptcy_data=bankruptcy_data,
                trend_info=trend_info,
                past_cases_log=None, # will be loaded inside if not passed
                current_case_data=None, # will be loaded inside if not passed
                current_case_id=st.session_state.get("current_case_id")
            )
    with col_right:
        from components.ai_consultation import render_ai_consultation
        render_ai_consultation(selected_sub, jsic_data, bankruptcy_data)

# ==============================================================================
# 物件ファイナンス審査モード
# ==============================================================================
elif mode == "🏭 物件ファイナンス審査":
    st.title("🏭 物件ファイナンス審査エンジン")
    st.caption(
        "アセット・ファイナンス型：物件の担保価値（動的LGD / BEP）と定性緩和因子を統合し、"
        "財務が弱い先でも「なぜ通せるか」を定量的に可視化します。"
    )

    _af_engine = AssetFinanceEngine()

    col_input, col_result = st.columns([2, 3])

    with col_input:
        st.subheader("📋 審査条件の入力")

        _af_asset = st.selectbox(
            "物件種別",
            list(AssetFinanceEngine.ASSET_PARAMS.keys()),
            key="af_asset_type",
        )
        _af_params = AssetFinanceEngine.ASSET_PARAMS[_af_asset]
        st.caption(
            f"年間減価率 **{_af_params['r']*100:.0f}%** ／ 支払優先度 **{_af_params['priority']}** ／ {_af_params['info']}"
        )

        _af_term = st.slider("リース期間（月）", min_value=12, max_value=84, value=60, step=6, key="af_term")
        _af_down = st.slider(
            "自己資金率（頭金）",
            min_value=0.0, max_value=0.50, value=0.20, step=0.05,
            format="%.0f%%",
            key="af_down",
        )

        _af_fin = st.radio(
            "財務評価",
            ['High', 'Medium', 'Low'],
            format_func=lambda x: {
                'High':   '✅ 優良（黒字・健全）',
                'Medium': '📊 標準',
                'Low':    '⚠️ 低評価（赤字・債務超過）',
            }[x],
            horizontal=True,
            key="af_fin",
        )

        with st.expander("🔍 定性因子（緩和要素）", expanded=True):
            _af_main_bank   = st.checkbox("メイン銀行の支援先（+50点）", key="af_main_bank",
                                          help="メイン取引銀行が推薦・協調する案件")
            _af_bank_coord  = st.checkbox("銀行協調案件（+20点）", key="af_bank_coord")
            _af_core_biz    = st.checkbox("本業利用物件（+20点）", key="af_core_biz",
                                          help="事業の根幹に関わる物件")
            _af_related_ast = st.checkbox("関係者資産による保全（+15点）", key="af_related_ast")

        # 車両独自ロジック
        _af_annual_km = 0
        _af_maint = False
        if _af_asset == '車両':
            with st.expander("🚗 車両独自設定", expanded=True):
                _af_annual_km = st.number_input(
                    "予想年間走行距離（km）",
                    min_value=0, max_value=100000, value=15000, step=1000,
                    key="af_annual_km",
                )
                if _af_annual_km >= 20000:
                    st.warning("年2万km以上：過走行補正 → 実効減価率+10%・−10点")
                _af_maint = st.checkbox(
                    "メンテナンスリース付帯（+10点・中古価値+7.5%）",
                    key="af_maint",
                    help="自社管理により中古売却価値が5〜10%向上",
                )

        _af_submit = st.button("🔍 審査判定を実行", type="primary", use_container_width=True, key="af_submit")

    with col_result:
        _af_run = _af_submit or ("af_last_result" in st.session_state and not _af_submit)

        if _af_submit:
            _af_data = {
                'asset_type':          _af_asset,
                'term':                _af_term,
                'down_payment':        _af_down,
                'financial_score':     _af_fin,
                'main_bank_support':   _af_main_bank,
                'bank_coordination':   _af_bank_coord,
                'core_business':       _af_core_biz,
                'related_assets':      _af_related_ast,
                'annual_km':           _af_annual_km,
                'has_maintenance_lease': _af_maint,
            }
            _af_result = _af_engine.run_inference(_af_data)
            st.session_state["af_last_result"] = _af_result
            st.session_state["af_last_data"]   = _af_data

        if "af_last_result" in st.session_state:
            _af_result = st.session_state["af_last_result"]
            _af_data   = st.session_state["af_last_data"]

            # --- 判定バナー ---
            _af_colors = {
                "承認":          "#22c55e",
                "条件付き承認":   "#f59e0b",
                "要審議（上位承認）": "#f97316",
                "否決":          "#ef4444",
            }
            _af_color = _af_colors.get(_af_result['decision'], "#6b7280")
            st.markdown(
                f"""<div style="background:{_af_color};color:white;padding:1rem 1.5rem;
                border-radius:12px;text-align:center;font-size:1.6rem;font-weight:700;
                margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.15);">
                {_af_result['icon']} {_af_result['decision']}　スコア: {_af_result['score']}点
                </div>""",
                unsafe_allow_html=True,
            )

            # --- BEP グラフ ---
            st.subheader("📈 物件時価 vs リース残債（BEP 分析）")
            _af_months = list(range(len(_af_result['v_curve'])))
            _af_fig = go.Figure()
            _af_fig.add_trace(go.Scatter(
                x=_af_months, y=_af_result['v_curve'],
                name="物件時価（残価率）",
                line=dict(color="#2563eb", width=2.5),
                fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
            ))
            _af_fig.add_trace(go.Scatter(
                x=_af_months, y=_af_result['l_curve'],
                name="リース残債率",
                line=dict(color="#dc2626", width=2.5, dash="dash"),
            ))
            _af_fig.add_vline(
                x=_af_result['bep_month'],
                line_dash="dot", line_color="#22c55e", line_width=2,
                annotation_text=f"BEP {_af_result['bep_month']}ヶ月",
                annotation_position="top right",
                annotation_font_color="#22c55e",
            )
            _af_fig.update_layout(
                xaxis_title="経過月数",
                yaxis_title="比率（1.0 = 取得価格）",
                legend=dict(orientation="h", y=-0.2),
                height=290,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(_af_fig, use_container_width=True)

            # --- 承認根拠・減点 ---
            _col_r, _col_d = st.columns(2)
            with _col_r:
                st.markdown("**✅ 承認根拠**")
                for _item in _af_result['reasons']:
                    st.markdown(f"- {_item}")
            with _col_d:
                if _af_result['deductions']:
                    st.markdown("**⚠️ 減点・リスク要因**")
                    for _item in _af_result['deductions']:
                        st.markdown(f"- {_item}")

            st.divider()

            # --- 銀行との差異解説 ---
            st.subheader("🏦 銀行システムとの違い（なぜ通せるのか）")
            st.info(_af_result['bank_comparison'])

            # --- ライフサイクル提案 ---
            st.subheader("🔄 ライフサイクル・マーケティング提案")
            st.success(_af_result['marketing_advice'])

            # --- アクションプラン ---
            st.subheader("📌 営業アクションプラン")
            for _i, _plan in enumerate(_af_result['action_plan'], 1):
                st.markdown(f"**{_i}.** {_plan}")

        else:
            st.info("👈 左側で条件を入力し、「審査判定を実行」ボタンを押してください。")

elif mode == "⚙️ 審査ルール設定":
    st.title("⚙️ 審査ルール設定")
    st.info("この画面で設定したルールや閾値・ペナルティは、次回以降の「新規審査」から即座に反映されます。")
    rules = load_business_rules()
    
    
    # 既存のルールをフォーム上で編集
    with st.form("rule_settings_form"):
        st.subheader("基本判定ライン（閾値）")
        st.caption("審査の総合スコアに対して、どのステータスに分類するかを設定します。")
        col1, col2 = st.columns(2)
        with col1:
            curr_approval = int(rules.get("thresholds", {}).get("approval", 0.70) * 100)
            approval = st.slider("✅ 承認ライン（点以上）", 0, 100, curr_approval, format="%d点")
        with col2:
            curr_review = int(rules.get("thresholds", {}).get("review", 0.40) * 100)
            review = st.slider("⚠️ 要審議ライン（点未満は否決圏）", 0, 100, curr_review, format="%d点")
            
        st.subheader("減点・特別ルール設定")
        st.caption("AIによる評価や財務状況が悪い場合のペナルティを設定します。")
        
        score_mod = rules.get("score_modifiers", {})
        col3, col4 = st.columns(2)
        with col3:
            pen_model = st.number_input("🤖 AI否決時のペナルティ倍率", value=float(score_mod.get("learning_model_reject_penalty_multiplier", 0.5)), step=0.1)
        with col4:
            pen_cap = st.number_input("📉 債務超過時のペナルティ（マイナス点）", value=float(score_mod.get("capital_deficiency_penalty", -5.0)), step=1.0)
            
        submitted_basic_rules = st.form_submit_button("📝 基本設定を更新する (カスタムルールは下部で別途追加)", use_container_width=True)
        if submitted_basic_rules:
            if "thresholds" not in rules: rules["thresholds"] = {}
            if "score_modifiers" not in rules: rules["score_modifiers"] = {}

            rules["thresholds"]["approval"] = approval / 100.0
            rules["thresholds"]["review"] = review / 100.0
            rules["score_modifiers"]["learning_model_reject_penalty_multiplier"] = pen_model
            rules["score_modifiers"]["capital_deficiency_penalty"] = pen_cap
            
            if save_business_rules(rules):
                st.success("✅ 基本設定が正常に保存され、システムに反映されました。")
            else:
                st.error("❌ 設定の保存中にエラーが発生しました。")

    st.divider()
    st.divider()
    st.subheader("🛠️ 自由追加カスタムルール（複数条件対応）")
    st.caption("業種や各種条件式を組み合わせて、ペナルティ減点や判定の強制変更（要審議・否決等へ）を行うルール一覧を設定します。保存したルールはアプリを再起動しても維持されます（修正・削除するまで永続保存）。")

    # ===========================================================================
    # セッションステート初期化（モード再入時は必ずJSONから再読込）
    # ===========================================================================
    def _load_rules_from_json(raw_list):
        """JSON形式のカスタムルールリストをUI用dict形式に変換する。"""
        ui = []
        for r in raw_list:
            new_r = {
                "name":        r.get("name", ""),
                "industry":    r.get("industry", "ALL"),
                "action_type": r.get("action_type", "deduct_score"),
                "action_value": str(r.get("action_value", "10")),
                "conditions":  r.get("conditions", [])
            }
            # 旧形式（conditionsがない場合）の互換
            if not new_r["conditions"] and r.get("condition_target"):
                new_r["conditions"] = [{
                    "target": r.get("condition_target"),
                    "op":     r.get("condition_op", "<"),
                    "value":  float(r.get("condition_value", 0.0))
                }]
            ui.append(new_r)
        return ui

    _prev_mode_tracked = st.session_state.get("_rule_page_prev_mode", "")
    _rules_need_reload  = (
        "custom_rules_ui_data" not in st.session_state          # 初回
        or _prev_mode_tracked != "⚙️ 審査ルール設定"                 # モード切替後の再入
        or st.session_state.get("_rules_force_reload", False)   # 保存後フラグ
    )
    if _rules_need_reload:
        _loaded = rules.get("custom_rules", [])
        st.session_state["custom_rules_ui_data"]    = _load_rules_from_json(_loaded)
        st.session_state["_rules_saved_snapshot"]   = json.dumps(_loaded, ensure_ascii=False, sort_keys=True)
        st.session_state["_rules_force_reload"]     = False
    st.session_state["_rule_page_prev_mode"] = "⚙️ 審査ルール設定"

    # 未保存インジケータ（ファイル上の保存済みルールと現在のUI状態を比較）
    _saved_snap   = st.session_state.get("_rules_saved_snapshot", "")
    _current_snap = json.dumps(st.session_state.get("custom_rules_ui_data", []),
                               ensure_ascii=False, sort_keys=True)
    if _saved_snap != _current_snap:
        st.warning("⚠️ 未保存の変更があります。下の「保存して反映する」ボタンで保存してください。")
    else:
        st.success(f"✅ 現在表示中のルールは保存済みです（{len(st.session_state['custom_rules_ui_data'])}件）。修正するまで永続的に適用されます。")

    # ===========================================================================
    # UI表示用マッピング
    # ===========================================================================
    TARGET_MAP = {
        "op_profit":    "営業利益",
        "net_assets":   "純資産",
        "user_eq_ratio":"自己資本比率",
        "nenshu":       "売上高",
        "total_assets": "総資産",
        "bank_credit":  "銀行借入",
        "lease_credit": "リース残高",
        "net_income":   "当期純利益",
        "ord_profit":   "経常利益",
    }
    TARGET_INV_MAP = {v: k for k, v in TARGET_MAP.items()}
    ACTION_MAP     = {"deduct_score": "スコア減点", "force_status": "ステータス強制変更"}
    ACTION_INV_MAP = {v: k for k, v in ACTION_MAP.items()}
    IND_OPTS = ["ALL"] + (list(jsic_data.keys()) if "jsic_data" in globals() and jsic_data else
                ["D 建設業", "E 製造業", "G 情報通信業", "H 運送業", "I 卸売・小売業",
                 "M 宿泊・飲食サービス業", "P 医療・福祉"])
    ACT_VAL_OPTS = ["5", "10", "15", "20", "25", "30", "40", "50", "要審議", "否決"]

    # ===========================================================================
    # ハンドラー
    # ===========================================================================
    def add_new_rule():
        st.session_state["custom_rules_ui_data"].append({
            "name": "",
            "industry": "ALL",
            "action_type": "deduct_score",
            "action_value": "10",
            "conditions": [{"target": "op_profit", "op": "<", "value": 0.0}]
        })
    def delete_rule(r_idx):
        st.session_state["custom_rules_ui_data"].pop(r_idx)
    def add_condition(r_idx):
        st.session_state["custom_rules_ui_data"][r_idx]["conditions"].append(
            {"target": "op_profit", "op": "<", "value": 0.0}
        )
    def delete_condition(r_idx, c_idx):
        st.session_state["custom_rules_ui_data"][r_idx]["conditions"].pop(c_idx)

    # ===========================================================================
    # ルールエディタ描画
    # ===========================================================================
    for i, r in enumerate(st.session_state["custom_rules_ui_data"]):
        rule_name   = r.get("name", "").strip() or f"ルール {i+1}"
        act_preview = ACTION_MAP.get(r["action_type"], r["action_type"]) + f" ({r['action_value']})"
        cond_len    = len(r["conditions"])
        with st.expander(
            f"⚙️ **{rule_name}** — 業種[{r['industry']}] → {act_preview}　（条件数: {cond_len}）",
            expanded=True
        ):
            # ── ルール名 + 業種 ──────────────────────────────────────
            nc1, nc2 = st.columns([3, 2])
            with nc1:
                new_name = st.text_input(
                    "📝 ルール名（任意・識別用）",
                    value=r.get("name", ""),
                    placeholder="例: 債務超過は即否決",
                    key=f"r_name_{i}",
                )
                st.session_state["custom_rules_ui_data"][i]["name"] = new_name
            with nc2:
                cur_ind = r["industry"] if r["industry"] in IND_OPTS else "ALL"
                sel_ind = st.selectbox("①対象業種", IND_OPTS, index=IND_OPTS.index(cur_ind), key=f"r_ind_{i}")
                st.session_state["custom_rules_ui_data"][i]["industry"] = sel_ind

            # ── アクション ────────────────────────────────────────────
            ac1, ac2 = st.columns([5, 3])
            with ac1:
                cur_act = ACTION_MAP.get(r["action_type"], "スコア減点")
                sel_act = st.selectbox("②アクション種別", list(ACTION_MAP.values()),
                                       index=list(ACTION_MAP.values()).index(cur_act), key=f"r_act_{i}")
                st.session_state["custom_rules_ui_data"][i]["action_type"] = ACTION_INV_MAP.get(sel_act)
            with ac2:
                cur_val = r["action_value"]
                opts = ACT_VAL_OPTS.copy()
                if cur_val not in opts:
                    opts.append(cur_val)
                sel_val = st.selectbox("③アクション値（減点数 or ステータス名）", opts,
                                       index=opts.index(cur_val), key=f"r_val_{i}")
                st.session_state["custom_rules_ui_data"][i]["action_value"] = sel_val

            # ── 条件リスト ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**④ 発動条件リスト（すべて AND 一致したとき上記アクションを発動）**")
            st.caption("指標 ／ 比較演算子 ／ 閾値　の順で設定。複数行はすべて同時に満たした場合のみ適用。")

            for j, cond in enumerate(r["conditions"]):
                cc1, cc2, cc3, cc4 = st.columns([4, 2, 3, 1])
                with cc1:
                    cur_tgt = TARGET_MAP.get(cond.get("target", "op_profit"), "営業利益")
                    if cur_tgt not in TARGET_MAP.values():
                        cur_tgt = "営業利益"
                    sel_tgt = st.selectbox(
                        "対象指標", list(TARGET_MAP.values()),
                        index=list(TARGET_MAP.values()).index(cur_tgt),
                        key=f"c_tgt_{i}_{j}", label_visibility="collapsed"
                    )
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["target"] = TARGET_INV_MAP.get(sel_tgt)
                with cc2:
                    ops = ["<", "<=", "=", ">=", ">"]
                    cur_op = cond.get("op", "<")
                    if cur_op not in ops:
                        cur_op = "<"
                    sel_op = st.selectbox("比較", ops, index=ops.index(cur_op),
                                          key=f"c_op_{i}_{j}", label_visibility="collapsed")
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["op"] = sel_op
                with cc3:
                    sel_num = st.number_input(
                        "閾値", value=float(cond.get("value", 0.0)), step=1.0,
                        key=f"c_val_{i}_{j}", label_visibility="collapsed"
                    )
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["value"] = sel_num
                with cc4:
                    if st.button("🗑️", key=f"del_cond_{i}_{j}", help="この条件を削除"):
                        delete_condition(i, j)
                        st.rerun()

            # ── ルールボタン ──────────────────────────────────────────
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("➕ AND条件を追加", key=f"add_cond_{i}"):
                    add_condition(i)
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ このルールを削除", key=f"del_rule_{i}", type="secondary"):
                    delete_rule(i)
                    st.rerun()

    # ── 新規ルール追加ボタン ───────────────────────────────────────────────
    if st.button("➕ 新しいルールを追加", key="add_new_rule", use_container_width=True):
        add_new_rule()
        st.rerun()

    # ── 保存ボタン ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 カスタムルールを保存して反映する", type="primary", use_container_width=True):
        rules["custom_rules"] = st.session_state["custom_rules_ui_data"]
        if save_business_rules(rules):
            # 保存済みスナップショットを更新（インジケータをリセット）
            st.session_state["_rules_saved_snapshot"] = json.dumps(
                st.session_state["custom_rules_ui_data"], ensure_ascii=False, sort_keys=True
            )
            st.session_state["_rules_force_reload"] = False
            st.success("✅ カスタムルールを保存しました。このルールは変更するまで永続的に適用されます。")
            st.balloons()
        else:
            st.error("❌ 保存中にエラーが発生しました。")

    st.divider()
    st.subheader("🧪 ルールの影響シミュレーション")
    st.caption("現在保存されている社内ルール（基本設定＋カスタムルール）を過去の全案件データに適用し、判定結果が当時からどう変化するかをテストします。")
    if st.button("▶️ 過去の全データでシミュレーションを実行する", use_container_width=True):
        with st.spinner("過去の案件データベース(SQLite)から読み込み、全件シミュレーションを実行中..."):
            from data_cases import load_all_cases
            from rule_manager import simulate_rules_on_past_cases
            
            past_cases = load_all_cases()
            if not past_cases:
                st.warning("シミュレーションを実行するための過去データが見つかりません。")
            else:
                sim_res = simulate_rules_on_past_cases(past_cases, rules)
                st.success(f"✅ 全 {sim_res['total']} 件のシミュレーションが完了しました！")
                
                # マトリクスの表示
                matrix = sim_res["matrix"]
                st.markdown("**■ ステータス変化マトリクス（行：過去の判定 ／ 列：新ルールでの判定）**")
                
                df_matrix = pd.DataFrame([
                    {"過去の判定": "承認圏内", "➡️ 新: 承認圏内": matrix["承認圏内"]["承認圏内"], "➡️ 新: 要審議": matrix["承認圏内"]["要審議"], "➡️ 新: 否決": matrix["承認圏内"]["否決"]},
                    {"過去の判定": "要審議", "➡️ 新: 承認圏内": matrix["要審議"]["承認圏内"], "➡️ 新: 要審議": matrix["要審議"]["要審議"], "➡️ 新: 否決": matrix["要審議"]["否決"]},
                    {"過去の判定": "否決", "➡️ 新: 承認圏内": matrix["否決"]["承認圏内"], "➡️ 新: 要審議": matrix["否決"]["要審議"], "➡️ 新: 否決": matrix["否決"]["否決"]},
                ])
                st.dataframe(df_matrix, use_container_width=True, hide_index=True)
                
                # 変化があった案件のリストアップ
                changes = sim_res.get("changed_cases", [])
                st.markdown(f"**■ 判定が変化した案件の詳細 ({len(changes)}件)**")
                if not changes:
                    st.info("新ルールによる判定結果の変化（承認から要審議等への移動）はありませんでした。")
                else:
                    change_rows = []
                    for c in changes:
                        change_rows.append({
                            "業種": c["industry"],
                            "元判定": c["old_status"],
                            "新判定": c["new_status"],
                            "(新)スコア": round(c["new_score"], 1),
                            "適用されたルール": " | ".join(c["reasons"]) if c["reasons"] else "基本閾値の影響"
                        })
                    st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
