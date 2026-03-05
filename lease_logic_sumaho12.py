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
# 共通機能 & キャッシュ最適化（データはリポジトリルートで sumaho8 と共通）
# ==============================================================================
BASE_DIR = _REPO_ROOT

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

def get_review_alert(res):
    """
    判定結果 res（last_result）を受け取り、要確認かどうかと理由リストを返す。
    戻り値: (needs_review: bool, reasons: list[str])
    """
    if not res:
        return False, []
    reasons = []
    score = res.get("score") or 0
    scr = res.get("scoring_result") or {}
    decision = (scr.get("decision") or "").strip()
    # 学習モデル否決時はスコアが0.5倍されているので、元スコアに戻して判定
    if decision == "否決":
        effective_original = score / SCORE_PENALTY_IF_LEARNING_REJECT
    else:
        effective_original = score
    if ALERT_BORDERLINE_MIN <= effective_original < APPROVAL_LINE:
        reasons.append("スコアが承認ライン（71）直下です。目視確認を推奨します。")
    if effective_original >= APPROVAL_LINE and decision == "否決":
        reasons.append("本社スコアは承認圏内ですが、学習モデルが否決です。要確認。")
    if effective_original < APPROVAL_LINE and decision == "承認":
        reasons.append("本社は要審議ですが、学習モデルは承認です。要確認。")
    return (len(reasons) > 0, reasons)

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
# AssetFinanceEngine ― アセット・ファイナンス型 物件審査エンジン
# ==============================================================================
class AssetFinanceEngine:
    """
    ベイジアンネットワーク思想に基づく「アセット・ファイナンス型」リース審査エンジン。
    物件の担保価値（動的LGD/BEP）と財務スコアを統合し、
    銀行審査を超えた判定・逆転承認・ライフサイクル提案を行う。
    """

    ASSET_PARAMS = {
        '建機':    {'r': 0.15, 'priority': '中',   'priority_score': 2, 'info': '海外需要による底値の硬さ'},
        '工作機械': {'r': 0.20, 'priority': '高',   'priority_score': 3, 'info': '精度維持による長寿命価値'},
        'PC/IT':   {'r': 0.40, 'priority': '低',   'priority_score': 1, 'info': '高速な陳腐化・資産価値なし'},
        '医療機器': {'r': 0.10, 'priority': '極高', 'priority_score': 5, 'info': '診療報酬直結（最優先支払）'},
        'ドローン': {'r': 0.50, 'priority': '中',   'priority_score': 2, 'info': '物理破損と超高速陳腐化'},
        '車両':    {'r': 0.25, 'priority': '中高', 'priority_score': 3, 'info': '高換金性と確立された中古市場'},
    }

    def get_effective_depreciation_rate(self, asset_type, annual_km=0, has_maintenance_lease=False):
        """実効減価率（走行距離補正・メンテリース補正を加えた実効値）"""
        r = self.ASSET_PARAMS[asset_type]['r']
        if asset_type == '車両':
            if annual_km >= 20000:
                r += 0.10  # 過走行補正（年2万km以上）
            if has_maintenance_lease:
                r -= 0.05  # メンテリースによる価値維持補正
        return max(r, 0.0)

    def get_maintenance_lgd_bonus(self, asset_type, has_maintenance_lease):
        """メンテナンスリース受託時のLGD（中古売却価値）加算率"""
        if asset_type == '車両' and has_maintenance_lease:
            return 0.075  # 5〜10%の中間値：管理品質による与信枠拡大
        return 0.0

    def calculate_bep(self, asset_type, term_months, down_payment_rate,
                      annual_km=0, has_maintenance_lease=False):
        """
        損益分岐点（BEP）算出。
        V(t) = (1 + maint_bonus) × (1 - r)^(t/12)
        L(t) = (1 - down_payment) × (1 - t/term)
        V(t) > L(t) となる最初の月をBEPとする。
        """
        r = self.get_effective_depreciation_rate(asset_type, annual_km, has_maintenance_lease)
        maint_bonus = self.get_maintenance_lgd_bonus(asset_type, has_maintenance_lease)

        bep_month = term_months
        v_curve = []
        l_curve = []

        for m in range(0, term_months + 1):
            v = (1.0 + maint_bonus) * ((1 - r) ** (m / 12))
            l = (1 - down_payment_rate) * (1 - m / term_months) if m < term_months else 0.0
            v_curve.append(round(v, 4))
            l_curve.append(round(l, 4))
            if m > 0 and v > l and bep_month == term_months:
                bep_month = m

        return bep_month, v_curve, l_curve

    def calculate_score(self, data):
        """
        総合承認スコアの算出。
        定量（財務）最大40点 ＋ 物件LGD/BEP緩和最大50点 ＋ 逆転定性因子最大105点
        """
        asset_type = data['asset_type']
        term = data['term']
        down_payment = data['down_payment']
        financial_score = data['financial_score']
        annual_km = data.get('annual_km', 0)
        has_maintenance_lease = data.get('has_maintenance_lease', False)

        bep_month, v_curve, l_curve = self.calculate_bep(
            asset_type, term, down_payment, annual_km, has_maintenance_lease
        )
        bep_ratio = bep_month / term
        priority_score = self.ASSET_PARAMS[asset_type]['priority_score']

        score = 0
        reasons = []
        deductions = []

        # --- 1. 財務スコア（銀行視点） ---
        fin_map = {'High': 40, 'Medium': 20, 'Low': -20}
        fin_pts = fin_map.get(financial_score, 0)
        score += fin_pts
        fin_labels = {'High': '優良', 'Medium': '標準', 'Low': '低評価（赤字・債務超過）'}
        if fin_pts >= 0:
            reasons.append(f"財務{fin_labels[financial_score]}（+{fin_pts}点）")
        else:
            deductions.append(f"財務{fin_labels[financial_score]}（{fin_pts}点）")

        # --- 2. 物件・LGD/BEP 緩和（リース独自視点） ---
        if bep_ratio < 0.3:
            pts = 35
            reasons.append(f"BEP早期達成（{bep_month}ヶ月目 / {term}ヶ月）→ 物件保全性が極めて高い（+{pts}点）")
        elif bep_ratio < 0.5:
            pts = 20
            reasons.append(f"BEP前半達成（{bep_month}ヶ月目）→ 物件保全性が高い（+{pts}点）")
        elif bep_ratio < 0.7:
            pts = 10
            reasons.append(f"BEP中盤（{bep_month}ヶ月目）→ 物件保全性は標準（+{pts}点）")
        else:
            pts = 0
            deductions.append(f"BEP後半（{bep_month}ヶ月目 / {term}ヶ月）→ 物件価値がリース期間全体でリスク（0点）")
        score += pts

        # 支払優先度ボーナス
        if priority_score >= 5:
            score += 15
            reasons.append("支払優先度「極高」（診療報酬直結）→ +15点")
        elif priority_score >= 3:
            score += 8
            reasons.append(f"支払優先度「{self.ASSET_PARAMS[asset_type]['priority']}」→ +8点")

        # メンテナンスリース補正（車両）
        if asset_type == '車両' and has_maintenance_lease:
            score += 10
            reasons.append("メンテナンスリース受託 → 管理品質による中古価値向上（+10点）")

        # --- 3. 逆転因子（定性） ---
        if data.get('main_bank_support'):
            score += 50
            reasons.append("メイン銀行の支援先（最強の緩和因子）→ +50点")
        if data.get('bank_coordination'):
            score += 20
            reasons.append("銀行協調案件 → +20点")
        if data.get('core_business'):
            score += 20
            reasons.append("本業利用物件（支払優先度UP）→ +20点")
        if data.get('related_assets'):
            score += 15
            reasons.append("関係者資産による保全 → +15点")

        # --- 4. 車両独自リスク（過走行） ---
        if asset_type == '車両' and annual_km >= 20000:
            score -= 10
            deductions.append(f"過走行リスク（年{annual_km:,}km）→ 実効減価率+10%補正、−10点")

        return score, bep_month, bep_ratio, v_curve, l_curve, reasons, deductions

    def get_decision(self, score):
        """スコアから判定区分を決定"""
        if score >= 80:
            return "承認", "✅"
        elif score >= 50:
            return "条件付き承認", "⚠️"
        elif score >= 30:
            return "要審議（上位承認）", "🔶"
        else:
            return "否決", "❌"

    def get_marketing_advice(self, asset_type, term_months):
        """ライフサイクル・マーケティング提案"""
        if asset_type == '車両':
            check_month = max(term_months - 6, 1)
            return (
                f"**車検タイミング戦略**: {check_month}ヶ月目（車検前6ヶ月）に"
                f"リプレイス提案を予約。3〜5年サイクルでの入れ替えニーズを先行受注。"
            )
        elif asset_type == '医療機器':
            return (
                "**長期伴走戦略**: 満了後の再リース移行率が高い物件です。"
                "7年以上を見据えた再リース・保守契約のセットプランを今から提示してください。"
            )
        elif asset_type == '建機':
            return (
                "**海外輸出戦略**: 4〜5年後の海外輸出相場（東南アジア・中東）に"
                "合わせた下取り提案が有効。中古ブローカーとの連携も検討。"
            )
        elif asset_type == '工作機械':
            return (
                "**精度保証戦略**: 満了前にオーバーホール費用の試算を提示し、"
                "リニューアルリースへの誘導を図ってください。"
            )
        elif asset_type == 'PC/IT':
            return (
                "**早期入れ替え戦略**: 陳腐化が速いため、2〜3年での早期入れ替えを前提に"
                "短期リース設計を推奨。バルク更新需要を狙う。"
            )
        elif asset_type == 'ドローン':
            return (
                "**物理リスク対策**: 損害保険とのセット提案が必須。"
                "法規制変化に合わせた入れ替えサイクル（2〜3年）を明示。"
            )
        return "満了の6ヶ月前に入れ替え需要を調査し、次案件の先行受注を狙ってください。"

    def get_bank_comparison(self, asset_type, financial_score):
        """銀行システムとの差異解説"""
        asset_info = self.ASSET_PARAMS[asset_type]['info']
        base = (
            f"銀行は「財務諸表の健全性」のみで判断しますが、"
            f"当エンジンは **{asset_type}**（{asset_info}）の物件価値・換金性を定量評価します。"
        )
        if financial_score == 'Low':
            base += (
                "\n\n財務が低評価でも、物件の保全性（BEP・残価）と定性緩和因子により、"
                "銀行が『否決』とする案件を承認圏に引き上げることができます。"
            )
        return base

    def get_action_plan(self, score, data, bep_month, bep_ratio):
        """営業アクションプラン"""
        plans = []
        asset_type = data['asset_type']
        down = data['down_payment']
        decision, _ = self.get_decision(score)

        if decision in ["否決", "要審議（上位承認）"]:
            needed_down = min(down + 0.10, 0.50)
            plans.append(
                f"自己資金を **{needed_down*100:.0f}%以上**（現在{down*100:.0f}%）に引き上げると"
                f"承認確率が大幅に上昇します。"
            )
            if not data.get('main_bank_support'):
                plans.append("**メイン銀行の推薦・協調案件**として申請することで、最大+50点の緩和が可能です。")
            if not data.get('core_business'):
                plans.append("**本業利用**であることを明示する書類（事業計画書等）を追加してください。（+20点）")
        if asset_type == '車両':
            plans.append(
                f"**{bep_month}ヶ月目**（BEP到達）以降は物件の換金価値がリース残債を上回ります。"
                "この点を保全力として稟議書に記載してください。"
            )
            if not data.get('has_maintenance_lease'):
                plans.append("**メンテナンスリース**を付帯させると中古売却価値が5〜10%向上します。（+10点）")
        if asset_type == '医療機器':
            plans.append("診療報酬との連動を稟議書に明記し、支払優先度の高さを強調してください。")
            plans.append(
                f"**{data['term'] + 12}ヶ月後**（満了後1年）の再リースプランを今から提示し、関係継続を確約。"
            )
        if bep_ratio > 0.7 and decision != "承認":
            plans.append("物件の頭金比率を上げるか、リース期間を短縮してBEPを前倒しにすることを検討してください。")
        if not plans:
            plans.append("現状の条件で承認可能です。満了に向けて次回リプレイス提案の準備を進めてください。")
        return plans

    def run_inference(self, data):
        """総合審査を実行し、結果辞書を返す"""
        score, bep_month, bep_ratio, v_curve, l_curve, reasons, deductions = self.calculate_score(data)
        decision, icon = self.get_decision(score)
        return {
            'score': score,
            'decision': decision,
            'icon': icon,
            'bep_month': bep_month,
            'bep_ratio': bep_ratio,
            'v_curve': v_curve,
            'l_curve': l_curve,
            'reasons': reasons,
            'deductions': deductions,
            'marketing_advice': self.get_marketing_advice(data['asset_type'], data['term']),
            'bank_comparison': self.get_bank_comparison(data['asset_type'], data['financial_score']),
            'action_plan': self.get_action_plan(score, data, bep_month, bep_ratio),
        }


# ==============================================================================
# 画面構成
# ==============================================================================
mode = st.sidebar.radio("モード切替", ["📋 審査・分析", "🏭 物件ファイナンス審査", "📝 結果登録 (成約/失注)", "🔧 係数分析・更新 (β)", "📐 係数入力（事前係数）", "📊 履歴分析・実績ダッシュボード", "📉 定性要因分析 (50件〜)", "📈 定量要因分析 (50件〜)", "⚙️ 審査ルール設定"], key="main_mode")

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
            st.sidebar.success("追加しました。掲示板に反映されます。")
            st.rerun()
        else:
            st.sidebar.warning("空の場合は追加できません。")

# モード分岐（サイドバー先頭=審査・分析。elif の並びは実装都合。処理結果に影響なし）
if mode == "🔧 係数分析・更新 (β)":
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

elif mode == "📐 係数入力（事前係数）":
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

elif mode == "📊 履歴分析・実績ダッシュボード":
    st.title("📊 履歴分析・実績ダッシュボード")
    analysis = run_contract_driver_analysis()
    if analysis is None:
        st.warning("成約データが5件以上貯まると表示されます。結果登録で「成約」を登録してください。")
    else:
        n = analysis["closed_count"]
        st.success(f"成約 {n} 件を分析しました。")
        try:
            pdf_bytes = build_contract_report_pdf(analysis)
            filename = f"成約の正体レポート_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("📥 分析結果をPDFでダウンロード", data=pdf_bytes, file_name=filename, mime="application/pdf", key="dl_contract_report_pdf")
        except Exception as e:
            st.caption(f"PDF生成をスキップしました: {e}")
        st.divider()
        # ---------- 成約要因分析 ----------
        st.subheader("📈 成約要因分析")
        st.caption("成約した案件だけを抽出し、共通項と成約に効く因子を分析した結果です。")
        st.markdown("**成約に最も寄与している上位3つの因子（ドライバー）**")
        for i, d in enumerate(analysis["top3_drivers"], 1):
            st.markdown(f"**{i}. {d['label']}** … 係数 {d['coef']:.4f}（{d['direction']}に効く）")
        st.divider()
        st.subheader("成約案件の平均的な財務数値")
        if analysis["avg_financials"]:
            rows = []
            for k, v in analysis["avg_financials"].items():
                if "自己資本" in k:
                    rows.append({"指標": k, "平均値": f"{v:.1f}%"})
                elif isinstance(v, float) and abs(v) >= 1:
                    rows.append({"指標": k, "平均値": f"{v:,.0f}"})
                else:
                    rows.append({"指標": k, "平均値": f"{v:.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("財務データが取得できませんでした。")
        st.divider()
        st.subheader("成約案件で頻出する定性タグ（ランキング）")
        if analysis["tag_ranking"]:
            for rank, (tag, count) in enumerate(analysis["tag_ranking"], 1):
                st.markdown(f"{rank}. **{tag}** … {count}件")
        else:
            st.caption("定性タグの登録がありません。")
        # 定性スコアリングの集計（成約案件）
        st.divider()
        st.subheader("成約案件の定性スコアリング")
        qs = analysis.get("qualitative_summary")
        if qs and (qs.get("avg_weighted") is not None or qs.get("avg_combined") is not None or qs.get("rank_distribution")):
            n_qual = qs.get("n_with_qual", 0)
            st.caption(f"成約{n}件のうち、定性スコアリングを入力していた案件: **{n_qual}件**")
            if qs.get("avg_weighted") is not None:
                st.metric("定性スコア（加重）の平均", f"{qs['avg_weighted']:.1f} / 100", help="項目別5段階の加重平均")
            if qs.get("avg_combined") is not None:
                st.metric("合計（総合×重み＋定性×重み）の平均", f"{qs['avg_combined']:.1f}", help="ランク算出の元となる合計点")
            if qs.get("rank_distribution"):
                st.markdown("**ランク（A〜E）の分布**")
                for r, cnt in sorted(qs["rank_distribution"].items(), key=lambda x: (-x[1], x[0])):
                    st.caption(f"- **{r}** … {cnt}件")
        else:
            st.caption("定性スコアリングを入力した成約案件がまだありません。審査入力で「定性スコアリング」を選択し、結果登録で成約にするとここに集計が表示されます。")

    # ---------------- 案件履歴一覧 ----------------
    st.divider()
    st.subheader("📋 最新の案件履歴")
    all_cases = load_all_cases()
    if all_cases:
        for case in reversed(all_cases[-15:]):  # 最新15件を表示
            c_date = case.get('timestamp', '')[:16]
            c_sub = case.get('industry_sub', '不明')
            c_score = case.get('result', {}).get('score', 0)
            c_status = case.get('final_status', '未登録')
            title_emoji = "✅" if c_status == "成約" else "❌" if c_status == "失注" else "📝"
            with st.expander(f"{title_emoji} {c_date} - {c_sub} (スコア: {c_score:.0f}) [{c_status}]"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write(f"**判定**: {case.get('result', {}).get('hantei', '不明')}")
                    if case.get('chat_summary'):
                        st.caption(case['chat_summary'])
                    if st.button("🔄 このデータを入力に復元", key=f"restore_hist_{case.get('id', c_date)}"):
                        i_data = case.get('inputs', {})
                        st.session_state["last_submitted_inputs"] = {
                            "selected_major": case.get("industry_major", ""),
                            "selected_sub": case.get("industry_sub", ""),
                        }
                        for k, v in i_data.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    st.session_state[sub_k] = sub_v
                            else:
                                st.session_state[k] = v
                        st.session_state["main_mode"] = "📋 審査・分析"
                        st.session_state["nav_index"] = 0
                        st.rerun()
                with c2:
                    if case.get('ai_industry_advice'):
                        st.markdown("##### 📈 AI業界分析アドバイス")
                        st.info(case['ai_industry_advice'])
                    if case.get('ai_byoki'):
                        st.markdown("##### 🤖 AIのぼやき")
                        st.caption(case['ai_byoki'])
    else:
        st.caption("まだ案件履歴がありません。")

elif mode == "📉 定性要因分析 (50件〜)":
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
                st.dataframe(lr_df, use_container_width=True, hide_index=True)
                if "lr_intercept" in result:
                    st.caption(f"切片: {result['lr_intercept']:.4f}")
            st.divider()
            st.subheader("LightGBM 特徴量重要度")
            if "lgb_importance" in result:
                imp_df = pd.DataFrame(result["lgb_importance"], columns=["項目", "重要度"])
                imp_df = imp_df.sort_values("重要度", ascending=False)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
        else:
            result = None
        if result is None and n_reg >= QUALITATIVE_ANALYSIS_MIN_CASES:
            st.info("上の「ロジスティック回帰とLightGBMを実行」ボタンで分析を開始してください。")

elif mode == "📈 定量要因分析 (50件〜)":
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

elif mode == "📝 結果登録 (成約/失注)":
    st.title("📝 案件結果登録")
    st.info("過去の審査案件に対して、最終的な結果（成約・失注）を登録します。")
    
    all_cases = load_all_cases()
    if not all_cases:
        st.warning("登録された案件がありません。")
    else:
        st.subheader("未登録の案件")
        pending_cases = [c for c in all_cases if c.get("final_status") == "未登録"]
        
        if not pending_cases:
            st.success("全ての案件が登録済みです！")
        
        for i, case in enumerate(reversed(pending_cases[-5:])): 
            case_id = case.get("id", "")
            res = case.get("result", {})
            score = res.get("score", 0)
            hantei = res.get("hantei", "不明")
            
            with st.expander(f"{case.get('timestamp', '')[:16]} - {case.get('industry_sub', '')} (スコア: {score:.0f})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**判定**: {hantei}")
                    summary = case.get("chat_summary", "")
                    st.caption((summary[:100] + "...") if summary else "サマリなし")
                
                with c2:
                    if st.button("🗑️ この案件を削除", key=f"del_pending_{case_id}", type="secondary", help="未登録のままこの案件を一覧から削除します"):
                        all_cases = [c for c in load_all_cases() if c.get("id") != case_id]
                        if save_all_cases(all_cases):
                            st.success("削除しました")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("保存に失敗しました。")
                    with st.form(f"status_form_{i}"):
                        res_status = st.radio("結果", ["成約", "失注"], horizontal=True)
                        final_rate = st.number_input("獲得レート (%)", value=0.0, step=0.01, format="%.2f", help="成約した場合の決定金利")
                        past_base_rate = case.get("pricing", {}).get("base_rate", 1.2)
                        base_rate_input = st.number_input("当時の基準金利 (%)", value=past_base_rate, step=0.01, format="%.2f")
                        lost_reason = st.text_input("失注理由 (失注の場合のみ)", placeholder="例: 金利で他社に負けた")
                        loan_condition_options = ["金融機関と協調", "本件限度", "次回格付まで本件限度", "その他"]
                        loan_conditions = st.multiselect("融資条件", loan_condition_options, help="該当する条件を複数選択")
                        competitor_name = st.text_input("競合他社情報", placeholder="例: 〇〇銀行、〇〇リース")
                        competitor_rate = st.number_input("他社提示金利 (%)", value=0.0, step=0.01, format="%.2f", help="競合の提示条件があれば入力")
                        
                        if st.form_submit_button("登録する"):
                            target_id = case.get("id")
                            updated = False
                            for c in all_cases:
                                if c.get("id") == target_id:
                                    c["final_status"] = res_status
                                    c["final_rate"] = final_rate
                                    c["base_rate_at_time"] = base_rate_input
                                    if res_status == "成約" and final_rate > 0:
                                        c["winning_spread"] = final_rate - base_rate_input
                                    if res_status == "失注":
                                        c["lost_reason"] = lost_reason
                                    c["loan_conditions"] = loan_conditions
                                    c["competitor_name"] = competitor_name.strip() or ""
                                    c["competitor_rate"] = competitor_rate if competitor_rate else None
                                    updated = True
                                    break
                            
                            if updated:
                                if save_all_cases(all_cases):
                                    st.success("登録しました！")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("保存に失敗しました。")

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
                st.header("📝 1. 審査データの入力")
                image_placeholder = st.empty()
                if 'current_image' not in st.session_state: st.session_state['current_image'] = "guide"
                img_path = get_image(st.session_state['current_image'])
                if img_path: image_placeholder.image(img_path, width=280)
                st.divider()

                # 業界・取引を expander で折りたたみ
                with st.expander("📌 業界選択・取引状況", expanded=True):
                    if not jsic_data:
                        st.error("業界データ(industry_trends_jsic.json)が見つかりません。")
                        major_keys = ["D 建設業"]
                    else:
                        major_keys = list(jsic_data.keys())
                    last_inp = st.session_state.get("last_submitted_inputs") or {}
                    idx_major = major_keys.index(last_inp["selected_major"]) if last_inp.get("selected_major") in major_keys else 0
                    selected_major = st.selectbox("大分類 (日本標準産業分類)", major_keys, index=idx_major, key="select_major")
                    if jsic_data:
                        sub_data = jsic_data[selected_major]["sub"]
                        sub_keys = list(sub_data.keys())
                        mapped_coeff_category = jsic_data[selected_major]["mapping"]
                    else:
                        sub_data = {}
                        sub_keys = ["06 総合工事業"]
                        mapped_coeff_category = "④建設業"
                    idx_sub = sub_keys.index(last_inp["selected_sub"]) if last_inp.get("selected_sub") in sub_keys else 0
                    selected_sub = st.selectbox("中分類", sub_keys, index=idx_sub, key="select_sub")
                    st.session_state["_frag_major"] = selected_major
                    st.session_state["_frag_sub"] = selected_sub
                    st.session_state["_frag_mapped_coeff"] = mapped_coeff_category
                    st.session_state["_frag_sub_data"] = sub_data
                    st.session_state["_frag_jsic_data"] = jsic_data
                    trend_info = sub_data.get(selected_sub, "情報なし")
                    past_stats = get_stats(selected_sub)
                    past_info_text = "過去データなし"
                    alert_msg = ""
                    if past_stats["count"] > 0:
                        past_info_text = f"過去{past_stats['count']}件 (平均: {past_stats['avg_score']:.1f}点)"
                        if past_stats["close_rate"] > 0:
                            past_info_text += f"\n成約率: {past_stats['close_rate']:.0%}"
                        if past_stats.get("avg_winning_rate") is not None and past_stats["avg_winning_rate"] > 0:
                            past_info_text += f"\n平均成約金利: {past_stats['avg_winning_rate']:.2f}%"
                        if past_stats.get("top_competitors_lost"):
                            past_info_text += f"\nよく負ける競合: {', '.join(past_stats['top_competitors_lost'][:5])}"
                        if past_stats["lost_reasons"]:
                            top_reason = max(set(past_stats["lost_reasons"]), key=past_stats["lost_reasons"].count)
                            alert_msg = f"\n⚠️ **注意**: この業種は「{top_reason}」による失注が多いです。"
                    st.info(f"💡 **業界トレンド ({selected_sub})**:\n{trend_info}\n\n📚 **社内実績**: {past_info_text}{alert_msg}")
                    with st.expander("🌐 ネットで最新ニュースを検索", expanded=False):
                        search_query = st.text_input("検索キーワード", value=f"{selected_sub} 動向 2025", key=f"news_search_query_{selected_sub}")
                        if st.button("検索", key="btn_news_search"):
                            try:
                                # まず ddgs（新パッケージ名）を優先的に利用し、なければ duckduckgo_search を使う
                                try:
                                    from ddgs import DDGS
                                    backend_name = "ddgs"
                                except ImportError:
                                    from duckduckgo_search import DDGS
                                    backend_name = "duckduckgo_search"

                                with st.spinner(f"検索中...（バックエンド: {backend_name}）"):
                                    raw_results = list(DDGS().text(search_query, region='jp-jp', max_results=10))
                                    if not raw_results:
                                        raw_results = list(DDGS().text(search_query, max_results=10))
                                    if not raw_results:
                                        st.warning("DuckDuckGo検索から結果が返ってきませんでした。ネットワーク制限や一時的な障害の可能性があります。")
                                        st.session_state.news_results = []
                                    else:
                                        jp_results = []
                                        for r in raw_results:
                                            title = (r.get("title") if isinstance(r, dict) else "") or ""
                                            body = (r.get("body") if isinstance(r, dict) else "") or ""
                                            if is_japanese_text(title + body):
                                                jp_results.append(r)
                                        if jp_results:
                                            st.session_state.news_results = jp_results[:3]
                                        else:
                                            st.info("日本語判定でヒットしなかったため、検索結果をそのまま表示します。")
                                            st.session_state.news_results = raw_results[:3]
                                    st.caption(f"検索結果件数: {len(st.session_state.news_results)} 件")
                            except ImportError:
                                st.error("検索機能には追加ライブラリが必要です: pip install duckduckgo-search または pip install ddgs")
                            except Exception as e:
                                st.error(f"検索エラー: {e}")
                        if 'news_results' in st.session_state and st.session_state.news_results:
                            for i, res in enumerate(st.session_state.news_results):
                                st.markdown(f"**[{res['title']}]({res['href']})**")
                                st.caption(res['body'])
                                if st.button(f"この記事をAIに読み込ませる", key=f"read_news_{i}"):
                                    with st.spinner(f"「{res['title']}」を読み込んでいます..."):
                                        content = scrape_article_text(res['href'])
                                        # 日本語記事のみAIに読み込ませる
                                        if content and isinstance(content, str) and not content.startswith("記事の読み込みに失敗しました"):
                                            if is_japanese_text(content):
                                                news_obj = {
                                                    "title": res['title'],
                                                    "url": res['href'],
                                                    "content": content,
                                                }
                                                st.session_state.selected_news_content = news_obj
                                                case_id = st.session_state.get("current_case_id")
                                                if case_id:
                                                    append_case_news({"case_id": case_id, **news_obj})
                                                st.success("日本語記事の読み込み完了！AIへの相談・ディベート時に内容が反映されます。")
                                            else:
                                                st.warning("この記事は日本語ではない可能性が高いため、AIへの読み込みをスキップしました。")
                                        elif isinstance(content, str) and content.startswith("記事の読み込みに失敗しました"):
                                            st.error(content)
                                        else:
                                            st.error("記事の本文を取得できませんでした。")
                                st.divider()
                    if 'selected_news_content' in st.session_state:
                        with st.container(border=True):
                            st.write("📖 **現在読み込み中の記事:**")
                            st.write(st.session_state.selected_news_content['title'])
                            if st.button("読み込みをクリア"):
                                del st.session_state.selected_news_content
                                st.rerun()
                    st.markdown("##### 🤝 取引・競合状況")
                    col_q1, col_q2 = st.columns(2)
                    with col_q1: main_bank = st.selectbox("取引区分", ["メイン先", "非メイン先"], key="main_bank", index=0 if (last_inp.get("main_bank") or "メイン先") == "メイン先" else 1)
                    with col_q2: competitor = st.selectbox("競合状況", ["競合なし", "競合あり"], key="competitor", index=0 if (last_inp.get("competitor") or "競合なし") == "競合なし" else 1)
                    # 競合ありの場合のみ「競合提示金利」を入力（金利差で成約率補正に利用）
                    if competitor == "競合あり":
                        comp_rate = st.number_input(
                            "競合提示金利 (%)",
                            min_value=0.0,
                            max_value=30.0,
                            value=float(st.session_state.get("competitor_rate") or 0.0),
                            step=0.1,
                            format="%.1f",
                            key="competitor_rate_input",
                            help="競合他社の提示金利を入力すると、自社が有利な場合に成約率をプラス補正します。"
                        )
                        st.session_state["competitor_rate"] = comp_rate if comp_rate > 0 else None
                    else:
                        st.session_state["competitor_rate"] = None
                st.caption("💡 数字入力で画面がガタつく場合：スライダーで大まかに合わせてから直接入力で微調整してください。")
                st.caption("📌 数値とスライダーは連動します。Enter は「入力確定」にだけ効き、判定には行きません。")
                if st.button("🆕 新しく入力する", help="全フィールドを初期値にリセットします", use_container_width=False):
                    _reset_shinsa_inputs()
                    st.rerun()
                with st.form("shinsa_form"):
                    st.info(
                        "**必須項目**: 売上高（1以上）、総資産（1以上）を入力しないと判定できません。\n\n"
                        "**推奨**: 営業利益・純資産も入力してください。未入力だと学習モデル（総資産・純資産必須）や自己資本比率が使えません。"
                    )
                    submitted_apply = st.form_submit_button("入力確定（Enterで反映）", type="secondary", help="数字入力でEnterを押したときはここが押された扱いになり、判定には行きません。")
                    with st.expander("📊 1. 損益計算書 (P/L)", expanded=True):
                        # ①売上高（フラグメント化で入力時のガタつき軽減）
                        _fragment_nenshu()

                        #  ②売上高総利益（スライダーは従来どおり、手入力のみ900億千円まで）
                        st.markdown("### 売上高総利益")
                        item9_gross = _slider_and_number("item9_gross", "sourieki", 10000, -500000, 1000000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #---------------------------------------------------------------------------------------------------------------

                        # #③営業利益
                        st.markdown("### 営業利益")
                        rieki = _slider_and_number("rieki", "rieki", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切

        #----------------------------------------------------------------------------------------------------------------------

                        st.markdown("### 経常利益")
                        item4_ord_profit = _slider_and_number("item4_ord_profit", "item4_ord_profit", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #-------------------------------------------------------------------------------------------

                        st.markdown("### 当期利益")
                        item5_net_income = _slider_and_number("item5_net_income", "item5_net_income", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切

                        # None対策（nenshu はフラグメント内で設定されるため session_state から取得）
                        c_nenshu = st.session_state.get("nenshu", 0) or 0
                        c_gross = item9_gross if item9_gross is not None else 0
                        c_rieki = rieki if rieki is not None else 0
                        c_ord = item4_ord_profit if item4_ord_profit is not None else 0
                        c_net = item5_net_income if item5_net_income is not None else 0
            
                        # [削除] 入力中のウォーターフォールグラフ表示 (分析タブに集約するため)
                        # if c_nenshu > 0: 
                        #     st.pyplot(plot_waterfall(c_nenshu, c_gross, c_rieki, c_ord, c_net))

                    with st.expander("🏢 2. 資産・経費・その他", expanded=False):
                    
                        st.markdown("### 減価償却費")
                        item10_dep = _slider_and_number("item10_dep", "item10_dep", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
    
        #--------------------------------------------------------------------------------------------------------
                        #⑦減価償却費（経費）
    
                        st.markdown("### 減価償却費(経費)")
                        item11_dep_exp = _slider_and_number("item11_dep_exp", "item11_dep_exp", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
    
        #----------------------------------------------------------------------------------------------------
    
                        # #⑧賃借料
                        st.markdown("### 賃借料")
                        item8_rent = _slider_and_number("item8_rent", "item8_rent", 10000, 0, 100000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
    
        #----------------------------------------------------------------------------------------------
    
                        #⑨賃借料（経費）
                        # h_item12_rent_exp=st.empty()
                        # item12_rent_exp = col3.select_slider("賃借料(経費）", options=range(0, 90000, 100), value=0)
                        # red_label(h_item12_rent_exp, f"賃借料(経費）:{item12_rent_exp:,} 千円")
                        # st.divider()
    
                        st.markdown("### 賃借料（経費）")
                        item12_rent_exp = _slider_and_number("item12_rent_exp", "item12_rent_exp", 10000, 0, 100000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
    
        #------------------------------------------------------------------------------------------------
    
                        #⑩機械装置
     
                        st.markdown("### 機械装置")
                        item6_machine = _slider_and_number("item6_machine", "item6_machine", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
    
        #--------------------------------------------------------------------------------------------
    
                        # #11その他資産
                        # h_item7_other=st.empty()
                        # item7_other = col4.select_slider("その他資産", options=range(0, 50000, 100), value=0)
                        # red_label(h_item7_other, f"その他資産:{ item7_other:,} 千円")
                        # st.divider()
    
                        st.markdown("### その他資産")
                        item7_other = _slider_and_number("item7_other", "item7_other", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #-------------------------------------------------------------------------------------------------------------
                        # #12純資産合計
    
                        st.markdown("### 純資産")
                        net_assets = _slider_and_number("net_assets", "net_assets", 10000, -30000, 500000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #--------------------------------------------------------------------------------
                        #13総資産
                        # h_total_assets=st.empty()
                        # total_assets = col4.select_slider("総資産（千円）", options=range(0, 900000, 1000), value=0)
                        # red_label(h_total_assets, f"総資産:{total_assets:,} 千円")
                        # st.divider()
    
                        st.markdown("### 総資産")
                        total_assets = _slider_and_number("total_assets", "total_assets", 10000, 0, 1000000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #------------------------------------------------------------------------------------------------------
                    with st.expander("💳 3. 信用情報", expanded=False):
    
                        # default値をリスト内の文字列と完全に一致させる必要があります
                        grade = st.segmented_control("格付", ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"], default=st.session_state.get("grade", "②4-6 (標準)"), key="grade")
        #---------------------------------------------------------------------------             
                    #     #14銀行与信
    
                        st.markdown("### うちの銀行与信")
                        st.caption("当社の与信です（総銀行与信ではありません）")
                        bank_credit = _slider_and_number("bank_credit", "bank_credit", 10000, 0, 3000000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #---------------------------------------------------------------------------------------------------------
      
                        # #15リース与信
    
                        st.markdown("### うちのリース与信")
                        st.caption("当社の与信です（総リース与信ではありません）")
                        lease_credit = _slider_and_number("lease_credit", "lease_credit", 10000, 0, 300000, 100, 1, max_val_number=90_000_000)
                        st.divider() # 次の項目との区切
        #--------------------------------------------------------------------------------------------------------
                        # #16契約数
                        st.markdown("### 契約数")
                        contracts = _slider_and_number("contracts", "contracts", 1, 0, 30, 1, 1, unit="件")
    
                        st.divider() # 次の項目との区切
    
        #------------------------------------------------------------------------------------------------------
    
    
                    with st.expander("📋 4. 契約条件・取得価格・リース物件", expanded=False):
                        customer_type = st.radio("顧客区分", ["既存先", "新規先"], horizontal=True, index=0 if st.session_state.get("customer_type", "既存先") == "既存先" else 1, key="customer_type")
                        st.divider()
                        st.markdown("##### 📈 契約条件・属性 (利回り予測用)")
                        with st.container():
                            c_y1, c_y2, c_y3 = st.columns(3)
                            contract_type = c_y1.radio("契約種類", ["一般", "自動車"], horizontal=True, index=0 if st.session_state.get("contract_type", "一般") == "一般" else 1, key="contract_type")
                            deal_source = c_y2.radio("商談ソース", ["銀行紹介", "その他"], horizontal=True, index=0 if st.session_state.get("deal_source", "その他") == "銀行紹介" else 1, key="deal_source")
                            lease_term = c_y3.select_slider("契約期間（月）", options=range(0, 121, 1), value=60)
                            st.divider()
                            c_l, c_r = st.columns([0.7, 0.3])
                            with c_l:
                                acceptance_year = st.number_input("検収年 (西暦)", value=2026, step=1)
                            st.session_state.lease_term = lease_term
                            st.session_state.acceptance_year = acceptance_year
                        st.markdown("### 取得価格")
                        acquisition_cost = _slider_and_number("acquisition_cost", "acquisition_cost", 1000, 0, 500000, 100, 100, label_slider="取得価格調整", max_val_number=90_000_000)
                        st.markdown("### リース物件")
                        if not LEASE_ASSETS_LIST:
                            selected_asset_id = "other"
                            asset_score = 50
                            asset_name = "未選択"
                            st.caption("lease_assets.json を配置すると物件リストから選択できます。")
                        else:
                            options = [f"{it.get('name', '')}（{it.get('score', 0)}点）" for it in LEASE_ASSETS_LIST]
                            default_idx = min(st.session_state.get("selected_asset_index", 0), len(options) - 1) if "selected_asset_index" in st.session_state else 0
                            sel_idx = st.selectbox("物件を選択（点数が判定に反映）", range(len(options)), format_func=lambda i: options[i], index=default_idx, key="lease_asset_select", help="選択した物件の点数を借手スコアに反映します。")
                            st.session_state["selected_asset_index"] = sel_idx
                            selected_item = LEASE_ASSETS_LIST[sel_idx]
                            selected_asset_id = selected_item.get("id", "other")
                            asset_score = int(selected_item.get("score", 50))
                            asset_name = selected_item.get("name", "その他")
                            if selected_item.get("note"):
                                st.caption(f"💡 {selected_item['note']}")
                        st.divider()
                        # ---------- 5. 定性スコアリング（総合×重み＋定性×重みでランクA〜E。定性未選択時は総合スコアのみ） ----------
                        with st.expander("📋 定性スコアリング", expanded=False):
                            st.caption("審査員が定性面を項目別に評価します。ランク（A〜E）は **総合スコア×重み＋定性×重み**（デフォルト60%/40%）で算出。定性を1件も選んでいない場合はランクは出さず、総合スコアのみで判定します。（未選択の項目は定性スコアに含めません）")
                            for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
                                opts = item.get("options") or QUALITATIVE_SCORING_LEVELS
                                opts_display = ["未選択"] + [o[1] for o in opts]
                                st.selectbox(
                                    f"{item['label']}（重み{item['weight']}%）",
                                    range(len(opts_display)),
                                    format_func=lambda i, d=opts_display: d[i],
                                    key=f"qual_corr_{item['id']}",
                                )
                            # 入力値は判定開始ブロックで session_state から取得
                    submitted_judge = st.form_submit_button("判定開始", type="primary", use_container_width=True)

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

            if submitted_judge or st.session_state.pop("_auto_judge", False):
                try:
                    # フラグメント利用時用: session_state の値で上書き（入力ガタつき軽減のため）
                    nenshu = st.session_state.get("nenshu", 0)
                    item9_gross = st.session_state.get("item9_gross", 0)
                    rieki = st.session_state.get("rieki", 0)
                    item4_ord_profit = st.session_state.get("item4_ord_profit", 0)
                    item5_net_income = st.session_state.get("item5_net_income", 0)
                    item10_dep = st.session_state.get("item10_dep", 0)
                    item11_dep_exp = st.session_state.get("item11_dep_exp", 0)
                    item8_rent = st.session_state.get("item8_rent", 0)
                    item12_rent_exp = st.session_state.get("item12_rent_exp", 0)
                    item6_machine = st.session_state.get("item6_machine", 0)
                    item7_other = st.session_state.get("item7_other", 0)
                    net_assets = st.session_state.get("net_assets", 0)
                    total_assets = st.session_state.get("total_assets", 0)
                    bank_credit = st.session_state.get("bank_credit", 0)
                    lease_credit = st.session_state.get("lease_credit", 0)
                    contracts = st.session_state.get("contracts", 0)
                    lease_term = st.session_state.get("lease_term", 0)
                    acquisition_cost = st.session_state.get("acquisition_cost", 0)
                    acceptance_year = st.session_state.get("acceptance_year", 2026)
                
                    # 変数の再マッピング (None -> 0)
                    nenshu = nenshu if nenshu is not None else 0
                    item9_gross = item9_gross if item9_gross is not None else 0
                    rieki = rieki if rieki is not None else 0
                    item4_ord_profit = item4_ord_profit if item4_ord_profit is not None else 0
                    item5_net_income = item5_net_income if item5_net_income is not None else 0
                    item10_dep = item10_dep if item10_dep is not None else 0
                    item11_dep_exp = item11_dep_exp if item11_dep_exp is not None else 0
                    item8_rent = item8_rent if item8_rent is not None else 0
                    item12_rent_exp = item12_rent_exp if item12_rent_exp is not None else 0
                    item6_machine = item6_machine if item6_machine is not None else 0
                    item7_other = item7_other if item7_other is not None else 0
                    net_assets = net_assets if net_assets is not None else 0
                    total_assets = total_assets if total_assets is not None else 0
                    bank_credit = bank_credit if bank_credit is not None else 0
                    lease_credit = lease_credit if lease_credit is not None else 0
                    contracts = contracts if contracts is not None else 0
                    lease_term = lease_term if lease_term is not None else 0
                    acquisition_cost = acquisition_cost if acquisition_cost is not None else 0
    
                    # 必須項目チェック（未入力・不正時は判定をブロック）
                    validation_ok = True
                    missing = []
                    for key, label, cond in REQUIRED_FIELDS:
                        val = locals().get(key)
                        if not cond(val):
                            missing.append(label)
                    if missing:
                        validation_ok = False
                        st.error(
                            f"**判定には次の必須項目を入力してください。**\n\n"
                            f"- 「{'」「'.join(missing)}」は **1以上** の値を入力してください。\n\n"
                            "売上高は比率計算に、総資産は自己資本比率・学習モデルに必要です。"
                        )
                    
                    if validation_ok:
                        # 指標計算
                        user_op_margin = (rieki / nenshu * 100) if nenshu > 0 else 0.0
                        user_equity_ratio = (net_assets / total_assets * 100) if total_assets > 0 else 0.0
                        # 流動比率の簡易算（流動資産≈総資産−固定資産、流動負債≈負債総額）
                        liability_total = total_assets - net_assets if (total_assets and net_assets is not None) else 0
                        current_assets_approx = max(0, total_assets - item6_machine - item7_other)
                        user_current_ratio = (current_assets_approx / liability_total * 100) if liability_total > 0 else 100.0
            
                        bench = benchmarks_data.get(selected_sub, {})
                        bench_op_margin = bench.get("op_margin", 0.0)
                        bench_equity_ratio = _equity_ratio_display(bench.get("equity_ratio")) or 0.0
                        bench_comment = bench.get("comment", "")
            
                        comp_margin = "高い" if user_op_margin >= bench_op_margin else "低い"
                        comp_equity = "高い" if user_equity_ratio >= bench_equity_ratio else "低い"
            
                        comparison_text = f"""
                        - **営業利益率**: {user_op_margin:.1f}% (業界目安: {bench_op_margin}%) → 平均より{comp_margin}
                        - **自己資本比率**: {user_equity_ratio:.1f}% (業界目安: {bench_equity_ratio}%) → 平均より{comp_equity}
                        - **業界特性**: {bench_comment}
                        ※ **銀行与信・リース与信**は総銀行与信・総リース与信ではなく、**当社（弊社）の与信**である。判定・アドバイスではこの点を踏まえること。
                        """
            
                        my_hints = hints_data.get(selected_sub, {"subsidies": [], "risks": [], "mandatory": ""})
    
                        # 財務ベース倒産確率と業界リスク検索（判定開始時に実行）
                        pd_percent = calculate_pd(user_equity_ratio, user_current_ratio, user_op_margin)
                        try:
                            network_risk_summary = search_bankruptcy_trends(selected_sub)
                        except Exception as _e:
                            network_risk_summary = f"（業界リスクの取得でエラー: {_e}。判定は続行します。）"
    
                        # ==========================================================================
                        # 🧮 スコア計算ロジック
                        # ==========================================================================
            
                        # モデル計算用データ (単位調整版)
                        data_scoring = {
                            # 対数項用 (千円単位のまま)
                            "nenshu": nenshu,             
                            "bank_credit": bank_credit,   
                            "lease_credit": lease_credit, 
                
                            # 線形項用 (百万円単位に変換) - 係数の桁から推測
                            "op_profit": rieki / 1000,
                            "ord_profit": item4_ord_profit / 1000,
                            "net_income": item5_net_income / 1000,
                            "gross_profit": item9_gross / 1000,
                            "machines": item6_machine / 1000,
                            "other_assets": item7_other / 1000,
                            "rent": item8_rent / 1000,
                            "depreciation": item10_dep / 1000,
                            "dep_expense": item11_dep_exp / 1000,
                            "rent_expense": item12_rent_exp / 1000,
                
                            # その他
                            "contracts": contracts,
                            "grade": grade,
                            "industry_major": selected_major,
                        }
            
                        # 安全なシグモイド関数 (オーバーフロー対策)
                        def safe_sigmoid(x):
                            try:
                                # xが大きすぎる、または小さすぎる場合の対策
                                if x > 700: return 1.0
                                if x < -700: return 0.0
                                return 1 / (1 + math.exp(-x))
                            except OverflowError:
                                return 0.0 if x < 0 else 1.0
    
                        def calculate_score_from_coeffs(data, coeff_set):
                            z = coeff_set["intercept"]
                
                            # ダミー変数の適用ロジック
                            major = data["industry_major"]
                            if "医療" in major or "福祉" in major or major.startswith("P"):
                                z += coeff_set.get("ind_medical", 0)
                            elif "運輸" in major or major.startswith("H"):
                                z += coeff_set.get("ind_transport", 0)
                            elif "建設" in major or major.startswith("D"):
                                z += coeff_set.get("ind_construction", 0)
                            elif "製造" in major or major.startswith("E"):
                                z += coeff_set.get("ind_manufacturing", 0)
                            elif "卸売" in major or "小売" in major or "サービス" in major or major[0] in ["I", "K", "M", "R"]:
                                 z += coeff_set.get("ind_service", 0)
                
                            # 対数項 (千円単位の値を対数化)
                            if data["nenshu"] > 0: z += np.log1p(data["nenshu"]) * coeff_set.get("sales_log", 0)
                            if data["bank_credit"] > 0: z += np.log1p(data["bank_credit"]) * coeff_set.get("bank_credit_log", 0)
                            if data["lease_credit"] > 0: z += np.log1p(data["lease_credit"]) * coeff_set.get("lease_credit_log", 0)
                
                            # 線形項 (既に百万円単位に変換済みの値を使用)
                            z += data["op_profit"] * coeff_set.get("op_profit", 0)
                            z += data["ord_profit"] * coeff_set.get("ord_profit", 0)
                            z += data["net_income"] * coeff_set.get("net_income", 0)
                            z += data["machines"] * coeff_set.get("machines", 0)
                            z += data["other_assets"] * coeff_set.get("other_assets", 0)
                            z += data["rent"] * coeff_set.get("rent", 0)
                            z += data["gross_profit"] * coeff_set.get("gross_profit", 0)
                            z += data["depreciation"] * coeff_set.get("depreciation", 0)
                            z += data["dep_expense"] * coeff_set.get("dep_expense", 0)
                            z += data["rent_expense"] * coeff_set.get("rent_expense", 0)
                
                            if "4-6" in data["grade"]: z += coeff_set.get("grade_4_6", 0)
                            elif "要注意" in data["grade"]: z += coeff_set.get("grade_watch", 0)
                            elif "無格付" in data["grade"]: z += coeff_set.get("grade_none", 0)
                
                            z += data["contracts"] * coeff_set.get("contracts", 0)
                
                            # 指標モデル用の追加変数 (比率)
                            z += data.get("ratio_op_margin", 0) * coeff_set.get("ratio_op_margin", 0)
                            z += data.get("ratio_gross_margin", 0) * coeff_set.get("ratio_gross_margin", 0)
                            z += data.get("ratio_ord_margin", 0) * coeff_set.get("ratio_ord_margin", 0)
                            z += data.get("ratio_net_margin", 0) * coeff_set.get("ratio_net_margin", 0)
                            z += data.get("ratio_fixed_assets", 0) * coeff_set.get("ratio_fixed_assets", 0)
                            z += data.get("ratio_rent", 0) * coeff_set.get("ratio_rent", 0)
                            z += data.get("ratio_depreciation", 0) * coeff_set.get("ratio_depreciation", 0)
                            z += data.get("ratio_machines", 0) * coeff_set.get("ratio_machines", 0)
                
                            return z
    
                        # 1. 全体モデル（成約/失注で更新した係数があればそれを優先）
                        z_main = calculate_score_from_coeffs(data_scoring, get_effective_coeffs("全体_既存先"))
                        score_prob = safe_sigmoid(z_main)
                        score_percent = score_prob * 100
            
                        # 2. 指標モデル (比率計算)
                        # マッピングロジック更新 (CSV指示に基づく)
                        # D, P, H -> 全体(指標)
                        # I, K, M, R -> サービス業(指標)
                        # E -> 製造業(指標)
            
                        bench_key = "全体_指標"
                        major_code_bench = selected_major.split(" ")[0]
            
                        if major_code_bench == "D":
                            bench_key = "全体_指標"
                        elif major_code_bench == "P":
                            bench_key = "医療_指標"
                        elif major_code_bench == "H":
                            bench_key = "運送業_指標"
                        elif major_code_bench in ["I", "K", "M", "R"]:
                            bench_key = "サービス業_指標"
                        elif major_code_bench == "E":
                            bench_key = "製造業_指標"
                
                        ratio_data = data_scoring.copy()
            
                        # 比率計算のために元の千円単位の値を使う
                        raw_nenshu = nenshu if nenshu > 0 else 1.0
            
                        raw_op = rieki if rieki is not None else 0
                        raw_gross = item9_gross if item9_gross is not None else 0
                        raw_ord = item4_ord_profit if item4_ord_profit is not None else 0
                        raw_net = item5_net_income if item5_net_income is not None else 0
                        raw_fixed = (item6_machine if item6_machine is not None else 0) + (item7_other if item7_other is not None else 0)
                        raw_rent = item12_rent_exp if item12_rent_exp is not None else 0
                        raw_dep = (item10_dep if item10_dep is not None else 0) + (item11_dep_exp if item11_dep_exp is not None else 0)
                        raw_machines = item6_machine if item6_machine is not None else 0
            
                        ratio_data["ratio_op_margin"] = raw_op / raw_nenshu
                        ratio_data["ratio_gross_margin"] = raw_gross / raw_nenshu
                        ratio_data["ratio_ord_margin"] = raw_ord / raw_nenshu
                        ratio_data["ratio_net_margin"] = raw_net / raw_nenshu
                        ratio_data["ratio_fixed_assets"] = raw_fixed / raw_nenshu
                        ratio_data["ratio_rent"] = raw_rent / raw_nenshu
                        ratio_data["ratio_depreciation"] = raw_dep / raw_nenshu
                        ratio_data["ratio_machines"] = raw_machines / raw_nenshu
            
                        # 指標モデル計算（既存先/新規先で更新係数があれば使用）
                        bench_key_with_type = f"{bench_key}_{'新規先' if customer_type == '新規先' else '既存先'}"
                        bench_coeffs = get_effective_coeffs(bench_key_with_type)
                        z_bench = calculate_score_from_coeffs(ratio_data, bench_coeffs)
                        score_prob_bench = safe_sigmoid(z_bench)
                        score_percent_bench = score_prob_bench * 100
            
                        # 3. 業種別モデル (分類ロジックの修正)
                        ind_key = "全体_既存先" # デフォルト
            
                        major_code = selected_major.split(" ")[0] # "D 建設業" -> "D"
            
                        # CSV定義に基づくマッピング
                        # H -> 運送業
                        # I, K, M, R -> サービス業
                        # E -> 製造業
                        # D, P -> 全体モデル (既存or新規)
            
                        if major_code == "H":
                            ind_key = "運送業_既存先"
                        elif major_code == "P":
                            ind_key = "医療_既存先"
                        elif major_code in ["I", "K", "M", "R"]:
                            ind_key = "サービス業_既存先"
                        elif major_code == "E":
                            ind_key = "製造業_既存先"
                        elif major_code == "D":
                            ind_key = "全体_既存先"
            
                        # 新規先の場合の切り替え
                        if customer_type == "新規先":
                            ind_key = ind_key.replace("既存先", "新規先")
                            # 万が一キーがない場合は全体_新規先へフォールバック
                            if ind_key not in COEFFS: ind_key = "全体_新規先"
            
                        ind_coeffs = get_effective_coeffs(ind_key)
                        z_ind = calculate_score_from_coeffs(data_scoring, ind_coeffs)
                        score_prob_ind = safe_sigmoid(z_ind)
                        score_percent_ind = score_prob_ind * 100
            
                        gap_val = score_percent - score_percent_bench
                        gap_sign = "+" if gap_val >= 0 else ""
                        gap_text = f"指標モデル差: {gap_sign}{gap_val:.1f}%"
    
                        # ========== 完全版ベイズ初期モデル: 継承＋補完（回帰で更新した係数も反映） ==========
                        effective = get_effective_coeffs()  # 成約/失注で更新した係数（既存+追加項目）があれば使用
                        # 逆転の鍵は削除済み（定性は定性スコアリングのみ）
                        strength_tags = []
                        passion_text = ""
                        n_strength = 0
                        contract_prob = score_percent
                        ai_completed_factors = []  # AIが補完した判定要因（表示・バトル用）
    
                        # メイン先（係数: 成約/失注で回帰更新されていればその値、なければ既定5）
                        # ※ 係数分析・更新モードで回帰分析を実行すると、成約/失注データから自動的に係数が再計算されます
                        main_bank_eff = effective.get("main_bank", 5)
                        if main_bank == "メイン先":
                            contract_prob += main_bank_eff
                            ai_completed_factors.append({"factor": "メイン取引先", "effect_percent": int(round(main_bank_eff)), "detail": "取引行として優位"})
    
                        # 競合: 競合あり=負の係数、競合なし=プラス（成約/失注で回帰更新されていればその値、なければ既定）
                        # ※ 係数分析・更新モードで回帰分析を実行すると、成約/失注データから自動的に係数が再計算されます
                        comp_present_eff = effective.get("competitor_present", BAYESIAN_PRIOR_EXTRA["competitor_present"])
                        comp_none_eff = effective.get("competitor_none", 5)
                        comp_effect = comp_present_eff if competitor == "競合あり" else comp_none_eff
                        contract_prob += comp_effect
                        if competitor == "競合あり":
                            ai_completed_factors.append({"factor": "競合他社の存在", "effect_percent": int(round(comp_effect)), "detail": "他社がいる場合は成約率を下げる補正"})
                        else:
                            ai_completed_factors.append({"factor": "競合なし", "effect_percent": int(round(comp_effect)), "detail": "競合優位で成約率を上げる補正"})
    
                        # 業界景気動向: Z化（-1,0,1）。係数は更新値 or 既定
                        _summary = (network_risk_summary or "").lower()
                        if "景気" in _summary or "好調" in _summary or "拡大" in _summary or "堅調" in _summary:
                            industry_z = 1.0
                            ind_label = "業界動向（ポジティブ）"
                        elif "倒産" in _summary or "減少" in _summary or "悪化" in _summary or "懸念" in _summary or "低下" in _summary:
                            industry_z = -1.0
                            ind_label = "業界動向（ネガティブ）"
                        else:
                            industry_z = 0.0
                            ind_label = "業界動向（中立）"
                        ind_coef = effective.get("industry_sentiment_z", BAYESIAN_PRIOR_EXTRA["industry_sentiment_per_z"])
                        ind_effect = ind_coef * industry_z
                        contract_prob += ind_effect
                        if industry_z != 0:
                            ai_completed_factors.append({"factor": ind_label, "effect_percent": int(round(ind_effect)), "detail": "業界の景気動向を成約率に反映"})
    
                        # 金利差は y_pred_adjusted 算出後に追加

                        # 定性スコア: タグスコア(0-10)と熱意(0/1)。係数は「1ポイントあたり」「熱意ありで」の効果（更新値 or 既定）
                        tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in strength_tags), 10)
                        tag_coef = effective.get("qualitative_tag_score", 2.0)   # 1ptあたり%効果
                        passion_coef = effective.get("qualitative_passion", BAYESIAN_PRIOR_EXTRA["qualitative_passion_bonus"])
                        tag_effect = tag_coef * tag_score
                        passion_effect = passion_coef if passion_text else 0
                        contract_prob += tag_effect + passion_effect
                        if n_strength > 0:
                            ai_completed_factors.append({"factor": "定性スコア（強みタグ）", "effect_percent": int(round(tag_effect)), "detail": f"特許・人脈等{n_strength}件を標準重みで加点"})
                        if passion_effect > 0:
                            ai_completed_factors.append({"factor": "熱意・裏事情の記述", "effect_percent": int(round(passion_effect)), "detail": "記述ありで加点"})
    
                        # 自己資本比率（追加項目）: 係数は「1%あたり」の効果（更新値 or 0）
                        equity_coef = effective.get("equity_ratio", 0)
                        equity_effect = equity_coef * user_equity_ratio
                        contract_prob += equity_effect
                        if abs(equity_effect) >= 0.5:
                            ai_completed_factors.append({"factor": "自己資本比率", "effect_percent": int(round(equity_effect)), "detail": f"自己資本比率 {user_equity_ratio:.1f}% を反映"})

                        from bayesian_engine import THRESHOLD_APPROVAL
                        approval_line = THRESHOLD_APPROVAL * 100
                        
                        # BNエンジン出力（スコア≤承認ライン かつ BN推論実行済の場合のみ有効）
                        _bn_res_sc = st.session_state.get("_bn_s_result")
                        if _bn_res_sc:
                            _bn_im_sc  = _bn_res_sc.get("intermediate") or {}
                            _bn_ap_v   = float(_bn_res_sc.get("approval_prob") or 0)
                            _bn_fc_v   = float(_bn_im_sc.get("Financial_Creditworthiness") or 0)
                            _bn_hc_v   = float(_bn_im_sc.get("Hedge_Condition") or 0)
                            _bn_av_v   = float(_bn_im_sc.get("Asset_Value") or 0)
                            _bn_ap_c   = effective.get("bn_approval_prob", 0)
                            _bn_fc_c   = effective.get("bn_fc", 0)
                            _bn_hc_c   = effective.get("bn_hc", 0)
                            _bn_av_c   = effective.get("bn_av", 0)
                            _bn_effect = (_bn_ap_c * _bn_ap_v + _bn_fc_c * _bn_fc_v
                                          + _bn_hc_c * _bn_hc_v + _bn_av_c * _bn_av_v)
                            contract_prob += _bn_effect
                            if abs(_bn_effect) >= 0.5:
                                ai_completed_factors.append({
                                    "factor": "BN承認確率",
                                    "effect_percent": int(round(_bn_effect)),
                                    "detail": (f"BN承認確率{_bn_ap_v:.0%}・"
                                               f"財務{_bn_fc_v:.0%}・"
                                               f"ヘッジ{_bn_hc_v:.0%}・"
                                               f"物件{_bn_av_v:.0%}"),
                                })

                        contract_prob = max(0, min(100, contract_prob))
    
                        # 利回り予測計算 (簡略化)
                        YIELD_COEFFS = {
                            "intercept": -132.213, "item10_dep": -5.2e-07, "item11_dep_exp": -5.9e-07,
                            "item12_rent_exp": -3.3e-07, "grade_1_3": 0.103051, "grade_4_6": 0.115129,
                            "grade_watch": 0.309849, "grade_none": 0.25737, "type_general": 0.032238,
                            "source_bank": 0.062498, "nenshu_log": -0.03134, "bank_credit_log": -0.00841,
                            "lease_credit_log": -0.02849, "term_log": -0.63635, "year": 0.067637,
                            "cost_log": -0.3945, "contracts_log": 0.130446
                        }
            
                        # 利回り予測モデルには「千円単位の生の数字」を使う (画像の例に従う)
                        # ただし、対数項は log1p(千円) を使用
                        y_pred = YIELD_COEFFS["intercept"]
                        y_pred += item10_dep * YIELD_COEFFS["item10_dep"]
                        y_pred += item11_dep_exp * YIELD_COEFFS["item11_dep_exp"]
                        y_pred += item12_rent_exp * YIELD_COEFFS["item12_rent_exp"]
            
                        if "1-3" in grade: y_pred += YIELD_COEFFS["grade_1_3"]
                        elif "4-6" in grade: y_pred += YIELD_COEFFS["grade_4_6"]
                        elif "要注意" in grade: y_pred += YIELD_COEFFS["grade_watch"]
                        elif "無格付" in grade: y_pred += YIELD_COEFFS["grade_none"]
            
                        if contract_type == "一般": y_pred += YIELD_COEFFS["type_general"]
                        if deal_source == "銀行紹介": y_pred += YIELD_COEFFS["source_bank"]
            
                        if nenshu > 0: y_pred += np.log1p(nenshu) * YIELD_COEFFS["nenshu_log"]
                        if bank_credit > 0: y_pred += np.log1p(bank_credit) * YIELD_COEFFS["bank_credit_log"]
                        if lease_credit > 0: y_pred += np.log1p(lease_credit) * YIELD_COEFFS["lease_credit_log"]
                        if lease_term > 0: y_pred += np.log1p(lease_term) * YIELD_COEFFS["term_log"]
                        if contracts > 0: y_pred += np.log1p(contracts) * YIELD_COEFFS["contracts_log"]
            
                        val_cost_log = np.log1p(acquisition_cost) if acquisition_cost > 0 else 0
                        y_pred += val_cost_log * YIELD_COEFFS["cost_log"]
                        y_pred += acceptance_year * YIELD_COEFFS["year"]
            
                        # 金利環境補正
                        BASE_DATE = "2025-03"
                        term_years = lease_term / 12
                        base_market_rate = get_market_rate(BASE_DATE, term_years)
                        today_str = datetime.date.today().strftime("%Y-%m")
                        current_market_rate = get_market_rate(today_str, term_years)
                        rate_diff = current_market_rate - base_market_rate
                        y_pred_adjusted = y_pred + rate_diff

                        # 金利差（競合比）: 係数は更新値 or 既定
                        competitor_rate_val = st.session_state.get("competitor_rate")
                        if competitor_rate_val is not None and isinstance(competitor_rate_val, (int, float)):
                            rate_diff_pt = float(y_pred_adjusted) - float(competitor_rate_val)
                            rate_z = max(-2, min(2, rate_diff_pt / 5.0))
                            rate_coef = effective.get("rate_diff_z", BAYESIAN_PRIOR_EXTRA["rate_diff_per_z"])
                            rate_effect = rate_coef * (-rate_z)
                            contract_prob += rate_effect
                            ai_completed_factors.append({"factor": "金利差（競合比）", "effect_percent": int(round(rate_effect)), "detail": f"自社が競合より{'有利' if rate_diff_pt < 0 else '不利'}な金利"})
                        contract_prob = max(0, min(100, contract_prob))

                        # 借手スコア + 物件スコア → 総合スコア（判定に反映）。重みは回帰最適化で変更可能。
                        w_borrower, w_asset, w_quant, w_qual = get_score_weights()
                        final_score = w_borrower * score_percent + w_asset * asset_score
                        st.session_state['current_image'] = "approve" if final_score >= APPROVAL_LINE else "challenge"
                
                        # [削除] AIアドバイス (1回目: 入力タブ側)
                        # ここにあった ai_question 生成と messages 追加ロジックは削除し、
                        # 分析結果タブでのみ参照するようにします。
                        # ただし、裏でプロンプト生成だけはしておく必要があるため、セッションステートへの保存は残します。
    
                        # 過去の類似案件（同業界・自己資本比率が近い）を最大3件取得
                        similar_cases = find_similar_past_cases(selected_sub, user_equity_ratio, max_count=3)
                        similar_cases_block = ""
                        if similar_cases:
                            similar_cases_block = "【参考：過去の類似案件の結末】\n"
                            for i, sc in enumerate(similar_cases, 1):
                                res = sc.get("result") or {}
                                eq = res.get("user_eq")
                                sc_score = res.get("score")
                                status = sc.get("final_status", "未登録")
                                eq_str = f"{_equity_ratio_display(eq) or 0:.1f}%" if eq is not None else "—"
                                score_str = f"{sc_score:.1f}%" if sc_score is not None else "—"
                                similar_cases_block += f"{i}. 業界: {sc.get('industry_sub', '—')}、自己資本比率: {eq_str}、スコア: {score_str}、結末: {status}\n"
                            similar_cases_block += "\n"
                        instruction_past = "過去に似た数値で承認された（または否決された）事例を参考にし、今回の案件との共通点や相違点を踏まえて、より精度の高い最終判定を出してください。\n\n"
    
                        ai_question_text = ""
                        if similar_cases_block:
                            ai_question_text += similar_cases_block + instruction_past
                        # 過去の競合・成約金利をコンテキストとして追加（競合に勝つ対策をAIに促す）
                        past_stats = get_stats(selected_sub)
                        if past_stats.get("top_competitors_lost") or (past_stats.get("avg_winning_rate") is not None and past_stats["avg_winning_rate"] > 0):
                            ai_question_text += "【過去の競合・成約金利】\n"
                            if past_stats.get("top_competitors_lost"):
                                ai_question_text += "よく負ける競合: " + "、".join(past_stats["top_competitors_lost"][:5]) + "。\n"
                            if past_stats.get("avg_winning_rate") and past_stats["avg_winning_rate"] > 0:
                                ai_question_text += f"同業種の平均成約金利: {past_stats['avg_winning_rate']:.2f}%。\n"
                            ai_question_text += "上記を踏まえ、競合に勝つための対策も考慮してアドバイスしてください。\n\n"
                        ai_question_text += "審査お疲れ様です。手元の決算書から、以下の**3点だけ**確認させてください。\n\n"
                        from bayesian_engine import THRESHOLD_APPROVAL
                        approval_line = THRESHOLD_APPROVAL * 100
                        
                        questions = []
                        if my_hints.get("mandatory"): questions.append(f"🏭 **業界確認**: {my_hints['mandatory']}")
                        if score_percent < approval_line: questions.append("💡 **実質利益**: 販管費の内訳に「役員報酬」は十分計上されていますか？")
                        elif user_op_margin < bench_op_margin: questions.append("📉 **利益率要因**: 今期の利益率低下は、一過性ですか？")
                        if score_percent < approval_line: questions.append("🏦 **資金繰り**: 借入金明細表で、返済が「約定通り」進んでいるか確認してください。")
                        if my_hints["risks"]: questions.append(f"⚠️ **業界リスク**: {my_hints['risks'][0]} はクリアしていますか？")
                
                        for q in questions[:3]: ai_question_text += f"- {q}\n"
                        ai_question_text += "\nこれらがクリアになれば、承認確率80%以上が見込めます。"
                        ai_question_text += f"\n\n業界の最新リスク情報も参照済みです。これらを総合して最終的なリスクと承認可否を判断してください。"
    
                        # チャット履歴に追加 (表示は分析タブのチャット欄で行う)
                        st.session_state.messages = [{"role": "assistant", "content": ai_question_text}]
                        st.session_state.debate_history = [] 
    
                        # 議論終了・判定プロンプト用に類似案件ブロックを保持
                        similar_past_for_prompt = (similar_cases_block + instruction_past) if similar_cases_block else ""
    
                        # 定性ワンホット（過去データ・RAG用）
                        qualitative_onehot = {tag: 1 for tag in STRENGTH_TAG_OPTIONS if tag in strength_tags}
                        qualitative_onehot.update({tag: 0 for tag in STRENGTH_TAG_OPTIONS if tag not in strength_tags})

                        # 定性スコアリングの集計（総合×60%＋定性×40%でランクA〜E）
                        qual_correction_items = {}
                        qual_weight_sum = 0
                        qual_weighted_total = 0.0
                        for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
                            idx = st.session_state.get(f"qual_corr_{item['id']}", 0)
                            opts = item.get("options") or QUALITATIVE_SCORING_LEVELS
                            val = opts[idx - 1][0] if 1 <= idx <= len(opts) else None
                            level_label = opts[idx - 1][1] if 1 <= idx <= len(opts) else None
                            qual_correction_items[item["id"]] = {
                                "value": val,
                                "label": item["label"],
                                "weight": item["weight"],
                                "level_label": level_label,
                            }
                            if val is not None:
                                qual_weight_sum += item["weight"]
                                qual_weighted_total += (val / 4.0) * 100 * (item["weight"] / 100.0)
                        qual_weighted_score = round((qual_weighted_total / qual_weight_sum * 100) if qual_weight_sum > 0 else 0)
                        qual_weighted_score = min(100, max(0, qual_weighted_score))
                        # ランクA〜Eは総合×重み＋定性×重みに基づく（重みは回帰最適化で変更可能）
                        combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                        combined_score = min(100, max(0, combined_score))
                        qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                        qualitative_scoring_correction = None
                        if qual_weight_sum > 0:
                            qualitative_scoring_correction = {
                                "items": qual_correction_items,
                                "weighted_score": qual_weighted_score,
                                "combined_score": combined_score,
                                "rank": qual_rank["label"],
                                "rank_text": qual_rank["text"],
                                "rank_desc": qual_rank["desc"],
                            }

                        # 学習モデル（業種別ハイブリッド）の予測（総資産・純資産が入力されている場合のみ）
                        scoring_result = None
                        if (total_assets or 0) > 0 and (net_assets is not None) and (net_assets >= 0):
                            try:
                                _scoring_dir = os.path.join(_SCRIPT_DIR, "scoring")
                                if _scoring_dir not in sys.path:
                                    sys.path.insert(0, _SCRIPT_DIR)
                                from scoring.predict_one import predict_one, map_industry_major_to_scoring
                                _base = os.environ.get("LEASE_SCORING_MODELS_DIR", os.path.join(_SCRIPT_DIR, "scoring", "models", "industry_specific"))
                                _industry = map_industry_major_to_scoring(selected_major)
                                scoring_result = predict_one(
                                    revenue=(nenshu or 0) * 1000,
                                    total_assets=(total_assets or 0) * 1000,
                                    equity=(net_assets or 0) * 1000,
                                    operating_profit=(rieki or 0) * 1000,
                                    net_income=(item5_net_income or 0) * 1000,
                                    machinery_equipment=(item6_machine or 0) * 1000,
                                    other_fixed_assets=(item7_other or 0) * 1000,
                                    depreciation=((item10_dep or 0) + (item11_dep_exp or 0)) * 1000,
                                    rent_expense=(item12_rent_exp or 0) * 1000,
                                    industry=_industry,
                                    base_path=_base,
                                )
                            except Exception:
                                scoring_result = None

                        # 学習モデル判定が「否決」のときはすべてのスコアを50%減
                        if scoring_result and (scoring_result.get("decision") or "").strip() == "否決":
                            final_score = final_score * SCORE_PENALTY_IF_LEARNING_REJECT
                            contract_prob = contract_prob * SCORE_PENALTY_IF_LEARNING_REJECT
                            score_percent = score_percent * SCORE_PENALTY_IF_LEARNING_REJECT
                            score_percent_bench = (score_percent_bench or 0) * SCORE_PENALTY_IF_LEARNING_REJECT
                            score_percent_ind = (score_percent_ind or 0) * SCORE_PENALTY_IF_LEARNING_REJECT
                            # 定性スコアリングの合計・ランクも否決後の総合で再計算
                            if qualitative_scoring_correction:
                                combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                                combined_score = min(100, max(0, combined_score))
                                qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                                qualitative_scoring_correction["combined_score"] = combined_score
                                qualitative_scoring_correction["rank"] = qual_rank["label"]
                                qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                                qualitative_scoring_correction["rank_desc"] = qual_rank["desc"]

                        # デフォルト率50%以上の場合、総合スコアから-50点
                        if pd_percent >= 50:
                            final_score = max(0, final_score - 50)
                            if qualitative_scoring_correction:
                                combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                                combined_score = min(100, max(0, combined_score))
                                qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                                qualitative_scoring_correction["combined_score"] = combined_score
                                qualitative_scoring_correction["rank"] = qual_rank["label"]
                                qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                                qualitative_scoring_correction["rank_desc"] = qual_rank["desc"]

                        # 新しい審査を実行したのでチャット履歴をリセット
                        st.session_state["messages"] = []
                        st.session_state["debate_history"] = []

                        # カスタムルールの適用
                        custom_rules_list = _rules.get("custom_rules", [])
                        if custom_rules_list:
                            context_data = {
                                "industry": selected_major,
                                "nenshu": nenshu,
                                "op_profit": rieki,
                                "ord_profit": item4_ord_profit,
                                "net_income": item5_net_income,
                                "net_assets": net_assets,
                                "total_assets": total_assets,
                                "user_eq_ratio": user_equity_ratio,
                                "term": lease_term,
                                "cost": acquisition_cost,
                                "bank_credit": bank_credit,
                                "lease_credit": lease_credit
                            }
                            cr_result = evaluate_custom_rules(custom_rules_list, context_data)
                            
                            # ペナルティ（減点）の適用
                            if cr_result["score_delta"] != 0:
                                delta = cr_result["score_delta"]
                                final_score = max(0, final_score + delta)
                            
                            # リストに理由を追加し、AI用プロンプトにも結合
                            if cr_result.get("applied_reasons"):
                                for reason in cr_result["applied_reasons"]:
                                    ai_completed_factors.append({
                                        "factor": "社内独自ルール",
                                        "effect_percent": int(round(cr_result["score_delta"])),
                                        "detail": reason
                                    })
                                    similar_past_for_prompt += f"\n【重要: 社内カスタムルール発動】\n- {reason}\n※AIは、このルールに抵触したことを踏まえて、顧客への改善アドバイスや留意点をアクションプランに含めてください。\n"
                                    
                            # 定性スコアリングの合計等も再計算
                            if qualitative_scoring_correction:
                                combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                                combined_score = min(100, max(0, combined_score))
                                qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                                qualitative_scoring_correction["combined_score"] = combined_score
                                qualitative_scoring_correction["rank"] = qual_rank["label"]
                                qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                            
                            # 強制ステータスの保持（後段の `hantei` トグル等で利用）
                            forced_custom_status = cr_result.get("forced_status")
                        else:
                            forced_custom_status = None

                        # 将来予測シミュレーション（AI連携用の事前計算）
                        future_sim_result = None
                        if (nenshu or 0) > 0:
                            try:
                                from future_simulation import run_business_simulation
                                future_sim_result = run_business_simulation(
                                    current_sales=nenshu,
                                    current_op_profit=rieki,
                                    drift=0.01,
                                    volatility=0.15,
                                    years=5,
                                    n_simulations=5000
                                )
                            except Exception:
                                pass

                        st.session_state['last_result'] = {
                            "score": final_score, "hantei": "承認圏内" if final_score >= APPROVAL_LINE else "要審議",
                            "score_borrower": score_percent, "asset_score": asset_score, "asset_name": asset_name,
                            "contract_prob": contract_prob, "z": z_main,
                            "ai_completed_factors": ai_completed_factors,
                            "comparison": comparison_text,
                            "user_op": user_op_margin, "bench_op": bench_op_margin,
                            "user_eq": user_equity_ratio, "bench_eq": bench_equity_ratio,
                            "hints": my_hints,
                            "pd_percent": pd_percent,
                            "network_risk_summary": network_risk_summary,
                            "similar_past_cases_prompt": similar_past_for_prompt,
                            "strength_tags": strength_tags,
                            "passion_text": passion_text,
                            "qualitative_onehot": qualitative_onehot,
                            "scoring_result": scoring_result,
                            "future_sim_result": future_sim_result,
                            "qualitative_scoring_correction": qualitative_scoring_correction,
                            "financials": {
                                "nenshu": nenshu,
                                "rieki": rieki,
                                "assets": total_assets,
                                "net_assets": net_assets,
                                "gross_profit": item9_gross,
                                "op_profit": rieki,
                                "ord_profit": item4_ord_profit,
                                "net_income": item5_net_income,
                                "machines": item6_machine,
                                "other_assets": item7_other,
                                "bank_credit": bank_credit,
                                "lease_credit": lease_credit,
                                "depreciation": item10_dep,
                            },
                            "yield_pred": y_pred_adjusted, "yield_base": y_pred, "rate_diff": rate_diff,
                            "gap_text": gap_text, "bench_score": score_percent_bench,
                            "ind_score": score_percent_ind, "ind_name": ind_key,
                            "industry_major": selected_major,
                            "industry_sub": selected_sub,
                            "industry_sentiment_z": industry_z,
                        }

                        # カスタムルールの影響や強制ステータスの上書き
                        if forced_custom_status:
                            st.session_state['last_result']["hantei"] = forced_custom_status
                            st.session_state['current_image'] = "challenge" if forced_custom_status in ["要審議", "否決"] else "approve"
                        elif final_score < REVIEW_LINE:
                            st.session_state['last_result']["hantei"] = "否決"
                            st.session_state['current_image'] = "challenge"
                        
                        # 審査委員会カードバトル用データ（分析タブで表示）
                        hp_card = int(min(999, max(1, net_assets / 1000))) if net_assets else int(min(999, max(1, user_equity_ratio * 5)))
                        atk_card = int(min(99, max(1, user_op_margin * 2)))
                        spd_card = int(min(99, max(1, user_current_ratio / 2)))
                        is_approved = final_score >= APPROVAL_LINE
                        # 補完要因をスキル・環境効果としてバトルに渡す
                        env_effects = [f"{f['factor']}: {f['effect_percent']:+.0f}%" for f in ai_completed_factors]
                        st.session_state["battle_data"] = {
                            "hp": hp_card, "atk": atk_card, "spd": spd_card,
                            "is_approved": is_approved,
                            "special_move_name": None, "special_effect": None,
                            "battle_log": [], "dice": None,
                            "score": final_score, "hantei": "承認圏内" if is_approved else "要審議",  # is_approved = (final_score >= APPROVAL_LINE)
                            "environment_effects": env_effects,
                            "ai_completed_factors": ai_completed_factors,
                        }
                        st.session_state["show_battle"] = False  # 別枠（開発中）のため判定後はダッシュボードへ直行

                        # ログ保存 (自動)
                        log_payload = {
                            "industry_major": selected_major,
                            "industry_sub": selected_sub,
                            "customer_type": customer_type,
                            "main_bank": main_bank,
                            "competitor": competitor,
                            "competitor_rate": st.session_state.get("competitor_rate"),
                            "inputs": {
                                "nenshu": nenshu,
                                "gross_profit": item9_gross,
                                "op_profit": rieki,
                                "ord_profit": item4_ord_profit,
                                "net_income": item5_net_income,
                                "machines": item6_machine,
                                "other_assets": item7_other,
                                "rent": item8_rent,
                                "depreciation": item10_dep,
                                "dep_expense": item11_dep_exp,
                                "rent_expense": item12_rent_exp,
                                "bank_credit": bank_credit,
                                "lease_credit": lease_credit,
                                "contracts": contracts,
                                "grade": grade,
                                "contract_type": contract_type,
                                "deal_source": deal_source,
                                "lease_term": lease_term,
                                "acceptance_year": acceptance_year,
                                "acquisition_cost": acquisition_cost,
                                "lease_asset_id": selected_asset_id,
                                "lease_asset_name": asset_name,
                                "lease_asset_score": asset_score,
                                "qualitative": {
                                    "strength_tags": strength_tags,
                                    "passion_text": passion_text,
                                    "onehot": qualitative_onehot,
                                },
                                "qualitative_scoring": qualitative_scoring_correction,
                            },
                            "result": st.session_state['last_result'],
                            "pricing": {
                                "base_rate": 1.2,
                                "pred_rate": y_pred_adjusted
                            },
                            "bn_engine": {
                                "evidence": st.session_state.get("_bn_s_evidence"),
                                "approval_prob": st.session_state.get("_bn_s_result", {}).get("approval_prob") if st.session_state.get("_bn_s_result") else None,
                                "decision": st.session_state.get("_bn_s_result", {}).get("decision") if st.session_state.get("_bn_s_result") else None,
                                "intermediate": st.session_state.get("_bn_s_result", {}).get("intermediate") if st.session_state.get("_bn_s_result") else None,
                            } if st.session_state.get("_bn_s_result") else None,
                        }
                        # 案件ログを保存し、案件IDをセッションに保持しておく
                        case_id = save_case_log(log_payload)
                        if case_id is None:
                            st.error("ログ保存に失敗しました。")
                        else:
                            st.session_state["current_case_id"] = case_id
                            # 戻ったときにクリアされないよう、今回の入力値をすべて保存（訂正で戻ったときに復元）
                            submitted_qual_corr = {f"qual_corr_{item['id']}": st.session_state.get(f"qual_corr_{item['id']}", 0) for item in QUALITATIVE_SCORING_CORRECTION_ITEMS}
                            st.session_state["last_submitted_inputs"] = {
                                "nenshu": nenshu, "item9_gross": item9_gross, "rieki": rieki,
                                "item4_ord_profit": item4_ord_profit, "item5_net_income": item5_net_income,
                                "item10_dep": item10_dep, "item11_dep_exp": item11_dep_exp,
                                "item8_rent": item8_rent, "item12_rent_exp": item12_rent_exp,
                                "item6_machine": item6_machine, "item7_other": item7_other,
                                "net_assets": net_assets, "total_assets": total_assets,
                                "bank_credit": bank_credit, "lease_credit": lease_credit,
                                "contracts": contracts, "lease_term": lease_term,
                                "acquisition_cost": acquisition_cost, "acceptance_year": acceptance_year,
                                "selected_major": selected_major, "selected_sub": selected_sub,
                                "grade": grade, "main_bank": main_bank, "competitor": competitor,
                                "customer_type": customer_type, "contract_type": contract_type,
                                "deal_source": deal_source,
                                "selected_asset_index": st.session_state.get("selected_asset_index", 0),
                                **submitted_qual_corr,
                            }
                            st.session_state["form_restored_from_submit"] = False
                            st.session_state.nav_index = 1  # 1番目（分析結果）に切り替える
                            st.session_state["_jump_to_analysis"] = True  # 判定直後の1回だけ分析結果に飛ぶ
                            st.rerun()  # 画面を読み込み直して、実際にタブを移動させる
                except Exception as e:
                    st.error("判定開始の処理中にエラーが発生しました。入力内容を確認するか、ページを再読み込みして再度お試しください。")
                    import traceback
                    with st.expander("エラー詳細", expanded=False):
                        st.code(traceback.format_exc())

        if nav_mode == "📊 分析結果":
            # ── クイック再入力パネル（全項目） ───────────────────────────
            with st.expander("✏️ 全項目編集して再判定", expanded=False):
                st.caption("すべての入力項目をここから変更できます。「🔄 再判定」で即座に再計算します。")

                # ─── 業種 ───────────────────────────────────────────────
                st.markdown("#### 🏭 業種")
                _q_major_keys = list(jsic_data.keys()) if jsic_data else ["D 建設業"]
                _q_cur_major = st.session_state.get("select_major") or st.session_state.get("last_submitted_inputs", {}).get("selected_major", _q_major_keys[0])
                _q_major_idx = _q_major_keys.index(_q_cur_major) if _q_cur_major in _q_major_keys else 0
                _q_major = st.selectbox("大分類", _q_major_keys, index=_q_major_idx, key="_quick_major")
                _q_sub_keys = list(jsic_data[_q_major]["sub"].keys()) if jsic_data and _q_major in jsic_data else ["06 総合工事業"]
                _q_cur_sub = st.session_state.get("select_sub") or st.session_state.get("last_submitted_inputs", {}).get("selected_sub", _q_sub_keys[0])
                _q_sub_idx = _q_sub_keys.index(_q_cur_sub) if _q_cur_sub in _q_sub_keys else 0
                _q_sub = st.selectbox("中分類", _q_sub_keys, index=_q_sub_idx, key="_quick_sub")

                st.divider()

                # ─── 損益計算書 ─────────────────────────────────────────
                st.markdown("#### 📊 損益計算書 P/L（千円）")
                _q1, _q2, _q3 = st.columns(3)
                with _q1:
                    _q_nenshu = st.number_input("売上高", min_value=0, max_value=90_000_000, value=int(st.session_state.get("nenshu", 0)), step=1000, key="_quick_nenshu")
                with _q2:
                    _q_gross = st.number_input("売上総利益（粗利）", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item9_gross", 0)), step=1000, key="_quick_gross")
                with _q3:
                    _q_rieki = st.number_input("営業利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("rieki", 0)), step=1000, key="_quick_rieki")
                _q4, _q5 = st.columns(2)
                with _q4:
                    _q_ord = st.number_input("経常利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item4_ord_profit", 0)), step=1000, key="_quick_ord")
                with _q5:
                    _q_net_income = st.number_input("当期利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item5_net_income", 0)), step=1000, key="_quick_net_income")

                st.divider()

                # ─── 資産・経費 ──────────────────────────────────────────
                st.markdown("#### 🏢 資産・経費（千円）")
                _qA1, _qA2, _qA3 = st.columns(3)
                with _qA1:
                    _q_dep = st.number_input("減価償却費（資産）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item10_dep", 0)), step=1000, key="_quick_dep")
                    _q_dep_exp = st.number_input("減価償却費（経費）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item11_dep_exp", 0)), step=1000, key="_quick_dep_exp")
                with _qA2:
                    _q_rent = st.number_input("賃借料（資産）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item8_rent", 0)), step=1000, key="_quick_rent")
                    _q_rent_exp = st.number_input("賃借料（経費）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item12_rent_exp", 0)), step=1000, key="_quick_rent_exp")
                with _qA3:
                    _q_machine = st.number_input("機械装置", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item6_machine", 0)), step=1000, key="_quick_machine")
                    _q_other = st.number_input("その他資産", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item7_other", 0)), step=1000, key="_quick_other")
                _qB1, _qB2 = st.columns(2)
                with _qB1:
                    _q_net = st.number_input("純資産", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("net_assets", 0)), step=1000, key="_quick_net")
                with _qB2:
                    _q_total = st.number_input("総資産", min_value=0, max_value=90_000_000, value=int(st.session_state.get("total_assets", 0)), step=1000, key="_quick_total")

                st.divider()

                # ─── 信用情報 ────────────────────────────────────────────
                st.markdown("#### 💳 信用情報")
                _qC1, _qC2 = st.columns(2)
                with _qC1:
                    _grade_opts = ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"]
                    _q_cur_grade = st.session_state.get("grade", "②4-6 (標準)")
                    _q_grade_idx = _grade_opts.index(_q_cur_grade) if _q_cur_grade in _grade_opts else 1
                    _q_grade = st.selectbox("格付", _grade_opts, index=_q_grade_idx, key="_quick_grade")
                    _q_bank = st.number_input("銀行与信（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("bank_credit", 0)), step=1000, key="_quick_bank")
                with _qC2:
                    _q_lease = st.number_input("リース与信（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("lease_credit", 0)), step=1000, key="_quick_lease")
                    _q_contracts = st.number_input("契約数（件）", min_value=0, max_value=200, value=int(st.session_state.get("contracts", 0)), step=1, key="_quick_contracts")

                st.divider()

                # ─── 契約条件 ────────────────────────────────────────────
                st.markdown("#### 📋 契約条件・物件")
                _qD1, _qD2, _qD3 = st.columns(3)
                with _qD1:
                    _q_ctype = st.radio("顧客区分", ["既存先", "新規先"], index=0 if st.session_state.get("customer_type", "既存先") == "既存先" else 1, horizontal=True, key="_quick_ctype")
                    _q_contract_type = st.radio("契約種類", ["一般", "自動車"], index=0 if st.session_state.get("contract_type", "一般") == "一般" else 1, horizontal=True, key="_quick_contract_type")
                with _qD2:
                    _q_deal_source = st.radio("商談ソース", ["銀行紹介", "その他"], index=0 if st.session_state.get("deal_source", "その他") == "銀行紹介" else 1, horizontal=True, key="_quick_deal_source")
                    _q_lease_term = st.number_input("契約期間（月）", min_value=0, max_value=120, value=int(st.session_state.get("lease_term", 0)), step=1, key="_quick_lease_term")
                with _qD3:
                    _q_acceptance_year = st.number_input("検収年（西暦）", min_value=2000, max_value=2100, value=int(st.session_state.get("acceptance_year", 2026)), step=1, key="_quick_acceptance_year")
                    _q_acq = st.number_input("取得価格（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("acquisition_cost", 0)), step=100, key="_quick_acq")
                if LEASE_ASSETS_LIST:
                    _q_asset_opts = [f"{it.get('name', '')}（{it.get('score', 0)}点）" for it in LEASE_ASSETS_LIST]
                    _q_asset_idx = min(st.session_state.get("selected_asset_index", 0), len(_q_asset_opts) - 1)
                    _q_asset_sel = st.selectbox("リース物件", range(len(_q_asset_opts)), format_func=lambda i: _q_asset_opts[i], index=_q_asset_idx, key="_quick_asset")
                else:
                    _q_asset_sel = None

                st.divider()

                # ─── 定性スコアリング ────────────────────────────────────
                st.markdown("#### 📝 定性スコアリング")
                _q_qual = {}
                for _qi, _qitem in enumerate(QUALITATIVE_SCORING_CORRECTION_ITEMS):
                    _qopts = _qitem.get("options") or QUALITATIVE_SCORING_LEVELS
                    _qopts_display = ["未選択"] + [o[1] for o in _qopts]
                    _qcur = st.session_state.get(f"qual_corr_{_qitem['id']}", 0)
                    _q_qual[_qitem["id"]] = st.selectbox(
                        f"{_qitem['label']}（重み{_qitem['weight']}%）",
                        range(len(_qopts_display)),
                        format_func=lambda i, d=_qopts_display: d[i],
                        index=_qcur,
                        key=f"_quick_qual_{_qitem['id']}",
                    )

                st.divider()
                if st.button("🔄 再判定", type="primary", use_container_width=True):
                    # 業種
                    st.session_state["select_major"] = _q_major
                    st.session_state["select_sub"] = _q_sub
                    # P/L
                    st.session_state["nenshu"] = _q_nenshu
                    st.session_state["item9_gross"] = _q_gross
                    st.session_state["rieki"] = _q_rieki
                    st.session_state["item4_ord_profit"] = _q_ord
                    st.session_state["item5_net_income"] = _q_net_income
                    # 資産・経費
                    st.session_state["item10_dep"] = _q_dep
                    st.session_state["item11_dep_exp"] = _q_dep_exp
                    st.session_state["item8_rent"] = _q_rent
                    st.session_state["item12_rent_exp"] = _q_rent_exp
                    st.session_state["item6_machine"] = _q_machine
                    st.session_state["item7_other"] = _q_other
                    st.session_state["net_assets"] = _q_net
                    st.session_state["total_assets"] = _q_total
                    # 信用情報
                    st.session_state["grade"] = _q_grade
                    st.session_state["bank_credit"] = _q_bank
                    st.session_state["lease_credit"] = _q_lease
                    st.session_state["contracts"] = _q_contracts
                    # 契約条件
                    st.session_state["customer_type"] = _q_ctype
                    st.session_state["contract_type"] = _q_contract_type
                    st.session_state["deal_source"] = _q_deal_source
                    st.session_state["lease_term"] = _q_lease_term
                    st.session_state["acceptance_year"] = _q_acceptance_year
                    st.session_state["acquisition_cost"] = _q_acq
                    if _q_asset_sel is not None:
                        st.session_state["selected_asset_index"] = _q_asset_sel
                    # 定性スコアリング
                    for _qid, _qval in _q_qual.items():
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
                          <div style="font-size:0.8rem;opacity:0.7;">総合スコア</div>
                          <div style="font-size:2rem;font-weight:bold;">{res['score']:.1f}%</div>
                        </div>
                        <div>
                          <div style="font-size:0.8rem;opacity:0.7;">契約期待度</div>
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
                    st.plotly_chart(plot_gauge_plotly(res['score'], "総合スコア"), use_container_width=True, key="gauge_score")

                st.divider()

                # ── 報告書PDF ──────────────────────────────────────────────────────
                with st.expander("📄 審査報告書 PDF出力", expanded=False):
                    if "last_result" not in st.session_state:
                        st.info("👈「新規審査」で審査を実行すると報告書を出力できます。")
                    else:
                        from screening_report import build_screening_report_pdf
                        _rep_res = st.session_state["last_result"]
                        st.caption(f"業種：{_rep_res.get('industry_sub','')}　スコア：{_rep_res.get('score',0):.1f}")

                        # ── BN逆転承認シミュレーター（スコア≤承認ライン のとき表示） ──────
                        from bayesian_engine import THRESHOLD_APPROVAL
                        approval_line = THRESHOLD_APPROVAL * 100
                        if _rep_res.get("score", 100) <= approval_line:
                            st.markdown("### 🔄 逆転承認シミュレーター（AI提案）")
                            st.write("現在の財務スコアおよびBN判定では「否決」です。以下の条件を追加することで承認に持ち込めるかシミュレーションします。")
                            st.caption("総合スコアが低い案件でも、追加条件を整えることで承認確率を高められます。入力項目をチェックして推論してください。")
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
                                    _bn_core      = st.checkbox("本業に不可欠な物件",      key="bn_s_core",      value=False)
                                    _bn_liquidity = st.checkbox("物件の中古流動性あり",     key="bn_s_liquidity", value=False)
                                    _bn_shorter   = st.checkbox("リース期間を短縮",         key="bn_s_shorter",   value=False)
                                    _bn_one_time  = st.checkbox("業況改善まで本件限り",      key="bn_s_one_time",  value=False)
                                # チェック変更のたびに即時再計算（ボタン不要）
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
                                    st.divider()
                                    _r1, _r2, _r3, _r4 = st.columns(4)
                                    _r1.metric("🎯 承認確率",  f"{_bn_prob:.1%}")
                                    _r2.metric("財務信用度",   f"{_im.get('Financial_Creditworthiness', 0):.1%}")
                                    _r3.metric("ヘッジ条件",   f"{_im.get('Hedge_Condition', 0):.1%}")
                                    _r4.metric("物件価値",     f"{_im.get('Asset_Value', 0):.1%}")
                                    st.markdown(f"**判定: <span style='color:{_bn_col}'>{_bn_dec}</span>**", unsafe_allow_html=True)
                                    _rev = _bn_reversal(_bne, top_n=5)
                                    if _rev:
                                        st.markdown("---")
                                        st.markdown("**💡 逆転提案 — これを取り付けると承認確率が上がります（PDFに赤字で記載されます）**")
                                        for _r in _rev:
                                            _gain_pct = int(_r["delta"] * 100)
                                            st.markdown(
                                                f"- **{_r['label']}** を ON にする →"
                                                f" {_r['before_prob']:.0%} → **{_r['after_prob']:.0%}**"
                                                f"（+{_gain_pct}%pt）　{_r['after_decision']}"
                                            )
                            except Exception as _bn_err:
                                st.caption(f"🧠 BNエンジン利用不可: {type(_bn_err).__name__}")

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
                                    _bn_result_for_pdf   = st.session_state.get("_bn_s_result")
                                    _bn_evidence_for_pdf = st.session_state.get("_bn_s_evidence") or {}
                                    _bn_reversal_for_pdf = []
                                    if _bn_result_for_pdf:
                                        try:
                                            from bayesian_engine import compute_reversal_suggestions as _bn_rev_fn
                                            _bn_reversal_for_pdf = _bn_rev_fn(_bn_evidence_for_pdf, top_n=5)
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
                            st.metric("総合スコア", f"{total_score:.1f}", help="借手スコア85%＋物件スコア15%")
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
                        st.metric("契約期待度", f"{res.get('contract_prob', 0):.1f}%", help="定性補正後の期待度")
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
                    st.subheader("🌐 業界リスク情報")
                    # ----- 業界リスク情報（ダッシュボード直下・フル幅） -----
                    net_summary = res.get("network_risk_summary", "") or ""
                    if net_summary.strip() and "取得できません" not in net_summary and "検索エラー" not in net_summary:
                        st.text_area("ネット検索で取得した倒産トレンド・リスク", value=net_summary[:1500] + ("…" if len(net_summary) > 1500 else ""), height=120, disabled=True, label_visibility="collapsed")
                    else:
                        st.caption("判定開始時に業界リスクを検索します。未取得の場合は審査入力で再実行してください。")

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
    with col_right:
        # Ensure selected_sub is up-to-date for chat
        if "last_result" in st.session_state:
            selected_sub = st.session_state["last_result"].get("industry_sub", selected_sub)
        st.header("💬 AI審査オフィサーに相談")
        st.caption(f"選択中の業種: {selected_sub}")
        
        tab_chat, tab_debate = st.tabs(["相談モード", "⚔️ 討論モード"])

        # 現在のAIエンジンとAPIキー状態を表示（Gemini時は「未設定」だと動かないので明示）
        _engine = st.session_state.get("ai_engine", "ollama")
        if _engine == "gemini":
            _key_ok = bool(
                (st.session_state.get("gemini_api_key") or "").strip()
                or GEMINI_API_KEY_ENV
                or _get_gemini_key_from_secrets()
            )
            st.caption(f"🤖 使用中: **Gemini API**　｜　APIキー: **{'設定済み' if _key_ok else '未設定（サイドバーで入力）'}**")
            with st.expander("🔧 Gemini デバッグ（動かないときに開く）", expanded=False):
                _dbg = st.session_state.get("last_gemini_debug", "まだ呼び出していません")
                st.text(_dbg)
                st.caption("相談で送信後、ここに「OK」またはエラー内容が表示されます。")
        else:
            st.caption("🤖 使用中: **Ollama（ローカル）**")
        
        with tab_chat:
            # ナレッジ参照トグル（マニュアル・事例集・FAQ）
            with st.expander("📚 マニュアル・事例集・FAQをAIに参照させる", expanded=False):
                st.caption("有効にすると「審査マニュアル」「業種別ガイド」「FAQ集」「事例集」の内容がAIへの質問に自動的に付加されます。")
                _kb_use_manual = st.checkbox("審査マニュアル・スコアリング基準", value=True, key="kb_use_manual")
                _kb_use_industry = st.checkbox("業種別ガイド（財務目安・審査ポイント）", value=True, key="kb_use_industry")
                _kb_use_faq = st.checkbox("FAQ集（よくある質問と回答）", value=True, key="kb_use_faq")
                _kb_use_cases = st.checkbox("審査事例集（Bランク・Cランク・Dランクの実例）", value=True, key="kb_use_cases")
                _kb_use_improvement = st.checkbox("スコア改善ガイド（短期・中期の改善アクション）", value=False, key="kb_use_improvement")

            # 音声入力から戻ったときのテキストを反映（URLの ?voice_text=... で渡される）
            if st.query_params.get("voice_text"):
                st.session_state["consultation_input"] = st.query_params.get("voice_text", "")
                try:
                    st.experimental_set_query_params()
                except Exception:
                    pass
                st.rerun()
            if "messages" not in st.session_state: st.session_state.messages = []
            if "consultation_input" not in st.session_state: st.session_state["consultation_input"] = ""
            # 送信済みの場合は入力欄を空にする（text_area 作成前にのみ session_state を変更可能）
            if "consultation_pending_q" in st.session_state:
                st.session_state["consultation_input"] = ""

            chat_box = st.container(height=400)
            with chat_box:
                for m in st.session_state.messages:
                    if m["role"] != "system":
                        with st.chat_message(m["role"]): st.markdown(m["content"])
            
            # バックグラウンドでAPI応答待ち中 → クルクル見せるためにポーリング
            # スレッド結果は _chat_result_holder で受け取る（session_state はスレッドから反映されないため）
            CHAT_LOADING_TIMEOUT = 125  # 秒（API側のタイムアウトより少し長め）
            if _chat_result_holder["done"]:
                result = _chat_result_holder["result"]
                _chat_result_holder["result"] = None
                _chat_result_holder["done"] = False
                st.session_state["chat_result"] = result
                st.session_state["chat_loading"] = False
                if st.session_state.get("ai_engine") == "gemini" and result:
                    c = (result.get("message") or {}).get("content", "")
                    st.session_state["last_gemini_debug"] = "OK" if c and "APIキーが" not in c and "Gemini API エラー:" not in c else (c[:200] + "..." if len(c or "") > 200 else (c or "（空）"))
            chat_loading = st.session_state.get("chat_loading", False)
            chat_result = st.session_state.get("chat_result")
            # 待機タイムアウト：一定時間応答がなければ強制解除
            loading_started = st.session_state.get("chat_loading_started_at")
            if chat_loading and loading_started is not None and (time.time() - loading_started) > CHAT_LOADING_TIMEOUT:
                st.session_state["chat_loading"] = False
                _chat_result_holder["done"] = True
                _chat_result_holder["result"] = {"message": {"content": "応答がタイムアウトしました（約2分）。\n\n・APIキー・ネット接続を確認するか、もう一度送信してください。\n・Gemini の場合は無料枠の制限に達している可能性もあります。"}}
                st.rerun()
            if chat_loading or chat_result is not None:
                with chat_box:
                    for m in st.session_state.messages:
                        if m["role"] != "system":
                            with st.chat_message(m["role"]): st.markdown(m["content"])
                    with st.chat_message("assistant"):
                        if chat_result is not None:
                            content = (chat_result.get("message") or {}).get("content", "")
                            if content and (
                                "APIキーが設定されていません" in content
                                or "Gemini API エラー:" in content
                                or "pip install" in content
                                or "応答が返りませんでした" in content
                                or "安全フィルターでブロック" in content
                            ):
                                st.error(content)
                            st.markdown(content or "（応答がありませんでした）")
                            st.session_state.messages.append({"role": "assistant", "content": content or "（応答がありませんでした）"})
                            # ホルダー経由の応答も相談メモに保存（話せば話すほど蓄積）
                            user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user"]
                            if user_msgs:
                                append_consultation_memory(user_msgs[-1], content or "（応答がありませんでした）")
                            st.session_state["chat_loading"] = False
                            st.session_state["chat_result"] = None
                        else:
                            with st.status("思考中...", state="running", expanded=True):
                                st.markdown("⏳ 応答を待っています...")
                                if st.button("待機をやめる", key="chat_cancel_loading"):
                                    st.session_state["chat_loading"] = False
                                    _chat_result_holder["done"] = True
                                    _chat_result_holder["result"] = {"message": {"content": "待機を解除しました。もう一度送信するか、APIキー・ネット接続を確認してください。"}}
                                    st.rerun()
                            time.sleep(1)
                            st.rerun()

            # 定性情報・相談入力（text_area + 音声入力ボタン + 送信）
            st.text_area("相談内容", value=st.session_state.get("consultation_input", ""), key="consultation_input", height=100, placeholder="相談する内容を入力...（下の🎤で音声入力もできます）", label_visibility="collapsed")
            voice_html = """
            <script>
            function startVoiceInput() {
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                    alert('お使いのブラウザは音声入力に対応していません。Chrome などでお試しください。');
                    return;
                }
                var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                var rec = new SpeechRecognition();
                rec.lang = 'ja-JP';
                rec.continuous = false;
                rec.interimResults = false;
                rec.onresult = function(e) {
                    var t = e.results[0][0].transcript;
                    var u = window.parent.location.pathname + '?voice_text=' + encodeURIComponent(t);
                    window.parent.location = u;
                };
                rec.onerror = function(e) {
                    if (e.error === 'not-allowed') alert('マイクの利用が許可されていません。');
                    else alert('音声認識エラー: ' + e.error);
                };
                rec.start();
            }
            </script>
            <button type="button" onclick="startVoiceInput()" style="padding: 8px 16px; font-size: 1rem; cursor: pointer; border-radius: 8px; background: #f0f2f6; border: 1px solid #ccc;">
            🎤 音声入力
            </button>
            """
            # コメント欄が右で切れないよう、入力行はカラム幅を抑える
            btn_col1, btn_col2 = st.columns([1, 3])
            with btn_col1:
                st.components.v1.html(voice_html, height=50)
            with btn_col2:
                send_clicked = st.button("送信", key="consultation_send", type="primary")
            if send_clicked and (st.session_state.get("consultation_input") or "").strip():
                st.session_state["consultation_pending_q"] = (st.session_state.get("consultation_input") or "").strip()
                st.rerun()

            q = None
            if st.session_state.get("consultation_pending_q"):
                q = st.session_state.pop("consultation_pending_q")
                st.session_state.messages.append({"role": "user", "content": q})
            if q:
                with chat_box:
                    with st.chat_message("user"): st.markdown(q)
                    with st.chat_message("assistant"):
                        if not is_ai_available():
                            if st.session_state.get("ai_engine") == "gemini":
                                st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                            else:
                                st.error(f"AIサーバー（Ollama）が起動していません。\nターミナルで `ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                        else:
                            # ── ai_chat.py へ分離したプロンプト構築ロジックを呼び出す ──
                            from ai_chat import get_ai_consultation_prompt
                            
                            with st.spinner("業種別トピックス等を取得中..."):
                                context_prompt = get_ai_consultation_prompt(
                                    q=q,
                                    res=_res,
                                    selected_sub=selected_sub,
                                    jsic_data=jsic_data,
                                    news_content=st.session_state.get('selected_news_content'),
                                    kb_use_faq=st.session_state.get("kb_use_faq", True),
                                    kb_use_cases=st.session_state.get("kb_use_cases", True),
                                    kb_use_manual=st.session_state.get("kb_use_manual", True),
                                    kb_use_industry=st.session_state.get("kb_use_industry", True),
                                    kb_use_improvement=st.session_state.get("kb_use_improvement", False)
                                )
                            _engine = st.session_state.get("ai_engine", "ollama")
                            _model = get_ollama_model()
                            _api_key = (st.session_state.get("gemini_api_key") or "").strip() or GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()
                            _gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
                            # スレッドではなくメインスレッドで同期的に呼ぶ（rerunでスレッドが消えるため応答が返らなくなる問題を回避）
                            with st.spinner("思考中..."):
                                ans = _chat_for_thread(_engine, _model, [{"role": "user", "content": context_prompt}], timeout_seconds=120, api_key=_api_key, gemini_model=_gemini_model)
                            content = (ans.get("message") or {}).get("content", "") or "（応答がありませんでした）"
                            if content and (
                                "APIキーが設定されていません" in content
                                or "Gemini API エラー:" in content
                                or "pip install" in content
                                or "応答が返りませんでした" in content
                                or "安全フィルターでブロック" in content
                            ):
                                st.error(content)
                            else:
                                st.markdown(content)
                            st.session_state.messages.append({"role": "assistant", "content": content})
                            # 相談1往復をメモに保存（話せば話すほど以後の相談で活用）
                            append_consultation_memory(q, content)
                            if st.session_state.get("ai_engine") == "gemini" and content and "APIキーが" not in content and "Gemini API エラー:" not in content:
                                st.session_state["last_gemini_debug"] = "OK"
                            elif st.session_state.get("ai_engine") == "gemini":
                                st.session_state["last_gemini_debug"] = (content[:200] + "...") if len(content or "") > 200 else (content or "（空）")

        with tab_debate:
            # 審査委員会モード：3ペルソナ（慎重派・推進派・審判）の性格定義
            PERSONA_CON = """あなたは「慎重派（守り）」のベテラン審査部長です。
・財務の欠点、業界リスク、倒産確率の不安を徹底的に突き、厳しい条件を出す立場です。
・発言には必ず【ネット検索結果】または【財務データ】の具体的な数値・事実を引用し、根拠を示してください。一般論のみの主張は禁止です。"""
            PERSONA_PRO = """あなたは「推進派（攻め）」の営業担当です。
・企業の情熱・将来性・ネットで見つけた好材料を強調し、前向きな支援を主張する立場です。
・発言には必ず【ネット検索結果】または【財務データ】の具体的な数値・好材料を引用し、根拠を示してください。一般論のみの主張は禁止です。"""
            PERSONA_JUDGE = """あなたは「審判（決裁者）」です。
・推進派と慎重派の議論を冷静に総括し、最終的な「承認確率(%)」と「具体的な融資条件」を算出する立場です。
・ネット検索結果や財務データに基づく根拠を踏まえ、両論を引用しつつ結論を出してください。"""

            st.info("審査委員会モード：慎重派・推進派・審判の3ペルソナでディベートし、最終決裁を出します。")
            if 'debate_history' not in st.session_state: st.session_state.debate_history = []
            
            # 議論ログの表示
            for m in st.session_state.debate_history:
                avatar = "🙆‍♂️" if m["role"] == "Pro" else "🙅‍♂️"
                if m["role"] == "User": avatar = "👤"
                role_name = "推進派" if m["role"] == "Pro" else ("慎重派" if m["role"] == "Con" else "あなた")
                
                with st.chat_message(m["role"], avatar=avatar):
                    st.markdown(f"**{role_name}**: {m['content']}")
            
            # 議論進行ボタン
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("⚔️ 議論を開始 / 進行 (1ターン進める)", use_container_width=True):
                    if 'last_result' not in st.session_state:
                        st.error("先に審査を実行してください。")
                    else:
                        # コンテキスト準備
                        res = st.session_state['last_result']
                        selected_major = res.get("industry_major", "D 建設業")
                        selected_sub = res.get("industry_sub", "06 総合工事業")
                        comparison_text = res.get("comparison", "")
                        if jsic_data and selected_major in jsic_data:
                            trend_info = jsic_data[selected_major]["sub"].get(selected_sub, "")
                        trend_extended_d = get_trend_extended(selected_sub)
                        if trend_extended_d:
                            trend_info = (trend_info or "") + "\n\n【拡充】\n" + trend_extended_d[:1500]
                        # --------------------------------------
                        score = res['score']
                        risk_context = ""
                        for b in bankruptcy_data:
                            risk_context += f"- {b['type']}: {b['signal']} ({b['check_point']})\n"
                        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.debate_history])

                        # ニュース記事の反映
                        news_context = ""
                        if 'selected_news_content' in st.session_state:
                            news = st.session_state.selected_news_content
                            news_context = f"\n\n【参考ニュース記事: {news['title']}】\n{news['content']}"
                        advice_extras_debate = get_advice_context_extras(selected_sub, selected_major)
                        advice_debate_block = ("補助金・リース・業界拡充: " + advice_extras_debate[:800]) if advice_extras_debate else ""
                        _debate_kb = build_knowledge_context(
                            query=f"{selected_sub} スコア{res.get('score',0):.0f}",
                            industry=selected_sub,
                            use_faq=True,
                            use_cases=True,
                            use_manual=True,
                            use_industry_guide=True,
                            use_improvement=False,
                            max_tokens_approx=1500,
                        )
                        _debate_kb_block = f"\n【審査マニュアル・FAQ・事例集（参考）】\n{_debate_kb}" if _debate_kb else ""

                        # ロール決定 & プロンプト作成（同一モデルでペルソナ切り替え）
                        if not st.session_state.debate_history:
                            next_role = "Pro"
                            from bayesian_engine import THRESHOLD_APPROVAL
                            approval_line = THRESHOLD_APPROVAL * 100
                            
                            prompt = f"""{PERSONA_PRO}

【財務データ】（必ず引用すること）
業種: {selected_sub}
スコア: {score:.1f}点 (承認ライン{approval_line:.0f}点)
財務評価: {comparison_text}

【ネット検索結果・業界材料】
{advice_debate_block}
{news_context if news_context else "（ニュース未読み込み）"}
{_debate_kb_block}

【指示】
- 上記の「財務データ」と「ネット検索結果」のいずれかから必ず1つ以上具体的に引用し、根拠を示したうえで主張すること。
- FAQや事例集に類似ケースがあれば引用してよい。
- 企業の情熱・将来性・好材料を強調し、前向きな支援を主張せよ。
- 140文字以内。
"""
                        else:
                            last_role = st.session_state.debate_history[-1]["role"]
                            if last_role == "User":
                                prev_ai = "Con" 
                                for m in reversed(st.session_state.debate_history[:-1]):
                                    if m["role"] in ["Pro", "Con"]:
                                        prev_ai = m["role"]
                                        break
                                next_role = "Con" if prev_ai == "Pro" else "Pro"
                            else:
                                next_role = "Con" if last_role == "Pro" else "Pro"
                            
                            if next_role == "Con":
                                advice_con_block = ("【補助金・リース判定等】" + advice_extras_debate[:500]) if advice_extras_debate else ""
                                prompt = f"""{PERSONA_CON}

【財務データ・リスク指標】（必ず引用すること）
スコア: {score:.1f}点、財務評価: {comparison_text}
【倒産リスクDB】
{risk_context}

【ネット検索結果・業界リスク】
{news_context if news_context else "（なし）"}
{advice_con_block}

【これまでの議論】
{history_text}

【指示】
- 上記の「財務データ」または「ネット検索結果」から必ず1つ以上具体的に引用し、根拠を示したうえで反論すること。
- 財務の欠点・業界リスク・倒産確率の不安を突き、厳しい条件を出せ。
- 140文字以内。
"""
                            else:  # Pro
                                advice_pro_block = ("【補助金・リース等】" + advice_extras_debate[:500]) if advice_extras_debate else ""
                                prompt = f"""{PERSONA_PRO}

【財務データ】（必ず引用すること）
財務評価: {comparison_text}
スコア: {score:.1f}点

【ネット検索結果・好材料】
{news_context if news_context else "業界の成長性、社長の覚悟"}
{advice_pro_block}

【これまでの議論】
{history_text}

【指示】
- 上記の「財務データ」または「ネット検索結果」から必ず1つ以上具体的に引用し、根拠を示したうえで慎重派に反論せよ。
- 企業の情熱・将来性・好材料を強調し、前向きな支援を主張せよ。
- 140文字以内。
"""
        
                        # AI思考中...
                        if not is_ai_available():
                            if st.session_state.get("ai_engine") == "gemini":
                                st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                            else:
                                st.error(f"AIサーバー（Ollama）が起動していません。\nターミナルで `ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                        else:
                            with st.spinner(f"{next_role}が思考中..."): 
                                try:
                                    # 討論モードではタイムアウトとリトライをやや厳しめに設定
                                    ans = chat_with_retry(
                                        model=get_ollama_model(),
                                        messages=[{"role": "user", "content": prompt}],
                                        retries=1,
                                        timeout_seconds=120,
                                    )
                            
                                    if not ans or 'message' not in ans:
                                        st.error("AIからの応答が不正です。")
                                    else:
                                        msg_content = ans['message']['content']
                                        if msg_content and (
                                            "APIキーが設定されていません" in msg_content
                                            or "Gemini API エラー:" in msg_content
                                            or "pip install" in msg_content
                                            or "応答が返りませんでした" in msg_content
                                            or "安全フィルターでブロック" in msg_content
                                        ):
                                            st.error(msg_content)
                                        st.session_state.debate_history.append({"role": next_role, "content": msg_content})
                                except Exception as e:
                                    st.error(f"AIエラー詳細: {e}")
                            
                            # 即座に再描画
                            st.rerun()
            
            # 終了判定ボタン（審判ペルソナで決裁）
            with col_btn2:
                if len(st.session_state.debate_history) >= 4:
                    res_judge = st.session_state.get("last_result") or {}
                    selected_sub_judge = res_judge.get("industry_sub", "")
                    if st.button("🏁 議論終了・判定", type="primary", use_container_width=True):
                        with st.spinner("審判が決裁中..."):
                            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.debate_history])
                            pd_val = res_judge.get("pd_percent")
                            net_risk = res_judge.get("network_risk_summary", "")
                            pd_str = f"{pd_val:.1f}%" if pd_val is not None else "（未算出）"
                            comparison_judge = res_judge.get("comparison", "")
                            similar_block = res_judge.get("similar_past_cases_prompt", "") or ""
                            judge_prompt = ""
                            if similar_block:
                                judge_prompt += similar_block
                            past_stats_judge = get_stats(selected_sub_judge)
                            if past_stats_judge.get("top_competitors_lost") or (past_stats_judge.get("avg_winning_rate") is not None and past_stats_judge.get("avg_winning_rate", 0) > 0):
                                judge_prompt += "\n【過去の競合・成約金利】\n"
                                if past_stats_judge.get("top_competitors_lost"):
                                    judge_prompt += "よく負ける競合: " + "、".join(past_stats_judge["top_competitors_lost"][:5]) + "\n"
                                if past_stats_judge.get("avg_winning_rate") and past_stats_judge["avg_winning_rate"] > 0:
                                    judge_prompt += f"同業種の平均成約金利: {past_stats_judge['avg_winning_rate']:.2f}%\n"
                                judge_prompt += "上記を踏まえ、融資条件には競合に勝つための対策も反映してください。\n\n"
                            judge_prompt += f"""{PERSONA_JUDGE}

【財務データ】（根拠として引用すること）
財務評価: {comparison_judge}

【ネット検索結果】
【業界の最新リスク情報】
{net_risk if net_risk else "（未取得）"}

【議論ログ（推進派・慎重派の発言）】
{history_text}

【指示】
- 上記の財務データとネット検索結果を根拠に、推進派と慎重派の議論を冷静に総括してください。
- 最終的な「承認確率(%)」と「具体的な融資条件」を算出し、理由を簡潔に述べてください。

出力形式（必ず守ること）:
承認確率: XX%
融資条件: （金利・担保・期間など具体的に）
理由: (80文字以内)
"""
                            if not is_ai_available():
                                if st.session_state.get("ai_engine") == "gemini":
                                    st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                                else:
                                    st.error("Ollama が起動していません。`ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                            else:
                                ans = chat_with_retry(
                                    model=get_ollama_model(),
                                    messages=[{"role": "user", "content": judge_prompt}],
                                    retries=1,
                                    timeout_seconds=120,
                                )
                                result_text = ans['message']['content']
                                
                                st.success("✅ **ディベート結果**")
                                st.write(result_text)
                                
                                save_debate_log({
                                    "industry": selected_sub_judge,
                                    "history": st.session_state.debate_history,
                                    "result": result_text
                                })
            
            # ユーザー介入（チャット入力）
            if user_input := st.chat_input("議論に介入する（回答・指示）", key="debate_input"):
                st.session_state.debate_history.append({"role": "User", "content": user_input})
                st.rerun()


        st.divider()

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
    st.caption("業種や各種条件式を組み合わせて、ペナルティ減点や判定の強制変更（要審議・否決等へ）を行うルール一覧を設定します。それぞれのルール中で「条件を追加」することで複数の条件（AND条件）を加えることができます。")

    # --- セッションステートの初期化 ---
    if "custom_rules_ui_data" not in st.session_state:
        # DB(ビジネスルールJSON)からロードしてUI用に構造変換
        loaded = rules.get("custom_rules", [])
        ui_data = []
        for r in loaded:
            new_r = {
                "industry": r.get("industry", "ALL"),
                "action_type": r.get("action_type", "deduct_score"),
                "action_value": str(r.get("action_value", "10")),
                "conditions": r.get("conditions", [])
            }
            # 旧形式（conditionsがない場合）の互換性維持
            if not new_r["conditions"] and r.get("condition_target"):
                new_r["conditions"] = [{
                    "target": r.get("condition_target"),
                    "op": r.get("condition_op", "<"),
                    "value": float(r.get("condition_value", 0.0))
                }]
            ui_data.append(new_r)
        st.session_state["custom_rules_ui_data"] = ui_data

    # UI表示用マッピング
    TARGET_MAP = {
        "op_profit": "営業利益", "net_assets": "純資産", "user_eq_ratio": "自己資本比率",
        "nenshu": "売上高", "total_assets": "総資産", "bank_credit": "銀行借入", "lease_credit": "リース残高"
    }
    TARGET_INV_MAP = {v: k for k, v in TARGET_MAP.items()}
    ACTION_MAP = {"deduct_score": "スコア減点", "force_status": "ステータス強制変更"}
    ACTION_INV_MAP = {v: k for k, v in ACTION_MAP.items()}
    IND_OPTS = ["ALL"] + (list(jsic_data.keys()) if "jsic_data" in globals() and jsic_data else ["D 建設業", "E 製造業", "G 情報通信業", "H 運送業", "I 卸売・小売業", "M 宿泊・飲食サービス業", "P 医療・福祉"])
    ACT_VAL_OPTS = ["5", "10", "15", "20", "25", "30", "40", "50", "要審議", "否決"]

    # 追加・削除のハンドラー
    def add_new_rule():
        st.session_state["custom_rules_ui_data"].append({
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

    # --- ルールエディタの描画 ---
    for i, r in enumerate(st.session_state["custom_rules_ui_data"]):
        # プレビュー文作成
        act_preview = ACTION_MAP.get(r['action_type'], r['action_type']) + f" ({r['action_value']})"
        cond_len = len(r['conditions'])
        with st.expander(f"⚙️ ルール {i+1} : 業種[{r['industry']}] → {act_preview} (条件数: {cond_len})", expanded=True):
            col1, col2, col3 = st.columns([2, 5, 3])
            
            with col1:
                cur_ind = r["industry"] if r["industry"] in IND_OPTS else "ALL"
                sel_ind = st.selectbox("①対象業種", IND_OPTS, index=IND_OPTS.index(cur_ind), key=f"r_ind_{i}")
                st.session_state["custom_rules_ui_data"][i]["industry"] = sel_ind
                
            with col2:
                cur_act = ACTION_MAP.get(r["action_type"], "スコア減点")
                sel_act = st.selectbox("②アクション種別", list(ACTION_MAP.values()), index=list(ACTION_MAP.values()).index(cur_act), key=f"r_act_{i}")
                st.session_state["custom_rules_ui_data"][i]["action_type"] = ACTION_INV_MAP.get(sel_act)
                
            with col3:
                cur_val = r["action_value"]
                opts = ACT_VAL_OPTS.copy()
                if cur_val not in opts: opts.append(cur_val)
                sel_val = st.selectbox("③アクション値 (減点数 or ステータス名)", opts, index=opts.index(cur_val), key=f"r_val_{i}")
                st.session_state["custom_rules_ui_data"][i]["action_value"] = sel_val

            st.markdown("---")
            st.markdown("**④発動条件リスト（すべての条件を満たした場合に上記アクションを発動）**")
            
            for j, cond in enumerate(r["conditions"]):
                cc1, cc2, cc3, cc4 = st.columns([4, 2, 3, 1])
                with cc1:
                    cur_tgt = TARGET_MAP.get(cond["target"], "営業利益")
                    sel_tgt = st.selectbox("対象指標", list(TARGET_MAP.values()), index=list(TARGET_MAP.values()).index(cur_tgt), key=f"c_tgt_{i}_{j}", label_visibility="collapsed")
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["target"] = TARGET_INV_MAP.get(sel_tgt)
                with cc2:
                    sel_op = st.selectbox("比較", ["<", "<=", "=", ">=", ">"], index=["<", "<=", "=", ">=", ">"].index(cond.get("op", "<")), key=f"c_op_{i}_{j}", label_visibility="collapsed")
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["op"] = sel_op
                with cc3:
                    sel_val = st.number_input("閾値", value=float(cond.get("value", 0.0)), step=1.0, key=f"c_val_{i}_{j}", label_visibility="collapsed")
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["value"] = sel_val
                with cc4:
                    if st.button("🗑️ 削除", key=f"del_cond_{i}_{j}", help="この条件を削除します"):
                        delete_condition(i, j)
                        st.rerun()

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("➕ AND条件を追加", key=f"add_cond_{i}"):
                    add_condition(i)
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ このルール全体を削除", key=f"del_rule_{i}", type="secondary"):
                    delete_rule(i)
                    st.rerun()

    if st.button("➕ 新しいルールを追加", key="add_new_rule", use_container_width=True):
        add_new_rule()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 上記の「カスタムルール」を保存して反映する", type="primary", use_container_width=True):
        rules["custom_rules"] = st.session_state["custom_rules_ui_data"]
        
        if save_business_rules(rules):
            
            # 再読み込みが必要な可能性がある場合のためにセッションからクリア
            # del st.session_state["custom_rules_ui_data"]
            # ただ、そのまま使い続けられるので残しておく。
            st.success("✅ カスタムルールの一覧が正常に保存され、新規の審査から適用されます。")
        else:
            st.error("❌ カスタムルールの保存中にエラーが発生しました。")

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
