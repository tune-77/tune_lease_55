"""
温水式リース審査AI - lease_logic_sumaho12
sumaho10(X) からモジュール分割（ai_chat / web_services）を完了した版。
起動: streamlit run lease_logic_sumaho12/lease_logic_sumaho12.py （リポジトリルートで実行）
"""
import sys
import os
import base64
from pathlib import Path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import streamlit as st
# set_page_config は必ず最初の st 呼び出しにする必要がある
st.set_page_config(page_title="温水式リース審査AI", page_icon="🏢", layout="wide")

@st.cache_resource
def _intro_video_b64() -> str:
    """動画を base64 に変換してキャッシュ（起動時1回のみ・以降はメモリから）"""
    _p = Path(__file__).parent / "static" / "intro_video.mp4"
    if not _p.exists():
        return ""
    with open(_p, "rb") as _f:
        return base64.b64encode(_f.read()).decode()
try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:
    style_metric_cards = None  # pip install streamlit-extras でメトリックをカード風に
import math
import json
import random
import html
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
from app_logger import log_error, log_info, log_warning

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
from constants import (
    APPROVAL_LINE, REVIEW_LINE, SCORE_PENALTY_IF_LEARNING_REJECT, ALERT_BORDERLINE_MIN,
    REQUIRED_FIELDS, RECOMMENDED_FIELDS,
    QUALITATIVE_SCORING_CORRECTION_ITEMS as _CONST_QUAL_ITEMS,
    QUALITATIVE_SCORING_LEVELS as _CONST_QUAL_LEVELS,
    QUALITATIVE_SCORE_RANKS as _CONST_QUAL_RANKS,
)
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
            {html.escape(str(text))}
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
    /*
     * デザインシステム — 色は役割で決める
     *   ブランド色  : #1e3a5f  ヘッダー・構造
     *   操作色      : #2563eb  ボタン・スライダー（赤は否決専用）
     *   承認        : #16a34a  (承認のみ)
     *   条件付き承認: #d97706  (条件付き承認のみ)
     *   要審議      : #ea580c  (要審議のみ)
     *   否決        : #dc2626  (否決のみ)
     */

    /* ── スライダー: 操作色（青）で統一 ── */
    div[data-baseweb="slider"] {
        min-width: min(100%, 320px) !important;
        width: 100% !important;
    }
    @media (max-width: 640px) {
        div[data-baseweb="slider"] { min-width: 100% !important; }
        .stSlider > div { width: 100% !important; }
    }
    /* つまみ: 操作色（青）— 赤は「否決」専用 */
    div[data-baseweb="slider"] div[role="slider"] {
        width: 22px !important;
        height: 22px !important;
        background-color: #2563eb !important;
        border: 2px solid #fff !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.35) !important;
    }
    /* レール: 細めで主張しすぎない */
    div[data-baseweb="slider"] > div {
        height: 7px !important;
    }
    /* ラベル: 読める程度・画面を占拠しない */
    .stSlider label p {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    /* 現在値表示 */
    .stSlider [data-baseweb="slider"] ~ div,
    .stSlider div[data-baseweb="slider"] + div,
    [data-testid="stSlider"] > div > div:last-child {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #1e3a5f !important;
    }
    /* ツールチップ（ドラッグ中） */
    [data-baseweb="tooltip"] span,
    [data-baseweb="tooltip"] div,
    [data-baseweb="popover"] span,
    [data-baseweb="popover"] div {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }

    /* ── グラフ・画像 ── */
    .stImage img, [data-testid="stImage"] img {
        border-radius: 8px !important;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.07) !important;
    }
    .js-plotly-plot .plotly, [data-testid="stPlotlyChart"] div {
        border-radius: 8px !important;
    }
    @media (min-width: 769px) {
        [data-testid="stPlotlyChart"] { max-width: 100% !important; width: 100% !important; margin-left: 0 !important; }
    }

    /* ── レイアウト: 右端切れ対策 ── */
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

    /* ── 左右カラム: 右カラム（AI相談）が切れないように ── */
    [data-testid="stHorizontalBlock"] {
        overflow-x: visible !important;
        max-width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child { min-width: 0 !important; }
    [data-testid="stHorizontalBlock"] > div {
        overflow-x: visible !important;
        overflow-y: visible !important;
    }
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
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"],
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] textarea,
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] > div {
        max-width: 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stHorizontalBlock"] { max-width: 100% !important; }
    [data-testid="stHorizontalBlock"] > div:last-child iframe { max-width: 100% !important; }
    [data-testid="stTextArea"] { max-width: 100% !important; }
    [data-testid="stTextArea"] > div,
    [data-testid="stTextArea"] textarea { max-width: 100% !important; box-sizing: border-box !important; }

    /* AI相談エリア: ブランド左ボーダー（役割を示す） */
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] {
        background: #f0f9ff !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border-left: 3px solid #1e3a5f !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 6px !important;
    }

    /* ── タブナビゲーション ── */
    [data-testid="stTabs"] > div > div { gap: 0 !important; }
    [data-testid="stTabs"] [role="tablist"] { margin-bottom: 0.5rem !important; }
    button[role="tab"] { color: #475569 !important; opacity: 1 !important; }
    button[role="tab"] p, button[role="tab"] span, button[role="tab"] div { color: #475569 !important; opacity: 1 !important; }
    button[role="tab"][aria-selected="true"] { color: #1e3a5f !important; font-weight: 700 !important; border-bottom: 2px solid #1e3a5f !important; }
    button[role="tab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] span,
    button[role="tab"][aria-selected="true"] div { color: #1e3a5f !important; font-weight: 700 !important; }
    button[role="tab"]:hover { color: #1e3a5f !important; background-color: rgba(30, 58, 95, 0.05) !important; }

    /* ── 電光掲示板 ── */
    .byoki-ticker-wrap { overflow: hidden; background: #1e293b; color: #94a3b8; padding: 6px 0; margin: 0 0 0.5rem 0; border-radius: 5px; font-size: 0.82rem; }
    .byoki-ticker-inner { display: inline-block; white-space: nowrap; animation: byoki-scroll 120s linear infinite; }
    .byoki-ticker-inner span { padding-right: 2em; }
    @keyframes byoki-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

    /* ── カード: 汎用 ── */
    .dashboard-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
    .dashboard-kpi-row { margin-bottom: 1.25rem; }
    .dashboard-section-title { color: #1e3a5f; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; }

    /* ── KPIメトリクスカード: グラデなし・役割ボーダー ── */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        margin-right: 0.6rem !important;
        margin-bottom: 0.6rem !important;
        padding: 0.7rem 0.6rem !important;
        min-width: 0 !important;
        background: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 3px solid #1e3a5f !important;
    }
    [data-testid="stMetric"] > div, [data-testid="metric-container"] > div { gap: 0.3rem !important; }
    [data-testid="stMetric"] p, [data-testid="metric-container"] p { margin-bottom: 0.15rem !important; line-height: 1.35 !important; }
    [data-testid="stMetric"] label, [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    /* ── フォーム要素: 標準サイズ ── */
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] p,
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] [role="listbox"],
    [data-testid="stSelectbox"] [role="option"] { font-size: 0.875rem !important; }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] div,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span { font-size: 0.875rem !important; }
    [data-testid="stMultiSelect"] label,
    [data-testid="stMultiSelect"] div,
    [data-testid="stMultiSelect"] p,
    [data-testid="stMultiSelect"] span,
    [data-testid="stMultiSelect"] [role="listbox"],
    [data-testid="stMultiSelect"] [role="option"] { font-size: 0.875rem !important; }
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] div,
    [data-testid="stNumberInput"] input { font-size: 0.875rem !important; }

    /* ── スマホ対応 ── */
    @media screen and (max-width: 768px) {
        .stColumns > div { width: 100% !important; }
        div[data-testid="metric-container"] { min-width: 70px !important; padding: 0.3rem !important; font-size: 0.8rem !important; }
        div[data-testid="stExpander"] { margin: 0.15rem 0; }
        .element-container table { font-size: 0.72rem; }
        .stButton > button { width: 100% !important; }
    }
</style>
    """, unsafe_allow_html=True)
	
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
_STATIC_DATA_DIR = os.path.join(BASE_DIR, "static_data")

@st.cache_data(ttl=3600)
def load_json_data(filename):
    # static_data/ を優先し、なければ BASE_DIR を確認（後方互換）
    for base in [_STATIC_DATA_DIR, BASE_DIR]:
        path = os.path.join(base, filename)
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
capex_lease_data = load_json_data("industry_capex_lease.json")  # e-Stat年度版: リース・設備投資ベンチマーク
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

# 審査判定の定数 ── 正規定義は constants.py。ここでは後方互換のため再エクスポートのみ。
# ⚠️  APPROVAL_LINE = 71 は根拠のない暫定値です（詳細は constants.py のコメントを参照）。
# APPROVAL_LINE / REVIEW_LINE / SCORE_PENALTY_IF_LEARNING_REJECT / ALERT_BORDERLINE_MIN
# は上の from constants import ... で取り込み済み。ここでの再定義は不要。

# REQUIRED_FIELDS / RECOMMENDED_FIELDS は constants.py で定義済み（上の import で取り込み済み）

# 過去案件・係数・相談メモ・ニュースのパスは data_cases で定義（CASES_FILE, COEFF_OVERRIDES_FILE 等を import 済み）
_DATA_DIR = os.path.join(BASE_DIR, "data")
DEBATE_FILE = os.path.join(_DATA_DIR, "debate_logs.jsonl") # ディベートログ
# ネットで取得した業界目安を中分類ごとに保存（年1回・4月1日を境に更新）
WEB_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "web_industry_benchmarks.json")
TRENDS_EXTENDED_FILE = os.path.join(_DATA_DIR, "industry_trends_extended.json")
ASSETS_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "industry_assets_benchmarks.json")
SALES_BAND_FILE = os.path.join(_DATA_DIR, "sales_band_benchmarks.json")
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
        # 環境変数で指定された assets フォルダ（長いファイル名）か判定
        use_long_names = bool(DASHBOARD_IMAGES_ASSETS) and base_dir == DASHBOARD_IMAGES_ASSETS.rstrip(os.sep)
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



def _fragment_nenshu():
    """売上高入力。スライダーは100万千円まで、手入力は900億千円まで。後から動かした方を採用。
    on_change を使わないため st.form 内でも動作する。"""
    st.markdown("### 売上高 📌 必須（1以上）")
    _slider_and_number(
        "nenshu", "nenshuu", 10000, 0, 1_000_000,
        step_slider=100, step_num=10000,
        unit="千円", label_slider="売上高調整",
        max_val_number=90_000_000,
    )
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
    base_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(base_path): return base_path
    desktop_path = os.path.join(os.path.expanduser("~/Desktop/"), filename)
    if os.path.exists(desktop_path): return desktop_path
    return None


# ==============================================================================
# 画面構成
# ==============================================================================
from components.sidebar import render_sidebar, load_byoki_list
mode = render_sidebar(benchmarks_data, useful_life_data, LEASE_ASSETS_LIST)


# ホーム以外のすべての画面に「ホームに戻る」ボタンを表示
if mode != "🏠 ホーム":
    st.write("")
    st.write("")
    if st.button("🏠 ホームに戻る", key="btn_go_home", help="ホーム画面に戻ります"):
        st.session_state["_pending_mode"] = "🏠 ホーム"
        st.rerun()

# モード分岐（サイドバー先頭=ホーム。elif の並びは実装都合。処理結果に影響なし）
if mode == "🏠 ホーム":
    from components.home import render_home
    render_home()

elif mode == "📄 審査レポート":
    from components.report import render_report
    render_report()

elif mode == "🤖 汎用エージェントハブ":
    from components.agent_hub import render_agent_hub
    render_agent_hub()

elif mode == "🔧 係数分析・更新 (β)":
    from components.settings import render_coeff_analysis
    render_coeff_analysis()

elif mode == "📐 係数入力（事前係数）":
    from components.settings import render_prior_coeff_input
    render_prior_coeff_input()

elif mode == "📋 係数変更履歴":
    from components.settings import render_coeff_history
    render_coeff_history()

elif mode == "🪵 アプリログ":
    from components.settings import render_app_logs
    render_app_logs()

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
        if (Path(__file__).parent / "static" / "intro_video.mp4").exists():
            st.markdown("""
<video width="10%" autoplay muted loop playsinline style="display:block; border-radius:6px;">
  <source src="/app/static/intro_video.mp4" type="video/mp4">
</video>
""", unsafe_allow_html=True)
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
                num_competitors = form_result.get("num_competitors", "未入力")
                deal_occurrence = form_result.get("deal_occurrence", "不明")
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
                    RECOMMENDED_FIELDS=RECOMMENDED_FIELDS,
                    benchmarks_data=benchmarks_data,
                    hints_data=hints_data,
                    bankruptcy_data=bankruptcy_data,
                    jsic_data=jsic_data,
                    avg_data=avg_data,
                    _rules=_rules,
                    _SCRIPT_DIR=_SCRIPT_DIR,
                    capex_lease_data=capex_lease_data,
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
                past_cases_log=None,
                current_case_data=None,
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
            # 各状態に専用色（色の役割を分離: 同じ色に複数の意味を持たせない）
            _af_colors = {
                "承認":              "#16a34a",   # 緑: 承認のみ
                "条件付き承認":       "#d97706",   # 琥珀: 条件付き承認のみ
                "要審議（上位承認）":  "#ea580c",   # オレンジ: 要審議のみ
                "否決":              "#dc2626",   # 赤: 否決のみ
            }
            _af_bgs = {
                "承認":              "#f0fdf4",
                "条件付き承認":       "#fffbeb",
                "要審議（上位承認）":  "#fff7ed",
                "否決":              "#fef2f2",
            }
            _af_color = _af_colors.get(_af_result['decision'], "#64748b")
            _af_bg    = _af_bgs.get(_af_result['decision'], "#f8fafc")
            st.markdown(
                f"""<div style="
                  background:{_af_bg};
                  border:1px solid #e2e8f0;
                  border-top:4px solid {_af_color};
                  border-radius:8px;
                  padding:1rem 1.5rem;
                  margin-bottom:1rem;
                  display:flex;
                  align-items:center;
                  gap:0.875rem;
                ">
                  <div style="font-size:1.75rem;line-height:1;">{_af_result['icon']}</div>
                  <div>
                    <div style="font-size:1.5rem;font-weight:800;color:{_af_color};line-height:1.1;">{_af_result['decision']}</div>
                    <div style="font-size:0.82rem;color:#64748b;margin-top:0.15rem;">スコア: <strong style="color:#1e3a5f;">{_af_result['score']}点</strong></div>
                  </div>
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

elif mode == "🤝 エージェントチーム議論":
    from components.agent_team import render_agent_team
    render_agent_team()

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
                log_info("基本設定（ビジネスルール）を保存しました。", context="設定保存")
            else:
                log_warning("基本設定の保存に失敗しました。", context="設定保存")
                st.error("❌ 設定の保存中にエラーが発生しました。（logs/app.log を確認してください）")

    # ── 🔬 Youden 指数で承認ライン自動最適化 ──────────────────────────────
    st.divider()
    st.subheader("🔬 承認ライン自動最適化（Youden 指数）")
    st.caption(
        "成約・失注の蓄積データから ROC 曲線を描き、**感度（成約を承認）と特異度（失注を却下）の合計が最大**になる"
        "閾値を自動で求めます。現在の承認ラインと比較し、ワンクリックで適用できます。"
    )
    st.info("💡 最低6件（成約3件・失注3件以上）のデータが必要です。件数が少ないほど結果は参考程度にしてください。")

    if st.button("📐 最適承認ラインを自動計算する", key="btn_youden_calc"):
        with st.spinner("ROC 曲線を計算中…"):
            from analysis_regression import calc_optimal_approval_line
            _youden_result = calc_optimal_approval_line()
        st.session_state["_youden_result"] = _youden_result

    _yr = st.session_state.get("_youden_result")
    if _yr:
        if "error" in _yr:
            st.error(f"❌ {_yr['error']}")
        else:
            curr = _yr["current_line"]
            opt  = _yr["optimal"]
            diff = opt - curr
            diff_str = f"+{diff}" if diff > 0 else str(diff)

            c_now, c_opt, c_auc = st.columns(3)
            with c_now:
                st.metric("現在の承認ライン", f"{curr} 点")
            with c_opt:
                st.metric("Youden 最適ライン", f"{opt} 点", delta=f"{diff_str} 点")
            with c_auc:
                st.metric("AUC", f"{_yr['auc']:.3f}", help="1.0 = 完全分類 / 0.5 = ランダム")

            st.markdown(f"""
| 指標 | 値 | 意味 |
|------|-----|------|
| Youden 指数 | **{_yr['youden_index']:.3f}** | 感度＋特異度−1（大きいほど良い） |
| 感度 | **{_yr['sensitivity']:.1%}** | 成約案件を「承認圏内」と判定できた割合 |
| 特異度 | **{_yr['specificity']:.1%}** | 失注案件を「要審議以下」と判定できた割合 |
| 分析件数 | 成約 {_yr['n_closed']}件 / 失注 {_yr['n_lost']}件 | |
""")

            # 候補一覧
            with st.expander("📋 上位5候補の詳細", expanded=False):
                import pandas as pd
                cand_df = pd.DataFrame(_yr["threshold_candidates"])
                cand_df.columns = ["閾値(点)", "Youden指数", "感度", "特異度"]
                st.dataframe(cand_df, use_container_width=True, hide_index=True)

            # 適用ボタン
            st.divider()
            if diff == 0:
                st.success(f"✅ 現在の承認ライン（{curr}点）は既に最適値と一致しています。変更不要です。")
            else:
                st.warning(
                    f"⚠️ 現在 **{curr}点** → 最適 **{opt}点** に変更すると、"
                    f"感度 {_yr['sensitivity']:.1%} / 特異度 {_yr['specificity']:.1%} になります。"
                )
                _apply_comment = st.text_input(
                    "変更理由コメント（履歴に記録されます）",
                    value=f"Youden指数自動最適化: {curr}点→{opt}点 (AUC={_yr['auc']:.3f}, 成約{_yr['n_closed']}件/失注{_yr['n_lost']}件)",
                    key="youden_apply_comment"
                )
                if st.button(f"✅ 承認ラインを {opt} 点に適用する", type="primary", key="btn_youden_apply"):
                    _rules_now = load_business_rules()
                    if "thresholds" not in _rules_now:
                        _rules_now["thresholds"] = {}
                    _rules_now["thresholds"]["approval"] = opt / 100.0
                    if save_business_rules(_rules_now):
                        # 変更履歴に記録
                        try:
                            from data_cases import save_coeff_overrides
                            # coeff_history に承認ライン変更を記録
                            from data_cases import _append_coeff_history
                            _before = {"approval_line": curr}
                            _after  = {"approval_line": opt}
                            _append_coeff_history("approval_line", _before, _after, _apply_comment)
                        except Exception as _he:
                            log_warning(f"承認ライン変更履歴の記録に失敗: {_he}", context="Youden適用")
                        log_info(f"承認ライン Youden 自動最適化: {curr}点→{opt}点", context="Youden適用")
                        st.success(f"✅ 承認ラインを **{opt}点** に更新しました。次回の判定から反映されます。")
                        st.session_state.pop("_youden_result", None)
                        st.rerun()
                    else:
                        st.error("❌ 保存に失敗しました。logs/app.log を確認してください。")
    # ── /Youden ─────────────────────────────────────────────────────────────

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
            {"target": "op_profit", "op": "<", "value_type": "number", "value": 0.0}
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
            st.caption("指標 ／ 比較演算子 ／ 値種別（数値 or 項目）／ 閾値または比較先項目　の順で設定。")

            for j, cond in enumerate(r["conditions"]):
                cc1, cc2, cc3, cc4, cc5 = st.columns([3, 1.5, 1.5, 3, 1])
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
                    cur_vtype = cond.get("value_type", "number")
                    if cur_vtype not in ("number", "field"):
                        cur_vtype = "number"
                    sel_vtype = st.selectbox(
                        "値種別", ["数値", "項目"],
                        index=0 if cur_vtype == "number" else 1,
                        key=f"c_vtype_{i}_{j}", label_visibility="collapsed"
                    )
                    vtype = "number" if sel_vtype == "数値" else "field"
                    st.session_state["custom_rules_ui_data"][i]["conditions"][j]["value_type"] = vtype
                with cc4:
                    if vtype == "number":
                        cur_num = cond.get("value", 0.0)
                        if not isinstance(cur_num, (int, float)):
                            cur_num = 0.0
                        sel_num = st.number_input(
                            "閾値", value=float(cur_num), step=1.0,
                            key=f"c_val_num_{i}_{j}", label_visibility="collapsed"
                        )
                        st.session_state["custom_rules_ui_data"][i]["conditions"][j]["value"] = sel_num
                    else:
                        cur_field = cond.get("value", "op_profit")
                        if not isinstance(cur_field, str) or cur_field not in TARGET_MAP:
                            cur_field = "op_profit"
                        cur_field_label = TARGET_MAP.get(cur_field, "営業利益")
                        sel_field = st.selectbox(
                            "比較先項目", list(TARGET_MAP.values()),
                            index=list(TARGET_MAP.values()).index(cur_field_label),
                            key=f"c_val_field_{i}_{j}", label_visibility="collapsed"
                        )
                        st.session_state["custom_rules_ui_data"][i]["conditions"][j]["value"] = TARGET_INV_MAP.get(sel_field)
                with cc5:
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
            log_info("カスタムルールを保存しました。", context="ルール保存")
            st.balloons()
        else:
            log_warning("カスタムルールの保存に失敗しました。", context="ルール保存")
            st.error("❌ 保存中にエラーが発生しました。（logs/app.log を確認してください）")

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
