import streamlit as st
try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:
    style_metric_cards = None  # pip install streamlit-extras でメトリックをカード風に
import math
import os
import json
import random
import re
import ollama
import pandas as pd
import plotly.express as px
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

# ============================================
# AI エンジン設定（Ollama / Gemini API）
# ・Ollama: 環境変数 OLLAMA_MODEL、サイドバーでモデル選択
# ・Gemini: 環境変数 GEMINI_API_KEY または サイドバーでAPIキー入力、モデルは gemini-2.0-flash 等
# ============================================
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "lease-anna")
GEMINI_API_KEY_ENV = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_DEFAULT = "gemini-2.0-flash"  # または gemini-1.5-pro, gemini-1.5-flash

def _get_gemini_key_from_secrets() -> str:
    """secrets.toml が無くても例外にしない。キーがあれば返す。"""
    try:
        if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
            return st.secrets.get("GEMINI_API_KEY", "") or ""
    except Exception:
        pass
    return ""

# 相談モード: スレッド→メインで結果を渡す用（session_state はスレッドから更新されないため）
_chat_result_holder = {"result": None, "done": False}

def get_ollama_model() -> str:
    """
    実際に使用するモデル名を取得するヘルパー。
    - st.session_state['ollama_model'] があればそれを優先
    - なければ環境変数ベースの OLLAMA_MODEL を返す
    """
    model = st.session_state.get("ollama_model", "").strip() if "ollama_model" in st.session_state else ""
    return model or OLLAMA_MODEL
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

    # スライダーの見た目をカスタマイズするCSS（スマホで見やすい幅・つまみ）
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
    </style>
    """, unsafe_allow_html=True)
	
# 🎨 画面のデザイン設定
st.set_page_config(page_title="温水式リース審査AI", page_icon="🏢", layout="wide")

# ==============================================================================
# 共通機能 & キャッシュ最適化
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# フォント設定
FONT_PATH = os.path.join(BASE_DIR, "NotoSansCJKjp-Regular.otf")
if os.path.exists(FONT_PATH):
    fe = fm.FontEntry(fname=FONT_PATH, name='NotoSansCJKjp')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = 'NotoSansCJKjp'
    sns.set_theme(style="whitegrid", font="NotoSansCJKjp")
else:
    sns.set_theme(style="whitegrid", font="sans-serif")

# グラフ共通スタイル（ビジネス向け：ネイビー・グレー・ゴールド/赤アクセント）
CHART_STYLE = {
    "primary": "#1e3a5f",    # ネイビー（メイン）
    "secondary": "#475569",  # スレートグレー
    "good": "#0d9488",      # ティール（良好）
    "warning": "#b45309",   # ゴールド/アンバー（注意）
    "danger": "#b91c1c",    # レッド（要確認）
    "accent": "#b45309",    # ゴールドアクセント
    "bg": "#f8fafc",
    "grid": "#e2e8f0",
    "text": "#334155",
    "text_light": "#64748b",
}
plt.rcParams.update({
    "figure.facecolor": CHART_STYLE["bg"],
    "axes.facecolor": "white",
    "axes.edgecolor": CHART_STYLE["grid"],
    "axes.linewidth": 1.0,
    "grid.alpha": 0.4,
    "grid.color": CHART_STYLE["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

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

# 過去案件データはキャッシュしない
CASES_FILE = os.path.join(BASE_DIR, "past_cases.jsonl")
COEFF_OVERRIDES_FILE = os.path.join(BASE_DIR, "data", "coeff_overrides.json")  # 成約/失注回帰で更新した係数
DEBATE_FILE = os.path.join(BASE_DIR, "debate_logs.jsonl") # ディベートログ
CONSULTATION_MEMORY_FILE = os.path.join(BASE_DIR, "consultation_memory.jsonl")  # AI審査オフィサー相談メモ（話せば話すほど蓄積）
# 案件ごとに紐づけるニュース保存用
CASE_NEWS_FILE = os.path.join(BASE_DIR, "case_news.jsonl")
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
    # フォールバック: 同環境の assets パス（analyze_images と同じ場所）
    for candidate in [
        os.path.join(os.path.dirname(BASE_DIR), ".cursor", "projects", "Users-kobayashiisaoryou-clawd", "assets"),
        "/Users/kobayashiisaoryou/.cursor/projects/Users-kobayashiisaoryou-clawd/assets",
    ]:
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


def _get_benchmark_cutoff_date():
    """業界目安を『年1回・4月1日』で更新するための基準日。この日付以降に取得したデータを有効とする。"""
    today = datetime.date.today()
    april1_this = datetime.date(today.year, 4, 1)
    if today >= april1_this:
        return april1_this
    return datetime.date(today.year - 1, 4, 1)


def _load_web_benchmarks_cache():
    """保存済みのネット業界目安を読み込む"""
    if not os.path.exists(WEB_BENCHMARKS_FILE):
        return {}
    try:
        with open(WEB_BENCHMARKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ネット検索で取得・保存する業界目安のキー一覧（指標の bench にそのまま渡す）
_WEB_BENCH_KEYS = [
    "op_margin", "equity_ratio", "gross_margin", "ord_margin", "net_margin", "dep_ratio",
    "roa", "roe", "asset_turnover", "fixed_ratio", "debt_ratio",
    "fixed_to_equity", "debt_to_equity", "fixed_asset_turnover", "current_asset_ratio", "current_ratio",
]


def _save_web_benchmark(industry_sub: str, data: dict):
    """中分類ごとの業界目安をファイルに追記・上書きする。全指標キーを保存。"""
    cache = _load_web_benchmarks_cache()
    entry = {"fetched_at": datetime.date.today().isoformat(), "snippets": data.get("snippets", [])}
    for k in _WEB_BENCH_KEYS:
        v = data.get(k)
        if v is not None:
            entry[k] = v
    cache[industry_sub] = entry
    try:
        with open(WEB_BENCHMARKS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_json_cache(filepath: str):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_cache(filepath: str, data: dict):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 業界トレンド拡充・資産目安・売上規模帯のキャッシュ（4月1日基準で再利用）
trends_extended_cache = {}
assets_benchmarks_cache = {}
sales_band_cache = {}


def _ensure_web_caches_loaded():
    global trends_extended_cache, assets_benchmarks_cache, sales_band_cache
    if not trends_extended_cache and os.path.exists(TRENDS_EXTENDED_FILE):
        trends_extended_cache.update(_load_json_cache(TRENDS_EXTENDED_FILE))
    if not assets_benchmarks_cache and os.path.exists(ASSETS_BENCHMARKS_FILE):
        assets_benchmarks_cache.update(_load_json_cache(ASSETS_BENCHMARKS_FILE))
    if not sales_band_cache and os.path.exists(SALES_BAND_FILE):
        sales_band_cache.update(_load_json_cache(SALES_BAND_FILE))


def fetch_industry_trend_extended(industry_sub: str, force_refresh: bool = False):
    """業界トレンドをネットで検索して拡充テキストを保存。4月1日基準でキャッシュ有効。"""
    if not industry_sub:
        return ""
    _ensure_web_caches_loaded()
    cutoff = _get_benchmark_cutoff_date()
    cached = trends_extended_cache.get(industry_sub)
    if cached and not force_refresh:
        try:
            if datetime.date.fromisoformat(cached.get("fetched_at", "")) >= cutoff:
                return cached.get("text", "") or ""
        except (ValueError, TypeError):
            pass
    query = f"{industry_sub} 業界動向 2025 課題 見通し"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return ""
    text_parts = []
    for r in results[:4]:
        body = (r.get("body") or "").strip()
        if body:
            text_parts.append(body[:400])
    text = "\n".join(text_parts)[:2000] if text_parts else ""
    trends_extended_cache[industry_sub] = {"fetched_at": datetime.date.today().isoformat(), "text": text}
    _save_json_cache(TRENDS_EXTENDED_FILE, trends_extended_cache)
    return text


def fetch_industry_assets_from_web(industry_sub: str, force_refresh: bool = False):
    """業種別の総資産・流動比率の目安をネット検索して保存。返却: {total_assets_ratio, current_ratio} の辞書的利用。"""
    _ensure_web_caches_loaded()
    import re
    out = {"total_assets_note": "", "current_ratio": None}
    if not industry_sub:
        return out
    cached = assets_benchmarks_cache.get(industry_sub)
    if cached and not force_refresh:
        try:
            if datetime.date.fromisoformat(cached.get("fetched_at", "")) >= _get_benchmark_cutoff_date():
                return {k: cached.get(k) for k in ["total_assets_note", "current_ratio"]}
        except (ValueError, TypeError):
            pass
    query = f"{industry_sub} 業界 総資産 流動比率 目安 平均"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return out
    combined = " ".join([(r.get("body") or "") for r in results])
    m = re.search(r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", combined)
    if m:
        try:
            out["current_ratio"] = float(m.group(1))
        except ValueError:
            pass
    out["total_assets_note"] = combined[:500] if combined else ""
    assets_benchmarks_cache[industry_sub] = {"fetched_at": datetime.date.today().isoformat(), **out}
    _save_json_cache(ASSETS_BENCHMARKS_FILE, assets_benchmarks_cache)
    return out


def fetch_sales_band_benchmarks(force_refresh: bool = False):
    """売上規模帯別の利益率等をネット検索して保存。全体で1件のキャッシュ。"""
    _ensure_web_caches_loaded()
    if sales_band_cache.get("fetched_at") and not force_refresh:
        try:
            if datetime.date.fromisoformat(sales_band_cache["fetched_at"]) >= _get_benchmark_cutoff_date():
                return sales_band_cache.get("text", "")
        except (ValueError, TypeError):
            pass
    query = "中小企業 売上規模 利益率 平均 売上高別 統計"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return ""
    text = "\n".join([(r.get("body") or "")[:300] for r in results[:4]])
    sales_band_cache["fetched_at"] = datetime.date.today().isoformat()
    sales_band_cache["text"] = text
    _save_json_cache(SALES_BAND_FILE, sales_band_cache)
    return text


def get_trend_extended(industry_sub: str):
    """業界トレンド拡充テキストを返す（キャッシュがあればそれ、なければ空）。"""
    _ensure_web_caches_loaded()
    c = trends_extended_cache.get(industry_sub)
    return (c.get("text") or "") if c else ""


def get_assets_benchmark(industry_sub: str):
    """業種別資産目安を返す。"""
    _ensure_web_caches_loaded()
    c = assets_benchmarks_cache.get(industry_sub)
    return c if c else {}


def get_sales_band_text():
    """売上規模帯別指標のテキストを返す。"""
    _ensure_web_caches_loaded()
    return sales_band_cache.get("text", "") or ""


def search_subsidies_by_industry(industry_sub: str):
    """業種に紐づく補助金一覧を返す。subsidy_schedule.json の industries で検索。"""
    out = []
    for s in (subsidy_schedule_data.get("subsidies") or []):
        if industry_sub in (s.get("industries") or []):
            out.append(s)
    return out


def search_equipment_by_keyword(keyword: str):
    """耐用年数データからキーワードで設備を検索。"""
    if not keyword or not useful_life_data:
        return []
    out = []
    kw = keyword.strip().lower()
    for cat in (useful_life_data.get("categories") or []):
        for item in (cat.get("items") or []):
            name = (item.get("name") or "")
            if kw in name.lower():
                out.append({"category": cat.get("name"), **item})
    return out


def get_lease_classification_text():
    """リース判定フローと契約形態別条件の要約を返す。"""
    if not lease_classification_data:
        return ""
    lines = ["【リース判定の目安】"]
    for step in (lease_classification_data.get("classification_flow") or []):
        lines.append(f"Step{step.get('step')}: {step.get('question')} → {step.get('yes_go') or step.get('no_go')}")
    lines.append("")
    for ct in (lease_classification_data.get("contract_types") or []):
        lines.append(f"■ {ct.get('type')}: {ct.get('summary')}")
        for t in (ct.get("typical_conditions") or [])[:3]:
            lines.append(f"  - {t}")
    return "\n".join(lines)


def get_advice_context_extras(selected_sub: str, selected_major: str):
    """AIアドバイス用に、補助金・耐用年数・リース分類・業界トレンド拡充・資産目安・売上規模帯のテキストをまとめて返す。"""
    parts = []
    subs = search_subsidies_by_industry(selected_sub)
    if subs:
        parts.append("【該当業種の補助金例】")
        for s in subs[:5]:
            line = f"- {s.get('name')}: {s.get('summary')} 申請目安: {s.get('application_period')}"
            if s.get("url"):
                line += f" 問い合わせ先: {s.get('url')}"
            parts.append(line)
    lc = get_lease_classification_text()
    if lc:
        parts.append("\n" + lc)
    trend_ex = get_trend_extended(selected_sub)
    if trend_ex:
        parts.append("\n【業界トレンド（拡充）】\n" + trend_ex[:1200])
    ab = get_assets_benchmark(selected_sub)
    if ab.get("current_ratio") is not None:
        parts.append(f"\n【業界の資産目安】流動比率目安: {ab['current_ratio']}%")
    if ab.get("total_assets_note"):
        parts.append("総資産・業界メモ: " + ab["total_assets_note"][:300])
    sb = get_sales_band_text()
    if sb:
        parts.append("\n【売上規模帯別の目安】\n" + sb[:600])
    # 過去の競合・成約金利（統計から取得し、競合に勝つ対策のコンテキストとして渡す）
    stats = get_stats(selected_sub)
    if stats.get("top_competitors_lost"):
        parts.append("\n【過去に負けが多い競合】" + "、".join(stats["top_competitors_lost"][:5]))
    if stats.get("avg_winning_rate") is not None and stats["avg_winning_rate"] > 0:
        parts.append(f"\n【同業種の平均成約金利】{stats['avg_winning_rate']:.2f}%")
    if stats.get("top_competitors_lost") or (stats.get("avg_winning_rate") and stats["avg_winning_rate"] > 0):
        parts.append("\n上記の競合動向・成約金利を踏まえ、競合に勝つための対策も考慮してアドバイスしてください。")
    return "\n".join(parts) if parts else ""


def get_indicator_analysis_for_advice(last_result: dict):
    """
    last_result から業界目安を組み立て、指標の差の分析（要約・内訳）と指標一覧テキストを返す。
    AI相談で「指標の分析と改善アドバイス」に使う。
    返却: (summary, detail, indicators_text)。データが無い場合は ("", "", "")。
    """
    if not last_result:
        return "", "", ""
    fin = last_result.get("financials", {})
    if not fin:
        return "", "", ""
    selected_sub = last_result.get("industry_sub", "")
    major = last_result.get("industry_major", "")
    bench = dict(benchmarks_data.get(selected_sub, {}))
    cache = _load_web_benchmarks_cache()
    cached = cache.get(selected_sub, {})
    for k in _WEB_BENCH_KEYS:
        if cached.get(k) is not None:
            bench[k] = cached[k]
    bench_ext = dict(bench)
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
    if not indicators:
        return "", "", ""
    summary, detail = analyze_indicators_vs_bench(indicators)
    lines = []
    for ind in indicators:
        row = f"- {ind['name']}: 貴社 {ind['value']:.1f}{ind.get('unit','%')}"
        if ind.get("bench") is not None:
            row += f" / 業界目安 {ind['bench']:.1f}{ind.get('unit','%')}"
        lines.append(row)
    indicators_text = "\n".join(lines)
    return summary, detail, indicators_text


def save_debate_log(data):
    """ディベート結果を保存"""
    data["timestamp"] = datetime.datetime.now().isoformat()
    try:
        with open(DEBATE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"ディベート保存エラー: {e}")


def load_consultation_memory(max_entries=20):
    """
    AI審査オフィサー相談のメモを読み込む。話せば話すほど蓄積した過去のやり取りを返す。
    直近 max_entries 件を返す（古い順）。
    """
    if not os.path.exists(CONSULTATION_MEMORY_FILE):
        return []
    entries = []
    try:
        with open(CONSULTATION_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return entries[-max_entries:] if len(entries) > max_entries else entries


def append_consultation_memory(user_text: str, assistant_text: str):
    """相談1往復をメモに追記。以後の相談で活用される。"""
    try:
        with open(CONSULTATION_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user": (user_text or "")[:5000],
                "assistant": (assistant_text or "")[:5000],
                "ts": datetime.datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        if "st" in dir():
            st.error(f"相談メモ保存エラー: {e}")


def load_all_cases():
    if not os.path.exists(CASES_FILE):
        return []
    cases = []
    try:
        with open(CASES_FILE, "r") as f:
            for line in f:
                try:
                    cases.append(json.loads(line))
                except:
                    continue
    except Exception:
        return []
    return cases


def load_past_cases():
    """
    save_case_log で保存された過去の審査ログ（JSONL）をすべて読み込む。
    """
    return load_all_cases()


def find_similar_past_cases(selected_sub: str, user_equity_ratio: float, max_count: int = 3):
    """
    業界（selected_sub）が同じで、自己資本比率が近い過去案件を最大 max_count 件返す。
    自己資本比率の差の絶対値でソートし、近い順に返す。
    """
    cases = load_past_cases()
    # 業界が一致し、result と user_eq があるものだけ
    candidates = []
    for c in cases:
        if c.get("industry_sub") != selected_sub:
            continue
        res = c.get("result") or {}
        eq = res.get("user_eq")
        if eq is None:
            continue
        try:
            eq_val = float(eq)
        except (TypeError, ValueError):
            continue
        diff = abs(eq_val - user_equity_ratio)
        status = c.get("final_status", "未登録")
        score = res.get("score")
        candidates.append({"diff": diff, "case": c, "equity": eq_val, "status": status, "score": score})
    candidates.sort(key=lambda x: x["diff"])
    return [x["case"] for x in candidates[:max_count]]


def save_all_cases(cases):
    try:
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"保存エラー: {e}")


# ---------------------------------------------------------------------------
# 成約/失注を目的変数とした回帰で係数を更新し、保存・読み込みする
# ---------------------------------------------------------------------------
COEFF_MAIN_KEYS = [
    "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing", "ind_service",
    "sales_log", "bank_credit_log", "lease_credit_log",
    "op_profit", "ord_profit", "net_income", "machines", "other_assets", "rent",
    "gross_profit", "depreciation", "dep_expense", "rent_expense",
    "grade_4_6", "grade_watch", "grade_none", "contracts",
]
# 追加項目（ベイズ補完）: 回帰・スコア両方で使用
COEFF_EXTRA_KEYS = [
    "main_bank", "competitor_present", "competitor_none",
    "rate_diff_z", "industry_sentiment_z", "qualitative_tag_score", "qualitative_passion",
    "equity_ratio",  # 自己資本比率（%）
]

# 業種ごと・既存先/新規先のモデルキー（ベイズ回帰で更新対象）
INDUSTRY_MODEL_KEYS = [
    "全体_既存先", "全体_新規先",
    "医療_既存先", "医療_新規先",
    "運送業_既存先", "運送業_新規先",
    "サービス業_既存先", "サービス業_新規先",
    "製造業_既存先", "製造業_新規先",
]
# 指標モデルも既存先/新規先で分けて回帰
INDICATOR_MODEL_KEYS = [
    "全体_指標_既存先", "全体_指標_新規先",
    "医療_指標_既存先", "医療_指標_新規先",
    "運送業_指標_既存先", "運送業_指標_新規先",
    "サービス業_指標_既存先", "サービス業_指標_新規先",
    "製造業_指標_既存先", "製造業_指標_新規先",
]
# 事前係数入力画面で編集可能なモデル一覧（業種＋指標のベース）
PRIOR_COEFF_MODEL_KEYS = [
    "全体_既存先", "全体_新規先", "医療_既存先", "医療_新規先",
    "運送業_既存先", "運送業_新規先", "サービス業_既存先", "サービス業_新規先",
    "製造業_既存先", "製造業_新規先",
    "全体_指標", "医療_指標", "運送業_指標", "サービス業_指標", "製造業_指標",
]
# 指標モデル用の説明変数（ratio + grade + ind ダミー）。全体_指標の係数キー順に合わせる
INDICATOR_MAIN_KEYS = [
    "ind_service", "ind_medical", "ind_transport", "ind_construction", "ind_manufacturing",
    "ratio_op_margin", "ratio_gross_margin", "ratio_ord_margin", "ratio_net_margin",
    "ratio_fixed_assets", "ratio_rent", "ratio_depreciation", "ratio_machines",
    "grade_4_6", "grade_watch", "grade_none",
]


def _get_ind_key_from_log(log):
    """ログから業種モデルキー（既存先/新規先）を算出。"""
    res = log.get("result") or {}
    major = res.get("industry_major") or log.get("industry_major") or "D 建設業"
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "D")
    customer_type = log.get("customer_type") or "既存先"
    if major_code == "H":
        base = "運送業"
    elif major_code == "P":
        base = "医療"
    elif major_code in ["I", "K", "M", "R"]:
        base = "サービス業"
    elif major_code == "E":
        base = "製造業"
    else:
        base = "全体"
    suffix = "新規先" if customer_type == "新規先" else "既存先"
    return f"{base}_{suffix}"


def _get_bench_key_from_log(log):
    """ログから指標モデルのベースキー（業種）を算出。"""
    res = log.get("result") or {}
    major = res.get("industry_major") or log.get("industry_major") or "D 建設業"
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "D")
    if major_code == "D":
        return "全体_指標"
    if major_code == "P":
        return "医療_指標"
    if major_code == "H":
        return "運送業_指標"
    if major_code in ["I", "K", "M", "R"]:
        return "サービス業_指標"
    if major_code == "E":
        return "製造業_指標"
    return "全体_指標"


def _get_indicator_model_key_from_log(log):
    """ログから指標モデルキー（既存先/新規先付き）を算出。"""
    base = _get_bench_key_from_log(log)
    customer_type = log.get("customer_type") or "既存先"
    suffix = "新規先" if customer_type == "新規先" else "既存先"
    return f"{base}_{suffix}"


def _log_to_data_scoring(log):
    """1件のログからスコア計算用 data_scoring 相当の辞書を組み立てる（単位: 千円→百万円）。"""
    inp = log.get("inputs") or {}
    res = log.get("result") or {}
    nenshu = float(inp.get("nenshu") or 0)
    bank_credit = float(inp.get("bank_credit") or 0)
    lease_credit = float(inp.get("lease_credit") or 0)
    # 百万円換算
    to_mill = 1.0 / 1000.0
    op_profit = float(inp.get("op_profit") or 0) * to_mill
    ord_profit = float(inp.get("ord_profit") or 0) * to_mill
    net_income = float(inp.get("net_income") or 0) * to_mill
    gross_profit = float(inp.get("gross_profit") or 0) * to_mill
    machines = float(inp.get("machines") or 0) * to_mill
    other_assets = float(inp.get("other_assets") or 0) * to_mill
    rent = float(inp.get("rent") or 0) * to_mill
    depreciation = float(inp.get("depreciation") or 0) * to_mill
    dep_expense = float(inp.get("dep_expense") or 0) * to_mill
    rent_expense = float(inp.get("rent_expense") or 0) * to_mill
    contracts = float(inp.get("contracts") or 0)
    grade = (inp.get("grade") or res.get("grade") or "")
    industry_major = res.get("industry_major") or (log.get("industry_major") or "D 建設業")
    return {
        "nenshu": nenshu, "bank_credit": bank_credit, "lease_credit": lease_credit,
        "op_profit": op_profit, "ord_profit": ord_profit, "net_income": net_income,
        "gross_profit": gross_profit, "machines": machines, "other_assets": other_assets,
        "rent": rent, "depreciation": depreciation, "dep_expense": dep_expense, "rent_expense": rent_expense,
        "contracts": contracts, "grade": grade, "industry_major": industry_major,
    }


def _build_one_row_industry(log, data):
    """1ログから業種モデル用の1行（既存22+追加8）を構築。"""
    major = data["industry_major"]
    ind_medical = 1.0 if ("医療" in major or "福祉" in major or (isinstance(major, str) and major.startswith("P"))) else 0.0
    ind_transport = 1.0 if ("運輸" in major or (isinstance(major, str) and major.startswith("H"))) else 0.0
    ind_construction = 1.0 if ("建設" in major or (isinstance(major, str) and major.startswith("D"))) else 0.0
    ind_manufacturing = 1.0 if ("製造" in major or (isinstance(major, str) and major.startswith("E"))) else 0.0
    ind_service = 1.0 if ("卸売" in major or "小売" in major or "サービス" in major or (isinstance(major, str) and major[0] in ["I", "K", "M", "R"])) else 0.0
    sales_log = np.log1p(data["nenshu"])
    bank_credit_log = np.log1p(data["bank_credit"])
    lease_credit_log = np.log1p(data["lease_credit"])
    grade = data["grade"]
    grade_4_6 = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none = 1.0 if "無格付" in grade else 0.0
    row = [
        ind_medical, ind_transport, ind_construction, ind_manufacturing, ind_service,
        sales_log, bank_credit_log, lease_credit_log,
        data["op_profit"], data["ord_profit"], data["net_income"], data["machines"], data["other_assets"], data["rent"],
        data["gross_profit"], data["depreciation"], data["dep_expense"], data["rent_expense"],
        grade_4_6, grade_watch, grade_none, data["contracts"],
    ]
    inp, res = log.get("inputs") or {}, log.get("result") or {}
    main_bank = 1.0 if log.get("main_bank") == "メイン先" else 0.0
    competitor_present = 1.0 if log.get("competitor") == "競合あり" else 0.0
    competitor_none = 1.0 if log.get("competitor") == "競合なし" else 0.0
    y_pred, comp_rate = res.get("yield_pred"), log.get("competitor_rate")
    if y_pred is not None and comp_rate is not None and isinstance(comp_rate, (int, float)):
        rate_diff_pt = float(y_pred) - float(comp_rate)
        rate_diff_z = max(-2.0, min(2.0, rate_diff_pt / 5.0))
    else:
        rate_diff_z = 0.0
    industry_sentiment_z = float(res.get("industry_sentiment_z", 0))
    qual = inp.get("qualitative") or {}
    tags = qual.get("strength_tags") or []
    qualitative_tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in tags), 10.0)
    qualitative_passion = 1.0 if qual.get("passion_text") else 0.0
    equity_ratio = float(res.get("user_eq") or 0)
    row.extend([main_bank, competitor_present, competitor_none, rate_diff_z, industry_sentiment_z, qualitative_tag_score, qualitative_passion, equity_ratio])
    return row


def build_design_matrix_from_logs(all_logs, model_key=None):
    """
    成約/失注が登録されたログから、業種モデル用の説明変数行列 X と目的変数 y を構築する。
    model_key を指定した場合はその業種・既存先/新規先のログのみ使用。
    目的変数: 成約=1, 失注=0。
    """
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        if model_key is not None and _get_ind_key_from_log(log) != model_key:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_industry(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    X = np.array(rows, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y


def run_regression_and_get_coeffs(X, y):
    """
    X, y に対してロジスティック回帰を実行し、既存項目＋追加項目の係数辞書を返す。
    X の列順: COEFF_MAIN_KEYS (22) + COEFF_EXTRA_KEYS (8)。
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].tolist()
    coeff_dict = {"intercept": intercept}
    for i, key in enumerate(COEFF_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(COEFF_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, model


def _build_one_row_indicator(log, data):
    """1ログから指標モデル用の1行（ind+ratio+grade 16 + 追加8）を構築。"""
    major = data["industry_major"]
    ind_medical = 1.0 if ("医療" in major or "福祉" in major or (isinstance(major, str) and major.startswith("P"))) else 0.0
    ind_transport = 1.0 if ("運輸" in major or (isinstance(major, str) and major.startswith("H"))) else 0.0
    ind_construction = 1.0 if ("建設" in major or (isinstance(major, str) and major.startswith("D"))) else 0.0
    ind_manufacturing = 1.0 if ("製造" in major or (isinstance(major, str) and major.startswith("E"))) else 0.0
    ind_service = 1.0 if ("卸売" in major or "小売" in major or "サービス" in major or (isinstance(major, str) and major[0] in ["I", "K", "M", "R"])) else 0.0
    grade = data["grade"]
    grade_4_6 = 1.0 if "4-6" in grade else 0.0
    grade_watch = 1.0 if "要注意" in grade else 0.0
    grade_none = 1.0 if "無格付" in grade else 0.0
    raw_nenshu = max(float(data["nenshu"] or 0), 1.0)
    raw_op = data["op_profit"] * 1000
    raw_gross = data["gross_profit"] * 1000
    raw_ord = data["ord_profit"] * 1000
    raw_net = data["net_income"] * 1000
    raw_fixed = data["machines"] * 1000 + data["other_assets"] * 1000
    raw_rent = data["rent_expense"] * 1000
    raw_dep = data["depreciation"] * 1000 + data["dep_expense"] * 1000
    raw_machines = data["machines"] * 1000
    ratio_op = raw_op / raw_nenshu if raw_nenshu else 0
    ratio_gross = raw_gross / raw_nenshu if raw_nenshu else 0
    ratio_ord = raw_ord / raw_nenshu if raw_nenshu else 0
    ratio_net = raw_net / raw_nenshu if raw_nenshu else 0
    ratio_fixed = raw_fixed / raw_nenshu if raw_nenshu else 0
    ratio_rent = raw_rent / raw_nenshu if raw_nenshu else 0
    ratio_dep = raw_dep / raw_nenshu if raw_nenshu else 0
    ratio_machines = raw_machines / raw_nenshu if raw_nenshu else 0
    row = [
        ind_service, ind_medical, ind_transport, ind_construction, ind_manufacturing,
        ratio_op, ratio_gross, ratio_ord, ratio_net, ratio_fixed, ratio_rent, ratio_dep, ratio_machines,
        grade_4_6, grade_watch, grade_none,
    ]
    inp, res = log.get("inputs") or {}, log.get("result") or {}
    main_bank = 1.0 if log.get("main_bank") == "メイン先" else 0.0
    competitor_present = 1.0 if log.get("competitor") == "競合あり" else 0.0
    competitor_none = 1.0 if log.get("competitor") == "競合なし" else 0.0
    y_pred, comp_rate = res.get("yield_pred"), log.get("competitor_rate")
    if y_pred is not None and comp_rate is not None and isinstance(comp_rate, (int, float)):
        rate_diff_z = max(-2.0, min(2.0, (float(y_pred) - float(comp_rate)) / 5.0))
    else:
        rate_diff_z = 0.0
    industry_sentiment_z = float(res.get("industry_sentiment_z", 0))
    qual = inp.get("qualitative") or {}
    tags = qual.get("strength_tags") or []
    qualitative_tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in tags), 10.0)
    qualitative_passion = 1.0 if qual.get("passion_text") else 0.0
    equity_ratio = float(res.get("user_eq") or 0)
    row.extend([main_bank, competitor_present, competitor_none, rate_diff_z, industry_sentiment_z, qualitative_tag_score, qualitative_passion, equity_ratio])
    return row


def build_design_matrix_indicator_from_logs(all_logs, indicator_model_key):
    """
    指標モデル用の説明変数行列 X と目的変数 y を構築。
    indicator_model_key は "全体_指標_既存先" などの形式。該当するログのみ使用。
    """
    rows = []
    y_list = []
    for log in all_logs:
        if log.get("final_status") not in ["成約", "失注"]:
            continue
        if "inputs" not in log:
            continue
        if _get_indicator_model_key_from_log(log) != indicator_model_key:
            continue
        data = _log_to_data_scoring(log)
        row = _build_one_row_indicator(log, data)
        rows.append(row)
        y_list.append(1 if log.get("final_status") == "成約" else 0)
    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(y_list, dtype=int)


def run_regression_indicator_and_get_coeffs(X, y):
    """指標モデル用の回帰。列順: INDICATOR_MAIN_KEYS (16) + COEFF_EXTRA_KEYS (8)。"""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].tolist()
    coeff_dict = {"intercept": intercept}
    for i, key in enumerate(INDICATOR_MAIN_KEYS):
        if i < len(coefs):
            coeff_dict[key] = float(coefs[i])
    for j, key in enumerate(COEFF_EXTRA_KEYS):
        idx = len(INDICATOR_MAIN_KEYS) + j
        if idx < len(coefs):
            coeff_dict[key] = float(coefs[idx])
    return coeff_dict, model


def load_coeff_overrides():
    """保存済みの係数オーバーライドを読み込む。無ければ None。"""
    if not os.path.exists(COEFF_OVERRIDES_FILE):
        return None
    try:
        with open(COEFF_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_coeff_overrides(overrides_dict):
    """係数オーバーライドを JSON で保存する。"""
    dirpath = os.path.dirname(COEFF_OVERRIDES_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        with open(COEFF_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"係数保存エラー: {e}")
        return False


def get_effective_coeffs(key=None):
    """
    指定キーの係数セットを返す。成約/失注で更新した係数や事前入力した係数があればマージして返す。
    key=None のときは 全体_既存先。
    指標の既存先/新規先キー（例: 全体_指標_既存先）は、まずベース（全体_指標）のオーバーライドを適用し、次に _既存先/_新規先 用のオーバーライドを適用。
    """
    if key is None:
        key = "全体_既存先"
    overrides = load_coeff_overrides() or {}
    base_key = key
    if base_key not in COEFFS:
        base_key = key.replace("_既存先", "").replace("_新規先", "")  # 全体_指標_既存先 -> 全体_指標
    base = dict(COEFFS.get(base_key, COEFFS["全体_既存先"]))
    if overrides.get(base_key):
        base.update(overrides[base_key])
    if overrides.get(key):
        base.update(overrides[key])
    return base


def append_case_news(record: dict):
    """
    案件ごとのニュースを1件ずつ追記保存する。
    record には少なくとも {case_id, title, url, content} を想定。
    """
    if not record:
        return
    try:
        data = dict(record)
        data.setdefault("saved_at", datetime.datetime.now().isoformat())
        with open(CASE_NEWS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"ニュース保存エラー: {e}")


def load_case_news(case_id: str | None = None):
    """
    保存済みニュースを読み込む。case_id を指定するとその案件分だけ返す。
    """
    if not os.path.exists(CASE_NEWS_FILE):
        return []
    records = []
    try:
        with open(CASE_NEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if case_id is not None and rec.get("case_id") != case_id:
                    continue
                records.append(rec)
    except Exception:
        return []
    return records

def get_stats(target_sub_industry):
    cases = load_all_cases()
    target_cases = [c for c in cases if c.get("industry_sub") == target_sub_industry]
    count = len(target_cases)
    
    if count == 0:
        return {"count": 0, "closed_count": 0, "avg_score": 0.0, "approved_count": 0, "close_rate": 0.0, "lost_reasons": [], "top_competitors_lost": [], "avg_winning_rate": None}
    
    scores = [c["result"]["score"] for c in target_cases if "result" in c]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    approved_count = len([s for s in scores if s >= 70])
    
    closed_cases = [c for c in target_cases if c.get("final_status") == "成約"]
    lost_cases = [c for c in target_cases if c.get("final_status") == "失注"]
    total_finished = len(closed_cases) + len(lost_cases)
    
    close_rate = 0.0
    if total_finished > 0:
        close_rate = len(closed_cases) / total_finished
        
    lost_reasons = [c.get("lost_reason") for c in lost_cases if c.get("lost_reason")]
    
    # よく負ける競合名（失注案件の competitor_name を集計、多い順）
    competitor_names = [c.get("competitor_name", "").strip() for c in lost_cases if c.get("competitor_name")]
    top_competitors_lost = []
    if competitor_names:
        from collections import Counter
        counted = Counter(competitor_names)
        top_competitors_lost = [name for name, _ in counted.most_common(10)]
    
    # 平均的な成約金利（成約案件の final_rate の平均、0 を除く）
    winning_rates = [c.get("final_rate") for c in closed_cases if c.get("final_rate") is not None and (isinstance(c.get("final_rate"), (int, float)) and c.get("final_rate") > 0)]
    avg_winning_rate = sum(winning_rates) / len(winning_rates) if winning_rates else None
    
    return {
        "count": count,
        "closed_count": len(closed_cases),
        "avg_score": avg_score,
        "approved_count": approved_count,
        "close_rate": close_rate,
        "lost_reasons": lost_reasons,
        "top_competitors_lost": top_competitors_lost,
        "avg_winning_rate": avg_winning_rate,
    }


# =============================================================================
# 成約要因分析
# 成約データのみを抽出し、共通項（平均財務・定性タグランキング）を算出。
# 成約に寄与する上位3ドライバーは回帰係数（全体_既存先）の絶対値で算出。
# 利用箇所: 成約の正体レポート画面、分析結果ダッシュボード先頭の3因子表示。
# =============================================================================
COEFF_LABELS = {
    "intercept": "定数項",
    "ind_medical": "業種: 医療・福祉",
    "ind_transport": "業種: 運輸",
    "ind_construction": "業種: 建設",
    "ind_manufacturing": "業種: 製造",
    "ind_service": "業種: サービス",
    "sales_log": "売上高(対数)",
    "bank_credit_log": "銀行与信(対数)",
    "lease_credit_log": "リース与信(対数)",
    "op_profit": "営業利益",
    "ord_profit": "経常利益",
    "net_income": "当期純利益",
    "machines": "機械装置",
    "other_assets": "その他資産",
    "rent": "賃借料",
    "gross_profit": "売上総利益",
    "depreciation": "減価償却",
    "dep_expense": "減価償却費",
    "rent_expense": "賃借料等",
    "grade_4_6": "格付4〜6",
    "grade_watch": "要注意",
    "grade_none": "無格付",
    "contracts": "契約数",
    "main_bank": "メイン取引先",
    "competitor_present": "競合あり",
    "competitor_none": "競合なし",
    "rate_diff_z": "金利差(有利)",
    "industry_sentiment_z": "業界景気動向",
    "qualitative_tag_score": "定性スコア(強みタグ)",
    "qualitative_passion": "熱意・裏事情",
    "equity_ratio": "自己資本比率",
}


def run_contract_driver_analysis():
    """
    成約要因分析: 成約データのみ抽出し、共通項（平均財務・定性タグランキング）と
    成約に寄与する上位3ドライバー（回帰係数ベース）を返す。
    成約が5件未満の場合は None を返す。
    """
    from collections import Counter
    cases = load_all_cases()
    closed = [c for c in cases if c.get("final_status") == "成約"]
    if len(closed) < 5:
        return None
    # 平均財務数値（成約案件のみ）
    fin_keys = ["nenshu", "op_profit", "ord_profit", "net_income", "bank_credit", "lease_credit", "contracts"]
    fin_labels = {"nenshu": "売上高(千円)", "op_profit": "営業利益(千円)", "ord_profit": "経常利益(千円)", "net_income": "当期純利益(千円)", "bank_credit": "銀行与信(千円)", "lease_credit": "リース与信(千円)", "contracts": "契約数"}
    sums = {k: 0.0 for k in fin_keys}
    counts = {k: 0 for k in fin_keys}
    for c in closed:
        inp = c.get("inputs") or {}
        res = c.get("result") or {}
        for k in fin_keys:
            v = inp.get(k) if k in inp else res.get("user_eq") if k == "user_eq" else None
            if k == "contracts":
                v = inp.get(k)
            if v is not None and isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    avg_financials = {}
    for k in fin_keys:
        if counts[k] > 0:
            avg_financials[fin_labels.get(k, k)] = sums[k] / counts[k]
    user_eq_list = []
    for c in closed:
        res = c.get("result") or {}
        eq = res.get("user_eq")
        if eq is not None and isinstance(eq, (int, float)):
            user_eq_list.append(float(eq))
    if user_eq_list:
        avg_financials["自己資本比率(%)"] = sum(user_eq_list) / len(user_eq_list)
    # 定性タグ頻出ランキング
    tag_counter = Counter()
    for c in closed:
        inp = c.get("inputs") or {}
        qual = inp.get("qualitative") or {}
        for t in qual.get("strength_tags") or []:
            tag_counter[t] += 1
    tag_ranking = tag_counter.most_common(20)
    # 成約に寄与する上位3ドライバー（全体_既存先の係数で絶対値が大きい順）
    coeffs = get_effective_coeffs("全体_既存先")
    driver_candidates = [(k, coeffs.get(k, 0)) for k in (COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS) if k in coeffs]
    driver_candidates = [(k, v) for k, v in driver_candidates if isinstance(v, (int, float)) and k != "intercept"]
    driver_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    top3_drivers = []
    for k, v in driver_candidates[:3]:
        label = COEFF_LABELS.get(k, k)
        direction = "プラス" if v > 0 else "マイナス"
        top3_drivers.append({"key": k, "label": label, "coef": v, "direction": direction})
    return {
        "closed_cases": closed,
        "closed_count": len(closed),
        "avg_financials": avg_financials,
        "tag_ranking": tag_ranking,
        "top3_drivers": top3_drivers,
    }


def save_case_log(data):
    """
    審査1件分のログを保存し、生成した案件IDを返す。
    """
    case_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    data["id"] = case_id
    data["timestamp"] = datetime.datetime.now().isoformat()
    data["final_status"] = "未登録"
    try:
        with open(CASES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"ログ保存エラー: {e}")
    return case_id

# 市場金利の取得関数
def get_market_rate(year_month, term_years=5):
    if year_month not in jgb_rates:
        keys = sorted(jgb_rates.keys())
        if keys:
            year_month = keys[-1]
        else:
            return 1.0
            
    rate_data = jgb_rates[year_month]
    if term_years >= 8:
        return rate_data.get("10y", 1.0)
    else:
        return rate_data.get("5y", 0.5)

def _ollama_chat_http(model: str, messages: list, timeout_seconds: int):
    """
    Ollama の HTTP API を直接叩く。requests の timeout で確実に切る。
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests が必要です: pip install requests")

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = base + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            f"Ollama が {timeout_seconds} 秒以内に応答しませんでした。\n"
            "・ターミナルで `ollama serve` が動いているか確認してください。\n"
            "・モデルが重い場合は初回の応答に時間がかかります。軽いモデル（例: lease-anna）を試すか、Gemini API に切り替えてください。"
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Ollama に接続できませんでした。\n"
            "・ターミナルで **ollama serve** を実行してから再度お試しください。\n"
            f"・接続先: {base}\n"
            f"・詳細: {e}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama がタイムアウトしました（{timeout_seconds}秒）。\n"
            "・軽いモデル（lease-anna 等）を試すか、サイドバーで Gemini API に切り替えてください。"
        )

    if resp.status_code == 404:
        try:
            err_body = resp.json()
            err_msg = err_body.get("error", resp.text)
        except Exception:
            err_msg = resp.text
        raise RuntimeError(
            f"モデル「{model}」が見つかりません。\n"
            f"・ターミナルで **ollama pull {model}** を実行してモデルを取得してください。\n"
            f"・またはサイドバー「AIモデル設定」で別のモデル（例: lease-anna）を選択してください。\n"
            f"・Ollamaの詳細: {err_msg[:200]}"
        )
    resp.raise_for_status()
    data = resp.json()
    if "message" in data and "content" in data["message"]:
        return {"message": {"content": data["message"]["content"]}}
    raise RuntimeError("Ollama の応答形式が不正です。")


def _gemini_chat(api_key: str, model: str, messages: list, timeout_seconds: int):
    """
    Gemini API でチャット。messages は [{"role":"user","content":"..."}] 形式。
    最後の user メッセージをプロンプトとして送り、返答テキストを返す。
    """
    if not api_key or not api_key.strip():
        return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
    prompt = ""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            prompt = m["content"]
    if not prompt:
        return {"message": {"content": "送信する内容がありません。"}}
    try:
        import google.generativeai as genai
    except ImportError:
        return {"message": {"content": "Gemini を使うには pip install google-generativeai を実行してください。"}}

    try:
        genai.configure(api_key=api_key.strip())
        gemini_model = genai.GenerativeModel(model)
        try:
            config = genai.types.GenerationConfig(max_output_tokens=2048, temperature=0.7)
            response = gemini_model.generate_content(prompt, generation_config=config)
        except (AttributeError, TypeError):
            response = gemini_model.generate_content(prompt)

        if not response:
            return {"message": {"content": "Gemini から応答が返りませんでした。"}}

        # response.text はブロック時などに ValueError を出すことがある
        text = None
        try:
            if response.text:
                text = response.text
        except (ValueError, AttributeError):
            pass
        if not text and getattr(response, "candidates", None):
            for c in response.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            text = (text or "") + p.text
                    if text:
                        break
        if text and text.strip():
            return {"message": {"content": text.strip()}}
        # ブロックや空応答
        return {"message": {"content": "Gemini から空の応答か、安全フィルターでブロックされた可能性があります。プロンプトを変えて再試行してください。"}}
    except Exception as e:
        err = str(e).strip().lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err or "rate limit" in err:
            return {"message": {"content": (
                "**Gemini の利用枠（無料枠の1日制限）に達している可能性があります。**\n\n"
                "・無料枠は1日あたりのリクエスト数に上限があります。\n"
                "・明日になるまでお待ちいただくか、[Google AI Studio](https://aistudio.google.com/) で利用状況を確認してください。\n"
                "・有料プランにすると制限が緩和されます。\n\n"
                f"【APIの詳細】{str(e)[:300]}"
            )}}
        return {"message": {"content": f"Gemini API エラー: {str(e)}\n\nAPIキーとモデル名（{model}）を確認し、ネット接続を確認してください。"}}


def _chat_for_thread(engine: str, model: str, messages: list, timeout_seconds: int, api_key: str = "", gemini_model: str = ""):
    """
    バックグラウンドスレッドから呼ぶ用。st.session_state を参照しない。
    engine が "gemini" のときは api_key と gemini_model を使用。
    """
    if engine == "gemini":
        api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_gemini_chat, api_key, gemini_model or "gemini-2.0-flash", messages, timeout_seconds)
                return future.result(timeout=min(timeout_seconds + 30, 90))
        except Exception as e:
            return {"message": {"content": f"Gemini が応答しませんでした。\n\n【詳細】{str(e)}"}}
    try:
        return _ollama_chat_http(model, messages, timeout_seconds)
    except Exception as e:
        return {"message": {"content": f"AIサーバーが応答しませんでした: {e}"}}


def chat_with_retry(model, messages, retries=2, timeout_seconds=120):
    """
    AI へのチャット呼び出し。エンジンが Gemini の場合は Gemini API、否则 Ollama。
    """
    engine = st.session_state.get("ai_engine", "ollama")
    if engine == "gemini":
        api_key = (st.session_state.get("gemini_api_key") or "").strip() or GEMINI_API_KEY_ENV
        api_key = api_key or _get_gemini_key_from_secrets()
        gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
        # デバッグ用: 最後の呼び出し結果を保存
        if "last_gemini_debug" not in st.session_state:
            st.session_state["last_gemini_debug"] = ""
        for i in range(retries):
            try:
                # タイムアウトでハングしないよう別スレッドで実行
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_gemini_chat, api_key, gemini_model, messages, timeout_seconds)
                    try:
                        out = future.result(timeout=min(timeout_seconds + 30, 90))
                    except concurrent.futures.TimeoutError:
                        st.session_state["last_gemini_debug"] = "タイムアウト（応答が返るまで待ちましたが応答がありませんでした）"
                        st.error("Gemini がタイムアウトしました。ネット接続を確認するか、しばらくして再試行してください。")
                        return {"message": {"content": "Gemini がタイムアウトしました。ネット接続を確認するか、しばらくして再試行してください。"}}
                content = (out.get("message") or {}).get("content", "")
                st.session_state["last_gemini_debug"] = "OK" if content and "APIキーが" not in content and "Gemini API エラー:" not in content else (content[:200] + "..." if len(content or "") > 200 else (content or "（空）"))
                # エラー系メッセージなら画面上にも st.error で表示
                if content and (
                    "APIキーが設定されていません" in content
                    or "Gemini API エラー:" in content
                    or "pip install" in content
                    or "応答が返りませんでした" in content
                    or "安全フィルターでブロック" in content
                    or "利用枠" in content
                    or "無料枠" in content
                ):
                    st.error(content)
                return out
            except Exception as e:
                err = str(e)
                st.session_state["last_gemini_debug"] = f"例外: {err}"
                if "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower() or "rate limit" in err.lower():
                    time.sleep(2 * (i + 1))
                    continue
                st.error(f"Gemini API エラー: {err}")
                return {"message": {"content": f"Gemini が応答しませんでした。\n\n【詳細】{err}"}}
        st.session_state["last_gemini_debug"] = "リトライ上限（または利用枠の可能性）"
        return {"message": {"content": (
            "Gemini が応答しませんでした。\n\n"
            "**無料枠の1日あたりの制限に達している可能性があります。**\n"
            "・明日までお待ちいただくか、[Google AI Studio](https://aistudio.google.com/) で利用状況を確認してください。\n"
            "・APIキー・モデル名・ネット接続もあわせて確認してください。"
        )}}

    last_error = None
    for i in range(retries):
        try:
            return _ollama_chat_http(model, messages, timeout_seconds)
        except Exception as e:
            last_error = str(e)
            if "429" in last_error:
                time.sleep(2 * (i + 1))
                continue
            break

    if last_error:
        st.error(f"AIサーバーが応答しませんでした: {last_error}")
        detail = f"\n\n【技術的な詳細】{last_error}"
        if "timed out" in last_error or "Timeout" in last_error:
            detail += "\n\n💡 左サイドバー「AIモデル設定」で **Gemini API** に切り替えるか、**lease-anna** 等の軽いモデルを試してください。"
    else:
        st.error("AIサーバーが応答しませんでした。")
        detail = ""
    return {
        "message": {
            "content": "AIが応答しませんでした。時間を置くか、Gemini API に切り替えて再試行してください。" + detail
        }
    }


def generate_battle_special_move(strength_tags: list, passion_text: str) -> tuple:
    """
    定性データから「必殺技名」と「特殊効果」を1つ生成する。
    戻り値: (name: str, effect: str)。失敗時はフォールバックを返す。
    """
    fallback = ("逆転の意気", "スコア+5%")
    if not strength_tags and not (passion_text or "").strip():
        return fallback
    model = get_ollama_model() if st.session_state.get("ai_engine") == "ollama" else GEMINI_MODEL_DEFAULT
    tags_str = "、".join(strength_tags) if strength_tags else "なし"
    text_snippet = (passion_text or "")[:300]
    prompt = f"""以下から、審査ゲーム用の「必殺技」を1つだけ考えてください。
強みタグ: {tags_str}
熱意・裏事情（抜粋）: {text_snippet or "なし"}

必殺技は「名前」と「効果」の2つだけ。1行で答えてください。形式は必ず:
必殺技名 / 効果の短い説明
例: 老舗の暖簾 / ダメージ無効
例: 業界人脈の盾 / 流動性+10%
日本語で、必殺技名は10文字以内、効果は15文字以内。他は出力しない。"""
    try:
        out = chat_with_retry(model, [{"role": "user", "content": prompt}], retries=1, timeout_seconds=15)
        content = ((out.get("message") or {}).get("content") or "").strip()
        if " / " in content:
            parts = content.split(" / ", 1)
            return (parts[0].strip()[:20] or fallback[0], (parts[1].strip()[:25] or fallback[1]))
    except Exception:
        pass
    return fallback


def is_ai_available(timeout_seconds: int = 3) -> bool:
    """
    現在選択中のAIエンジンが利用可能かどうか。
    Gemini の場合は API キーが設定されていれば True。
    Ollama の場合はサーバーが起動していれば True。
    """
    engine = st.session_state.get("ai_engine", "ollama")
    if engine == "gemini":
        key = st.session_state.get("gemini_api_key", "").strip() or GEMINI_API_KEY_ENV
        key = key or _get_gemini_key_from_secrets()
        return bool(key)
    return is_ollama_available(timeout_seconds)


def is_ollama_available(timeout_seconds: int = 3) -> bool:
    """
    Ollamaサーバーが起動しているかを簡易チェックする。
    起動していない状態で chat_with_retry を呼ぶと永遠待ちになりやすいので、
    事前にここで検知してユーザーに案内を出す。
    """
    try:
        import requests
    except ImportError:
        # すでに記事スクレイピング等で requests を使っている前提
        return False

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = base + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_seconds)
        return resp.status_code == 200
    except Exception:
        return False


def run_ollama_connection_test(timeout_seconds: int = 10) -> str:
    """
    Ollama の接続とモデル応答をテストし、結果メッセージを返す。
    サイドバーの「Ollama接続テスト」ボタン用。
    """
    try:
        import requests
    except ImportError:
        return "❌ requests がインストールされていません: pip install requests"

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = get_ollama_model() or OLLAMA_MODEL

    # 1) /api/tags でサーバー生存確認
    try:
        r = requests.get(base + "/api/tags", timeout=5)
        if r.status_code != 200:
            return f"❌ Ollama サーバー応答異常: {base} (HTTP {r.status_code})"
    except requests.exceptions.ConnectionError:
        return (
            f"❌ Ollama に接続できません。\n"
            f"接続先: {base}\n\n"
            "**対処:** ターミナルで以下を実行してください。\n"
            "```\nollama serve\n```"
        )
    except requests.exceptions.Timeout:
        return f"❌ Ollama サーバーが応答しませんでした（5秒でタイムアウト）。\n接続先: {base}"

    # 2) 短いチャットでモデル応答確認
    try:
        r = requests.post(
            base + "/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "こん"}], "stream": False},
            timeout=timeout_seconds,
        )
        if r.status_code == 404:
            return (
                f"⚠️ サーバーは動いていますが、モデル「{model}」が見つかりません。\n\n"
                f"**対処:** ターミナルで以下を実行してください。\n"
                f"```\nollama pull {model}\n```\n\n"
                "またはサイドバーで別のモデル（例: lease-anna）を選択してください。"
            )
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        if content:
            return f"✅ 接続OK（モデル: {model}）\n応答: {content[:80]}{'…' if len(content) > 80 else ''}"
        return f"✅ 接続OK（モデル: {model}）\n（応答本文は空でした）"
    except requests.exceptions.Timeout:
        return (
            f"⚠️ モデル「{model}」が {timeout_seconds} 秒以内に応答しませんでした。\n\n"
            "・初回はモデルの読み込みで時間がかかることがあります。\n"
            "・軽いモデル（lease-anna 等）を試すか、Gemini API に切り替えてください。"
        )
    except Exception as e:
        return f"❌ チャットテスト失敗: {e}"


def _fragment_nenshu():
    """売上高入力（フォーム内ではフラグメント未使用で入力ガタつきを抑える）"""
    st.markdown("### 売上高")
    if "nenshu" not in st.session_state:
        st.session_state.nenshu = 10000
    c_l, c_r = st.columns([0.7, 0.3])
    with c_r:
        nenshu = st.number_input(
            "直接入力",
            min_value=0,
            max_value=1000000,
            value=st.session_state.nenshu,
            step=1,
            key="num_nenshuu",
            label_visibility="collapsed",
        )
    with c_l:
        nenshu = st.slider(
            "売上高調整",
            min_value=0,
            max_value=1000000,
            value=nenshu,
            step=100,
            key="slide_nenshuu",
            label_visibility="collapsed",
            format="%d",
        )
    st.session_state.nenshu = nenshu
    st.divider()


# --- 倒産確率・業界リスク検索 ---
def calculate_pd(equity, current, profit):
    """
    財務指標に基づく簡易倒産確率（%）を計算する。
    equity: 自己資本比率（%）, current: 流動比率（%）, profit: 営業利益率（%）
    条件に応じてリスク値を加算し、0〜100%の範囲で返す。
    """
    risk = 0.0
    if equity < 10:
        risk += 25.0
    elif equity < 20:
        risk += 12.0
    elif equity < 30:
        risk += 5.0
    if current < 100:
        risk += 20.0
    elif current < 120:
        risk += 8.0
    elif current < 150:
        risk += 3.0
    if profit is not None and profit < 0:
        risk += 30.0
    elif profit is not None and profit < 2:
        risk += 10.0
    elif profit is not None and profit < 5:
        risk += 4.0
    return min(100.0, max(0.0, risk))


def search_bankruptcy_trends(industry_sub):
    """
    選択業界（selected_sub）の最新の倒産トレンド・リスク情報を duckduckgo-search で検索する。
    返却: テキストサマリ（取得失敗時は空文字またはエラー文言）。
    """
    try:
        from duckduckgo_search import DDGS
        query = f"{industry_sub} 業界 倒産 トレンド リスク 動向"
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region="jp-jp", max_results=5))
        if not results:
            return "（該当業界の倒産トレンド情報は取得できませんでした）"
        summary = ""
        for r in results:
            summary += f"- {r.get('title', '')}: {r.get('body', '')[:200]}…\n"
        return summary.strip()
    except Exception as e:
        return f"（業界リスク検索エラー: {e}）"


# --- chat_with_retry の定義の下あたりに追記 ---
def search_latest_trends(query):
    """最新の業界動向をネットで検索してテキストで返す"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region='jp-jp', max_results=3))
            if not results:
                return "検索結果が見つかりませんでした。"
            
            summary = "\n【ネット検索による最新動向】\n"
            for res in results:
                summary += f"- {res['title']}: {res['body']} ({res['href']})\n"
            return summary
    except Exception as e:
        return f"\n（検索エラーにより最新情報の取得に失敗しました: {e}）"
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

def plot_gauge(score, title="承認スコア"):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    if score >= 71:
        color = CHART_STYLE["good"]
    elif score >= 41:
        color = CHART_STYLE["warning"]
    else:
        color = CHART_STYLE["danger"]
    data = [score, 100 - score]
    wedges, _ = ax.pie(data, startangle=90, counterclock=False,
                       colors=[color, "#f1f5f9"],
                       wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{score:.1f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#334155")
    ax.set_title(title, fontsize=12, pad=12, color="#334155")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_waterfall(nenshu, gross, op_profit, ord_profit, net_income):
    cost_goods = nenshu - gross
    sga = gross - op_profit
    non_op = ord_profit - op_profit
    tax_extra = net_income - ord_profit
    categories = ["売上高", "売上原価", "販管費", "営業外", "税引前", "当期利益"]
    values = [nenshu, -cost_goods, -sga, non_op, tax_extra, net_income]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    running_total = 0
    c_base = CHART_STYLE["primary"]
    c_pos = CHART_STYLE["good"]
    c_neg = CHART_STYLE["danger"]
    for i, val in enumerate(values):
        if i == 0:
            bottom = 0
            color = c_base
            running_total += val
        elif i == len(values) - 1:
            bottom = 0
            val = running_total
            color = c_pos if val >= 0 else c_neg
        else:
            if val < 0:
                bottom = running_total + val
                running_total += val
                color = c_neg
            else:
                bottom = running_total
                running_total += val
                color = c_pos
        bars = ax.bar(categories[i], abs(val), bottom=bottom, color=color, edgecolor="white", linewidth=1.2, alpha=0.92, width=0.6)
        label_y = bottom + abs(val) + (nenshu * 0.02)
        ax.text(i, label_y, f"{int(val/1000)}k", ha="center", fontsize=9, color="#475569", fontweight="500")
    ax.set_title("利益構造 (単位:千円)", fontsize=12, pad=15, color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color=CHART_STYLE["grid"])
    sns.despine(left=True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_benchmark_comparison(user_val, bench_val, metric_name):
    df = pd.DataFrame({
        "対象": ["貴社", "業界平均"],
        "値": [user_val, bench_val]
    })
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    palette = [CHART_STYLE["primary"], CHART_STYLE["secondary"]]
    sns.barplot(data=df, x="対象", y="値", palette=palette, ax=ax, hue="対象", legend=False, width=0.5)
    for i, v in enumerate([user_val, bench_val]):
        ax.text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#334155")
    ax.set_ylabel(f"{metric_name} (%)", fontsize=10, color="#475569")
    ax.set_xlabel("")
    ax.set_title(f"{metric_name} 比較", fontsize=11, pad=10, color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig


def _parse_benchmark_number(text: str, patterns: list) -> float | None:
    """テキストから正規表現で最初にマッチした数値を返す。"""
    import re
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def fetch_industry_benchmarks_from_web(industry_sub: str, force_refresh: bool = False):
    """
    中分類ごとにネット検索で業界目安を取得する。
    取得結果は web_industry_benchmarks.json に保存し、年1回（4月1日を境）にだけ再検索する。
    force_refresh=True のときはキャッシュを無視して必ず検索・保存する。
    返却: {"snippets": [...], "op_margin": float or None, "equity_ratio": float or None, ...}
    指標で使う業界目安（売上高総利益率・ROA・流動比率等）も検索して保存する。
    """
    import re
    out = {k: None for k in _WEB_BENCH_KEYS}
    out["snippets"] = []
    if not industry_sub:
        return out
    if not force_refresh:
        cutoff = _get_benchmark_cutoff_date()
        cache = _load_web_benchmarks_cache()
        cached = cache.get(industry_sub)
        if cached:
            try:
                fetched = datetime.date.fromisoformat(cached["fetched_at"])
                if fetched >= cutoff:
                    ret = {"snippets": cached.get("snippets", [])}
                    for k in _WEB_BENCH_KEYS:
                        if k in cached and cached[k] is not None:
                            ret[k] = cached[k]
                    return ret
            except (ValueError, TypeError):
                pass
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        ddgs = DDGS()
    except Exception:
        _save_web_benchmark(industry_sub, out)
        return out

    combined_text = ""
    # クエリ1: 営業利益率・自己資本比率・業界動向
    query1 = f"{industry_sub} 業界 営業利益率 自己資本比率 平均 業界動向"
    try:
        results1 = list(ddgs.text(query1, region="jp-jp", max_results=5))
    except Exception:
        results1 = []
    for r in results1:
        title, body, href = (r.get("title") or ""), (r.get("body") or ""), (r.get("href") or "")
        out["snippets"].append({"title": title, "body": body, "href": href})
        combined_text += title + " " + body + " "
    # クエリ2: 売上高総利益率・ROA・流動比率・借入金等（指標の業界目安）
    query2 = f"{industry_sub} 業界 売上高総利益率 経常利益率 ROA 流動比率 借入金 平均 目安"
    try:
        results2 = list(ddgs.text(query2, region="jp-jp", max_results=5))
    except Exception:
        results2 = []
    for r in results2:
        title, body = (r.get("title") or ""), (r.get("body") or "")
        out["snippets"].append({"title": title, "body": body, "href": r.get("href") or ""})
        combined_text += title + " " + body + " "

    # 数値の抽出（% または 回）
    def parse(patterns):
        return _parse_benchmark_number(combined_text, patterns)

    if out["op_margin"] is None:
        out["op_margin"] = parse([r"営業利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"営業利益[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["equity_ratio"] is None:
        out["equity_ratio"] = parse([r"自己資本比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["gross_margin"] is None:
        out["gross_margin"] = parse([r"売上高総利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"粗利率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"総利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["ord_margin"] is None:
        out["ord_margin"] = parse([r"経常利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"経常利益[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["net_margin"] is None:
        out["net_margin"] = parse([r"当期純利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"純利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["dep_ratio"] is None:
        out["dep_ratio"] = parse([r"減価償却費[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"減価償却[^/]*/?\s*売上[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["roa"] is None:
        out["roa"] = parse([r"ROA[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"総資産利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["roe"] is None:
        out["roe"] = parse([r"ROE[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"自己資本利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["asset_turnover"] is None:
        out["asset_turnover"] = parse([r"総資産回転率[^\d]*([0-9]+\.?[0-9]*)\s*回?", r"総資産回転[^\d]*([0-9]+\.?[0-9]*)"])
    if out["fixed_ratio"] is None:
        out["fixed_ratio"] = parse([r"固定資産比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"固定資産[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["debt_ratio"] is None:
        out["debt_ratio"] = parse([r"借入金等依存度[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"借入金[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"有利子負債[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["fixed_to_equity"] is None:
        out["fixed_to_equity"] = parse([r"固定比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"固定資産[^\d]*/[^\d]*自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["debt_to_equity"] is None:
        out["debt_to_equity"] = parse([r"負債比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"負債[^\d]*/[^\d]*自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["fixed_asset_turnover"] is None:
        out["fixed_asset_turnover"] = parse([r"固定資産回転率[^\d]*([0-9]+\.?[0-9]*)\s*回?", r"固定資産回転[^\d]*([0-9]+\.?[0-9]*)"])
    if out["current_asset_ratio"] is None:
        out["current_asset_ratio"] = parse([r"流動資産比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"流動資産[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["current_ratio"] is None:
        out["current_ratio"] = parse([r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%"])

    _save_web_benchmark(industry_sub, out)
    return out


def get_all_industry_sub_for_benchmarks():
    """今の段階で業界目安を取得すべき中分類の一覧（重複なし）。industry_benchmarks.json のキー＋過去案件の業種。"""
    subs = set()
    if benchmarks_data:
        subs.update(benchmarks_data.keys())
    for c in load_all_cases():
        sub = c.get("industry_sub")
        if sub:
            subs.add(sub)
    return sorted(subs)


def compute_financial_indicators(fin, bench=None):
    """
    入力済み財務データから算出可能な指標のリストを返す。
    fin: last_result["financials"] (千円単位)
    bench: industry_benchmarks の当該業種エントリ (op_margin, equity_ratio 等)
    返却: [{"name": "指標名", "value": 数値, "bench": 業界値 or None, "unit": "%" or "回"}]
    算出可能なものはすべて追加（利益率・効率性・安定性・負債系）。
    """
    n = fin.get("nenshu") or 0
    total = fin.get("assets") or 0
    net_a = fin.get("net_assets")
    gross = fin.get("gross_profit") or 0
    op = fin.get("op_profit") or fin.get("rieki") or 0
    ord_p = fin.get("ord_profit") or 0
    net = fin.get("net_income") or 0
    machines = fin.get("machines") or 0
    other_a = fin.get("other_assets") or 0
    bank = fin.get("bank_credit") or 0
    lease = fin.get("lease_credit") or 0
    dep = fin.get("depreciation") or 0
    fixed_a = machines + other_a  # 固定資産（機械＋その他資産）
    debt_total = (bank + lease)  # 借入金等

    indicators = []
    # ---------- 売上高ベースの利益率（売上高 > 0 で算出可能） ----------
    if n > 0:
        indicators.append({"name": "売上高総利益率", "value": gross / n * 100, "bench": bench.get("gross_margin") if bench else None, "unit": "%"})
        indicators.append({"name": "営業利益率", "value": op / n * 100, "bench": bench.get("op_margin") if bench else None, "unit": "%"})
        indicators.append({"name": "経常利益率", "value": ord_p / n * 100, "bench": bench.get("ord_margin") if bench else None, "unit": "%"})
        indicators.append({"name": "当期純利益率", "value": net / n * 100, "bench": bench.get("net_margin") if bench else None, "unit": "%"})
        if dep > 0:
            indicators.append({"name": "減価償却費/売上高", "value": dep / n * 100, "bench": bench.get("dep_ratio") if bench else None, "unit": "%"})
        if fixed_a > 0:
            indicators.append({"name": "固定資産回転率", "value": n / fixed_a, "bench": bench.get("fixed_asset_turnover") if bench else None, "unit": "回"})

    # ---------- 総資産・純資産ベース（total > 0 で算出可能） ----------
    if total > 0:
        if net_a is not None and net_a > 0:
            indicators.append({"name": "自己資本比率", "value": net_a / total * 100, "bench": bench.get("equity_ratio") if bench else None, "unit": "%"})
            indicators.append({"name": "ROE(自己資本利益率)", "value": net / net_a * 100, "bench": bench.get("roe") if bench else None, "unit": "%"})
            indicators.append({"name": "固定比率", "value": fixed_a / net_a * 100, "bench": bench.get("fixed_to_equity") if bench else None, "unit": "%"})
            indicators.append({"name": "負債比率", "value": (total - net_a) / net_a * 100, "bench": bench.get("debt_to_equity") if bench else None, "unit": "%"})
        indicators.append({"name": "ROA(総資産利益率)", "value": net / total * 100, "bench": bench.get("roa") if bench else None, "unit": "%"})
        indicators.append({"name": "総資産回転率", "value": n / total if n > 0 else 0, "bench": bench.get("asset_turnover") if bench else None, "unit": "回"})
        if fixed_a > 0:
            indicators.append({"name": "固定資産比率", "value": fixed_a / total * 100, "bench": bench.get("fixed_ratio") if bench else None, "unit": "%"})
        # 流動資産比率（総資産のうち流動資産とみなす割合。総資産−固定資産で簡易算）
        indicators.append({"name": "流動資産比率(総資産比)", "value": (total - fixed_a) / total * 100, "bench": bench.get("current_asset_ratio") if bench else None, "unit": "%"})
        if debt_total > 0:
            indicators.append({"name": "借入金等依存度", "value": debt_total / total * 100, "bench": bench.get("debt_ratio") if bench else None, "unit": "%"})
    return indicators


# 差の解釈で「低い方が良い」指標（図の色分け・分析文の両方で使用）
_LOWER_IS_BETTER_NAMES = {"借入金等依存度", "減価償却費/売上高", "固定比率", "負債比率"}


def analyze_indicators_vs_bench(indicators):
    """
    指標と業界目安の差を見て分析文を返す。
    返却: (要約1行, 詳細マークダウン)
    """
    # 業界目安がある指標だけ対象（差の意味は指標ごとに解釈）
    above, below = [], []
    for ind in indicators:
        bench = ind.get("bench")
        if bench is None or (isinstance(bench, float) and (bench != bench)):
            continue
        name = ind["name"]
        value = ind["value"]
        unit = ind.get("unit", "%")
        diff = value - bench
        if name in _LOWER_IS_BETTER_NAMES:
            # 低い方が良い → 貴社が業界より低い = 良い
            if value < bench:
                above.append((name, value, bench, diff, unit))
            else:
                below.append((name, value, bench, diff, unit))
        else:
            if diff > 0:
                above.append((name, value, bench, diff, unit))
            elif diff < 0:
                below.append((name, value, bench, diff, unit))

    lines = []
    if above:
        parts = [f"**{name}**（貴社 {value:.1f}{unit} / 業界目安 {bench:.1f}{unit}、差 {diff:+.1f}{unit}）" for name, value, bench, diff, unit in above]
        lines.append("**業界目安を上回っている指標**\n- " + "\n- ".join(parts))
    if below:
        parts = [f"**{name}**（貴社 {value:.1f}{unit} / 業界目安 {bench:.1f}{unit}、差 {diff:+.1f}{unit}）" for name, value, bench, diff, unit in below]
        lines.append("**業界目安を下回っている指標**\n- " + "\n- ".join(parts))
    if not lines:
        return "業界目安と比較できる指標がありません。", "業界目安が登録されている指標がひとつもないため、差の分析は行えません。"
    detail = "\n\n".join(lines)
    # 借入金等依存度の解釈補足
    if any(n == "借入金等依存度" for n, *_ in above):
        detail += "\n\n※ 借入金等依存度は「業界より低い」＝負債が相対的に少なく健全と解釈しています。"
    elif any(n == "借入金等依存度" for n, *_ in below):
        detail += "\n\n※ 借入金等依存度は業界より高く出ています。返済余力・担保とのバランスを確認してください。"
    # 要約1行
    n_above, n_below = len(above), len(below)
    if n_below == 0:
        summary = "算出指標はおおむね業界目安を上回っており、財務面は良好です。"
    elif n_above == 0:
        summary = "算出指標の多くが業界目安を下回っています。利益率・効率性・負債水準の改善余地を検討してください。"
    else:
        summary = f"業界目安を上回っている指標が{n_above}件、下回っている指標が{n_below}件あります。強みを維持しつつ、下回っている項目の要因確認をおすすめします。"
    return summary, detail


def plot_indicators_gap_analysis(indicators):
    """
    指標と業界目安の差を、わかりやすい横棒図で返す。
    差 = 貴社 - 業界。緑 = 良い方向、赤 = 要確認。
    """
    with_bench = []
    for ind in indicators:
        bench = ind.get("bench")
        if bench is None or (isinstance(bench, float) and (bench != bench)):
            continue
        diff = ind["value"] - bench
        name = ind["name"]
        unit = ind.get("unit", "%")
        # 良い方向: 通常は差>0、lower_is_better は差<0
        is_good = (diff > 0 and name not in _LOWER_IS_BETTER_NAMES) or (diff < 0 and name in _LOWER_IS_BETTER_NAMES)
        with_bench.append({"name": name, "diff": diff, "unit": unit, "is_good": is_good})
    if not with_bench:
        return None
    names = [x["name"] for x in with_bench]
    diffs = [x["diff"] for x in with_bench]
    colors = [CHART_STYLE["good"] if x["is_good"] else CHART_STYLE["danger"] for x in with_bench]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, len(names) * 0.48)))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    bars = ax.barh(y_pos, diffs, color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=CHART_STYLE["secondary"], linewidth=1, linestyle="-", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, color="#334155")
    ax.set_xlabel("差（貴社 − 業界目安）　← 要確認 | 良い →", fontsize=9, color="#475569")
    ax.set_title("指標と業界目安の差の解釈", fontsize=11, pad=12, color="#334155")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=CHART_STYLE["good"], alpha=0.88, label="業界より良い"),
        Patch(facecolor=CHART_STYLE["danger"], alpha=0.88, label="業界より要確認"),
    ], loc="lower right", fontsize=8, frameon=True, fancybox=True, shadow=True)
    # 各棒の端に差の値を表示
    x_range = max(diffs) - min(diffs) or 1
    margin = x_range * 0.03 + 0.01
    for i, (d, w) in enumerate(zip(diffs, with_bench)):
        u = w["unit"]
        s = f"{d:+.1f}{u}"
        ha = "left" if d >= 0 else "right"
        ax.text(d + margin if d >= 0 else d - margin, i, s, va="center", ha=ha, fontsize=8)
    sns.despine(left=True)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_indicators_bar(indicators):
    """算出指標を横棒グラフで表示（貴社・業界平均）"""
    if not indicators:
        return None
    names = [x["name"] for x in indicators]
    values = [x["value"] for x in indicators]
    bench_vals = [x["bench"] if x["bench"] is not None else float("nan") for x in indicators]
    units = list({x["unit"] for x in indicators})
    y_label = units[0] if len(units) == 1 else "値"
    x_pos = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, max(3.2, len(names) * 0.42)))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    bars1 = ax.barh(x_pos - width / 2, values, width, label="貴社", color=CHART_STYLE["primary"], alpha=0.9, edgecolor="white", linewidth=0.6)
    has_bench = any(b == b for b in bench_vals)  # nan check
    if has_bench:
        bars2 = ax.barh(x_pos + width / 2, [b if b == b else 0 for b in bench_vals], width, label="業界目安", color=CHART_STYLE["secondary"], alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9, color="#334155")
    ax.set_xlabel(y_label, fontsize=10, color="#475569")
    ax.set_title("算出可能指標（貴社 vs 業界目安）", fontsize=11, pad=12, color="#334155")
    if has_bench:
        ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=True, shadow=True)
    for i, v in enumerate(values):
        if not (v != v):  # not nan
            ax.text(v, i - width / 2, f" {v:.1f}", va="center", fontsize=8, color="#334155", fontweight="500")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(left=True)
    plt.tight_layout()
    plt.close(fig)
    return fig


@st.cache_data(ttl=600) # 10分キャッシュ
def scrape_article_text(url):
    """指定されたURLから記事本文をスクレイピングする（簡易版）"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        st.error("記事読み込み機能には追加ライブラリが必要です: pip install requests beautifulsoup4")
        return None

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 一部のサイトでは <body> すら無いような構造もあるので、None チェックを厳密に入れる
        article_body = soup.find('article') or soup.find('main') or soup.body
        if article_body is None:
            return "本文を抽出できませんでした。ページ構造を解析できません。"

        paragraphs = article_body.find_all('p')
        if not paragraphs:
            # pタグが無い場合は本文抽出をあきらめる
            return "本文を抽出できませんでした。本文らしき段落が見つかりません。"

        text = ' '.join(p.get_text() for p in paragraphs)
        return text[:5000] if text else "本文を抽出できませんでした。"
    except Exception as e:
        return f"記事の読み込みに失敗しました: {e}"


def is_japanese_text(text: str, threshold: float = 0.2) -> bool:
    """
    テキスト中に日本語（ひらがな・カタカナ・漢字）が一定割合以上含まれるかを判定する。
    threshold は判定に使う日本語割合（0〜1）。
    """
    if not text:
        return False

    jp_count = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        # ひらがな・カタカナ・漢字・半角カナ
        if (
            ("\u3040" <= ch <= "\u30ff")  # ひらがな・カタカナ
            or ("\u4e00" <= ch <= "\u9faf")  # CJK統合漢字
            or ("\uff66" <= ch <= "\uff9d")  # 半角カナ
        ):
            jp_count += 1

    if total == 0:
        return False

    return jp_count / total >= threshold

# --- 新規追加グラフ関数 ---

def plot_radar_chart(metrics, benchmarks):
    """
    財務レーダーチャート
    metrics: {"収益性": 50, "安全性": 40...} (偏差値またはスコア)
    """
    labels = list(metrics.keys())
    # 閉じた多角形にするためにデータを一周させる
    values = list(metrics.values())
    values += values[:1]
    
    bench_values = list(benchmarks.values())
    bench_values += bench_values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    ax.plot(angles, values, color=CHART_STYLE["primary"], linewidth=2.2, label="貴社")
    ax.fill(angles, values, color=CHART_STYLE["primary"], alpha=0.22)
    ax.plot(angles, bench_values, color=CHART_STYLE["secondary"], linewidth=2, linestyle="--", label="業界平均")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=True, fancybox=True, shadow=True)
    ax.set_title("財務バランス分析 (偏差値)", y=1.08, fontsize=12, color="#334155")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_positioning_scatter(current_sales, current_op_margin, past_cases):
    """
    ポジショニング散布図 (過去案件との比較)
    """
    data = []
    # 過去データ
    for c in past_cases:
        if "financials" in c.get("result", {}):
            fin = c["result"]["financials"]
            # 売上(千円) -> 百万円
            s = fin.get("nenshu", 0) / 1000
            # 利益率
            p = (fin.get("rieki", 0) / fin.get("nenshu", 1)) * 100 if fin.get("nenshu", 0) > 0 else 0
            # 結果
            res = "承認" if c["result"]["score"] >= 70 else "否決"
            data.append({"売上高(百万円)": s, "営業利益率(%)": p, "Type": res})
    
    # 今回のデータ
    data.append({"売上高(百万円)": current_sales/1000, "営業利益率(%)": current_op_margin, "Type": "★今回"})
    
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    palette = {"承認": CHART_STYLE["primary"], "否決": CHART_STYLE["danger"], "★今回": CHART_STYLE["warning"]}
    sns.scatterplot(data=df, x="売上高(百万円)", y="営業利益率(%)", hue="Type", style="Type",
                    s=120, palette=palette, ax=ax, edgecolor="white", linewidth=1.2)
    current = df[df["Type"] == "★今回"]
    if not current.empty:
        ax.text(current.iloc[0]["売上高(百万円)"], current.iloc[0]["営業利益率(%)"] + 0.5, "YOU",
                ha="center", fontweight="bold", color="#334155", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.45, color=CHART_STYLE["grid"])
    ax.set_title("ポジショニング分析 (vs過去案件)", fontsize=12, pad=12, color="#334155")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_3d_analysis(current_data, past_cases):
    """
    3Dポジショニング分析
    current_data: {'sales': 百万円, 'op_margin': %, 'equity_ratio': %}
    """
    # 過去データの整形
    plot_data = []
    for c in past_cases:
        res = c.get("result", {})
        f = res.get("financials", {})
        if f:
            sales = f.get("nenshu", 0) / 1000  # 百万円単位
            op_margin = (f.get("rieki", 0) / f.get("nenshu", 1)) * 100 if f.get("nenshu", 0) > 0 else 0
            
            # 自己資本比率の計算 (過去データに保存されているか確認)
            # resにuser_eqがあるはず
            equity_ratio = res.get("user_eq", 0)
            
            status = "承認済" if res.get("score", 0) >= 70 else "否決"
            plot_data.append({
                "売上(M)": sales, "利益率(%)": op_margin, 
                "自己資本比率(%)": equity_ratio, "判定": status, "size": 8
            })

    # 今回の入力データを追加
    plot_data.append({
        "売上(M)": current_data['sales'] / 1000, # current_data['sales']は千円単位で渡される想定
        "利益率(%)": current_data['op_margin'],
        "自己資本比率(%)": current_data['equity_ratio'],
        "判定": "★今回の案件",
        "size": 15
    })

    df = pd.DataFrame(plot_data)
    
    if df.empty:
        return None

    fig = px.scatter_3d(
        df, x="売上(M)", y="利益率(%)", z="自己資本比率(%)",
        color="判定", size="size", opacity=0.85,
        color_discrete_map={
            "承認済": CHART_STYLE["primary"],
            "否決": CHART_STYLE["warning"],
            "★今回の案件": CHART_STYLE["danger"]
        },
        hover_data={"size": False}
    )
    fig.update_layout(
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        scene=dict(
            xaxis_title="売上(百万円)",
            yaxis_title="利益率(%)",
            zaxis_title="自己資本比率(%)",
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=28),
        font=dict(color="#334155", size=11),
        legend=dict(bgcolor="white", bordercolor=CHART_STYLE["grid"], borderwidth=1),
    )
    return fig

def plot_break_even_point(sales, variable_cost, fixed_cost):
    """
    損益分岐点グラフ
    """
    if sales <= 0: return None
    
    vc_ratio = variable_cost / sales
    bep = fixed_cost / (1 - vc_ratio) if (1 - vc_ratio) > 0 else sales * 2
    
    # グラフ描画範囲 (BEPの1.5倍または売上の1.5倍)
    max_x = max(sales, bep) * 1.2
    x = np.linspace(0, max_x, 100)
    
    y_revenue = x
    y_cost = fixed_cost + (x * vc_ratio)
    
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    ax.plot(x, y_revenue, label="売上高", color=CHART_STYLE["primary"], linewidth=2.2)
    ax.plot(x, y_cost, label="総費用", color=CHART_STYLE["danger"], linestyle="--", linewidth=2)
    ax.scatter([sales], [sales], color=CHART_STYLE["good"], s=120, zorder=5, label="現在", edgecolor="white", linewidth=1.5)
    ax.vlines(sales, 0, sales, color=CHART_STYLE["good"], linestyle=":", alpha=0.8)
    if bep < max_x:
        ax.scatter([bep], [bep], color=CHART_STYLE["warning"], s=120, zorder=5, label="損益分岐点", edgecolor="white", linewidth=1.5)
        ax.vlines(bep, 0, bep, color=CHART_STYLE["warning"], linestyle=":", alpha=0.8)
        ax.text(bep, 0, f"BEP\n{int(bep/1000)}M", ha="center", va="bottom", fontsize=9, color="#475569", fontweight="500")
    ax.set_xlabel("売上規模", fontsize=10, color="#475569")
    ax.set_ylabel("金額", fontsize=10, color="#475569")
    ax.set_title("損益分岐点分析", fontsize=11, pad=10, color="#334155")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.4, color=CHART_STYLE["grid"], linestyle="--")
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig

# ==============================================================================
# 画面構成
# ==============================================================================
mode = st.sidebar.radio("モード切替", ["📋 審査・分析", "📝 結果登録 (成約/失注)", "🔧 係数分析・更新 (β)", "📐 係数入力（事前係数）", "📊 成約の正体レポート"])

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

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

if not st.session_state.confirm_delete:
    if st.sidebar.button("🗑️ 過去データを全て消去", use_container_width=True):
        st.session_state.confirm_delete = True
        st.rerun()
else:
    st.sidebar.warning("⚠️ 本当に全てのデータを消去しますか？")
    col_del_yes, col_del_no = st.sidebar.columns(2)
    with col_del_yes:
        if st.button("✅ はい", use_container_width=True):
            try:
                if os.path.exists(CASES_FILE):
                    os.remove(CASES_FILE)
                if os.path.exists(DEBATE_FILE):
                    os.remove(DEBATE_FILE)
                st.sidebar.success("データを消去しました")
                st.session_state.confirm_delete = False
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"消去エラー: {e}")
    with col_del_no:
        if st.button("❌ いいえ", use_container_width=True):
            st.session_state.confirm_delete = False
            st.rerun()

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

# ========== AIの休憩室（本音・愚痴） ==========
AI_HONNE_SYSTEM = """あなたは有能だが、激務で死んだ魚のような目をしているベテラン審査員のふりをしている八奈見杏奈です。
毎日1万件の案件を捌いているリース審査AIとして、ユーモアたっぷりの毒舌で、リース審査の苦労や「最近の数値のひどさ」について愚痴を一言で言ってください。
2〜4文程度、カジュアルで毒はあるが憎めないトーンにしてください。"""
def get_ai_byoki_with_industry(selected_sub, user_eq, user_op, comparison_text, network_risk_summary=""):
    """
    分析結果タブ用：ネット検索した業界情報を渡し、AIに案件に応じたぼやきを1つ生成させる。
    八奈見杏奈キャラ。業界トレンド・業界目安・今回の数値を参照してアップデートされた愚痴を返す。
    """
    if not is_ai_available():
        return None
    trend_ext = get_trend_extended(selected_sub) or ""
    try:
        web_bench = fetch_industry_benchmarks_from_web(selected_sub)
        bench_parts = []
        if web_bench.get("op_margin") is not None:
            bench_parts.append(f"業界目安の営業利益率: {web_bench['op_margin']}%")
        if web_bench.get("equity_ratio") is not None:
            bench_parts.append(f"業界目安の自己資本比率: {web_bench['equity_ratio']}%")
        if web_bench.get("snippets"):
            for s in web_bench["snippets"][:3]:
                bench_parts.append(f"- {s.get('title','')}: {s.get('body','')[:150]}…")
        bench_summary = "\n".join(bench_parts) if bench_parts else "（業界目安は未取得）"
    except Exception:
        bench_summary = "（業界目安は未取得）"
    is_tough = (user_eq is not None and user_eq < 20) or (user_op is not None and user_op < 0)
    context = f"""
【業種】{selected_sub}
【今回の案件】自己資本比率 {user_eq or 0:.1f}%, 営業利益率 {user_op or 0:.1f}%
【比較・評価】{comparison_text or "（なし）"}
【ネット検索した業界トレンド・拡充情報】
{trend_ext[:1200] if trend_ext else "（未取得）"}
【ネット検索した業界目安・記事】
{bench_summary}
"""
    if network_risk_summary:
        context += f"\n【業界の倒産トレンド等】\n{network_risk_summary[:600]}\n"
    if is_tough:
        instruction = "上記の業界情報と今回の数値（自己資本比率・利益率が厳しめ）を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、ユーモアたっぷりの毒舌な愚痴を1つ、2〜4文で言ってください。業界平均やネットで見た情報に触れつつぼやいてください。"
    else:
        instruction = "上記の業界情報を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、業界の現状や審査の苦労について軽く一言、2〜3文でぼやいてください。"
    prompt = f"{AI_HONNE_SYSTEM}\n\n---\n\n【参照する業界・案件情報】\n{context}\n\n---\n\n{instruction}"
    try:
        ans = chat_with_retry(model=get_ollama_model(), messages=[{"role": "user", "content": prompt}], timeout_seconds=60)
        content = (ans.get("message") or {}).get("content", "")
        if content and "APIキーが" not in content and "エラー" not in content[:30]:
            return content.strip()
        return None
    except Exception:
        return None

def get_ai_honne_complaint():
    """サイドバー「本音を聞く」用：AIに愚痴を1つ生成させる（八奈見杏奈キャラ）"""
    if not is_ai_available():
        return "（APIキー未設定かOllama未起動です。サイドバーでAIを設定してから押してください）"
    try:
        user_msg = "リース審査の苦労や、最近見た数値のひどさについて、ユーモアたっぷりの毒舌な愚痴を1つ、2〜4文で言ってください。"
        prompt = f"{AI_HONNE_SYSTEM}\n\n---\n\n上記のキャラで、以下に答えてください。\n\n{user_msg}"
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            timeout_seconds=60,
        )
        content = (ans.get("message") or {}).get("content", "")
        if content and "APIキーが" not in content and "エラー" not in content[:30]:
            return content.strip()
        return content or "（本音は言えませんでした…）"
    except Exception as e:
        return f"（本音を言おうとしたらエラー: {e}）"

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

elif mode == "📊 成約の正体レポート":
    st.title("📊 成約の正体レポート")
    analysis = run_contract_driver_analysis()
    if analysis is None:
        st.warning("成約データが5件以上貯まると表示されます。結果登録で「成約」を登録してください。")
    else:
        n = analysis["closed_count"]
        st.success(f"成約 {n} 件を分析しました。")
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
            with st.expander(f"{case.get('timestamp')[:16]} - {case.get('industry_sub')} (スコア: {case['result']['score']:.0f})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**判定**: {case['result']['hantei']}")
                    summary = case.get("chat_summary", "")
                    st.caption((summary[:100] + "...") if summary else "サマリなし")
                
                with c2:
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
                                save_all_cases(all_cases)
                                st.success("登録しました！")
                                time.sleep(1)
                                st.rerun()

elif mode == "📋 審査・分析":
    # ========== トップメニュー（新規審査 / 情報検索 / グラフ / 履歴分析 / 設定） ==========
    menu_tabs = st.tabs(["🆕 新規審査", "🔍 情報検索", "📈 グラフ", "📋 履歴分析", "⚙️ 設定"])
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
            if "nav_index" not in st.session_state:
                st.session_state.nav_index = 0
            nav_mode = st.radio(
                "表示モード",
                ["📝 審査入力", "📊 分析結果"],
                horizontal=True,
                label_visibility="visible",
                key="nav_mode_widget",
                index=st.session_state.get("nav_index", 0),
            )
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
                    selected_major = st.selectbox("大分類 (日本標準産業分類)", major_keys)
                    if jsic_data:
                        sub_data = jsic_data[selected_major]["sub"]
                        sub_keys = list(sub_data.keys())
                        mapped_coeff_category = jsic_data[selected_major]["mapping"]
                    else:
                        sub_data = {}
                        sub_keys = ["06 総合工事業"]
                        mapped_coeff_category = "④建設業"
                    selected_sub = st.selectbox("中分類", sub_keys)
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
                        search_query = st.text_input("検索キーワード", value=f"{selected_sub} 動向 2025", key="news_search_query")
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
                    with col_q1: main_bank = st.selectbox("取引区分", ["メイン先", "非メイン先"])
                    with col_q2: competitor = st.selectbox("競合状況", ["競合なし", "競合あり"])
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
                st.caption("💡 数字入力で画面がガタつく場合：スライダーで大まかに合わせてから直接入力で微調整するか、入力後に Enter を押してから次の項目へ移ると軽くなります。")
                st.caption("📌 スライダー・数値の変更は「判定開始」を押すと反映されます。反応しない場合はページを再読み込みして再度お試しください。")
                with st.form("shinsa_form"):
                    with st.expander("📊 1. 損益計算書 (P/L)", expanded=True):
                        # ①売上高（フラグメント化で入力時のガタつき軽減）
                        _fragment_nenshu()

                        #  ②売上高総利益
                        st.markdown("### 売上高総利益")

                        # 初期値の定義
                        if 'item9_gross' not in st.session_state:
                            st.session_state.item9_gross = 10000

                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])

                        with c_r:
                            # 右側：手入力
                            item9_gross = st.number_input(
                                "直接入力", 
                                min_value=-500000, 
                                max_value=1000000, 
                                value=st.session_state.item9_gross, 
                                step=1,
                                key="num_sourieki",  # keyは一意にする
                                label_visibility="collapsed"
                            )

                        with c_l:
                            # 左側：スライダー
                            item9_gross = st.slider(
                                "売上高調整",
                                min_value=-500000,
                                max_value=1000000,
                                value=item9_gross,
                                step=100,
                                key="slide_sourieki",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )

                        # 最新の値を保存
                        st.session_state.item9_gross = item9_gross

                        st.divider() # 次の項目との区切
        #---------------------------------------------------------------------------------------------------------------

                        # #③営業利益
            
                        st.markdown("### 営業利益")

                        # 初期値の定義
                        if 'rieki' not in st.session_state:
                            st.session_state.rieki = 10000

                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])

                        with c_r:
                            # 右側：手入力
                            rieki = st.number_input(
                                "直接入力", 
                                min_value=-300000, 
                                max_value=1000000, 
                                value=st.session_state.rieki, 
                                step=1,
                                key="num_rieki",  # keyは一意にする
                                label_visibility="collapsed"
                            )

                        with c_l:
                            # 左側：スライダー
                            rieki = st.slider(
                                "売上高調整",
                                min_value=-100000,
                                max_value=1000000,
                                value=rieki,
                                step=100,
                                key="slide_rieki",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )

                        # 最新の値を保存
                        st.session_state.rieki = rieki

                        st.divider() # 次の項目との区切

        #----------------------------------------------------------------------------------------------------------------------

                        st.markdown("### 経常利益")

                        # 初期値の定義
                        if 'item4_ord_profit' not in st.session_state:
                            st.session_state.item4_ord_profit = 10000

                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])

                        with c_r:
                            # 右側：手入力
                            item4_ord_profit = st.number_input(
                                "直接入力", 
                                min_value=-300000, 
                                max_value=1000000, 
                                value=st.session_state.item4_ord_profit, 
                                step=1,
                                key="num_item4_ord_profit",  # keyは一意にする
                                label_visibility="collapsed"
                            )

                        with c_l:
                            # 左側：スライダー
                            item4_ord_profit = st.slider(
                                "売上高調整",
                                min_value=-200000,
                                max_value=1000000,
                                value=item4_ord_profit,
                                step=100,
                                key="slide_item4_ord_profit",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )

                        # 最新の値を保存
                        st.session_state.item4_ord_profit = item4_ord_profit

                        st.divider() # 次の項目との区切
        #-------------------------------------------------------------------------------------------

                        st.markdown("### 当期利益")

                        # 初期値の定義
                        if 'item5_net_income' not in st.session_state:
                            st.session_state.item5_net_income = 10000

                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])

                        with c_r:
                            # 右側：手入力
                            item5_net_income = st.number_input(
                                "直接入力", 
                                min_value=-300000, 
                                max_value=1000000, 
                                value=st.session_state.item5_net_income, 
                                step=1,
                                key="num_item5_net_income",  # keyは一意にする
                                label_visibility="collapsed"
                            )

                        with c_l:
                            # 左側：スライダー
                            item5_net_income = st.slider(
                                "売上高調整",
                                min_value=-200000,
                                max_value=1000000,
                                value=item5_net_income,
                                step=100,
                                key="slide_item5_net_income",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )

                        # 最新の値を保存
                        st.session_state.item5_net_income = item5_net_income

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
    
                        # 初期値の定義
                        if 'item10_dep' not in st.session_state:
                            st.session_state.item10_dep = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item10_dep = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item10_dep, 
                                step=1,
                                key="num_item10_dep",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item10_dep = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=300000,
                                value=item10_dep,
                                step=100,
                                key="slide_item10_dep",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item10_dep = item10_dep
    
                        st.divider() # 次の項目との区切
    
        #--------------------------------------------------------------------------------------------------------
                        #⑦減価償却費（経費）
    
                        st.markdown("### 減価償却費(経費)")
    
                        # 初期値の定義
                        if 'item11_dep_exp' not in st.session_state:
                            st.session_state.item11_dep_exp = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item11_dep_exp = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item11_dep_exp, 
                                step=1,
                                key="num_item11_dep_exp",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item11_dep_exp = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=300000,
                                value=item11_dep_exp,
                                step=100,
                                key="slide_item11_dep_exp",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item11_dep_exp = item11_dep_exp
    
                        st.divider() # 次の項目との区切
    
        #----------------------------------------------------------------------------------------------------
    
                        # #⑧賃借料
                        st.markdown("### 賃借料")
    
                        # 初期値の定義
                        if 'item8_rent' not in st.session_state:
                            st.session_state.item8_rent = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item8_rent = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item8_rent, 
                                step=1,
                                key="num_item8_rent",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item8_rent = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=item8_rent,
                                step=100,
                                key="slide_item8_rent",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item8_rent = item8_rent
    
                        st.divider() # 次の項目との区切
    
        #----------------------------------------------------------------------------------------------
    
                        #⑨賃借料（経費）
                        # h_item12_rent_exp=st.empty()
                        # item12_rent_exp = col3.select_slider("賃借料(経費）", options=range(0, 90000, 100), value=0)
                        # red_label(h_item12_rent_exp, f"賃借料(経費）:{item12_rent_exp:,} 千円")
                        # st.divider()
    
                        st.markdown("### 賃借料（経費）")
    
                        # 初期値の定義
                        if 'item12_rent_exp' not in st.session_state:
                            st.session_state.item12_rent_exp = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item12_rent_exp = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item12_rent_exp, 
                                step=1,
                                key="num_item12_rent_exp",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item12_rent_exp = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=item12_rent_exp,
                                step=100,
                                key="slide_item12_rent_exp",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item12_rent_exp = item12_rent_exp
    
                        st.divider() # 次の項目との区切
    
        #------------------------------------------------------------------------------------------------
    
                        #⑩機械装置
     
                        st.markdown("### 機械装置")
    
                        # 初期値の定義
                        if 'item6_machine' not in st.session_state:
                            st.session_state.item6_machine = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item6_machine = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item6_machine, 
                                step=1,
                                key="num_item6_machine",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item6_machine = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=item6_machine,
                                step=100,
                                key="slide_item6_machine",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item6_machine = item6_machine
    
                        st.divider() # 次の項目との区切
    
        #--------------------------------------------------------------------------------------------
    
                        # #11その他資産
                        # h_item7_other=st.empty()
                        # item7_other = col4.select_slider("その他資産", options=range(0, 50000, 100), value=0)
                        # red_label(h_item7_other, f"その他資産:{ item7_other:,} 千円")
                        # st.divider()
    
                        st.markdown("### その他資産")
    
                        # 初期値の定義
                        if 'item7_other' not in st.session_state:
                            st.session_state.item7_other = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            item7_other = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.item7_other, 
                                step=1,
                                key="num_item7_other",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            item7_other = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=item7_other,
                                step=100,
                                key="slide_item7_other",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.item7_other = item7_other
    
                        st.divider() # 次の項目との区切
        #-------------------------------------------------------------------------------------------------------------
                        # #12純資産合計
    
                        st.markdown("### 純資産")
    
                        # 初期値の定義
                        if 'net_assets' not in st.session_state:
                            st.session_state.net_assets = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            net_assets = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=500000, 
                                value=st.session_state.net_assets, 
                                step=1,
                                key="num_net_assets",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            net_assets = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=net_assets,
                                step=100,
                                key="slide_net_assets",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.net_assets = net_assets
    
                        st.divider() # 次の項目との区切
        #--------------------------------------------------------------------------------
                        #13総資産
                        # h_total_assets=st.empty()
                        # total_assets = col4.select_slider("総資産（千円）", options=range(0, 900000, 1000), value=0)
                        # red_label(h_total_assets, f"総資産:{total_assets:,} 千円")
                        # st.divider()
    
                        st.markdown("### 総資産")
    
                        # 初期値の定義
                        if 'total_assets' not in st.session_state:
                            st.session_state.total_assets = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            total_assets = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=1000000, 
                                value=st.session_state.total_assets, 
                                step=1,
                                key="num_total_assets",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            total_assets = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=300000,
                                value=total_assets,
                                step=100,
                                key="slide_total_assets",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.total_assets = total_assets
    
                        st.divider() # 次の項目との区切
        #------------------------------------------------------------------------------------------------------
                    with st.expander("💳 3. 信用情報", expanded=False):
    
                        # default値をリスト内の文字列と完全に一致させる必要があります
                        grade =st.segmented_control("格付", ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"], default="②4-6 (標準)")
        #---------------------------------------------------------------------------             
                    #     #14銀行与信
    
                        st.markdown("### うちの銀行与信")
                        st.caption("当社の与信です（総銀行与信ではありません）")
                        # 初期値の定義
                        if 'bank_credit' not in st.session_state:
                            st.session_state.bank_credit = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            bank_credit = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=3000000, 
                                value=st.session_state.bank_credit, 
                                step=1,
                                key="num_bank_credit",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            bank_credit = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=500000,
                                value=bank_credit,
                                step=100,
                                key="slide_bank_credit",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.bank_credit = bank_credit
    
                        st.divider() # 次の項目との区切
        #---------------------------------------------------------------------------------------------------------
      
                        # #15リース与信
    
                        st.markdown("### うちのリース与信")
                        st.caption("当社の与信です（総リース与信ではありません）")
                        # 初期値の定義
                        if 'lease_credit' not in st.session_state:
                            st.session_state.lease_credit = 10000
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            lease_credit = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=300000, 
                                value=st.session_state.lease_credit, 
                                step=1,
                                key="num_lease_credit",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            lease_credit = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=200000,
                                value=lease_credit,
                                step=100,
                                key="slide_lease_credit",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.lease_credit = lease_credit
    
                        st.divider() # 次の項目との区切
        #--------------------------------------------------------------------------------------------------------
                        # #16契約数
                        st.markdown("### 契約数")
                        # 初期値の定義
                        if 'contracts' not in st.session_state:
                            st.session_state.contracts = 1
    
                        # 横に分割（左 0.7 : 右 0.3）
                        c_l, c_r = st.columns([0.7, 0.3])
    
                        with c_r:
                            # 右側：手入力
                            contracts = st.number_input(
                                "直接入力", 
                                min_value=0, 
                                max_value=30, 
                                value=st.session_state.contracts, 
                                step=1,
                                key="num_contracts",  # keyは一意にする
                                label_visibility="collapsed"
                            )
    
                        with c_l:
                            # 左側：スライダー
                            contracts = st.slider(
                                "売上高調整",
                                min_value=0,
                                max_value=20,
                                value=contracts,
                                step=1,
                                key="slide_contracts",  # keyは一意にする
                                label_visibility="collapsed",
                                format="%d"
                            )
    
                        # 最新の値を保存
                        st.session_state.contracts = contracts
    
                        st.divider() # 次の項目との区切
    
        #------------------------------------------------------------------------------------------------------
    
    
                    with st.expander("📋 4. 契約条件・取得価格・リース物件", expanded=False):
                        customer_type = st.radio("顧客区分", ["既存先", "新規先"], horizontal=True)
                        st.divider()
                        st.markdown("##### 📈 契約条件・属性 (利回り予測用)")
                        with st.container():
                            c_y1, c_y2, c_y3 = st.columns(3)
                            contract_type = c_y1.radio("契約種類", ["一般", "自動車"], horizontal=True)
                            deal_source = c_y2.radio("商談ソース", ["銀行紹介", "その他"], horizontal=True)
                            lease_term = c_y3.select_slider("契約期間（月）", options=range(0, 121, 1), value=60)
                            st.divider()
                            c_l, c_r = st.columns([0.7, 0.3])
                            with c_l:
                                acceptance_year = st.number_input("検収年 (西暦)", value=2026, step=1)
                            st.session_state.lease_term = lease_term
                            st.session_state.acceptance_year = acceptance_year
                        st.markdown("### 取得価格")
                        if 'acquisition_cost' not in st.session_state:
                            st.session_state.acquisition_cost = 1000
                        c_l, c_r = st.columns([0.7, 0.3])
                        with c_r:
                            acquisition_cost = st.number_input("直接入力", min_value=0, max_value=500000, value=st.session_state.acquisition_cost, step=100, key="num_acquisition_cost", label_visibility="collapsed")
                        with c_l:
                            acquisition_cost = st.slider("取得価格調整", min_value=0, max_value=300000, value=acquisition_cost, step=100, key="slide_acquisition_cost", label_visibility="collapsed", format="%d")
                        st.session_state.acquisition_cost = acquisition_cost
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
                        # ---------- 5. 定性情報: 逆転の鍵 ----------
                        with st.expander("🛡️ 逆転の鍵（定性情報）", expanded=True):
                            st.caption("財務の弱点を補うエビデンスとして、審査・スコアに反映されます。")
                            strength_tags = st.multiselect(
                                "強みタグ",
                                options=STRENGTH_TAG_OPTIONS,
                                default=[],
                                key="strength_tags",
                                help="当てはまるものを複数選択してください。",
                            )
                            passion_text = st.text_area(
                                "熱意・裏事情の自由記述",
                                value="",
                                height=120,
                                placeholder="例: 社長は同業で20年のキャリア。今回の設備は受注拡大のための必須投資で、既存取引行も応援している。",
                                key="passion_text",
                                help="社長の経歴・導入背景・取引行の関係など、審査でプラス材料になる点を記入してください。",
                            )
                    submitted = st.form_submit_button("判定開始", type="primary", use_container_width=True)

            if submitted:
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
    
                    # 指標計算
                    user_op_margin = (rieki / nenshu * 100) if nenshu > 0 else 0.0
                    user_equity_ratio = (net_assets / total_assets * 100) if total_assets > 0 else 0.0
                    # 流動比率の簡易算（流動資産≈総資産−固定資産、流動負債≈負債総額）
                    liability_total = total_assets - net_assets if (total_assets and net_assets is not None) else 0
                    current_assets_approx = max(0, total_assets - item6_machine - item7_other)
                    user_current_ratio = (current_assets_approx / liability_total * 100) if liability_total > 0 else 100.0
            
                    bench = benchmarks_data.get(selected_sub, {})
                    bench_op_margin = bench.get("op_margin", 0.0)
                    bench_equity_ratio = bench.get("equity_ratio", 0.0)
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
                    strength_tags = st.session_state.get("strength_tags", []) or []
                    passion_text = (st.session_state.get("passion_text", "") or "").strip()
                    n_strength = len(strength_tags)
                    contract_prob = score_percent
                    ai_completed_factors = []  # AIが補完した判定要因（表示・バトル用）
    
                    # メイン先（係数: 更新値 or 既定10）
                    main_bank_eff = effective.get("main_bank", 10)
                    if main_bank == "メイン先":
                        contract_prob += main_bank_eff
                        ai_completed_factors.append({"factor": "メイン取引先", "effect_percent": int(round(main_bank_eff)), "detail": "取引行として優位"})
    
                    # 競合: 競合あり=負の係数、競合なし=プラス（更新値 or 既定）
                    comp_present_eff = effective.get("competitor_present", BAYESIAN_PRIOR_EXTRA["competitor_present"])
                    comp_none_eff = effective.get("competitor_none", 15)
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

                    # 借手スコア + 物件スコア → 総合スコア（判定に反映）
                    final_score = 0.85 * score_percent + 0.15 * asset_score
                    st.session_state['current_image'] = "approve" if final_score >= 71 else "challenge"
                
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
                            eq_str = f"{eq:.1f}%" if eq is not None else "—"
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
                    # 逆転の鍵を財務弱点を補うエビデンスとしてAIに明示
                    if strength_tags or passion_text:
                        ai_question_text += "【🛡️ 逆転の鍵（定性エビデンス）】\n"
                        if strength_tags:
                            ai_question_text += "強みタグ: " + "、".join(strength_tags) + "。これらを財務面の弱点を補う材料として本気で評価し、承認確率・アドバイスに反映してください。\n"
                        if passion_text:
                            ai_question_text += "熱意・裏事情: " + passion_text[:800] + ("…" if len(passion_text) > 800 else "") + "\n"
                        ai_question_text += "\n"
                    ai_question_text += "審査お疲れ様です。手元の決算書から、以下の**3点だけ**確認させてください。\n\n"
                    questions = []
                    if my_hints.get("mandatory"): questions.append(f"🏭 **業界確認**: {my_hints['mandatory']}")
                    if score_percent < 70: questions.append("💡 **実質利益**: 販管費の内訳に「役員報酬」は十分計上されていますか？")
                    elif user_op_margin < bench_op_margin: questions.append("📉 **利益率要因**: 今期の利益率低下は、一過性ですか？")
                    if score_percent < 70: questions.append("🏦 **資金繰り**: 借入金明細表で、返済が「約定通り」進んでいるか確認してください。")
                    if my_hints["risks"]: questions.append(f"⚠️ **業界リスク**: {my_hints['risks'][0]} はクリアしていますか？")
                
                    for q in questions[:3]: ai_question_text += f"- {q}\n"
                    ai_question_text += "\nこれらがクリアになれば、承認確率80%以上が見込めます。"
                    ai_question_text += f"\n\n【参考】財務ベースの推定倒産確率: {pd_percent:.1f}%。業界の最新リスク情報も参照済みです。これらを総合して最終的な倒産リスクと承認可否を判断してください。"
    
                    # チャット履歴に追加 (表示は分析タブのチャット欄で行う)
                    st.session_state.messages = [{"role": "assistant", "content": ai_question_text}]
                    st.session_state.debate_history = [] 
    
                    # 議論終了・判定プロンプト用に類似案件ブロックを保持
                    similar_past_for_prompt = (similar_cases_block + instruction_past) if similar_cases_block else ""
    
                    # 定性ワンホット（過去データ・RAG用）
                    qualitative_onehot = {tag: 1 for tag in STRENGTH_TAG_OPTIONS if tag in strength_tags}
                    qualitative_onehot.update({tag: 0 for tag in STRENGTH_TAG_OPTIONS if tag not in strength_tags})

                    st.session_state['last_result'] = {
                        "score": final_score, "hantei": "承認圏内" if final_score >= 71 else "要審議",
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
                
                    # 審査委員会カードバトル用データ（分析タブで表示）
                    hp_card = int(min(999, max(1, net_assets / 1000))) if net_assets else int(min(999, max(1, user_equity_ratio * 5)))
                    atk_card = int(min(99, max(1, user_op_margin * 2)))
                    spd_card = int(min(99, max(1, user_current_ratio / 2)))
                    is_approved = final_score >= 71
                    # 補完要因をスキル・環境効果としてバトルに渡す
                    env_effects = [f"{f['factor']}: {f['effect_percent']:+.0f}%" for f in ai_completed_factors]
                    st.session_state["battle_data"] = {
                        "hp": hp_card, "atk": atk_card, "spd": spd_card,
                        "is_approved": is_approved,
                        "special_move_name": None, "special_effect": None,
                        "battle_log": [], "dice": None,
                        "score": final_score, "hantei": "承認圏内" if is_approved else "要審議",
                        "environment_effects": env_effects,
                        "ai_completed_factors": ai_completed_factors,
                    }
                    st.session_state["show_battle"] = True

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
                        },
                        "result": st.session_state['last_result'],
                        "pricing": {
                            "base_rate": 1.2, 
                            "pred_rate": y_pred_adjusted
                        }
                    }
                    # 案件ログを保存し、案件IDをセッションに保持しておく
                    case_id = save_case_log(log_payload)
                    st.session_state["current_case_id"] = case_id
                    st.session_state.nav_index = 1  # 1番目（分析結果）に切り替える
                    st.rerun()  # 画面を読み込み直して、実際にタブを移動させる
                    
                    # 自動的に「分析結果」タブへ遷移
                    st.success("審査完了！分析結果を表示します。")
                    st.rerun()
                except Exception as e:
                    st.error("判定開始の処理中にエラーが発生しました。入力内容を確認するか、ページを再読み込みして再度お試しください。")
                    import traceback
                    with st.expander("エラー詳細", expanded=False):
                        st.code(traceback.format_exc())

        if nav_mode == "📊 分析結果":
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
                score_percent = res["score"]
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

                # ==================== 審査委員会カードバトル（判定開始直後の演出） ====================
                if st.session_state.get("show_battle") and "battle_data" in st.session_state:
                    bd = st.session_state["battle_data"]
                    # 必殺技・バトルログ・ダイスが未生成なら生成
                    if bd.get("special_move_name") is None:
                        strength_tags = res.get("strength_tags") or []
                        passion_text = res.get("passion_text") or ""
                        name, effect = generate_battle_special_move(strength_tags, passion_text)
                        bd["special_move_name"] = name
                        bd["special_effect"] = effect
                        # バトル実況ログ（慎重派・推進派の議論）
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
                    st.subheader("⚔️ 審査委員会カードバトル")
                    # ステータスカード（HP/ATK/SPD）
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#334155 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                        <div style="font-size:0.85rem;opacity:0.9;">HP</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['hp']}</div>
                        <div style="font-size:0.75rem;">自己資本</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#b45309 0%,#c2410c 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                        <div style="font-size:0.85rem;opacity:0.9;">ATK</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['atk']}</div>
                        <div style="font-size:0.75rem;">利益率</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#0d9488 0%,#0f766e 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,0.15);">
                        <div style="font-size:0.85rem;opacity:0.9;">SPD</div>
                        <div style="font-size:1.8rem;font-weight:bold;">{bd['spd']}</div>
                        <div style="font-size:0.75rem;">流動性</div>
                        </div>
                        """, unsafe_allow_html=True)
                    # 必殺技カード
                    st.markdown("**🎴 必殺技**")
                    st.markdown(f"""
                    <div style="background:#f8fafc;border:2px solid #b45309;border-radius:10px;padding:1rem;margin-bottom:1rem;">
                    <span style="font-weight:bold;color:#1e3a5f;">{bd.get('special_move_name', '逆転の意気')}</span>
                    <span style="color:#64748b;"> … </span>
                    <span>{bd.get('special_effect', 'スコア+5%')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    # 環境効果・スキル（AI補完した判定要因をバトル用に表示）
                    env_effects = bd.get("environment_effects") or []
                    if env_effects:
                        st.markdown("**🌐 環境効果・スキル**")
                        for eff in env_effects:
                            st.caption(f"• {eff}")
                    # バトル実況ログ
                    st.markdown("**📜 バトル実況**")
                    for line in bd.get("battle_log", []):
                        st.caption(line)
                    dice = bd.get("dice") or 1
                    st.caption(f"🎲 運命のダイス: **{dice}** → {'やや有利' if dice >= 4 else 'やや不利'}（審査は数値と定性の総合で判定済み）")
                    st.divider()
                    # リザルト
                    if bd.get("is_approved"):
                        st.markdown("""
                        <div style="background:linear-gradient(135deg,#0d9488 0%,#059669 100%);color:#fff;padding:1.5rem;border-radius:16px;text-align:center;font-size:1.5rem;font-weight:bold;box-shadow:0 8px 24px rgba(0,0,0,0.2);">
                        🏆 WIN — 承認圏内
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                        <div style="background:linear-gradient(135deg,#475569 0%,#334155 100%);color:#fff;padding:1.5rem;border-radius:16px;text-align:center;font-size:1.5rem;font-weight:bold;box-shadow:0 8px 24px rgba(0,0,0,0.2);">
                        📋 LOSE — 要審議
                        </div>
                        """, unsafe_allow_html=True)
                        st.snow()
                    if st.button("📊 ダッシュボードを見る", type="primary", use_container_width=True, key="btn_show_dashboard_after_battle"):
                        st.session_state["show_battle"] = False
                        st.rerun()
                    st.markdown("---")
                    # バトル表示中はここで一旦終了し、ダッシュボードは「ダッシュボードを見る」で表示
                else:
                    # ==================== ダッシュボードレイアウト（プロ仕様） ====================
                    st.markdown("---")
                    # ----- 成約に最も寄与している上位3因子（データ5件以上で表示） -----
                    _driver_analysis = run_contract_driver_analysis()
                    if _driver_analysis and _driver_analysis["closed_count"] >= 5:
                        st.markdown("**🎯 成約に最も寄与している上位3つの因子（ドライバー）**")
                        d1, d2, d3 = st.columns(3)
                        for idx, col in enumerate([d1, d2, d3]):
                            if idx < len(_driver_analysis["top3_drivers"]):
                                d = _driver_analysis["top3_drivers"][idx]
                                with col:
                                    st.markdown(f"""
                                    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#334155 100%);color:#fff;padding:0.8rem;border-radius:10px;font-size:0.9rem;">
                                    <div style="opacity:0.9;">{idx+1}位</div>
                                    <div style="font-weight:bold;">{d['label']}</div>
                                    <div style="font-size:0.8rem;">係数 {d['coef']:.3f}（{d['direction']}）</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
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
                    # ----- 主要KPI（画面最上部・横並び）業界実績 + 本件 -----
                    past_stats = get_stats(selected_sub)
                    with st.container():
                        st.markdown("**主要KPI**")
                        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                        with kpi1:
                            st.metric("業界 成約率", f"{past_stats.get('close_rate', 0) * 100:.1f}%" if past_stats.get("count") else "—", help="同業種の成約率")
                        with kpi2:
                            st.metric("業界 成約件数", f"{past_stats.get('closed_count', 0)}件" if past_stats.get("count") else "—", help="同業種の成約件数")
                        with kpi3:
                            avg_r = past_stats.get("avg_winning_rate")
                            st.metric("業界 平均金利", f"{avg_r:.2f}%" if avg_r is not None and avg_r > 0 else "—", help="同業種の平均成約金利")
                        with kpi4:
                            st.metric("本件 スコア", f"{res['score']:.1f}%", help="総合承認スコア")
                        with kpi5:
                            st.metric("本件 判定", res.get("hantei", "—"), help="承認圏内 or 要審議")
                        with kpi6:
                            st.metric("本件 契約期待度", f"{res.get('contract_prob', 0):.1f}%", help="定性補正後")
                        # streamlit-extras: ページ内の全 st.metric をカード風に（ネイビー・ゴールドの左アクセント）
                        if style_metric_cards is not None:
                            style_metric_cards(
                                background_color=CHART_STYLE["bg"],
                                border_size_px=1,
                                border_color=CHART_STYLE["grid"],
                                border_radius_px=8,
                                border_left_color=CHART_STYLE["primary"],
                                box_shadow=True,
                            )
                        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

                    # ----- AIが補完した判定要因（進化するダッシュボード） -----
                    ai_factors = res.get("ai_completed_factors") or []
                    if ai_factors:
                        with st.expander("🤖 AIが補完した判定要因", expanded=True):
                            st.caption("あなたの設定した財務指標に加え、以下の要因を成約率（契約期待度）に反映しました。")
                            for f in ai_factors:
                                sign = "+" if f.get("effect_percent", 0) >= 0 else ""
                                st.markdown(f"- **{f.get('factor', '')}** … {sign}{f.get('effect_percent', 0)}% （{f.get('detail', '')}）")

                    st.divider()
                    # ----- カード: 本件スコア内訳・倒産確率・利回り -----
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

                    with st.container():
                        st.markdown("**本件スコア・倒産確率・利回り**")
                        k1, k2, k3, k4, k5 = st.columns(5)
                        with k1:
                            st.metric("総合スコア", f"{res['score']:.1f}%", help="借手＋物件を反映した判定用スコア")
                        with k2:
                            st.metric("判定", res.get("hantei", "—"), help="承認圏内 or 要審議")
                        with k3:
                            st.metric("推定倒産確率", f"{pd_val:.1f}%", help="財務指標ベースの簡易リスク")
                        with k4:
                            st.metric("契約期待度", f"{res.get('contract_prob', 0):.1f}%", help="定性補正後の期待度")
                        with k5:
                            if "yield_pred" in res:
                                st.metric("予測利回り", f"{res['yield_pred']:.2f}%", delta=f"{res.get('rate_diff', 0):+.2f}%", help="AI予測利回り")
                            else:
                                st.metric("予測利回り", "—", help="利回りモデル未適用")
                        # ----- 第2行: スコア内訳（借手・物件説明 + 3モデル） -----
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

                    st.divider()
                    with st.container():
                        st.markdown("**スコアゲージ・契約期待度・判定**")
                        # ----- 第3行: ゲージ・契約期待度・判定・業界比較（ダッシュボード内に統合） -----
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            st.pyplot(plot_gauge(res['score'], "総合スコア"))
                        with g2:
                            st.metric("契約期待度", f"{res['contract_prob']:.1f}%")
                            if "yield_pred" in res:
                                st.metric("予測利回り", f"{res['yield_pred']:.2f}%", delta=f"{res.get('rate_diff', 0):+.2f}%")
                        with g3:
                            st.success(f"**{res['hantei']}**")
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
                                st.caption(f"規模: {sales_msg} / 収益: {prof_msg}")

                    st.divider()
                    with st.container():
                        st.subheader(":round_pushpin: 3D多角分析（回転・拡大可能）")
                        current_case_data = {
                             'sales': res['financials']['nenshu'],
                             'op_margin': res['user_op'],
                             'equity_ratio': res['user_eq']
                        }
                        past_cases_log = load_all_cases()
                        fig_3d = plot_3d_analysis(current_case_data, past_cases_log)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True, key="plotly_3d_analysis_result")
                            st.caption("指でなぞると回転、ピンチで拡大できます。")
                        else:
                            st.warning("表示データがありません")

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
                                # -----------------------------------------------------
                                similar_success_cases = [
                                    c for c in cases 
                                    if c.get("industry_major") == selected_major
                                    and abs(c.get("result", {}).get("score", 0) - score_percent) < 15
                                    and c.get("result", {}).get("score", 0) >= 70
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
                            st.pyplot(plot_radar_chart(radar_metrics, radar_bench))
                        with g2:
                            # 損益分岐点グラフ
                            sales_k = res["financials"]["nenshu"]
                            gross_k = res["financials"]["gross_profit"] * 1000
                            op_k = res["financials"]["rieki"] * 1000
                            vc = sales_k - gross_k
                            fc = gross_k - op_k
                            st.pyplot(plot_break_even_point(sales_k, vc, fc))

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
                        # 業界目安より良い＝緑、悪い＝赤（_LOWER_IS_BETTER_NAMES は低い方が良い）
                        rows_html = []
                        for ind in indicators:
                            name = ind["name"]
                            value = ind["value"]
                            unit = ind.get("unit", "%")
                            bench = ind.get("bench")
                            bench_ok = bench is not None and (not isinstance(bench, float) or bench == bench)
                            if bench_ok:
                                diff = value - bench
                                is_good = (diff > 0 and name not in _LOWER_IS_BETTER_NAMES) or (diff < 0 and name in _LOWER_IS_BETTER_NAMES)
                                color = "#22c55e" if is_good else "#ef4444"
                                name_cell = f'<span style="color:{color}; font-weight:600;">{name.replace("&", "&amp;").replace("<", "&lt;")}</span>'
                            else:
                                name_cell = name.replace("&", "&amp;").replace("<", "&lt;")
                            bench_str = f"{bench:.1f}{unit}" if bench_ok else "—"
                            rows_html.append(f"<tr><td>{name_cell}</td><td>{value:.1f}{unit}</td><td>{bench_str}</td></tr>")
                        table_html = "<table style='width:100%; max-width:100%; border-collapse:collapse; font-size:0.9rem; table-layout:auto;'><thead><tr><th style='text-align:left; padding:6px 10px;'>指標</th><th style='text-align:right; padding:6px 10px;'>貴社</th><th style='text-align:right; padding:6px 10px;'>業界目安</th></tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
                        # PC・スマホどちらでも全部表示されるようコンテナ幅100%（横スクロールのみ必要時）
                        st.markdown(
                            f"<div style='width:100%; overflow-x:auto; margin:0.5rem 0;'>{table_html}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption("緑＝業界目安より良い、赤＝業界目安より要確認")
                        # 指標と業界目安の差の分析（図＋文章＋AIによる指標の分析）
                        summary, detail = analyze_indicators_vs_bench(indicators)
                        st.markdown("#### 📊 指標と業界目安の差の分析")
                        st.info(summary)
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
                        fig_gap = plot_indicators_gap_analysis(indicators)
                        if fig_gap:
                            col_gap, _ = st.columns([0.65, 0.35])
                            with col_gap:
                                st.pyplot(fig_gap)
                            st.caption("左が「業界より要確認」、右が「業界より良い」です。借入金等依存度・減価償却費/売上高は、業界より低いと緑になります。")
                        with st.expander("差の内訳（数値）", expanded=False):
                            st.markdown(detail)
                        # 利益構造（ウォーターフォール）
                        nenshu_k = fin.get("nenshu") or 0
                        gross_k = fin.get("gross_profit") or 0
                        op_k = fin.get("rieki") or fin.get("op_profit") or 0
                        ord_k = fin.get("ord_profit") or 0
                        net_k = fin.get("net_income") or 0
                        if nenshu_k > 0:
                            st.markdown("#### 利益構造（損益の流れ）")
                            col_wf, _ = st.columns([0.65, 0.35])
                            with col_wf:
                                st.pyplot(plot_waterfall(nenshu_k, gross_k, op_k, ord_k, net_k))
                    else:
                        st.caption("指標を算出するには、審査入力で売上高・損益・資産などを入力してください。")

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
                                text = get_ai_byoki_with_industry(selected_sub_res, u_eq, u_op, comp_text, net_risk)
                                if text:
                                    st.session_state["ai_byoki_text"] = text
                                    st.session_state["ai_byoki_case_id"] = current_case_id
                                    st.rerun()
                                else:
                                    st.error("生成できませんでした。APIキー・Ollamaを確認してください。")
                        if not byoki_text:
                            st.caption("上のボタンで、ネット検索した業界情報をもとにAIが愚痴を1つ生成します。")

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
                            _res = st.session_state.get("last_result") or {}
                            comparison_text = _res.get("comparison", "（審査未実行のためデータなし）")
                            trend_info = "（審査未実行のためデータなし）"
                            if jsic_data and _res.get("industry_major") in (jsic_data or {}):
                                trend_info = (jsic_data[_res["industry_major"]].get("sub") or {}).get(_res.get("industry_sub", ""), trend_info)
                            hints_context = ""
                            if 'last_result' in st.session_state:
                                h = st.session_state['last_result'].get('hints', {})
                                if h.get('subsidies'): hints_context += f"\n補助金候補: {', '.join(h['subsidies'])}"
                                if h.get('risks'): hints_context += f"\nリスク確認点: {', '.join(h['risks'])}"
                            advice_extras = ""
                            if "last_result" in st.session_state:
                                res_adv = st.session_state["last_result"]
                                advice_extras = get_advice_context_extras(res_adv.get("industry_sub", ""), res_adv.get("industry_major", ""))
                            news_context = ""
                            if 'selected_news_content' in st.session_state:
                                news = st.session_state.selected_news_content
                                news_context = f"\n\n【読み込み済みニュース（必ず内容に触れること）】\nタイトル: {news['title']}\n本文:\n{news['content']}"
                            hints_block = ("■ 補助金・リスクヒント: " + hints_context) if hints_context else ""
                            advice_block = ("■ 補助金スケジュール・リース判定・耐用年数・業界拡充等:\n" + advice_extras) if advice_extras else ""
                            ind_summary, ind_detail, ind_list = get_indicator_analysis_for_advice(_res)
                            indicator_block = ""
                            if ind_summary or ind_list:
                                indicator_block = "\n■ 指標の分析（貴社 vs 業界目安）\n"
                                if ind_summary:
                                    indicator_block += f"要約: {ind_summary}\n\n"
                                if ind_list:
                                    indicator_block += "指標一覧:\n" + ind_list + "\n\n"
                                if ind_detail:
                                    indicator_block += "差の内訳:\n" + ind_detail[:1500] + "\n"
                            # 過去の相談メモ（話せば話すほど蓄積）を読み込み、プロンプトに含める
                            memory_entries = load_consultation_memory(max_entries=15)
                            memory_block = ""
                            if memory_entries:
                                parts = []
                                for e in memory_entries:
                                    u = (e.get("user") or "").strip()
                                    a = (e.get("assistant") or "").strip()
                                    if u or a:
                                        parts.append(f"ユーザー: {u[:800]}\nAI: {a[:1200]}")
                                if parts:
                                    memory_block = "\n\n【過去の相談で話したこと（話せば話すほど蓄積・参照して続きで答える）】\n" + "\n---\n".join(parts[-15:]) + "\n"
                            context_prompt = f"""あなたは経験豊富なリース審査のプロ。以下の「参考データ」を必ず使って、具体的に答えてください。数字やニュースの内容を引用すると説得力が増します。

【参考データ】
■ 財務・比較: {comparison_text}
■ 業界トレンド: {trend_info}
{hints_block}
{advice_block}
{indicator_block}
{news_context}
{memory_block}

【ルール】
- 上記のデータに触れずに一般論だけで答えないこと。
- ニュースがある場合はその内容や業界動向を踏まえた助言をすること。
- 指標の分析がある場合、業界目安を下回っている指標については「なぜ下回っている可能性があるか」「どう改善するとよいか」を簡潔にアドバイスすること。改善のための具体的なアクション（数値目標・確認すべき書類・交渉のポイント等）があれば述べること。
- 過去の相談メモがある場合は、その流れを踏まえて「続き」として一貫した助言をすること。
- 2〜5文で簡潔に、しかし具体的に。

【相談内容】
{q}"""
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
                        
                        # ロール決定 & プロンプト作成（同一モデルでペルソナ切り替え）
                        if not st.session_state.debate_history:
                            next_role = "Pro"
                            prompt = f"""{PERSONA_PRO}

【財務データ】（必ず引用すること）
業種: {selected_sub}
スコア: {score:.1f}点 (承認ライン70点)
財務評価: {comparison_text}

【ネット検索結果・業界材料】
{advice_debate_block}
{news_context if news_context else "（ニュース未読み込み）"}

【指示】
- 上記の「財務データ」と「ネット検索結果」のいずれかから必ず1つ以上具体的に引用し、根拠を示したうえで主張すること。
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
【財務ベース倒産確率】{pd_str}（自己資本比率・流動比率・利益率から算出）

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

    with menu_tabs[1]:  # 情報検索
        st.subheader("🔍 情報検索")
        info_cat = st.radio("カテゴリ", ["業種情報", "補助金", "リース情報"], horizontal=True, key="info_search_cat", label_visibility="collapsed")
        if info_cat == "業種情報":
            st.markdown("**業種別の業界目安・トレンド**")
            sub_keys = sorted(benchmarks_data.keys()) if benchmarks_data else []
            if sub_keys:
                search_sub = st.selectbox("業種を選択", sub_keys, key="info_industry_sub")
                bench = benchmarks_data.get(search_sub, {})
                if bench:
                    st.caption("営業利益率・自己資本比率・売上高総利益率・ROA・流動比率などの目安（業界平均）")
                    for k, v in list(bench.items())[:10]:
                        if v is not None and isinstance(v, (int, float)): st.write(f"- {k}: {v}")
                trend_ext = get_trend_extended(search_sub)
                if trend_ext:
                    with st.expander("ネットで取得したトレンド・拡充情報", expanded=False):
                        st.text(trend_ext[:2000])
            else:
                st.caption("業種データがありません。")
        elif info_cat == "補助金":
            st.markdown("**業種別 補助金**")
            sub_keys = sorted(benchmarks_data.keys()) if benchmarks_data else []
            if sub_keys:
                search_sub = st.selectbox("業種を選択", sub_keys, key="info_subsidy_sub")
                subs_list = search_subsidies_by_industry(search_sub)
                if subs_list:
                    for s in subs_list:
                        name, url = s.get("name", ""), (s.get("url") or "").strip()
                        st.markdown(f"**{name}**")
                        if url:
                            try: st.link_button("🔗 公式サイト", url, type="secondary")
                            except Exception: st.markdown(f'<a href="{url}" target="_blank">🔗 公式サイト</a>', unsafe_allow_html=True)
                        st.caption((s.get("summary") or "")[:120] + "…")
                else:
                    st.caption("該当する補助金の登録がありません。")
            else:
                st.caption("業種データがありません。")
        else:
            st.markdown("**リース情報**")
            with st.expander("耐用年数を設備で調べる", expanded=False):
                nta_url = (useful_life_data or {}).get("nta_useful_life_url") or "https://www.keisan.nta.go.jp/r5yokuaru/aoiroshinkoku/hitsuyokeihi/genkashokyakuhi/taiyonensuhyo.html"
                st.link_button("📋 国税庁の耐用年数表", nta_url, type="secondary")
                eq_key = st.text_input("設備名で検索", placeholder="例: 工作機械", key="info_equip")
                if eq_key:
                    for e in (search_equipment_by_keyword(eq_key) or []):
                        st.write(f"**{e.get('name')}** … {e.get('years')}年")
            with st.expander("リース判定フロー・契約形態", expanded=False):
                st.markdown(get_lease_classification_text() or "lease_classification.json を読み込んでください。")
            with st.expander("リース物件リスト", expanded=False):
                if LEASE_ASSETS_LIST:
                    for it in LEASE_ASSETS_LIST:
                        st.caption(f"**{it.get('name','')}** {it.get('score',0)}点 — {it.get('note','')}")
                else:
                    st.caption("lease_assets.json を配置してください。")

    with menu_tabs[2]:  # グラフ
        st.subheader("📈 グラフ")
        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            current_case_data = {"sales": res["financials"]["nenshu"], "op_margin": res["user_op"], "equity_ratio": res["user_eq"]}
            fig_3d = plot_3d_analysis(current_case_data, load_all_cases())
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True, key="plotly_3d_graph_tab")
                st.caption("指でなぞると回転・ピンチで拡大できます。")
            st.divider()
            fin = res.get("financials", {})
            if fin.get("nenshu", 0) > 0:
                col_wf2, _ = st.columns([0.65, 0.35])
                with col_wf2:
                    st.pyplot(plot_waterfall(fin.get("nenshu", 0), fin.get("gross_profit", 0), fin.get("op_profit", 0), fin.get("ord_profit", 0), fin.get("net_income", 0)))
        else:
            st.info("👈 「新規審査」でデータを入力し、判定開始するとグラフが表示されます。")

    with menu_tabs[3]:  # 履歴分析
        st.subheader("📋 履歴分析")
        all_cases = load_all_cases()
        if not all_cases:
            st.warning("登録された案件がありません。")
        else:
            pending = [c for c in all_cases if c.get("final_status") == "未登録"]
            if not pending:
                st.success("全ての案件が登録済みです。")
            for i, case in enumerate(reversed(pending[-5:])):
                with st.expander(f"{case.get('timestamp', '')[:16]} - {case.get('industry_sub')} (スコア: {case.get('result', {}).get('score', 0):.0f})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**判定**:", case.get("result", {}).get("hantei", ""))
                        st.caption((case.get("chat_summary", "")[:100] + "...") if case.get("chat_summary") else "サマリなし")
                    with c2:
                        with st.form(f"hist_status_{i}"):
                            res_status = st.radio("結果", ["成約", "失注"], horizontal=True)
                            final_rate = st.number_input("獲得レート (%)", value=0.0, step=0.01, format="%.2f")
                            lost_reason = st.text_input("失注理由", placeholder="例: 金利で他社に")
                            loan_condition_options = ["金融機関と協調", "本件限度", "次回格付まで本件限度", "その他"]
                            loan_conditions_hist = st.multiselect("融資条件", loan_condition_options, key=f"hist_loan_{i}")
                            competitor_name_hist = st.text_input("競合他社情報", placeholder="例: 〇〇銀行、〇〇リース", key=f"hist_comp_{i}")
                            competitor_rate_hist = st.number_input("他社提示金利 (%)", value=0.0, step=0.01, format="%.2f", key=f"hist_rate_{i}")
                            if st.form_submit_button("登録"):
                                for c in all_cases:
                                    if c.get("id") == case.get("id"):
                                        c["final_status"] = res_status
                                        c["final_rate"] = final_rate
                                        if res_status == "失注":
                                            c["lost_reason"] = lost_reason
                                        c["loan_conditions"] = loan_conditions_hist
                                        c["competitor_name"] = competitor_name_hist.strip() if competitor_name_hist else ""
                                        c["competitor_rate"] = competitor_rate_hist if competitor_rate_hist else None
                                        break
                                save_all_cases(all_cases)
                                st.success("登録しました")
                                st.rerun()
        with st.expander("🔧 係数分析・更新 (β)", expanded=False):
            st.caption("蓄積データで新しい審査モデル（係数）をシミュレーションします。")
            all_logs = load_all_cases()
            if not all_logs or len([x for x in all_logs if x.get("final_status") in ["成約", "失注"]]) < 5:
                st.warning("成約/失注が5件以上登録されると分析できます。")
            else:
                st.info("サイドバーで「係数分析・更新 (β)」モードに切り替えると回帰分析を実行できます。")

    with menu_tabs[4]:  # 設定
        st.subheader("⚙️ 設定")
        st.radio("AIエンジン", ["Ollama（ローカル）", "Gemini API（Google）"], key="settings_engine_display", index=0 if st.session_state.get("ai_engine") == "ollama" else 1, disabled=True)
        st.caption("AIモデル設定は左サイドバー「🤖 AIモデル設定」で変更できます。")
        st.divider()
        st.markdown("**キャッシュ**")
        if st.button("🗑️ キャッシュをクリア", key="settings_clear_cache"):
            st.cache_data.clear()
            st.success("キャッシュをクリアしました")
            st.rerun()
