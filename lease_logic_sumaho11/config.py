# -*- coding: utf-8 -*-
"""
設定・定数・パス。lease_logic_sumaho10 用。
データファイルは sumaho8/9 と共通にするため BASE_DIR はリポジトリルート（親ディレクトリ）を指す。
学習モデル（業種別ハイブリッド）は scoring/models/ または環境変数で指定。
"""
import os

# このパッケージのディレクトリ
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# データファイルはリポジトリルートに置く（sumaho8/9 と共通）
BASE_DIR = os.path.dirname(_PKG_DIR)

# 学習モデル（スコアリング）のパス。未設定なら scoring/models/industry_specific を参照
SCORING_MODELS_DIR = os.environ.get("LEASE_SCORING_MODELS_DIR", os.path.join(_PKG_DIR, "scoring", "models", "industry_specific"))

# AI エンジン
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "lease-anna")
GEMINI_API_KEY_ENV = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_DEFAULT = "gemini-2.0-flash"

# ファイルパス
CASES_FILE = os.path.join(BASE_DIR, "past_cases.jsonl")
COEFF_OVERRIDES_FILE = os.path.join(BASE_DIR, "data", "coeff_overrides.json")
DEBATE_FILE = os.path.join(BASE_DIR, "debate_logs.jsonl")
CONSULTATION_MEMORY_FILE = os.path.join(BASE_DIR, "consultation_memory.jsonl")
CASE_NEWS_FILE = os.path.join(BASE_DIR, "case_news.jsonl")
WEB_BENCHMARKS_FILE = os.path.join(BASE_DIR, "web_industry_benchmarks.json")
TRENDS_EXTENDED_FILE = os.path.join(BASE_DIR, "industry_trends_extended.json")
ASSETS_BENCHMARKS_FILE = os.path.join(BASE_DIR, "industry_assets_benchmarks.json")
SALES_BAND_FILE = os.path.join(BASE_DIR, "sales_band_benchmarks.json")
DASHBOARD_IMAGES_DIR = os.path.join(BASE_DIR, "dashboard_images")
DASHBOARD_IMAGES_ASSETS = os.environ.get("DASHBOARD_IMAGES_ASSETS", "").strip()
BYOKI_JSON = os.path.join(BASE_DIR, "byoki_list.json")

# グラフスタイル
CHART_STYLE = {
    "primary": "#1e3a5f",
    "secondary": "#475569",
    "good": "#0d9488",
    "warning": "#b45309",
    "danger": "#b91c1c",
    "accent": "#b45309",
    "bg": "#f8fafc",
    "grid": "#e2e8f0",
    "text": "#334155",
    "text_light": "#64748b",
}

# 定例の愚痴デフォルト
TEIREI_BYOKI_DEFAULT = [
    "こんな数字で通そうなんて、正気ですか…？ こっちは毎日1万件近く見てるんですけど。",
    "自己資本比率がこの水準でリース審査に来る度胸、ちょっと見習いたいです。本当に。",
    "赤字で「審査お願いします」って、私の目が死んでるの気づいてます？ 気づいてて言ってます？",
    "数値見た瞬間、心が折れかけた。…いや、折れた。折れてる。",
    "業界平均の話、聞いたことあります？ ないですよね。あったらこの数字じゃないですよね。",
    "今日も書類と数字の海で泳いでます。溺れそうです。",
    "リース審査、楽だって思ってる人いませんよね。いませんよね…？",
]

STRENGTH_TAG_OPTIONS = [
    "技術力", "業界人脈", "特許", "立地", "後継者あり",
    "関係者資産あり", "取引行と付き合い長い", "既存返済懸念ない",
]
