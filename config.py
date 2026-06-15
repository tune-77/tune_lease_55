# -*- coding: utf-8 -*-
"""
設定・定数・パス。lease_logic_sumaho10 用。
データファイルは sumaho8/9 と共通にするため BASE_DIR はリポジトリルート（親ディレクトリ）を指す。
学習モデル（業種別ハイブリッド）は scoring/models/ または環境変数で指定。
"""
import os

# このパッケージのディレクトリ
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# サブパッケージ (tune_lease_55/) として動かす場合は親を BASE_DIR とし、
# フラット展開（worktree 等）では _PKG_DIR 自身を使う
BASE_DIR = os.path.dirname(_PKG_DIR) if os.path.basename(_PKG_DIR) == "tune_lease_55" else _PKG_DIR
_DATA_DIR = os.path.join(BASE_DIR, "data")

# 学習モデル（スコアリング）のパス。未設定なら scoring/models/industry_specific を参照
SCORING_MODELS_DIR = os.environ.get("LEASE_SCORING_MODELS_DIR", os.path.join(_PKG_DIR, "scoring", "models", "industry_specific"))

# AI エンジン
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "lease-anna")
GEMINI_API_KEY_ENV = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"

# ファイルパス
CASES_FILE = os.path.join(BASE_DIR, "past_cases.jsonl")
COEFF_OVERRIDES_FILE = os.path.join(BASE_DIR, "data", "coeff_overrides.json")
DEBATE_FILE = os.path.join(_DATA_DIR, "debate_logs.jsonl")
CONSULTATION_MEMORY_FILE = os.path.join(_DATA_DIR, "consultation_memory.jsonl")
CASE_NEWS_FILE = os.path.join(_DATA_DIR, "case_news.jsonl")
WEB_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "web_industry_benchmarks.json")
TRENDS_EXTENDED_FILE = os.path.join(_DATA_DIR, "industry_trends_extended.json")
ASSETS_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "industry_assets_benchmarks.json")
SALES_BAND_FILE = os.path.join(_DATA_DIR, "sales_band_benchmarks.json")
DASHBOARD_IMAGES_DIR = os.path.join(BASE_DIR, "dashboard_images")
DASHBOARD_IMAGES_ASSETS = os.environ.get("DASHBOARD_IMAGES_ASSETS", "").strip()
BYOKI_JSON = os.path.join(_DATA_DIR, "byoki_list.json")

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

# LightGBM ハイパーパラメータ（analysis_regression.py の比較実験で使用）
# チューニング時はここだけ変更すれば比較用LGBMに反映される
LGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "num_leaves": 4,
    "min_child_samples": 10,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": -1,
}

# 本体モデルとして使う RandomForest の既定パラメータ
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "min_samples_split": 4,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}
