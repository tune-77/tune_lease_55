import os
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "lease_logic_sumaho12" else _SCRIPT_DIR

# ==============================================================================
# UI スタイル定義
# ==============================================================================
MAIN_CSS = """
<style>
    /* 全体のフォント・背景調整 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif !important;
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* 余白を詰める */
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
"""

# ==============================================================================
# 定数定義
# ==============================================================================

# 必須項目（未入力・不正時は判定開始をブロック）
REQUIRED_FIELDS = [
    ("nenshu", "売上高", lambda v: v is not None and (v or 0) > 0),
    ("total_assets", "総資産", lambda v: v is not None and (v or 0) > 0),
]

# 推奨項目（0のとき警告を表示するが判定は続行）
RECOMMENDED_FIELDS = [
    ("rieki",      "営業利益", "営業利益率が 0% として計算されます（業界比較・スコアへの影響あり）"),
    ("net_assets", "純資産",   "自己資本比率が 0% となり、学習モデル精度が低下します"),
]

# === 判定用定数 ===
APPROVAL_LINE = 71                      # 社内承認ライン（過去実績ベース）
REVIEW_LINE = 40                        # これ未満は即否決圏
SCORE_PENALTY_IF_LEARNING_REJECT = 0.5  # AIモデル否決時の乗算ペナルティ
ALERT_BORDERLINE_MIN = 68               # 承認ライン直下の要確認ゾーン下限
CAPITAL_DEFICIENCY_PENALTY_DEFAULT = -5.0  # 債務超過時の減点

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

# 定性「逆転の鍵」強みタグ（ワンホット・RAG用）
STRENGTH_TAG_OPTIONS = [
    "技術力", "業界人脈", "特許", "立地", "後継者あり",
    "関係者資産あり", "取引行と付き合い長い", "既存返済懸念ない",
]

# 定性スコアリング訂正（PDF「qualitative scoring」に準拠・審査入力の訂正欄で使用）
QUALITATIVE_SCORING_CORRECTION_ITEMS = [
    {
        "id": "company_history",
        "label": "設立・経営年数",
        "weight": 10,
        "options": [(4, "20年以上"), (3, "10年〜20年"), (2, "5年〜10年"), (1, "3年〜5年"), (0, "3年未満")],
    },
    {
        "id": "customer_stability",
        "label": "顧客安定性",
        "weight": 20,
        "options": [(4, "非常に安定（大口・長期）"), (3, "安定（分散良好）"), (2, "普通"), (1, "やや不安定（集中あり）"), (0, "不安定・依存大")],
    },
    {
        "id": "repayment_history",
        "label": "返済履歴",
        "weight": 25,
        "options": [(4, "5年以上問題なし"), (3, "3年以上問題なし"), (2, "遅延少ない"), (1, "遅延・リスケあり"), (0, "問題あり・要確認")],
    },
    {
        "id": "business_future",
        "label": "事業将来性",
        "weight": 15,
        "options": [(4, "有望（成長・ニーズ確実）"), (3, "やや有望"), (2, "普通"), (1, "やや懸念"), (0, "懸念（縮小・競争激化）")],
    },
    {
        "id": "equipment_purpose",
        "label": "設備目的",
        "weight": 15,
        "options": [(4, "収益直結・受注必須"), (3, "生産性向上・省力化"), (2, "更新・維持・法定対応"), (1, "やや不明確"), (0, "不明確・要説明")],
    },
    {
        "id": "main_bank",
        "label": "メイン取引銀行",
        "weight": 15,
        "options": [(4, "メイン先で取引良好・支援表明"), (3, "メイン先"), (2, "サブ扱い・取引あり"), (1, "取引浅い・他社メイン"), (0, "取引なし・不安")],
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

# キャッシュファイルへのパス系（data/ サブディレクトリに移動済み）
_DATA_DIR = os.path.join(BASE_DIR, "data")
DEBATE_FILE = os.path.join(_DATA_DIR, "debate_logs.jsonl") # ディベートログ
WEB_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "web_industry_benchmarks.json")
TRENDS_EXTENDED_FILE = os.path.join(_DATA_DIR, "industry_trends_extended.json")
ASSETS_BENCHMARKS_FILE = os.path.join(_DATA_DIR, "industry_assets_benchmarks.json")
SALES_BAND_FILE = os.path.join(_DATA_DIR, "sales_band_benchmarks.json")

# 分析ダッシュボード用画像フォルダ
DASHBOARD_IMAGES_DIR = os.path.join(BASE_DIR, "dashboard_images")
DASHBOARD_IMAGES_ASSETS = os.environ.get("DASHBOARD_IMAGES_ASSETS", "").strip()

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

# ==============================================================================
# Helper / 共通パス解決系
# ==============================================================================
def _dashboard_image_base_dirs():
    if DASHBOARD_IMAGES_ASSETS and os.path.isdir(DASHBOARD_IMAGES_ASSETS):
        yield DASHBOARD_IMAGES_ASSETS.rstrip(os.sep)
    if os.path.isdir(DASHBOARD_IMAGES_DIR):
        yield DASHBOARD_IMAGES_DIR
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
    """
    is_approved = (hantei or "").strip() == "承認圏内"

    def pick_fname(base_dir):
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
        # 短い名前向け
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
