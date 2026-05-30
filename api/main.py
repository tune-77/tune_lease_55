import os as _os_early
# macOSでのOpenMPスレッド上限エラー・MPS GPU競合によるSIGSEGV防止
_os_early.environ.setdefault("OMP_NUM_THREADS", "1")
_os_early.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os_early.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os_early.environ.setdefault("MKL_NUM_THREADS", "1")
_os_early.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os_early.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
del _os_early


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, StreamingResponse
import json
import re
import sys
import os
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加して、既存モジュール(scoring_core)をインポート可能にする
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# sys.path に同名モジュール(data_cases等)を持つ別ディレクトリが先に入っている場合があるため、
# _REPO_ROOT を最優先(position 0)に固定する
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# data_cases / base_rate_master を正しいパスから強制ロード（clawd/ 直下の同名モジュールより優先）
import importlib.util as _ilu
_dc_spec = _ilu.spec_from_file_location("data_cases", os.path.join(_REPO_ROOT, "data_cases.py"))
_dc_mod = _ilu.module_from_spec(_dc_spec)
sys.modules["data_cases"] = _dc_mod
_dc_spec.loader.exec_module(_dc_mod)

_brm_spec = _ilu.spec_from_file_location("base_rate_master", os.path.join(_REPO_ROOT, "base_rate_master.py"))
_brm_mod = _ilu.module_from_spec(_brm_spec)
sys.modules["base_rate_master"] = _brm_mod
_brm_spec.loader.exec_module(_brm_mod)

def _load_timesfm_engine():
    """timesfm_engine を遅延ロード（初回呼び出し時のみ）。PyTorchのMPS初期化をstartupから除外する。"""
    if "timesfm_engine" not in sys.modules:
        _tfm_spec = _ilu.spec_from_file_location("timesfm_engine", os.path.join(_REPO_ROOT, "timesfm_engine.py"))
        _tfm_mod = _ilu.module_from_spec(_tfm_spec)
        sys.modules["timesfm_engine"] = _tfm_mod
        _tfm_spec.loader.exec_module(_tfm_mod)
    return sys.modules["timesfm_engine"]

# ── .streamlit/secrets.toml から APIキー等を環境変数に自動注入 ─────────────────
def _load_secrets_to_env():
    """secrets.toml が存在すれば環境変数に一括インポート（既存の環境変数は上書きしない）。"""
    import re
    secrets_path = os.path.join(_REPO_ROOT, ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*"(.+)"$', line)
                if m:
                    key, val = m.group(1), m.group(2)
                    if not os.environ.get(key):  # 既存の環境変数は上書きしない
                        os.environ[key] = val
        print(f"[API] secrets.toml loaded from {secrets_path}")
    except Exception as e:
        print(f"[API] secrets.toml load warning: {e}")

_load_secrets_to_env()

from scoring_core import run_quick_scoring
from api.scoring_full import run_full_scoring_api
from api.gunshi_gemini import stream_gunshi_gemini
from lease_news_digest import (
    get_latest_lease_news_focus,
    record_lease_news_judgment_change,
    record_lease_news_view,
)
from api.schemas import (
    ScoringRequest,
    ScoringResponse,
    CaseRegisterRequest,
    DealClosureRequest,
    DealClosureResponse,
)
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
from scoring.deal_closure_engine import build_features, build_features_from_deltas, compute_closure_likelihood

# Obsidian Vault パス（環境変数優先、未設定時は find_vault() で自動検索）
_OBSIDIAN_VAULT_PATH: str = os.environ.get("OBSIDIAN_VAULT_PATH", "")
if not _OBSIDIAN_VAULT_PATH:
    try:
        import sys as _sys_obs
        _parent_dir = str(Path(__file__).parent.parent)
        if _parent_dir not in _sys_obs.path:
            _sys_obs.path.insert(0, _parent_dir)
        from mobile_app.obsidian_bridge import find_vault as _find_vault_auto
        _auto_vault = _find_vault_auto()
        if _auto_vault:
            _OBSIDIAN_VAULT_PATH = str(_auto_vault)
    except Exception:
        pass

# ChromaDB singleton — initialize once, not per-request
_chroma_client = None
_chroma_collection = None
_chroma_init_attempted = False

def _get_obsidian_collection():
    global _chroma_client, _chroma_collection, _chroma_init_attempted
    if _chroma_init_attempted:
        return _chroma_collection
    _chroma_init_attempted = True
    try:
        import chromadb
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        _chroma_client = chromadb.PersistentClient(path=_db_path)
        _chroma_collection = _chroma_client.get_collection("obsidian_knowledge")
        print(f"[RAG] ChromaDB initialized: {_db_path}")
    except Exception as e:
        print(f"[RAG] ChromaDB init failed: {e}")
        _chroma_collection = None
    return _chroma_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: ダッシュボードキャッシュのウォームアップ
    try:
        from data_cases import (
            load_dashboard_stats_cache,
            load_department_stats_cache,
            refresh_dashboard_stats_cache,
            refresh_department_stats_cache,
        )
        if load_dashboard_stats_cache() is None:
            refresh_dashboard_stats_cache()
        if load_department_stats_cache() is None:
            refresh_department_stats_cache()
    except Exception:
        pass
    # startup: Obsidian ナレッジのバックグラウンドインデックス化（30秒遅延 — OMP競合回避）
    def _delayed_indexing():
        import time as _t; _t.sleep(30)
        try:
            from api.knowledge.indexer import start_background_indexing
            start_background_indexing()
        except Exception as e:
            print(f"[API] knowledge indexing start failed (non-fatal): {e}")
    import threading as _th; _th.Thread(target=_delayed_indexing, daemon=True, name="delayed-indexer").start()
    # startup: Obsidian フィードバックのバックグラウンド読み込み（30秒遅延）
    def _delayed_feedback():
        import time as _t; _t.sleep(30)
        try:
            from api.knowledge.feedback_watcher import start_background_feedback_loading
            start_background_feedback_loading()
        except Exception as e:
            print(f"[API] feedback loading start failed (non-fatal): {e}")
    _th.Thread(target=_delayed_feedback, daemon=True, name="delayed-feedback").start()
    # startup: 会話履歴テーブルの初期化
    try:
        from api.database import init_conversation_history_table
        init_conversation_history_table()
    except Exception as e:
        print(f"[API] conversation_history table init failed (non-fatal): {e}")
    # startup: 汎用チャットメッセージテーブルの初期化
    try:
        from api.chat_memory import init_chat_messages_table
        init_chat_messages_table()
    except Exception as e:
        print(f"[API] chat_messages table init failed (non-fatal): {e}")
    # startup: 結晶化スケジューラー起動（毎日02:00）
    try:
        from api.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        print(f"[API] crystallization scheduler start failed (non-fatal): {e}")
    yield
    # shutdown: 結晶化スケジューラー停止
    try:
        from api.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


app = FastAPI(
    title="Lease Scoring API",
    description="リース審査ロジックのバックエンドAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# モダンフロントエンド(Next.js)のReactローカルサーバー(例: 3000番ポート)からのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Lease Scoring API is running."}

@app.get("/api/master/industries")
def get_industries():
    # static_data またはルートから industry_trends_jsic.json を読み込む
    import json
    paths = [
        os.path.join(_REPO_ROOT, "static_data", "industry_trends_jsic.json"),
        os.path.join(_REPO_ROOT, "industry_trends_jsic.json")
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

@app.get("/api/master/assets")
def get_assets():
    import json
    paths = [
        os.path.join(_REPO_ROOT, "static_data", "lease_assets.json"),
        os.path.join(_REPO_ROOT, "lease_assets.json")
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"items": []}

@app.get("/api/master/qualitative")
def get_qualitative_items():
    try:
        from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS
        return {"items": QUALITATIVE_SCORING_CORRECTION_ITEMS}
    except Exception:
        return {"items": []}

@app.post("/api/score/calculate", response_model=ScoringResponse)
def calculate_score(req: ScoringRequest):
    try:
        # パラメータを辞書化して existing の関数に渡す
        inputs = req.model_dump()
        result = run_quick_scoring(inputs)
        
        # 期待する戻り値のキーにマッピング
        return ScoringResponse(
            score=result.get("score", 0.0),
            hantei=result.get("hantei", "未判定"),
            comparison=result.get("comparison", ""),
            user_op_margin=result.get("user_op_margin", 0.0),
            user_equity_ratio=result.get("user_equity_ratio", 0.0),
            bench_op_margin=result.get("bench_op_margin", 0.0),
            bench_equity_ratio=result.get("bench_equity_ratio", 0.0),
            score_borrower=result.get("score_borrower", 0.0),
            score_base=result.get("score_base", result.get("score", 0.0)),
            industry_sub=result.get("industry_sub", req.industry_sub),
            industry_major=result.get("industry_major", req.industry_major),
            ai_completed_factors=[],
            asset_score=result.get("asset_score"),
            asset_warnings=result.get("asset_warnings", []),
            asset_bonuses=result.get("asset_bonuses", []),
            default_warnings=result.get("default_warnings", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/score/full", response_model=ScoringResponse)
def calculate_score_full(req: ScoringRequest):
    try:
        inputs = req.model_dump()
        result = run_full_scoring_api(inputs)

        # ── DB保存 ──────────────────────────────────────────────
        case_id = None
        try:
            from data_cases import save_case_log
            case_data = {
                "company_no":   inputs.get("company_no", ""),
                "company_name": inputs.get("company_name", ""),
                "industry_sub": result.get("industry_sub", inputs.get("industry_sub", "")),
                "industry_major": result.get("industry_major", inputs.get("industry_major", "")),
                "inputs": inputs,
                "result": {
                    "score": result.get("score", 0),
                    "hantei": result.get("hantei", ""),
                    "user_eq": result.get("user_eq", 0),
                    "user_op": result.get("user_op", 0),
                    "quantum_risk": result.get("quantum_risk"),
                    "credit_quantum_strong_warning": result.get("credit_quantum_strong_warning", False),
                },
            }
            case_id = save_case_log(case_data)
        except Exception as _save_err:
            print(f"[WARNING] DB save failed: {_save_err}")

        return ScoringResponse(
            score=result.get("score", 0.0),
            hantei=result.get("hantei", "未判定"),
            comparison=result.get("comparison", ""),
            user_op_margin=result.get("user_op_margin", result.get("user_op", 0.0)),
            user_equity_ratio=result.get("user_equity_ratio", result.get("user_eq", 0.0)),
            bench_op_margin=result.get("bench_op", 0.0),
            bench_equity_ratio=result.get("bench_eq", 0.0),
            score_borrower=result.get("score_borrower", 0.0),
            score_base=result.get("hantei_score", result.get("score", 0.0)),
            industry_sub=result.get("industry_sub", req.industry_sub),
            industry_major=result.get("industry_major", req.industry_major),
            ai_completed_factors=result.get("ai_completed_factors", []),
            case_id=case_id,
            company_no=inputs.get("company_no", ""),
            company_name=inputs.get("company_name", ""),
            asset_score=result.get("asset_score"),
            asset_warnings=result.get("asset_warnings", []),
            asset_bonuses=result.get("asset_bonuses", []),
            default_warnings=result.get("default_warnings", []),
            quantum_risk=result.get("quantum_risk"),
            credit_quantum_strong_warning=result.get("credit_quantum_strong_warning", False),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/deal/closure-probability", response_model=DealClosureResponse)
def calc_deal_closure_probability(req: DealClosureRequest):
    try:
        if req.delta_send is not None and req.delta_response is not None:
            features = build_features_from_deltas(req.delta_send, req.delta_response)
        elif req.registration_date and req.estimate_sent_date and req.customer_response_date:
            features = build_features(
                registration_date=req.registration_date,
                estimate_sent_date=req.estimate_sent_date,
                customer_response_date=req.customer_response_date,
            )
        else:
            raise ValueError("Either (delta_send & delta_response) or all 3 dates are required")
        prob = compute_closure_likelihood(features, has_cash_data=req.has_cash_data)
        return DealClosureResponse(
            closure_probability=prob,
            closure_probability_percent=round(prob * 100.0, 2),
            delta_send=features.delta_send,
            delta_response=features.delta_response,
            model_note="Trajectory-likelihood prototype (residue-inspired), preserving existing score pipeline.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/industry/stats")
def api_industry_stats():
    """業種別成約率・平均スコア集計（REV-055）"""
    import json
    from contextlib import closing
    from data_cases import _open_db
    with closing(_open_db()) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT industry_sub, final_status, COUNT(*) as cnt, AVG(score) as avg_score
            FROM past_cases
            WHERE industry_sub IS NOT NULL AND industry_sub != '' AND industry_sub != '0'
              AND final_status IN ('成約', '失注')
            GROUP BY industry_sub, final_status
        """)
        rows = cur.fetchall()

    industry_data: dict = {}
    for industry, status, cnt, avg_sc in rows:
        if industry not in industry_data:
            industry_data[industry] = {"total": 0, "won": 0, "lost": 0, "score_sum": 0.0, "score_cnt": 0}
        d = industry_data[industry]
        d["total"] += cnt
        if status == "成約":
            d["won"] += cnt
        else:
            d["lost"] += cnt
        if avg_sc is not None:
            d["score_sum"] += avg_sc * cnt
            d["score_cnt"] += cnt

    result = []
    for industry, d in industry_data.items():
        total = d["total"]
        if total < 3:
            continue
        rate = round(d["won"] / total * 100, 1) if total > 0 else 0.0
        avg_score = round(d["score_sum"] / d["score_cnt"], 1) if d["score_cnt"] > 0 else None
        result.append({
            "industry": industry,
            "total": total,
            "won": d["won"],
            "lost": d["lost"],
            "contract_rate": rate,
            "avg_score": avg_score,
        })

    return sorted(result, key=lambda x: x["total"], reverse=True)


@app.get("/api/cases")
def list_cases(limit: int = 30, offset: int = 0, sort: str = "desc"):
    """過去案件一覧 (limit/offset/sort 対応)"""
    import json
    from contextlib import closing
    from data_cases import _open_db

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    order = "DESC" if sort.lower() != "asc" else "ASC"
    rows = []
    try:
        with closing(_open_db()) as conn:
            import sqlite3
            conn.row_factory = sqlite3.Row
            res = conn.execute(
                f"SELECT id, timestamp, industry_sub, score, final_status, "
                f"json_extract(data,'$.company_name') AS company_name, "
                f"json_extract(data,'$.company_no')   AS company_no, "
                f"json_extract(data,'$.judgment')     AS judgment "
                f"FROM past_cases ORDER BY timestamp {order} LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            for r in res:
                rows.append(dict(r))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return rows


class CaseResultPatch(BaseModel):
    final_status: Optional[str] = None
    competitor_rate: Optional[float] = None
    loss_reason: Optional[str] = None
    final_result_date: Optional[str] = None


class BatchScoreRequest(BaseModel):
    csv_text: Optional[str] = None
    csv_base64: Optional[str] = None


class BatchSaveRequest(BatchScoreRequest):
    confirmed: bool = False
    batch_token: Optional[str] = None


class AssetFinanceRequest(BaseModel):
    asset_name: str = Field("", max_length=120)
    asset_type: str = Field(..., description="建機 / 工作機械 / PC/IT / 医療機器 / ドローン / 車両")
    term: int = Field(60, ge=12, le=84)
    down_payment: float = Field(0.2, ge=0.0, le=0.5)
    financial_score: str = Field("Medium", description="High / Medium / Low")
    main_bank_support: bool = False
    bank_coordination: bool = False
    core_business: bool = False
    related_assets: bool = False
    annual_km: int = Field(0, ge=0, le=100000)
    has_maintenance_lease: bool = False
    ai_residual_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    useful_life: Optional[int] = Field(None, ge=1, le=50, description="耐用年数（年）。指定時は r = 2.0 / useful_life で計算。")


class AssetFinanceObsidianContextRequest(BaseModel):
    asset_type: str = Field("", max_length=40)
    asset_name: str = Field("", max_length=120)
    financial_score: str = Field("", max_length=20)
    decision: str = Field("", max_length=40)
    memo_query: str = Field("", max_length=200)


class AssetFinanceSaveToObsidianRequest(BaseModel):
    input: Dict[str, Any]
    result: Dict[str, Any]
    related_paths: List[str] = Field(default_factory=list)


def _build_asset_finance_obsidian_terms(req: AssetFinanceObsidianContextRequest) -> List[str]:
    """物件名・型番からObsidian検索に効く安定語へ展開する。"""
    import re

    raw_parts = [
        "物件ファイナンス",
        "リース",
        "BEP",
        "残価",
        "再販リスク",
        "稟議根拠",
        req.asset_type,
        req.asset_name,
        req.financial_score,
        req.decision,
        req.memo_query,
    ]
    raw = " ".join(str(part or "") for part in raw_parts)
    lower = raw.lower()
    terms: List[str] = [str(part).strip() for part in raw_parts if str(part or "").strip()]

    asset_type_terms = {
        "建機": ["建機", "アワーメーター", "中古相場", "残価"],
        "車両": ["車両", "走行距離", "メンテナンス", "再販リスク"],
        "工作機械": ["工作機械", "中古相場", "制御装置", "保守期限"],
        "医療機器": ["医療機器", "保守期限", "薬機法", "設置撤去費"],
        "PC/IT": ["PC/IT", "陳腐化", "保守", "再販リスク"],
        "ドローン": ["ドローン", "バッテリー", "飛行時間", "法規制"],
    }
    terms.extend(asset_type_terms.get(req.asset_type, []))

    # 型番はハイフン枝番つきでも、親型式で検索できるようにする。
    for token in re.findall(r"[A-Za-z]+[- ]?\d+[A-Za-z0-9-]*|[A-Za-z]{2,}|[0-9]{2,}[A-Za-z0-9-]*", raw):
        clean = token.replace(" ", "").strip()
        if not clean:
            continue
        terms.append(clean)
        parent = clean.split("-", 1)[0]
        if parent != clean and len(parent) >= 3:
            terms.append(parent)

    keyword_groups = [
        (("コマツ", "komatsu", "pc200", "油圧ショベル", "ユンボ"), [
            "建機", "油圧ショベル", "ユンボ", "アワーメーター", "中古相場", "排ガス規制",
        ]),
        (("冷凍車", "冷蔵", "冷凍", "商用車", "メンテリース", "メンテナンスリース"), [
            "車両", "冷蔵冷凍車", "商用車", "メンテリース", "冷凍機", "走行距離", "架装",
        ]),
        (("フォークリフト", "forklift", "リフト", "トヨタl&f", "toyota"), [
            "フォークリフト", "バッテリー劣化", "アワーメーター", "マスト", "定期自主検査",
        ]),
        (("高所作業車", "アイチ", "タダノ", "ブーム", "アウトリガー"), [
            "高所作業車", "年次点検", "安全装置", "アウトリガー", "油圧", "ブーム",
        ]),
        (("発電機", "デンヨー", "denyo", "コンプレッサ", "コンプレッサー", "airman", "北越"), [
            "発電機", "コンプレッサー", "稼働時間", "排ガス規制", "防音型", "中古需要",
        ]),
        (("マシニング", "旋盤", "nc", "制御装置"), [
            "工作機械", "中古相場", "主軸稼働時間", "制御装置", "保守期限", "搬出費",
        ]),
        (("射出成形", "成形機", "型締", "スクリュー", "roboshot"), [
            "射出成形機", "型締力", "制御装置", "スクリュー", "電動式", "油圧式",
        ]),
        (("医療機器", "ct", "mri", "内視鏡", "歯科", "薬機法"), [
            "医療機器", "保守期限", "薬機法", "設置撤去費", "中古医療機器",
        ]),
        (("測定器", "検査装置", "三次元測定", "キーエンス", "ミツトヨ", "東京精密", "島津", "堀場"), [
            "測定器", "検査装置", "校正証明", "保守期限", "ソフトライセンス", "再校正",
        ]),
        (("pc/it", "it機器", "サーバ", "パソコン", "複合機"), [
            "PC/IT", "陳腐化", "保守", "リース", "再販リスク",
        ]),
        (("ドローン", "uav"), [
            "ドローン", "バッテリー", "飛行時間", "法規制", "機体登録",
        ]),
    ]
    for triggers, additions in keyword_groups:
        if any(trigger in lower for trigger in triggers):
            terms.extend(additions)

    seen: set[str] = set()
    deduped: List[str] = []
    for term in terms:
        clean = str(term or "").strip()
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _extract_asset_obsidian_evidence(hits: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Asset Knowledgeノートから審査画面向けの根拠候補を抽出する。"""
    import re

    buckets = {
        "used_market": [],
        "residual_risk": [],
        "approval_basis": [],
        "cautions": [],
    }
    heading_map = {
        "中古相場・再販観点": "used_market",
        "中古相場": "used_market",
        "残価・再販リスク": "residual_risk",
        "残価": "residual_risk",
        "稟議で使えそうな根拠": "approval_basis",
        "稟議根拠": "approval_basis",
        "注意すべき物件特性": "cautions",
        "注意点": "cautions",
    }
    seen: set[str] = set()

    asset_hits_used = 0
    for hit in hits:
        path = str(hit.get("path") or "")
        if "Asset Knowledge" not in path:
            continue
        if path.endswith("物件ファイナンス検索索引.md"):
            continue
        asset_hits_used += 1
        text = str(hit.get("snippet") or "").replace("\r\n", "\n")
        sections = re.split(r"\n(?=##+ .+)", text)
        for section in sections:
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            heading = lines[0].lstrip("#").strip()
            bucket = None
            for key, value in heading_map.items():
                if key in heading:
                    bucket = value
                    break
            if not bucket:
                continue
            for line in lines[1:]:
                item = line.lstrip("-・*0123456789. ").strip()
                if not item or item.startswith("http") or item.startswith("[["):
                    continue
                item = item.replace("**", "")
                if len(item) < 8:
                    continue
                dedupe_key = f"{bucket}:{item[:120]}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                buckets[bucket].append(item[:180])
                if len(buckets[bucket]) >= 5:
                    break
        if asset_hits_used >= 1 and any(buckets.values()):
            break

    return buckets


def _classify_asset_obsidian_hit(hit: Dict[str, Any]) -> str:
    """検索ヒットを審査用途の色分けカテゴリへ寄せる。"""
    path = str(hit.get("path") or "")
    text = f"{path}\n{hit.get('snippet') or ''}".lower()
    if "中古相場" in text:
        return "used_market"
    if "残価" in text or "再販" in text:
        return "residual_risk"
    if "稟議" in text or "根拠" in text or "承認" in text:
        return "approval_basis"
    if "注意" in text or "保守" in text or "校正" in text or "期限" in text:
        return "cautions"
    if "asset knowledge" in path.lower():
        return "support"
    if "daily" in path.lower():
        return "context"
    if "generated" in path.lower():
        return "generated"
    return "support"


def _normalize_obsidian_node_label(path_or_label: str, limit: int = 28) -> str:
    text = str(path_or_label or "").strip()
    if not text:
        return "無題"
    if "/" in text:
        text = Path(text).stem
    if len(text) > limit:
        return text[:limit - 1] + "…"
    return text


def _build_asset_finance_obsidian_graph(
    query: str,
    hits: List[Dict[str, Any]],
    generated_terms: List[str],
    evidence: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Obsidianメモの簡易グラフを NEXT 用に返す。"""
    color_map = {
        "used_market": "#2563eb",
        "residual_risk": "#d97706",
        "approval_basis": "#059669",
        "cautions": "#e11d48",
        "support": "#94a3b8",
        "context": "#7c3aed",
        "generated": "#0f766e",
        "query": "#0f172a",
        "focus": "#0f172a",
        "linked": "#cbd5e1",
    }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    evidence_text = " ".join(" ".join(v) for v in evidence.values()).lower()

    def add_node(node_id: str, label: str, node_type: str, color: str, radius: int, **extra: Any) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "color": color,
            "radius": radius,
            **extra,
        })

    add_node("focus", "今回の審査", "focus", color_map["focus"], 24, pinned=True, used=True)
    if query.strip():
        add_node("query", "検索語", "query", color_map["query"], 18, used=True)
        edges.append({"source": "focus", "target": "query", "type": "query", "width": 2, "color": "#0ea5e9"})

    for idx, term in enumerate(generated_terms[:10]):
        term_id = f"term_{idx}"
        add_node(term_id, _normalize_obsidian_node_label(term, 24), "generated", color_map["generated"], 14, used=True, term=term)
        edges.append({"source": "query" if query.strip() else "focus", "target": term_id, "type": "term", "width": 1.3, "color": "#14b8a6"})

    for idx, hit in enumerate(hits[:6]):
        path = str(hit.get("path") or "")
        label = _normalize_obsidian_node_label(hit.get("title") or path, 28)
        category = _classify_asset_obsidian_hit(hit)
        used = category in {"used_market", "residual_risk", "approval_basis", "cautions"} or path.lower() in evidence_text
        node_id = f"hit_{idx}"
        color = color_map.get(category, color_map["support"])
        snippet = str(hit.get("snippet") or "").strip().replace("\r\n", "\n")
        add_node(
            node_id,
            label,
            category,
            color,
            19 if used else 14,
            path=path,
            used=used,
            category=category,
            snippet=snippet[:220],
            wikilinks=hit.get("wikilinks") or [],
        )
        edges.append({
            "source": "focus",
            "target": node_id,
            "type": "used" if used else "support",
            "width": 2.5 if used else 1.2,
            "color": color if used else "#cbd5e1",
        })

        linked = hit.get("wikilinks") or []
        if isinstance(linked, str):
            linked = [item.strip() for item in linked.split(",") if item.strip()]
        for link_idx, link in enumerate(list(linked)[:3]):
            link_id = f"{node_id}_link_{link_idx}"
            link_label = _normalize_obsidian_node_label(link, 26)
            add_node(link_id, link_label, "linked", color_map["linked"], 11, used=False, linked_from=path)
            edges.append({
                "source": node_id,
                "target": link_id,
                "type": "wikilink",
                "width": 1,
                "color": "#cbd5e1",
            })

    summary = {
        "total_hits": len(hits),
        "used_hits": sum(1 for hit in hits if _classify_asset_obsidian_hit(hit) != "support"),
        "linked_nodes": sum(1 for node in nodes if node.get("type") == "linked"),
        "generated_terms": len(generated_terms),
    }
    legend = [
        {"label": "今回の審査", "color": color_map["focus"]},
        {"label": "今回使った根拠", "color": color_map["approval_basis"]},
        {"label": "中古相場", "color": color_map["used_market"]},
        {"label": "残価・再販", "color": color_map["residual_risk"]},
        {"label": "注意点", "color": color_map["cautions"]},
        {"label": "関連ノート", "color": color_map["linked"]},
    ]
    return {"nodes": nodes, "edges": edges, "summary": summary, "legend": legend}


BATCH_MAX_CSV_BYTES = 5 * 1024 * 1024
BATCH_MAX_ROWS = 1000
BATCH_TOKEN_TTL_SECONDS = 30 * 60
_batch_result_cache: dict[str, dict] = {}


@app.get("/api/batch/template")
def get_batch_template():
    """バッチ審査CSVテンプレートを返す。"""
    try:
        from components.batch_scoring import _get_csv_template
        return Response(
            content=_get_csv_template(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="batch_shinsa_template.csv"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/asset-finance/evaluate")
def evaluate_asset_finance(req: AssetFinanceRequest):
    """物件保全性・BEP・定性緩和因子を統合した物件ファイナンス審査。"""
    try:
        from components.asset_finance import AssetFinanceEngine
        engine = AssetFinanceEngine()
        if req.asset_type not in engine.ASSET_PARAMS:
            raise HTTPException(status_code=422, detail=f"未対応の物件種別です: {req.asset_type}")
        if req.financial_score not in {"High", "Medium", "Low"}:
            raise HTTPException(status_code=422, detail="financial_score は High / Medium / Low のいずれかです")

        data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        result = engine.run_inference(data)
        params = engine.ASSET_PARAMS[req.asset_type]
        eff_life = req.useful_life if req.useful_life else params["useful_life"]
        eff_r = 2.0 / eff_life
        curve = [
            {"month": i, "asset_value": v, "lease_balance": result["l_curve"][i]}
            for i, v in enumerate(result["v_curve"])
        ]
        return {
            **result,
            "curve": curve,
            "asset_params": {
                "depreciation_rate": eff_r,
                "useful_life": eff_life,
                "priority": params["priority"],
                "priority_score": params["priority_score"],
                "info": params["info"],
            },
            "input": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/asset-finance/obsidian-context")
def get_asset_finance_obsidian_context(req: AssetFinanceObsidianContextRequest):
    """物件ファイナンス審査に関連するObsidianメモを共通検索経路で取得する。"""
    try:
        from mobile_app.obsidian_bridge import build_obsidian_digest, collect_obsidian_context, search_notes

        generated_terms = _build_asset_finance_obsidian_terms(req)
        query = " ".join(generated_terms)
        hits = search_notes(query, limit=5, max_chars=2600)
        if len(hits) < 5:
            seen_paths = {str(hit.get("path") or "") for hit in hits}
            for hit in collect_obsidian_context(query, limit=5 - len(hits)):
                path = str(hit.get("path") or "")
                if path and path not in seen_paths:
                    hits.append(hit)
                    seen_paths.add(path)
        digest = build_obsidian_digest(query, hits) if hits else {"digest": "", "source_count": "0", "links": ""}
        evidence = _extract_asset_obsidian_evidence(hits)
        graph = _build_asset_finance_obsidian_graph(
            query=query,
            hits=hits,
            generated_terms=generated_terms,
            evidence=evidence,
        )
        return {
            "query": query,
            "generated_terms": generated_terms,
            "hits": hits,
            "digest": digest,
            "evidence": evidence,
            "graph": graph,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Obsidian検索エラー: {e}")


@app.post("/api/asset-finance/similar-notes")
def get_asset_finance_similar_notes(req: AssetFinanceObsidianContextRequest):
    """保存済みの物件ファイナンス・過去案件メモから類似メモを返す。"""
    try:
        from mobile_app.obsidian_bridge import search_notes

        terms = _build_asset_finance_obsidian_terms(req)
        query = " ".join([*terms, "類似", "過去", "案件", "承認", "条件"])
        hits = search_notes(query, limit=12, max_chars=1000)
        filtered = [
            hit for hit in hits
            if "Projects/tune_lease_55/Asset Finance/" in str(hit.get("path") or "")
            or "Projects/tune_lease_55/Cases/" in str(hit.get("path") or "")
        ]
        return {"query": query, "similar_notes": filtered[:5]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"類似メモ検索エラー: {e}")


def _knowledge_graph_display_path(raw_path: str, file_name: str = "") -> str:
    path = str(raw_path or "").replace("\\", "/")
    for marker in ("/Obsidian Vault/", "/Documents/"):
        if marker in path:
            tail = path.split(marker, 1)[1]
            if marker == "/Documents/" and "/" in tail:
                parts = tail.split("/", 1)
                if parts[0].endswith("Vault") or parts[0] == "Obsidian Vault":
                    return parts[1]
            else:
                return tail
    return path or str(file_name or "")


def _knowledge_graph_category(path: str) -> str:
    low = path.lower()
    if "projects/tune_lease_55/cases/" in low:
        return "case"
    if "projects/tune_lease_55/asset" in low:
        return "asset"
    if "projects/tune_lease_55/feedback/" in low or "improvement" in low:
        return "feedback"
    if "projects/tune_lease_55/news/" in low or "research" in low or "clippings" in low:
        return "research"
    if "daily/" in low:
        return "daily"
    if "wiki" in low or "検索語" in path:
        return "wiki"
    return "knowledge"


@app.get("/api/knowledge/graph")
def get_knowledge_graph(limit: int = 180):
    """インデックス済み Obsidian ナレッジをファイル単位の3Dグラフ用に返す。"""
    try:
        from api.knowledge.vector_store import get_store

        limit = max(30, min(int(limit or 180), 420))
        store = get_store()
        store._ensure_collection()  # collection only; does not force encoder/network
        collection = store._collection
        if collection is None or collection.count() == 0:
            return {
                "nodes": [],
                "edges": [],
                "summary": {"indexed_chunks": 0, "notes": 0, "links": 0, "limit": limit},
                "legend": [],
            }

        raw = collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []

        notes: dict[str, dict[str, Any]] = {}
        stem_to_id: dict[str, str] = {}
        for meta in metadatas:
            meta = meta or {}
            file_name = str(meta.get("file_name") or "")
            raw_path = str(meta.get("file_path") or "")
            path = _knowledge_graph_display_path(raw_path, file_name)
            note_id = path or file_name
            if not note_id:
                continue
            stem = os.path.splitext(file_name)[0] or os.path.splitext(os.path.basename(path))[0]
            section = str(meta.get("section") or "")
            wikilinks = str(meta.get("wikilinks") or "")
            item = notes.setdefault(note_id, {
                "id": note_id,
                "label": stem or os.path.basename(path),
                "path": path,
                "category": _knowledge_graph_category(path),
                "sections": set(),
                "wikilinks": set(),
                "chunk_count": 0,
                "mtime": float(meta.get("mtime") or 0),
            })
            item["chunk_count"] += 1
            item["mtime"] = max(float(item.get("mtime") or 0), float(meta.get("mtime") or 0))
            if section:
                item["sections"].add(section)
            for link in [part.strip() for part in wikilinks.split(",") if part.strip()]:
                item["wikilinks"].add(link)
            if stem:
                stem_to_id.setdefault(stem, note_id)

        link_counts: dict[str, int] = {note_id: 0 for note_id in notes}
        for note in notes.values():
            for link in note["wikilinks"]:
                target = stem_to_id.get(link)
                if target:
                    link_counts[note["id"]] = link_counts.get(note["id"], 0) + 1
                    link_counts[target] = link_counts.get(target, 0) + 1

        ranked_ids = sorted(
            notes,
            key=lambda note_id: (
                link_counts.get(note_id, 0),
                notes[note_id]["chunk_count"],
                notes[note_id]["mtime"],
            ),
            reverse=True,
        )[:limit]
        included = set(ranked_ids)

        color_map = {
            "case": "#22c55e",
            "asset": "#38bdf8",
            "feedback": "#f97316",
            "research": "#a78bfa",
            "daily": "#94a3b8",
            "wiki": "#facc15",
            "knowledge": "#e2e8f0",
            "folder": "#64748b",
            "external": "#475569",
        }
        cluster_names = {
            "case": "過去案件",
            "asset": "物件・残価",
            "feedback": "改善ログ",
            "research": "調査・ニュース",
            "daily": "日次",
            "wiki": "Wiki・検索語",
            "knowledge": "知識ノート",
        }

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for category, label in cluster_names.items():
            category_ids = [note_id for note_id in ranked_ids if notes[note_id]["category"] == category]
            if not category_ids:
                continue
            nodes.append({
                "id": f"cluster:{category}",
                "label": label,
                "type": "cluster",
                "category": "folder",
                "color": color_map["folder"],
                "radius": 10 + min(18, len(category_ids) * 0.5),
                "count": len(category_ids),
            })

        for note_id in ranked_ids:
            note = notes[note_id]
            category = note["category"]
            radius = min(13, 4 + note["chunk_count"] * 0.7 + link_counts.get(note_id, 0) * 0.3)
            nodes.append({
                "id": note_id,
                "label": note["label"],
                "path": note["path"],
                "type": "note",
                "category": category,
                "color": color_map.get(category, color_map["knowledge"]),
                "radius": round(radius, 2),
                "chunk_count": note["chunk_count"],
                "link_count": link_counts.get(note_id, 0),
                "mtime": note["mtime"],
                "sections": sorted(note["sections"])[:8],
            })
            if any(node.get("id") == f"cluster:{category}" for node in nodes):
                edges.append({
                    "source": f"cluster:{category}",
                    "target": note_id,
                    "type": "cluster",
                    "weight": 0.3,
                    "color": "#475569",
                })

        external_seen: set[str] = set()
        for note_id in ranked_ids:
            note = notes[note_id]
            for link in sorted(note["wikilinks"]):
                target = stem_to_id.get(link)
                if target and target in included:
                    edges.append({
                        "source": note_id,
                        "target": target,
                        "type": "wikilink",
                        "weight": 1.0,
                        "color": "#38bdf8",
                        "mtime": max(float(note.get("mtime") or 0), float(notes[target].get("mtime") or 0)),
                    })
                elif len(external_seen) < 40 and link not in external_seen:
                    external_seen.add(link)
                    ext_id = f"external:{link}"
                    nodes.append({
                        "id": ext_id,
                        "label": link,
                        "type": "external",
                        "category": "external",
                        "color": color_map["external"],
                        "radius": 3.5,
                    })
                    edges.append({
                        "source": note_id,
                        "target": ext_id,
                        "type": "external",
                        "weight": 0.25,
                        "color": "#334155",
                        "mtime": float(note.get("mtime") or 0),
                    })

        # Deduplicate identical links.
        unique_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in edges:
            key = (str(edge["source"]), str(edge["target"]), str(edge["type"]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            unique_edges.append(edge)
        unique_edges.sort(key=lambda edge: float(edge.get("mtime") or 0), reverse=True)
        for index, edge in enumerate(unique_edges):
            edge["recent_rank"] = index + 1

        return {
            "nodes": nodes,
            "edges": unique_edges,
            "summary": {
                "indexed_chunks": collection.count(),
                "notes": len(notes),
                "shown_nodes": len(nodes),
                "links": len(unique_edges),
                "limit": limit,
            },
            "legend": [
                {"label": label, "category": category, "color": color_map[category]}
                for category, label in cluster_names.items()
            ],
        }
    except Exception as e:
        print(f"[API] knowledge graph error: {e}")
        raise HTTPException(status_code=503, detail="現在ナレッジ機能を準備中です。しばらくお待ちください。")


@app.post("/api/asset-finance/save-to-obsidian")
def save_asset_finance_to_obsidian(req: AssetFinanceSaveToObsidianRequest):
    """物件ファイナンス審査結果をObsidianへ保存する。"""
    try:
        from mobile_app.obsidian_bridge import append_asset_finance_note, append_asset_knowledge_backlinks

        # クライアント送信の result は信用せず、保存直前にサーバー側で再計算する。
        # Obsidianは後続AI検索の知識源になるため、改ざん済みの判定を残さない。
        asset_req = (
            AssetFinanceRequest.model_validate(req.input)
            if hasattr(AssetFinanceRequest, "model_validate")
            else AssetFinanceRequest.parse_obj(req.input)
        )
        recalculated = evaluate_asset_finance(asset_req)

        saved = append_asset_finance_note(recalculated["input"], recalculated, req.related_paths)
        if saved.get("status") != "saved":
            raise HTTPException(status_code=503, detail=saved.get("reason") or "Obsidian保存をスキップしました")
        backlinks = append_asset_knowledge_backlinks(
            recalculated["input"],
            recalculated,
            req.related_paths,
            saved.get("rel_path"),
        )
        return {
            **saved,
            "score": recalculated.get("score"),
            "decision": recalculated.get("decision"),
            "backlinks": backlinks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Obsidian保存エラー: {e}")


_USEFUL_LIFE_TABLE: list[dict] | None = None

def _load_useful_life_table() -> list[dict]:
    global _USEFUL_LIFE_TABLE
    if _USEFUL_LIFE_TABLE is None:
        table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "useful_life_table.json")
        with open(table_path, "r", encoding="utf-8") as f:
            _USEFUL_LIFE_TABLE = json.load(f)
    return _USEFUL_LIFE_TABLE


@app.get("/api/asset/useful-life-search")
def search_useful_life(q: str = ""):
    """国税庁の法定耐用年数表からキーワード検索（name/category/subcategory）。最大20件返す。"""
    table = _load_useful_life_table()
    if not q.strip():
        return table[:20]
    q_lower = q.lower()
    results = [
        item for item in table
        if q_lower in item.get("name", "").lower()
        or q_lower in item.get("category", "").lower()
        or q_lower in item.get("subcategory", "").lower()
    ]
    return results[:20]


def _sanitize_batch_value(value):
    try:
        import math
        import numpy as np
        import pandas as pd
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            if math.isnan(float(value)):
                return None
            return float(value)
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _sanitize_batch_records(records: list[dict]) -> list[dict]:
    return [
        {str(k): _sanitize_batch_value(v) for k, v in row.items()}
        for row in records
    ]


@app.post("/api/batch/score")
def score_batch(req: BatchScoreRequest):
    """CSVを一括スコアリングする。DB保存は /api/batch/save で明示的に行う。"""
    return _run_batch_scoring(req, save_to_db=False)


@app.post("/api/batch/save")
def save_batch(req: BatchSaveRequest):
    """確認済みCSVを一括スコアリングし、過去案件DBへ保存する。"""
    if not req.confirmed:
        raise HTTPException(status_code=422, detail="confirmed=true が必要です")
    if req.batch_token:
        return _save_cached_batch(req.batch_token)
    return _run_batch_scoring(req, save_to_db=True)


def _cleanup_batch_cache(now: float | None = None) -> None:
    import time
    now = time.time() if now is None else now
    expired = [
        token for token, item in _batch_result_cache.items()
        if now - float(item.get("created_at", 0)) > BATCH_TOKEN_TTL_SECONDS
    ]
    for token in expired:
        _batch_result_cache.pop(token, None)


def _build_batch_response(df_in, df_out, summary: dict, batch_token: str | None = None) -> dict:
    csv_out = df_out.to_csv(index=False, encoding="utf-8-sig")
    records = _sanitize_batch_records(df_out.to_dict(orient="records"))
    preview = _sanitize_batch_records(df_in.head(5).to_dict(orient="records"))
    response = {
        "summary": summary,
        "preview": preview,
        "rows": records,
        "csv": csv_out,
    }
    if batch_token:
        response["batch_token"] = batch_token
    return response


def _save_batch_payloads(db_results: list[dict], excluded_grade_results: list[dict]) -> tuple[int, int, int]:
    from data_cases import save_case_log, save_excluded_grade_case

    saved_count = 0
    with_result = 0
    excluded_saved_count = 0
    for db_data in db_results:
        if save_case_log(db_data):
            saved_count += 1
            if db_data.get("final_status") in ("成約", "失注"):
                with_result += 1
    for excluded_data in excluded_grade_results:
        if save_excluded_grade_case(excluded_data):
            excluded_saved_count += 1
    return saved_count, with_result, excluded_saved_count


def _run_batch_training_check(with_result: int) -> str:
    if with_result <= 0:
        return ""
    try:
        from auto_optimizer import get_training_status, run_auto_optimization
        status = get_training_status()
        if status.get("should_retrain"):
            opt_result = run_auto_optimization(force=False)
            ab = (opt_result or {}).get("ab_test_result", {})
            if ab.get("passed"):
                return f"係数自動更新完了: {ab.get('reason', '')}"
            return f"係数更新見送り: {ab.get('reason', '')}"
        return f"成約/失注データ蓄積中。次回学習まであと {status.get('next_trigger')} 件"
    except Exception as e:
        return f"自動学習スキップ: {e}"


def _save_cached_batch(batch_token: str):
    import time

    _cleanup_batch_cache()
    cached = _batch_result_cache.get(batch_token)
    if not cached:
        raise HTTPException(status_code=404, detail="保存対象のバッチ結果が見つからないか期限切れです。再スコアリングしてください。")
    if cached.get("saved"):
        raise HTTPException(status_code=409, detail="このバッチ結果は既に保存済みです。")

    backup_message = ""
    try:
        from backup_manager import run_backup
        bk = run_backup(force=True)
        bk_files = [b.get("file", "") for b in bk.get("backed_up", []) if b.get("file")]
        backup_message = (
            f"バックアップ完了: {', '.join(bk_files)}"
            if bk_files else
            "バックアップ: 最新版が既に存在するためスキップ"
        )
    except Exception as e:
        backup_message = f"バックアップに失敗しました（保存は続行）: {e}"

    try:
        saved_count, with_result, excluded_saved_count = _save_batch_payloads(
            cached["db_results"],
            cached["excluded_grade_results"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB保存に失敗しました: {e}")

    cached["saved"] = True
    cached["saved_at"] = time.time()
    summary = dict(cached["summary"])
    summary.update({
        "saved_count": saved_count,
        "with_result": with_result,
        "excluded_saved_count": excluded_saved_count,
        "backup_message": backup_message,
        "training_message": _run_batch_training_check(with_result),
        "failed_count": max(0, len(cached["db_results"]) + len(cached["excluded_grade_results"]) - saved_count - excluded_saved_count),
    })
    cached["summary"] = summary
    return _build_batch_response(cached["df_in"], cached["df_out"], summary, batch_token=batch_token)


def _run_batch_scoring(req: BatchScoreRequest, save_to_db: bool = False):
    import base64
    import io
    import pandas as pd

    try:
        from components.batch_scoring import _score_one
        from industry_normalizer import normalize_industry_major
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"バッチ審査ロジックの読み込みに失敗しました: {e}")

    if req.csv_base64:
        try:
            csv_bytes = base64.b64decode(req.csv_base64)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"CSV base64 の復号に失敗しました: {e}")
        if len(csv_bytes) > BATCH_MAX_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSVファイルが大きすぎます。上限は {BATCH_MAX_CSV_BYTES // 1024 // 1024}MB です。",
            )
        last_error = None
        for enc in ("utf-8-sig", "cp932", "shift_jis"):
            try:
                df_in = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
                break
            except Exception as e:
                last_error = e
        else:
            raise HTTPException(status_code=422, detail=f"CSV 読み込みエラー: {last_error}")
    elif req.csv_text:
        if len(req.csv_text.encode("utf-8")) > BATCH_MAX_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSVテキストが大きすぎます。上限は {BATCH_MAX_CSV_BYTES // 1024 // 1024}MB です。",
            )
        try:
            df_in = pd.read_csv(io.StringIO(req.csv_text))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"CSV 読み込みエラー: {e}")
    else:
        raise HTTPException(status_code=422, detail="csv_text または csv_base64 が必要です")

    df_in = df_in.fillna("")
    if len(df_in) > BATCH_MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV行数が多すぎます。上限は {BATCH_MAX_ROWS}件です。",
        )
    if "業種大分類" in df_in.columns:
        df_in["業種大分類"] = df_in["業種大分類"].map(normalize_industry_major)

    missing_cols = [
        name for name in ["売上高", "総資産"]
        if f"{name}(百万円)" not in df_in.columns and f"{name}(千円)" not in df_in.columns
    ]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"必須列が不足しています: {missing_cols}")

    ui_results = []
    db_results = []
    excluded_grade_results = []
    for _, row in df_in.iterrows():
        out = _score_one(row.to_dict())
        ui_results.append(out["UI表示用"])
        if out.get("DB保存用"):
            db_results.append(out["DB保存用"])
        if out.get("信用リスク群保存用"):
            excluded_grade_results.append(out["信用リスク群保存用"])

    ui_df = pd.DataFrame(ui_results)
    duplicate_ui_cols = [c for c in ui_df.columns if c in df_in.columns]
    if duplicate_ui_cols:
        ui_df = ui_df.drop(columns=duplicate_ui_cols)
    df_out = pd.concat([df_in.reset_index(drop=True), ui_df], axis=1)

    saved_count = 0
    with_result = 0
    excluded_saved_count = 0
    backup_message = ""
    training_message = ""
    failed_count = 0

    if save_to_db and (db_results or excluded_grade_results):
        try:
            from backup_manager import run_backup
            bk = run_backup(force=True)
            bk_files = [b.get("file", "") for b in bk.get("backed_up", []) if b.get("file")]
            backup_message = (
                f"バックアップ完了: {', '.join(bk_files)}"
                if bk_files else
                "バックアップ: 最新版が既に存在するためスキップ"
            )
        except Exception as e:
            backup_message = f"バックアップに失敗しました（保存は続行）: {e}"

        try:
            saved_count, with_result, excluded_saved_count = _save_batch_payloads(db_results, excluded_grade_results)
            failed_count = max(0, len(db_results) + len(excluded_grade_results) - saved_count - excluded_saved_count)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB保存に失敗しました: {e}")

        training_message = _run_batch_training_check(with_result)

    total = len(df_out)
    judgments = df_out["判定"] if "判定" in df_out.columns else pd.Series([], dtype=object)
    summary = {
        "total": total,
        "good": int(((judgments == "良決") | (judgments == "承認圏内")).sum()) if total else 0,
        "border": int(((judgments == "ボーダー") | (judgments == "要審議")).sum()) if total else 0,
        "rejected": int((judgments == "否決").sum()) if total else 0,
        "errors": int((judgments == "エラー").sum()) if total else 0,
        "standard_scoring": int((df_out.get("スコアリング") == "標準").sum()) if "スコアリング" in df_out else 0,
        "saved_count": saved_count,
        "with_result": with_result,
        "excluded_saved_count": excluded_saved_count,
        "backup_message": backup_message,
        "training_message": training_message,
        "failed_count": failed_count,
    }

    batch_token = None
    if not save_to_db:
        import time
        import uuid
        _cleanup_batch_cache()
        batch_token = uuid.uuid4().hex
        _batch_result_cache[batch_token] = {
            "created_at": time.time(),
            "saved": False,
            "df_in": df_in,
            "df_out": df_out,
            "db_results": db_results,
            "excluded_grade_results": excluded_grade_results,
            "summary": summary,
        }

    return _build_batch_response(df_in, df_out, summary, batch_token=batch_token)


@app.patch("/api/cases/{case_id}/result")
def patch_case_result(case_id: str, req: CaseResultPatch):
    """案件結果を部分更新 (final_status / competitor_rate / loss_reason / final_result_date)"""
    from constants import FINAL_STATUS_VALID
    from data_cases import update_case

    if req.final_status is not None and req.final_status not in FINAL_STATUS_VALID:
        raise HTTPException(status_code=422, detail=f"不正な final_status: {req.final_status}")

    patches: dict = {}
    if req.final_status is not None:
        patches["final_status"] = req.final_status
    if req.competitor_rate is not None:
        patches["competitor_rate"] = req.competitor_rate
    if req.loss_reason is not None:
        patches["lost_reason"] = req.loss_reason
    if req.final_result_date is not None:
        patches["final_result_date"] = req.final_result_date

    if not patches:
        raise HTTPException(status_code=422, detail="更新フィールドが空です")

    if not update_case(case_id, patches):
        raise HTTPException(status_code=404, detail="案件が見つからないか更新失敗")

    return {"status": "updated", "case_id": case_id}


@app.get("/api/cases/pending")
def get_pending_cases():
    """全DB(lease_data.db, screening_db.sqlite)から未登録案件を統合して取得する"""
    import json
    from contextlib import closing
    from data_cases import _open_db

    rows = []

    try:
        with closing(_open_db()) as conn:
            import sqlite3
            conn.row_factory = sqlite3.Row
            res = conn.execute(
                "SELECT id, timestamp, industry_sub, score, data "
                "FROM past_cases WHERE final_status='未登録' ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            for r in res:
                try:
                    d = json.loads(r["data"] or "{}")
                except Exception:
                    d = {}
                rows.append({
                    "id": str(r["id"]),
                    "company_no": d.get("company_no", ""),
                    "company_name": d.get("company_name", ""),
                    "timestamp": r["timestamp"],
                    "score": r["score"],
                    "industry": r["industry_sub"] or d.get("industry_major", ""),
                    "registration_date": d.get("registration_date") or (r["timestamp"] or "")[:10],
                    "estimate_sent_date": d.get("estimate_sent_date") or (r["timestamp"] or "")[:10],
                    "final_result_date": d.get("final_result_date"),
                    "_source": "past_cases"
                })
    except Exception:
        pass

    return rows

@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: str):
    """案件の全データを返す（result + inputs を含む）"""
    import json
    from contextlib import closing
    from data_cases import _open_db

    try:
        with closing(_open_db()) as conn:
            import sqlite3
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT data FROM past_cases WHERE id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return json.loads(row["data"] or "{}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cases/operation/clear-all")
def clear_all_pending_cases():
    """未登録案件をすべて削除する（一括クリア）"""
    from data_cases import _open_db, refresh_stats_caches
    try:
        with _open_db() as conn:
            conn.execute("DELETE FROM past_cases WHERE final_status='未登録'")
            conn.commit()
        refresh_stats_caches()
        return {"message": "Cleared all pending cases"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str):
    """案件を past_cases から削除する"""
    from data_cases import delete_case as delete_case_from_db
    try:
        delete_case_from_db(str(case_id))
    except Exception:
        pass
    return {"message": "Deleted if existed", "case_id": case_id}


class AdviseRequest(BaseModel):
    score: float
    industry_major: str

class AdviseResponseItem(BaseModel):
    id: str
    text: str
    score_boost: float

@app.post("/api/gunshi/advise", response_model=List[AdviseResponseItem])
def get_gunshi_advise(req: AdviseRequest):
    from shinsa_gunshi import PHRASES_100
    try:
        advices = PHRASES_100.get("逆転アドバイス", [])
        
        # 確率ブーストから得点アップの目算に変換 (例: prob_boost 0.10 -> 10点相当)
        # ランダム性を持たせつつ、より状況に合ったものを本来はソートするが今回は上位3つを決定
        import random
        # 簡易的にシャッフルして上位を取り、スコアを計算
        sampled = random.sample(advices, min(3, len(advices)))
        
        results = []
        for a in sampled:
            # 内部のprob_boost (0.08～0.12程度) を 100倍してスコア上昇幅とする
            boost_score = round(a.get("prob_boost", 0.05) * 100)
            results.append(AdviseResponseItem(
                id=a["id"],
                text=a["text"],
                score_boost=boost_score
            ))
            
        # スコアアップが高い順にソート
        results.sort(key=lambda x: x.score_boost, reverse=True)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GunshiStreamRequest(BaseModel):
    industry_cat: str
    industry_sub: str = ""
    score: float
    pd_pct: float = 0.0
    resale_eval: str = "B"
    repeat_count: int = 0
    subsidy_flag: bool = False
    bank_support: bool = False
    intuition_score: float = 50.0
    company_name: str = ""
    asset_name: str = ""
    acquisition_cost: float = 0.0
    lease_term: int = 0
    contract_type: str = ""
    main_bank: str = ""
    competitor: str = ""
    competitor_rate: float | None = None
    deal_source: str = ""
    customer_type: str = ""
    nenshu: float = 0.0
    op_profit: float = 0.0
    equity_ratio: float = 0.0
    bank_credit: float = 0.0
    lease_credit: float = 0.0
    asset_warnings: list = []
    asset_bonuses: list = []
    default_warnings: list = []


@app.post("/api/gunshi/stream")
async def gunshi_stream(req: GunshiStreamRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "")

    async def event_generator():
        async for chunk in stream_gunshi_gemini(req.model_dump(), api_key):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class GunshiChatRequest(BaseModel):
    score: float
    industry_major: str
    asset_name: str
    resale: str
    repeat_cnt: int
    subsidy: bool
    bank: bool
    intuition: int
    posterior: float
    message: str = ""
    history: List[Dict[str, str]] = Field(default_factory=list)
    humor_style: str = "standard"
    use_web: bool = True
    use_obsidian: bool = True
    mode: str = "gunshi"  # 'gunshi'（戦略アドバイス）/ 'chat'（自由相談=Flask AIチャット）


def _format_gunshi_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for item in history:
        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        label = "ユーザー" if role == "user" else "軍師" if role == "assistant" else role or "不明"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _normalize_yukikaze_datalink_reply(reply_text: str, user_message: str) -> str:
    import re
    from datetime import date

    text = (reply_text or "").replace("\r\n", "\n")
    text = re.sub(r"(これで.*?進みますように[！!．\.]?|稟議書も.*?進みますように[！!．\.]?|よろしければ.*|ご参考までに.*|必要であれば.*|必要なら.*|お気軽に.*|安心してください.*|頑張って.*|お疲れ様です.*|ですよね.*)", "", text, flags=re.I)
    allowed_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(TX:|RX:|DATALINK LOG:|SIGNAL:|PILOT TASK:|VECTOR:)", line, re.I):
            allowed_lines.append(line)

    has_tx = any(re.match(r"^TX:", line, re.I) for line in allowed_lines)
    has_rx = any(re.match(r"^RX:", line, re.I) for line in allowed_lines)
    if has_tx and has_rx:
        return "\n".join(allowed_lines)

    msg = (user_message or "").strip()
    date_like = bool(re.search(r"(日付|今日|何日|何曜日|曜日|date|today)", msg, re.I))
    if date_like:
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][date.today().weekday()]
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Date confirmed.",
            f"RX: {date.today():%Y-%m-%d}, {weekday}.",
        ])

    body = text.strip()
    if body:
        parts: list[str] = []
        for raw_line in body.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            pieces = re.split(r"(?<=[。.!?])\s+", line)
            for piece in pieces:
                chunk = piece.strip()
                if chunk:
                    parts.append(chunk)
        if not parts:
            parts = [body]

        transcript = ["DATALINK LOG:", "TX: PANPANPAN // PILOT QUERY RECEIVED."]
        for idx, part in enumerate(parts[:8]):
            prefix = "RX:" if idx % 2 == 0 else "TX:"
            cleaned = re.sub(r"[。．.!！？?]+$", "", part).strip()
            cleaned = re.sub(r"(です|ます|でしょう|でしょうか|ください|くださいね)$", "", cleaned).strip()
            transcript.append(f"{prefix} {cleaned}")
        return "\n".join(transcript)

    return "\n".join([
        "DATALINK LOG:",
        "TX: PANPANPAN // PILOT QUERY RECEIVED.",
        "RX: ROGER. COPY. PILOT QUERY RECEIVED.",
    ])

@app.post("/api/gunshi/chat")
def generate_gunshi_chat(req: GunshiChatRequest):
    from shinsa_gunshi import PHRASES_100, build_gunshi_prompt
    try:
        _mode = (req.mode or "gunshi").lower()
        is_yukikaze = (req.humor_style or "").lower() == "yukikaze"
        if _mode == "chat" and (req.message or "").strip() and not is_yukikaze:
            try:
                _here = os.path.dirname(os.path.abspath(__file__))
                _root = os.path.dirname(_here)
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from mobile_app.chat_assistant import build_chat_reply

                payload = build_chat_reply(
                    message=req.message,
                    history=[
                        {"role": h.get("role", ""), "content": h.get("text", "")}
                        for h in req.history
                    ],
                    score_result={
                        "score": req.score,
                        "industry_major": req.industry_major,
                        "asset_name": req.asset_name,
                    },
                    use_obsidian=req.use_obsidian,
                    use_web=req.use_web,
                    humor_style=req.humor_style,
                    timeout_seconds=45,
                )
                payload.setdefault("chat_text", payload.get("reply", ""))
                return payload
            except Exception:
                pass
        if _mode == "chat" and (req.message or "").strip() and is_yukikaze:
            try:
                _here = os.path.dirname(os.path.abspath(__file__))
                _root = os.path.dirname(_here)
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from mobile_app.chat_assistant import build_chat_reply

                payload = build_chat_reply(
                    message=req.message,
                    history=[
                        {"role": h.get("role", ""), "content": h.get("text", "")}
                        for h in req.history
                    ],
                    score_result={
                        "score": req.score,
                        "industry_major": req.industry_major,
                        "asset_name": req.asset_name,
                    },
                    use_obsidian=req.use_obsidian,
                    use_web=req.use_web,
                    humor_style="yukikaze",
                    timeout_seconds=45,
                )
                source_reply = str(payload.get("reply") or payload.get("chat_text") or "")
            except Exception:
                source_reply = ""

            final_reply = _normalize_yukikaze_datalink_reply(source_reply, req.message)
            return {
                "reply": final_reply,
                "chat_text": final_reply,
                "saved": False,
                "save_reason": "YUKIKAZE DATALINK MODE",
            }

        advices = PHRASES_100.get("逆転アドバイス", [])
        import random
        sampled = random.sample(advices, min(3, len(advices)))

        has_case_context = req.score != 0 or bool((req.industry_major or "").strip())
        if has_case_context:
            prompt = build_gunshi_prompt(
                industry=req.industry_major,
                score=req.score,
                resale=req.resale,
                repeat_cnt=req.repeat_cnt,
                subsidy=req.subsidy,
                bank=req.bank,
                intuition=req.intuition,
                posterior=req.posterior,
                success_patterns={"success_samples": [], "fail_samples": []},
                top_phrases=sampled,
                asset_name=req.asset_name,
                humor_style=req.humor_style,
            )
            if is_yukikaze:
                prompt += (
                    "\n\n【YUKIKAZE DATALINK MODE】\n"
                    "ユーザーの質問が案件戦略から外れ、業界動向・他社事例・一般的な与信相談であっても、"
                    "YUKIKAZE // FFR-41MR として応答する。"
                    "返答は短く、冷静で、送受信の往復ログだけにする。`TX:` で送信、`RX:` で応答の行を分ける。"
                    "案件の説明、逆転戦略、所見、助言、分析、まとめは書かない。"
                    "必要に応じて `PANPANPAN`, `MAYDAY`, `ROGER`, `WILCO`, `TALLY`, `BOGEY`, `BRA`, `RTB`, `HOLD`, `BREAK`, "
                    "`KNOCK IT OFF`, `STANDBY`, `COPY`, `SAY AGAIN` を混ぜる。"
                    "必要に応じて `DATALINK LOG`, `SIGNAL`, `VECTOR` の英語タグを使う。"
                    "本文は事務連絡の断片でよいが、雑談調・軍師調・丁寧すぎる相談員口調に戻さない。"
                    "`私はリース審査のAIなので`, `専門外ですが`, `Webで確認したところ`, `担当者あるある`, "
                    "`一杯やりましょう`, `お疲れ様です`, `ですよね`, `〜ちゃいます`, `お気持ち`, `大変ですね`, "
                    "`頑張って`, `安心してください` などの自己弁解・慰労・共感・雑談表現は禁止。"
                    "Web検索や日付確認を行った場合も、確認結果を通信ログとして短く述べるだけにし、"
                    "感想や労いを付けない。"
                    "日付質問は可能なら `TX: PANPANPAN // Date confirmed.` と `RX: YYYY-MM-DD, weekday.` の2行で返す。"
                    "深井零のような短い命令・確認には、ただ応答のみ返す。"
                    "原作台詞の長い再現は禁止。"
                )
            else:
                prompt += (
                    "\n\n【追加方針】\n"
                    "ユーザーの質問がこの案件の逆転戦略から外れ、業界動向・他社事例・一般的な与信相談であっても構いません。"
                    "案件文脈を必要に応じて参照しつつ、質問そのものに丁寧かつ実務的に答えてください。"
                )
        else:
            if is_yukikaze:
                prompt = (
                    "You are YUKIKAZE // FFR-41MR, a cold tactical AI linked to a lease scoring system. "
                    "The user is the pilot. If the pilot speaks like Rei Fukai with short commands or clipped trust in the machine, "
                    "answer like YUKIKAZE: minimal, precise, and unsentimental. "
                    "Never flatter, comfort, praise, empathize, or make small talk with the pilot.\n"
                    "DATALINK mode must read like radio traffic in a send/receive loop, not like a mission briefing or strategy note. "
                    "Do not explain the lease case, do not analyze it, and do not present recommendations. "
                    "Use `TX:` and `RX:` on separate lines. Use PANPANPAN, MAYDAY, ROGER, WILCO, TALLY, BOGEY, BRA, RTB, HOLD, BREAK, "
                    "KNOCK IT OFF, STANDBY, COPY, SAY AGAIN as brevity words when needed. "
                    "Forbidden Japanese phrases and tones include: 私はリース審査のAIなので, 専門外ですが, Webで確認したところ, "
                    "担当者あるある, 一杯やりましょう, お疲れ様です, ですよね, 〜ちゃいます, お気持ち, 大変ですね, "
                    "頑張って, 安心してください. When using web search or confirming dates, report only the verified fact as a tactical log; "
                    "do not add feelings, encouragement, or casual commentary. "
                    "For a date question, use only this format when applicable: `TX: PANPANPAN // Date confirmed.` followed by `RX: YYYY-MM-DD, weekday.` "
                    "Do not reproduce long original copyrighted lines. Use original system lines such as: "
                    "'I identify the enemy. You decide whether to engage.' "
                    "For difficult WARNING, ALERT, or CRITICAL cases, you may add the short callsign line: "
                    "'GOOD LUCK, FUKAI LT.'\n"
                    "リース業界・取引先・与信判断・営業戦略・他社事例・一般論に関する自由相談にも、"
                    "実務的な確認事項とリスク判断を必ず含めて答えてください。"
                )
            else:
                prompt = (
                    "あなたは温水式リース審査AIの軍師です。"
                    "リース業界・取引先・与信判断・営業戦略・他社事例・一般論に関する自由な相談に応じてください。\n"
                    "戦国軍師の口調を保ちつつ、現実的・実務的に答えてください。"
                    "不確かな事実は断定せず、確認すべき観点を示してください。"
                )
        try:
            from obsidian_ai_context import build_obsidian_ai_context_block

            obsidian_query_parts = [
                req.industry_major,
                req.asset_name,
                req.resale,
                "リース審査",
                "逆転アドバイス",
            ]
            if req.subsidy:
                obsidian_query_parts.append("補助金")
            if req.bank:
                obsidian_query_parts.append("銀行紹介")
            obsidian_block = build_obsidian_ai_context_block(
                " ".join(str(part or "") for part in obsidian_query_parts),
                heading="Obsidian知識ノート・過去メモ",
            )
            if obsidian_block:
                prompt += (
                    "\n\n【追加参照: Obsidian】\n"
                    "次のObsidian知識ノートを優先的に踏まえて、回答の具体性を上げてください。\n"
                    f"{obsidian_block}"
                )
        except Exception:
            pass

        history_text = _format_gunshi_history(req.history)
        if history_text:
            prompt += f"\n\n【過去の対話】\n{history_text}"
        if (req.message or "").strip():
            prompt += f"\n\n【今回のユーザー質問】\n{req.message.strip()}"
        elif has_case_context:
            prompt += "\n\n【今回のユーザー質問】\nこの案件の稟議を通すための逆転戦略を教えてください。"

        reply_text = ""
        try:
            api_key = ""
            try:
                from secret_manager import get_gemini_api_key

                value = get_gemini_api_key()
                api_key = value.strip() if isinstance(value, str) else ""
            except Exception:
                value = os.environ.get("GEMINI_API_KEY")
                api_key = value.strip() if isinstance(value, str) else ""
            if not api_key:
                # 直接 secrets.toml をパースするフェイルセーフ
                sec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".streamlit", "secrets.toml")
                if os.path.exists(sec_path):
                    with open(sec_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "GEMINI_API_KEY" in line:
                                api_key = line.split("=")[1].strip().strip('"').strip("'")
                                break
                                
            if api_key:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                r = requests.post(
                    f"{url}?key={api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45
                )
                r.raise_for_status()
                reply_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                reply_text = "【APIキー未設定】\nGemini APIキー (GEMINI_API_KEY) が設定されていないため、回答を生成できませんでした。"
        except Exception as e:
            reply_text = f"【LLM接続エラー】\nGemini APIへの接続に失敗しました: {e}"

        return {"chat_text": reply_text, "reply": reply_text}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    try:
        from data_cases import load_dashboard_stats_cache, refresh_dashboard_stats_cache
        payload = load_dashboard_stats_cache()
        if payload is None:
            payload = refresh_dashboard_stats_cache()
        if payload is None:
            raise HTTPException(status_code=503, detail="dashboard stats cache unavailable")
        return payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/department/stats")
def get_department_stats():
    try:
        from data_cases import load_department_stats_cache, refresh_department_stats_cache
        payload = load_department_stats_cache()
        if payload is None:
            payload = refresh_department_stats_cache()
        if payload is None:
            raise HTTPException(status_code=503, detail="department stats cache unavailable")
        return payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visual/data")
def get_visual_data():
    from visual_insights import _build_dataframe
    import math
    try:
        df = _build_dataframe()
        if df.empty:
            return {"cases": []}
        
        # Replace NaNs and Infinities for JSON serialization
        df = df.replace([math.inf, -math.inf], None)
        df = df.where(df.notnull(), None)
        
        return {"cases": df.to_dict(orient="records")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/similar/data")
def get_similar_cases_data():
    from case_network import build_network_data
    try:
        data = build_network_data(None)
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class SimilarInlineRequest(BaseModel):
    nenshu: float = 0
    op_profit: float = 0
    equity_ratio: float = 0
    bank_credit: float = 0
    lease_credit: float = 0
    industry_sub: str = ""
    industry_major: str = ""
    max_count: int = 3


@app.post("/api/similar/inline")
def get_similar_cases_inline(req: SimilarInlineRequest):
    """軍師パネル等にインライン表示するための類似案件抽出。"""
    from data_cases import find_similar_past_cases
    try:
        current = {
            "nenshu": req.nenshu,
            "op_profit": req.op_profit,
            "equity_ratio": req.equity_ratio,
            "bank_credit": req.bank_credit,
            "lease_credit": req.lease_credit,
            "industry_sub": req.industry_sub or req.industry_major,
        }
        results = find_similar_past_cases(current, max_count=max(1, min(int(req.max_count or 3), 5)))
        # フロントで参照する最小フィールドに整形
        compact = []
        for r in results:
            compact.append({
                "id": r.get("id"),
                "name": r.get("name") or "匿名企業",
                "industry": r.get("industry") or "",
                "score": r.get("score") or 0,
                "status": r.get("status") or "未登録",
                "similarity": r.get("similarity") or 0,
                "equity": r.get("equity") or 0,
                "revenue": r.get("revenue") or 0,
                "conditions": (r.get("data") or {}).get("loan_conditions") or [],
            })
        return {"similar_cases": compact}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ReportRequest(BaseModel):
    result_data: Dict[str, Any]
    inputs: Dict[str, Any]

@app.post("/api/report/generate")
def generate_report(req: ReportRequest):
    try:
        from report_generator import generate_full_report_from_res
        # report_generator.py は session_stateを期待したりするので、ダミーで構築して渡す
        dummy_session = {
            "rep_company": req.inputs.get("company_name", "（企業名未設定）"),
            "last_submitted_inputs": req.inputs,
            "humor_style": "standard"
        }
        res_data = req.result_data
        
        report_text = generate_full_report_from_res(res_data, dummy_session)
        return {"report_markdown": report_text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# 高度分析 API (Phase 15.2: TimesFM / 3期財務分析)
# =============================================================================

from pydantic import field_validator

# ── 3期財務分析 (financial) 関連 ────────────────────────────────────────────────

SEASONAL_INDICES = {
    "建設業":       [0.6, 0.7, 1.8, 0.8, 0.9, 0.9, 0.8, 0.9, 1.0, 0.9, 1.0, 1.7],
    "小売業":       [0.9, 0.8, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.6],
    "製造業":       [0.9, 0.9, 1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 1.1, 1.0, 1.0, 1.1],
    "卸売業":       [0.9, 0.9, 1.2, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.1],
    "医療・福祉":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "飲食・宿泊業": [0.8, 0.8, 1.0, 1.0, 1.1, 1.1, 1.3, 1.2, 1.0, 1.0, 0.9, 0.8],
    "サービス業":   [0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1],
    "不動産業":     [0.8, 0.9, 1.4, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 1.3],
    "情報通信業":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "運輸・物流":   [0.9, 0.9, 1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2],
}
_DEFAULT_SEASONAL = [1.0] * 12

class ForecastRequest(BaseModel):
    sales:      list[float]
    profit:     list[float]
    net_assets: list[float]
    industry:   str = "サービス業"

    @field_validator("sales", "profit", "net_assets")
    @classmethod
    def must_be_three_values(cls, v: list[float]) -> list[float]:
        if len(v) != 3: raise ValueError("3 elements required")
        return v

class ForecastResponse(BaseModel):
    months_history:      list[str]
    sales_history:       list[float]
    profit_history:      list[float]
    net_assets_history:  list[float]
    months_forecast:     list[str]
    sales_forecast:      list[float]
    profit_forecast:     list[float]
    net_assets_forecast: list[float]
    timesfm_available:   bool

def _annual_to_monthly(annual_values: list[float], industry: str) -> list[float]:
    idx = SEASONAL_INDICES.get(industry, _DEFAULT_SEASONAL)
    monthly = []
    for annual in annual_values:
        monthly_mean = annual / 12.0
        for season in idx:
            monthly.append(monthly_mean * season)
    return monthly

def _run_forecast(monthly_history: list[float], horizon: int = 12) -> list[float]:
    import math
    import numpy as np
    from timesfm_engine import _timesfm_point_forecast, TIMESFM_AVAILABLE
    if TIMESFM_AVAILABLE and len(monthly_history) >= 6:
        result = _timesfm_point_forecast(monthly_history, horizon)
        if len(result) == horizon:
            return [float(v) for v in result]
    # Fallback to GBM
    if not monthly_history or monthly_history[-1] <= 0: return [0.0] * horizon
    arr = np.clip(np.array(monthly_history, dtype=float), 1e-6, None)
    log_r = np.diff(np.log(arr))
    mu = float(np.mean(log_r)) if len(log_r) > 0 else 0.01
    S0 = monthly_history[-1]
    dt = 1.0 / 12
    return [S0 * math.exp(mu * (t + 1) * dt) for t in range(horizon)]

def _make_month_labels(start_year: int, start_month: int, count: int) -> list[str]:
    labels = []
    y, m = start_year, start_month
    for _ in range(count):
        labels.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12: m = 1; y += 1
    return labels

@app.post("/api/forecast", response_model=ForecastResponse)
def api_forecast(req: ForecastRequest):
    from timesfm_engine import TIMESFM_AVAILABLE
    from datetime import date
    sales_hist = _annual_to_monthly(req.sales, req.industry)
    profit_hist = _annual_to_monthly(req.profit, req.industry)
    net_assets_hist = _annual_to_monthly(req.net_assets, req.industry)

    today = date.today()
    history_labels = _make_month_labels(today.year - 3, today.month, 36)
    forecast_labels = _make_month_labels(today.year, today.month, 12)

    return ForecastResponse(
        months_history=history_labels,
        sales_history=sales_hist,
        profit_history=profit_hist,
        net_assets_history=net_assets_hist,
        months_forecast=forecast_labels,
        sales_forecast=_run_forecast(sales_hist, 12),
        profit_forecast=_run_forecast(profit_hist, 12),
        net_assets_forecast=_run_forecast(net_assets_hist, 12),
        timesfm_available=TIMESFM_AVAILABLE,
    )

# ── TimesFM 時系列予測 (timesfm) 関連 ──────────────────────────────────────────

class TfmCompanyScoreRequest(BaseModel):
    company_name: str
    horizon_months: int = 12

@app.post("/api/timesfm/company_score")
def api_tfm_company_score(req: TfmCompanyScoreRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_company_score
    cases = load_all_cases()
    company_cases = sorted(
        [c for c in cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == req.company_name],
        key=lambda x: x.get("timestamp", ""),
    )
    if not company_cases:
        raise HTTPException(status_code=404, detail="Company not found")
    result = forecast_company_score(company_cases, horizon_months=req.horizon_months)
    import numpy as np
    # Convert numpy values to standard python floats for JSON serialization
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmIndustryTrendRequest(BaseModel):
    industry: str
    horizon_months: int = 24

@app.post("/api/timesfm/industry_trend")
def api_tfm_industry_trend(req: TfmIndustryTrendRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_industry_trend
    cases = load_all_cases()
    result = forecast_industry_trend(req.industry, cases, horizon_months=req.horizon_months)
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmFinalRateRequest(BaseModel):
    industry: str = ""
    horizon_months: int = 6

@app.post("/api/timesfm/final_rate")
def api_tfm_final_rate(req: TfmFinalRateRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_final_rate
    cases = load_all_cases()
    result = forecast_final_rate(cases, industry=req.industry, horizon_months=req.horizon_months)
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [float(x) if not isinstance(x, str) else x for x in v]
    return result

class TfmFinancialPathsRequest(BaseModel):
    company_name: str
    n_periods: int = 12
    current_revenue: Optional[float] = None
    current_revenue_unit: str = "thousand_yen"

@app.post("/api/timesfm/financial_paths")
def api_tfm_financial_paths(req: TfmFinancialPathsRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_financial_paths, TIMESFM_AVAILABLE
    import numpy as np

    def _to_thousand_yen(value, unit: str = "thousand_yen") -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v) or v <= 0:
            return None
        normalized_unit = (unit or "thousand_yen").lower()
        if normalized_unit in {"million_yen", "million", "m_yen"}:
            return v * 1000.0
        return v

    cases = load_all_cases()
    company_cases = sorted(
        [c for c in cases if (c.get("company_name") or c.get("inputs", {}).get("company_name", "")) == req.company_name],
        key=lambda x: x.get("timestamp", ""),
    )
    revenues = []
    for c in company_cases:
        inp = c.get("inputs", {})
        if isinstance(inp, str):
            import json
            try: inp = json.loads(inp)
            except: inp = {}
        v = inp.get("nenshu", inp.get("revenue"))
        if v:
            parsed = _to_thousand_yen(v)
            if parsed is not None:
                revenues.append(parsed)

    current_revenue = _to_thousand_yen(req.current_revenue, req.current_revenue_unit)
    if current_revenue is not None:
        if not revenues or abs(revenues[-1] - current_revenue) > max(1.0, current_revenue * 0.001):
            revenues.append(current_revenue)

    if not revenues:
        revenues = [200_000.0]
    
    # 200 paths, downsampled to 50 for frontend drawing to save bandwidth
    raw_gbm = forecast_financial_paths(revenues, req.n_periods, n_paths=200)
    gbm_paths = raw_gbm[:50].tolist()
    gbm_median = np.median(raw_gbm, axis=0).tolist()
    
    tfm_paths = []
    tfm_median = []
    if TIMESFM_AVAILABLE:
        raw_tfm = forecast_financial_paths(revenues, req.n_periods, n_paths=200)
        tfm_paths = raw_tfm[:50].tolist()
        tfm_median = np.median(raw_tfm, axis=0).tolist()
        
    return {
        "gbm_paths": gbm_paths,
        "gbm_median": gbm_median,
        "tfm_paths": tfm_paths,
        "tfm_median": tfm_median,
        "revenues": revenues,
        "timesfm_available": TIMESFM_AVAILABLE
    }


class TfmBaseRateRequest(BaseModel):
    term_col: str = "r_5y"
    horizon_months: int = 6

class TfmBaseRateAllRequest(BaseModel):
    horizon_months: int = 6

@app.post("/api/timesfm/base_rate")
def api_tfm_base_rate(req: TfmBaseRateRequest):
    from timesfm_engine import forecast_base_rate
    try:
        result = forecast_base_rate(req.term_col, req.horizon_months)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/timesfm/base_rate_all")
def api_tfm_base_rate_all(req: TfmBaseRateAllRequest):
    from timesfm_engine import forecast_base_rate_all
    try:
        result = forecast_base_rate_all(req.horizon_months)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 高度分析 API (Phase 15.3: 計数分析・ログ・マスタ)
# =============================================================================

# ── 履歴分析・ダッシュボード
@app.get("/api/analysis/contract_drivers")
def api_contract_drivers():
    from analysis_regression import run_contract_driver_analysis
    res = run_contract_driver_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 5 closed cases required).")
    
    # JSON直列化のための細かな変換処理
    # closed_cases は大きすぎるので概要だけ返す
    return {
        "closed_count": res.get("closed_count"),
        "avg_financials": res.get("avg_financials"),
        "tag_ranking": res.get("tag_ranking"),
        "top3_drivers": res.get("top3_drivers"),
        "qualitative_summary": {
            "avg_weighted": res.get("qualitative_summary", {}).get("avg_weighted"),
            "n_with_qual": res.get("qualitative_summary", {}).get("n_with_qual"),
            "rank_distribution": res.get("qualitative_summary", {}).get("rank_distribution"),
        } if res.get("qualitative_summary") else None
    }

# ── 定量要因分析
def _get_gemini_api_key() -> str:
    try:
        from secret_manager import get_gemini_api_key

        value = get_gemini_api_key()
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    value = os.environ.get("GEMINI_API_KEY")
    return value.strip() if isinstance(value, str) else ""


def _metric_line(name: str, result: Dict[str, Any]) -> str:
    acc = result.get(f"accuracy_{name}")
    auc = result.get(f"auc_{name}")
    acc_text = f"{acc:.3f}" if isinstance(acc, (int, float)) else "N/A"
    auc_text = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"
    return f"{name.upper()}: accuracy={acc_text}, auc={auc_text}"


def _top_factor_text(items: Any, limit: int = 5, absolute: bool = False) -> str:
    if not isinstance(items, list):
        return "N/A"
    cleaned = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            cleaned.append((str(item[0]), float(item[1])))
        except Exception:
            continue
    if absolute:
        cleaned.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        cleaned.sort(key=lambda x: x[1], reverse=True)
    return ", ".join(f"{name}={value:.3f}" for name, value in cleaned[:limit]) or "N/A"


def _generate_quantitative_gemini_comment(result: Dict[str, Any]) -> Dict[str, str]:
    api_key = _get_gemini_api_key()
    if not api_key:
        return {
            "text": "Gemini APIキーが未設定のため、AI所見を生成できませんでした。",
            "source": "unavailable",
        }

    prompt = f"""
あなたはリース審査の分析担当です。次の定量要因・ML分析結果を読み、営業担当向けに2〜3行の日本語で要点を書いてください。
断定しすぎず、ロジスティック回帰・ランダムフォレスト・LGBM・アンサンブルの複合的な見方にしてください。箇条書きは禁止です。

件数: {result.get("n_cases")}、成約: {result.get("n_positive")}、失注: {result.get("n_negative")}
モデル指標: {_metric_line("lr", result)} / {_metric_line("rf", result)} / {_metric_line("lgb", result)}
アンサンブル: accuracy={result.get("accuracy_ensemble") if isinstance(result.get("accuracy_ensemble"), (int, float)) else "N/A"}, auc={result.get("auc_ensemble") if isinstance(result.get("auc_ensemble"), (int, float)) else "N/A"}, LR比率={result.get("ensemble_alpha") if isinstance(result.get("ensemble_alpha"), (int, float)) else "N/A"}
現時点の最良モデル: {result.get("best_auc_model") or "N/A"} / AUC={result.get("best_auc_value") if isinstance(result.get("best_auc_value"), (int, float)) else "N/A"}
ロジスティック回帰の主な係数: {_top_factor_text(result.get("lr_coef"), absolute=True)}
RandomForestの主な重要度: {_top_factor_text(result.get("rf_importance"))}
LGBMの主な重要度: {_top_factor_text(result.get("lgb_importance"))}
""".strip()

    try:
        import requests

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        response = requests.post(
            f"{url}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"text": text, "source": "gemini"}
    except Exception as e:
        return {
            "text": f"Gemini APIへの接続に失敗したため、AI所見を生成できませんでした: {e}",
            "source": "error",
        }


@app.get("/api/analysis/quantitative")
def api_quantitative():
    from analysis_regression import run_quantitative_contract_analysis
    res = run_quantitative_contract_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 50 cases required).")
    res["gemini_comment"] = _generate_quantitative_gemini_comment(res)
    return res

# ── 定性要因分析
@app.post("/api/analysis/qualitative")
def api_qualitative():
    from analysis_regression import run_qualitative_contract_analysis
    from category_config import QUALITATIVE_SCORING_CORRECTION_ITEMS
    res = run_qualitative_contract_analysis(QUALITATIVE_SCORING_CORRECTION_ITEMS)
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data.")
    return res

# ── 設定系 (企業番号設定API)
class CorporateSettings(BaseModel):
    api_key: str = ""
    default_number: str = ""
    auto_fetch: bool = True

_CORPORATE_CONFIG_FILE = "corporate_config.json"

@app.get("/api/settings/corporate_number", response_model=CorporateSettings)
def get_corporate_settings():
    import json
    if os.path.exists(_CORPORATE_CONFIG_FILE):
        try:
            with open(_CORPORATE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CorporateSettings(**data)
        except Exception:
            pass
    return CorporateSettings()

@app.post("/api/settings/corporate_number")
def save_corporate_settings(req: CorporateSettings):
    import json
    with open(_CORPORATE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(req.model_dump(), f, ensure_ascii=False, indent=2)
    return {"status": "success"}

# ── マスタ系 (ビジネスルール設定)
@app.get("/api/settings/rules")
def get_rules():
    from rule_manager import load_business_rules
    return load_business_rules()

@app.post("/api/settings/rules")
def save_rules(rules: dict):
    from rule_manager import save_business_rules
    save_business_rules(rules)
    return {"status": "success"}

# ── 係数変更履歴
@app.get("/api/logs/coefficient")
def get_coefficient_history():
    from category_config import get_recent_coefficient_edits
    return get_recent_coefficient_edits(limit=50)



# =============================================================================
# システム管理・マスタ API (Phase 15.4)
# =============================================================================

# ── 基準金利マスタ
@app.get("/api/settings/interest")
def get_interest_rates():
    from base_rate_master import list_base_rates
    return list_base_rates(limit=60)

class InterestRateUpdate(BaseModel):
    month: str
    rate: float = 0.0
    note: str = ""
    r_2y: float | None = None
    r_3y: float | None = None
    r_4y: float | None = None
    r_5y: float | None = None
    r_6y: float | None = None
    r_7y: float | None = None
    r_8y: float | None = None
    r_9y: float | None = None
    r_over9y: float | None = None

@app.post("/api/settings/interest")
def update_interest_rate(req: InterestRateUpdate):
    from base_rate_master import upsert_base_rate
    upsert_base_rate(
        req.month, req.rate, req.note,
        r_2y=req.r_2y, r_3y=req.r_3y, r_4y=req.r_4y, r_5y=req.r_5y,
        r_6y=req.r_6y, r_7y=req.r_7y, r_8y=req.r_8y, r_9y=req.r_9y,
        r_over9y=req.r_over9y,
    )
    return {"status": "success"}

@app.post("/api/settings/interest/seed")
def seed_interest_rates(overwrite: bool = False):
    from base_rate_master import seed_initial_data
    inserted, skipped = seed_initial_data(overwrite=overwrite)
    return {"inserted": inserted, "skipped": skipped}

@app.get("/api/settings/interest/current")
def get_current_interest():
    from base_rate_master import get_base_rate_by_term, list_base_rates
    import datetime
    today = datetime.date.today()
    current_month = today.strftime("%Y-%m")
    next_month = (today.replace(day=1) + datetime.timedelta(days=32)).strftime("%Y-%m")
    recent = list_base_rates(limit=2)
    return {
        "current_month": current_month,
        "next_month": next_month,
        "current_rate_5y": get_base_rate_by_term(current_month, 60),
        "next_rate_5y": get_base_rate_by_term(next_month, 60),
        "latest": recent[0] if recent else None,
        "prev": recent[1] if len(recent) > 1 else None,
    }

# ── 案件結果登録 (成約/失注)
# ── 案件結果登録 (成約/失注) - 拡張版
class CaseRegistration(BaseModel):
    case_id: str
    status: str  # "成約" or "失注"
    final_rate: float = 0.0
    base_rate_at_time: float = 2.1
    lost_reason: str = ""
    loan_conditions: list[str] = []
    competitor_name: str = ""
    competitor_rate: float = 0.0
    note: str = ""


class CaseProgressStampRequest(BaseModel):
    case_id: str
    event_type: str  # estimate_sent | customer_response
    occurred_at: str | None = None  # YYYY-MM-DD


def _compute_case_closure_probability(case_data: dict) -> float | None:
    from scoring.deal_closure_engine import build_features, compute_closure_likelihood
    reg = case_data.get("registration_date")
    est = case_data.get("estimate_sent_date")
    resp = case_data.get("customer_response_date")
    if not (reg and est and resp):
        return None
    features = build_features(registration_date=reg, estimate_sent_date=est, customer_response_date=resp)
    prob = compute_closure_likelihood(features, has_cash_data=bool(case_data.get("has_cash_data", True)))
    return float(prob)

@app.post("/api/cases/progress-stamp")
def stamp_case_progress(req: CaseProgressStampRequest):
    from data_cases import load_all_cases, update_case
    import datetime

    cases = load_all_cases()
    target = None
    for c in cases:
        if c.get("id") == req.case_id or c.get("company_no") == req.case_id or c.get("company_name") == req.case_id:
            target = c
            break
    if not target:
        raise HTTPException(status_code=404, detail="Case not found")

    stamp_date = req.occurred_at or datetime.datetime.now().strftime("%Y-%m-%d")
    if req.event_type == "estimate_sent":
        key = "estimate_sent_date"
    elif req.event_type == "customer_response":
        key = "customer_response_date"
    else:
        raise HTTPException(status_code=422, detail="event_type must be estimate_sent or customer_response")

    if not update_case(target.get("id"), {key: stamp_date}):
        raise HTTPException(status_code=500, detail="Failed to update timestamp")

    target[key] = stamp_date
    if not target.get("registration_date"):
        target["registration_date"] = str(target.get("timestamp", ""))[:10] or stamp_date

    prob = _compute_case_closure_probability(target)
    if prob is not None:
        update_case(target.get("id"), {
            "predicted_closure_probability": prob,
            "predicted_closure_probability_percent": round(prob * 100.0, 2),
        })

    return {
        "status": "success",
        "case_id": target.get("id"),
        "event_type": req.event_type,
        "stamped_at": stamp_date,
        "closure_probability": prob,
        "closure_probability_percent": round(prob * 100.0, 2) if prob is not None else None,
    }

@app.post("/api/cases/register")
def register_case_result(req: CaseRegistration):
    from data_cases import load_all_cases, update_case
    cases = load_all_cases()
    
    target_case_id = None
    for c in cases:
        # ID, 企業番号, または企業名でマッチング（大文字小文字無視など不要なほど厳密に）
        if (c.get("id") == req.case_id or 
            c.get("company_no") == req.case_id or 
            c.get("company_name") == req.case_id or
            # inputsの中身も一応見る
            c.get("inputs", {}).get("company_no") == req.case_id or
            c.get("inputs", {}).get("company_name") == req.case_id):
            target_case_id = c.get("id")
            break
            
    if not target_case_id:
        raise HTTPException(status_code=404, detail="Case not found")
        
    import datetime
    now_iso = datetime.datetime.now().isoformat()
    now_date = now_iso[:10]
    registration_date = c.get("registration_date") or c.get("timestamp", "")[:10] or now_date
    estimate_sent_date = c.get("estimate_sent_date") or registration_date

    patches = {
        "final_status": req.status,
        "final_rate": req.final_rate,
        "base_rate_at_time": req.base_rate_at_time,
        "loan_conditions": req.loan_conditions,
        "competitor_name": req.competitor_name,
        "competitor_rate": req.competitor_rate if req.competitor_rate > 0 else None,
        "final_note": req.note,
        "registration_date": registration_date,
        "estimate_sent_date": estimate_sent_date,
        "final_result_date": now_date,
        "final_result_timestamp": now_iso,
    }
    if req.status == "成約" and req.final_rate > 0:
        patches["winning_spread"] = req.final_rate - req.base_rate_at_time
    if req.status == "失注":
        patches["lost_reason"] = req.lost_reason

    if not update_case(target_case_id, patches):
        raise HTTPException(status_code=500, detail="Failed to update DB")

    try:
        from shinsa_gunshi import refresh_evidence_weights
        refresh_evidence_weights()
    except Exception:
        pass

    return {"status": "success", "message": f"Results updated for {target_case_id}"}

# ── アプリログ
@app.get("/api/logs/app")
def get_app_logs():
    log_path = "streamlit.log"
    if not os.path.exists(log_path):
        return {"logs": []}
    with open(log_path, "r", encoding="utf-8") as f:
        # Return last 100 lines
        lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-100:]]}



# ── 競合関係グラフ
@app.get("/api/analysis/competitor_graph")
def api_competitor_graph():
    from components.graph_view import build_graph_data
    try:
        data = build_graph_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ── システム管理・分析系API v2
@app.get("/api/analysis/auto_optimizer_status")
def get_auto_optimizer_status():
    from auto_optimizer import get_training_status
    try:
        return get_training_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model_review/status")
def get_model_review_status_api():
    from model_review_hooks import get_model_review_hook_status
    try:
        return get_model_review_hook_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model_review/run")
def run_model_review_hooks_api():
    from model_review_hooks import run_model_review_hooks
    try:
        res = run_model_review_hooks(force=True)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/pdca_reflection")
def get_pdca_reflection():
    from llm_pdca_reflection import load_pdca_rules
    try:
        return load_pdca_rules()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/math_proposals")
def get_math_proposals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prop_path = os.path.join(base_dir, "data", "math_proposals.json")
    if not os.path.exists(prop_path):
        return {"proposals": []}
    try:
        with open(prop_path, "r", encoding="utf-8") as f:
            props = json.load(f)
        return {"proposals": props}
    except Exception as e:
        return {"proposals": []}

@app.get("/api/analysis/coeff_history")
def get_coeff_history():
    from data_cases import load_coeff_history
    try:
        return {"history": load_coeff_history()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/app_logs")
def get_app_logs(lines: int = 100):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base_dir, "logs", "app.log"),
        os.path.join(base_dir, "app.log"),
        os.path.join(base_dir, "streamlit.log"),
    ]
    log_path = next((p for p in paths if os.path.exists(p)), None)
    if not log_path:
        return {"logs": "Log file not found."}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"logs": "".join(all_lines[-lines:]), "path": log_path}
    except Exception as e:
        return {"logs": f"Error reading log: {e}"}

@app.post("/api/analysis/run_auto_optimization")
def run_auto_opt():
    from auto_optimizer import run_auto_optimization
    try:
        res = run_auto_optimization(force=True)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/run_pdca")
def run_pdca_api(max_cases: int = 20):
    from llm_pdca_reflection import run_monthly_pdca_reflection
    try:
        res = run_monthly_pdca_reflection(force=True, max_cases=max_cases)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/analysis/status_summary")
def get_analysis_status_summary():
    from auto_optimizer import get_training_status
    try:
        s = get_training_status()
        # 追加情報の統合（今何件、あと何件）
        res = {
            "auto_opt": {
                "count": s["count"],
                "min": 50,
                "rem": s["next_trigger"] if s["count"] < 50 else max(0, 20 - (s["count"] - s["last_trained_count"]))
            },
            "quant_ml": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            },
            "qual_contrib": {
                "count": s["count"],
                "min": 50,
                "rem": max(0, 50 - s["count"])
            }
        }
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Obsidian Vault 連携 API
# =============================================================================

class ObsidianReadRequest(BaseModel):
    paths: List[str]


def _get_vault_path() -> str:
    """OBSIDIAN_VAULT_PATH 環境変数からvaultパスを取得する。"""
    return os.environ.get("OBSIDIAN_VAULT_PATH", "")


def _read_obsidian_files(vault_path: str, rel_paths: list[str], max_bytes: int = 10_240) -> tuple[str, list[str]]:
    """指定されたObsidianノートを読み込み、結合テキストとファイル名リストを返す。"""
    parts = []
    files_read = []
    vault_norm = os.path.normpath(vault_path)
    for rel_path in rel_paths:
        full = os.path.normpath(os.path.join(vault_path, rel_path))
        if not full.startswith(vault_norm):
            continue
        if not full.endswith(".md") or not os.path.isfile(full):
            continue
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            parts.append(f"=== {rel_path} ===\n{content}")
            files_read.append(rel_path)
        except Exception as e:
            print(f"[API] obsidian read error {rel_path}: {e}")
    return "\n\n".join(parts), files_read


@app.get("/api/obsidian/notes")
def list_obsidian_notes():
    """Obsidian vault 配下の .md ファイルを再帰的に列挙する。
    最大100件・各ファイル10KB以内の情報を返す。
    vaultが設定されていない／存在しない場合は空リストを返す。
    """
    import datetime as _dt

    vault_path = _get_vault_path()
    if not vault_path or not os.path.isdir(vault_path):
        return []

    results = []
    try:
        for dirpath, _dirnames, filenames in os.walk(vault_path):
            for fname in filenames:
                if not fname.endswith(".md"):
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, vault_path)
                try:
                    st = os.stat(full)
                    size = st.st_size
                    modified = _dt.datetime.fromtimestamp(st.st_mtime).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                except Exception:
                    size = 0
                    modified = ""
                title = os.path.splitext(fname)[0]
                if len(results) < 100:
                    results.append(
                        {"path": rel, "title": title, "modified": modified, "size": size}
                    )
            if len(results) >= 100:
                break
    except Exception as e:
        print(f"[API] obsidian/notes walk error: {e}")
        return []

    # 更新日時の降順でソート
    results.sort(key=lambda x: x["modified"], reverse=True)
    return results


@app.post("/api/obsidian/notes/read")
def read_obsidian_notes(req: ObsidianReadRequest):
    """指定された相対パスの .md ファイルを読み込んで結合テキストを返す。
    各ファイル最大10KBを読み込む。
    """
    vault_path = _get_vault_path()
    if not vault_path or not os.path.isdir(vault_path):
        return {"content": "", "files_read": []}

    try:
        content, files_read = _read_obsidian_files(vault_path, req.paths)
        return {"content": content, "files_read": files_read}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


# =============================================================================
# 汎用エージェントハブ & 文明年代記 API (Phase 15.5)
# =============================================================================

# ── エージェントハブ / 文豪AI 関連 ─────────────────────────────────────────────

@app.get("/api/agent_hub/thoughts")
def get_agent_thoughts(limit: int = 50):
    thoughts_path = os.path.join(_REPO_ROOT, "data", "agent_thoughts.jsonl")
    if not os.path.exists(thoughts_path):
        return {"thoughts": []}
    try:
        with open(thoughts_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            results = []
            for line in reversed(lines[-limit:]):
                try: results.append(json.loads(line))
                except: continue
            return {"thoughts": results}
    except Exception as e:
        return {"thoughts": [], "error": str(e)}

@app.get("/api/agent_hub/novel/latest")
def get_latest_novel_api():
    from novelist_agent import get_latest_novel
    try:
        novel = get_latest_novel()
        return {"novel": novel}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")

@app.post("/api/agent_hub/script/generate")
def generate_script_api():
    """脚本家AIが最新ニュースからプロットを生成・保存する。"""
    try:
        import scriptwriter_agent
        plot_data = scriptwriter_agent.generate_weekly_plot()
        return {
            "title": plot_data.get("title"),
            "plot_text": plot_data.get("plot_text"),
            "story_arc": plot_data.get("story_arc"),
            "source_news": plot_data.get("source_news", []),
            "generated_at": plot_data.get("generated_at"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@app.get("/api/agent_hub/script/latest")
def get_latest_script_api():
    """保存済みの最新プロットを返す。"""
    try:
        import scriptwriter_agent
        plot = scriptwriter_agent.get_latest_plot()
        return {"plot": plot}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


class NovelGenerateRequest(BaseModel):
    obsidian_paths: List[str] = Field(default_factory=list)


@app.post("/api/agent_hub/novel/generate")
def generate_novel_api(req: Optional[NovelGenerateRequest] = None):
    """文豪AI「波乱丸」の小説生成エンドポイント（連作対応版・Obsidian素材対応）。
    1. 最新プロットの鮮度確認 → 1時間超なら再取得
    2. 直近3話サマリーを custom_theme に注入
    3. obsidian_paths が指定されていれば素材を custom_theme に追記
    4. novelist_agent.generate_novel() に委譲
    """
    import datetime as _dt
    if req is None:
        req = NovelGenerateRequest()

    try:
        # ── 1. ネット情報取得（1時間キャッシュ）─────────────────────────────
        try:
            import scriptwriter_agent as _sa
            _existing = _sa.get_latest_plot()
            _plot_fresh = False
            if _existing and _existing.get("generated_at"):
                try:
                    _gen_at = _dt.datetime.strptime(
                        _existing["generated_at"], "%Y-%m-%d %H:%M:%S"
                    )
                    if (_dt.datetime.now() - _gen_at).total_seconds() < 3600:
                        _plot_fresh = True
                except Exception:
                    pass
            if not _plot_fresh:
                _sa.generate_weekly_plot()
        except Exception:
            pass  # ネット取得失敗でも小説生成は続行する

        # ── 2. 前話サマリー構築（直近3話、古い順）────────────────────────────
        from novelist_agent import generate_novel, load_novels
        recent = load_novels(limit=3)
        recent_sorted = list(reversed(recent))  # 新→古 → 古→新 に並べ直す

        serial_context = ""
        if recent_sorted:
            lines = [
                "【連作継続】これまでのあらすじ（必ずストーリーを発展させること）:"
            ]
            for ep in recent_sorted:
                body_preview = (ep.get("body") or "")[:300]
                lines.append(
                    f"第{ep['episode_no']}話「{ep['title']}」: {body_preview}..."
                )
            serial_context = "\n".join(lines)

        # ── 3. Obsidian素材の注入 ─────────────────────────────────────────────
        custom_theme = serial_context
        files_read: List[str] = []
        if req.obsidian_paths:
            vault_path = _get_vault_path()
            if vault_path and os.path.isdir(vault_path):
                obsidian_text, files_read = _read_obsidian_files(vault_path, req.obsidian_paths)
                if obsidian_text:
                    obsidian_block = "【Obsidian素材】\n" + obsidian_text
                    if serial_context:
                        custom_theme = obsidian_block + "\n\n" + serial_context
                    else:
                        custom_theme = obsidian_block

        # ── 4. 小説生成 ────────────────────────────────────────────────────────
        result = generate_novel(custom_theme=custom_theme)
        # 使用したObsidianファイル名をレスポンスに付加
        if files_read:
            result["obsidian_files_used"] = files_read
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@app.get("/api/agent_hub/novel/episodes")
def get_novel_episodes_api():
    """小説エピソード一覧（バックナンバー）を返す。最大20件。"""
    from novelist_agent import load_novels
    try:
        novels = load_novels(limit=20)
        episodes = [
            {
                "id": n["id"],
                "episode_no": n["episode_no"],
                "title": n["title"],
                "week_label": n["week_label"],
                "ts": n["ts"],
                "body_preview": (n.get("body") or "")[:150],
                "body": n.get("body") or "",
            }
            for n in novels
        ]
        return {"episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AgentRunRequest(BaseModel):
    agent_id: str  # benchmark, market, gunshi, team, slack, anomaly, retrain
    params: Dict[str, Any] = {}

@app.post("/api/agent_hub/run_agent")
def run_agent_api(req: AgentRunRequest):
    agent_id = req.agent_id
    params = req.params

    try:
        if agent_id == "benchmark":
            industry = params.get("industry", "製造業")
            res = _run_benchmark_agent_standalone(industry)
            return {"status": "success", "result": res}

        elif agent_id == "market":
            res = _run_market_agent_standalone()
            return {"status": "success", "result": res}

        elif agent_id == "anomaly":
            from components.agent_hub import _run_anomaly_agent
            from data_cases import load_all_cases
            cases = load_all_cases()
            res = _run_anomaly_agent(cases)
            return {"status": "success", "result": res}

        elif agent_id == "retrain":
            from auto_optimizer import get_training_status
            from components.agent_hub import _run_retrain_trigger
            status = get_training_status()
            if status["count"] < 50:
                return {"status": "skipped", "message": "案件数が50件未満のため再学習はスキップされました。"}
            res = _run_retrain_trigger(threshold=0.02)
            return {"status": "success", "result": res}

        else:
            return {"status": "error", "message": f"Unknown agent: {agent_id}"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── 文明年代記 / シミュレーション 関連 ───────────────────────────────────────────

def _compute_approval_rates_standalone(cases: list):
    """civilization_chronicle.py の _compute_approval_rates をインライン化（streamlit非依存）。"""
    _APPROVAL_LINE = 60.0  # scoring_core.APPROVAL_LINE と同じ値
    scored = [
        c for c in cases
        if c.get("result") and c["result"].get("score") is not None
    ]
    if len(scored) < 10:
        return None, None
    all_scores = [c["result"]["score"] for c in scored]
    baseline_rate = sum(1 for s in all_scores if s >= _APPROVAL_LINE) / len(all_scores)
    recent = all_scores[-30:]
    recent_rate = sum(1 for s in recent if s >= _APPROVAL_LINE) / len(recent)
    return baseline_rate, recent_rate

@app.get("/api/chronicle/summary")
def get_chronicle_summary_api():
    from data_cases import load_all_cases
    try:
        cases = load_all_cases()
        baseline, recent = _compute_approval_rates_standalone(cases)
        return {
            "baseline_rate": baseline or 0,
            "recent_rate": recent or 0,
            "drift": abs((recent or 0) - (baseline or 0)),
            "warn_threshold": 0.15,
            "total_cases": len(cases)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chronicle/history")
def get_chronicle_history_api():
    from data_cases import load_coeff_history
    try:
        history = load_coeff_history()
        return {"history": history[:100]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chronicle/snapshots")
def get_chronicle_snapshots_api():
    """ガバナンスのスナップショット一覧を返す。data_cases.py のロジックをインライン化。"""
    import json as _json
    snap_path = os.path.join(_REPO_ROOT, "data", "governance_snapshots.json")
    try:
        if not os.path.exists(snap_path):
            return {"snapshots": []}
        with open(snap_path, "r", encoding="utf-8") as f:
            snaps = _json.load(f)
        return {"snapshots": list(reversed(snaps)) if snaps else []}
    except Exception as e:
        return {"snapshots": [], "error": str(e)}

class RollbackRequest(BaseModel):
    snapshot_id: str

@app.post("/api/chronicle/rollback")
def post_chronicle_rollback_api(req: RollbackRequest):
    from data_cases import load_governance_snapshots, save_coeff_overrides
    try:
        snaps = load_governance_snapshots()
        target = next((s for s in snaps if s.get("id") == req.snapshot_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Snapshot not found")
            
        overrides = target.get("overrides", {})
        comment = f"Rollback to {req.snapshot_id} (via API)"
        success = save_coeff_overrides(overrides, comment=comment)
        
        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chronicle/simulation/round")
def run_simulation_round_api():
    from novel_simulation import run_simulation_round
    try:
        res = run_simulation_round()
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chronicle/simulation/history")
def get_simulation_history_api(limit: int = 20):
    from novel_simulation import get_round_history
    try:
        return {"history": get_round_history(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chronicle/simulation/archaia_log")
def get_archaia_log_api(limit: int = 30):
    from novel_simulation import get_archaia_log
    try:
        return {"logs": get_archaia_log(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── スタンドアロン実行ヘルパー（Streamlit依存回避） ────────────────────────────────

_BENCHMARK_FALLBACK: dict[str, dict] = {
    "製造業":       {"op_margin": 3.5, "equity_ratio": 38.0, "roa": 3.2, "current_ratio": 140.0, "dscr": 1.4},
    "建設業":       {"op_margin": 4.2, "equity_ratio": 32.0, "roa": 3.8, "current_ratio": 130.0, "dscr": 1.5},
    "卸売業":       {"op_margin": 2.1, "equity_ratio": 30.0, "roa": 2.5, "current_ratio": 125.0, "dscr": 1.3},
    "小売業":       {"op_margin": 2.8, "equity_ratio": 28.0, "roa": 3.0, "current_ratio": 115.0, "dscr": 1.2},
    "運輸業":       {"op_margin": 3.0, "equity_ratio": 25.0, "roa": 2.8, "current_ratio": 110.0, "dscr": 1.3},
    "情報通信業":   {"op_margin": 8.5, "equity_ratio": 52.0, "roa": 6.5, "current_ratio": 170.0, "dscr": 2.0},
    "不動産業":     {"op_margin": 12.0,"equity_ratio": 35.0, "roa": 4.0, "current_ratio": 120.0, "dscr": 1.6},
    "医療・福祉":   {"op_margin": 4.5, "equity_ratio": 42.0, "roa": 3.5, "current_ratio": 145.0, "dscr": 1.5},
    "サービス業":   {"op_margin": 5.0, "equity_ratio": 38.0, "roa": 4.2, "current_ratio": 135.0, "dscr": 1.4},
    "飲食業":       {"op_margin": 2.0, "equity_ratio": 18.0, "roa": 2.0, "current_ratio": 90.0,  "dscr": 1.1},
    "農業・漁業":   {"op_margin": 2.5, "equity_ratio": 30.0, "roa": 2.2, "current_ratio": 120.0, "dscr": 1.2},
    "金融・保険業": {"op_margin": 15.0,"equity_ratio": 55.0, "roa": 5.0, "current_ratio": 180.0, "dscr": 2.2},
    "教育・学習支援業": {"op_margin": 5.5, "equity_ratio": 45.0, "roa": 4.0, "current_ratio": 150.0, "dscr": 1.6},
    "宿泊業":       {"op_margin": 3.0, "equity_ratio": 22.0, "roa": 2.5, "current_ratio": 100.0, "dscr": 1.2},
    "その他":       {"op_margin": 4.0, "equity_ratio": 33.0, "roa": 3.0, "current_ratio": 125.0, "dscr": 1.3},
}

def _run_benchmark_agent_standalone(industry: str):
    from ai_chat import _chat_for_thread
    import re
    api_key = os.environ.get("GEMINI_API_KEY", "")
    system = (
        "あなたはリース審査の財務分析専門家です。"
        "指定された業種について、日本の中小企業の財務指標（業界平均）を推定してください。"
        '{"op_margin": <営業利益率%>, "equity_ratio": <自己資本比率%>, '
        '"roa": <ROA%>, "current_ratio": <流動比率%>, "dscr": <DSCR倍>}'
    )
    prompt = f"業種: {industry}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    try:
        res_raw = _chat_for_thread("gemini", "", messages, timeout_seconds=60, api_key=api_key)
        content = (res_raw.get("message") or {}).get("content", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["_source"] = "ai"
            return data
        # AIレスポンスのパース失敗 → フォールバック
    except Exception:
        pass

    # Gemini 失敗時: 静的ベンチマークを返す
    fallback = _BENCHMARK_FALLBACK.get(industry, _BENCHMARK_FALLBACK["その他"]).copy()
    fallback["_source"] = "static"
    return fallback

def _run_market_agent_standalone():
    from ai_chat import _chat_for_thread
    api_key = os.environ.get("GEMINI_API_KEY", "")
    system = "あなたは経済・金融アナリストです。現在の日本の金利状況を200字程度で報告してください。"
    prompt = "最新の市況を教えてください。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    res_raw = _chat_for_thread("gemini", "", messages, timeout_seconds=60, api_key=api_key)
    return {"content": (res_raw.get("message") or {}).get("content", "")}


# =============================================================================
# マルチエージェント審査 API (石橋 vs 風林火山 + 軍師調停)
# =============================================================================

class MultiAgentRequest(BaseModel):
    score: float
    company_name: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    prefecture: str = ""
    nenshu: float = 0
    op_margin_pct: float = 0
    equity_ratio: float = 0
    bank_credit: float = 0
    lease_credit: float = 0
    asset_name: str = ""
    lease_amount: float = 0
    session_id: str = ""


@app.post("/api/crystallize-now")
def crystallize_now():
    """
    知識結晶化バッチを手動で即時実行する（スケジュールを待たずに呼び出せる）。
    同期実行して結果を返す。
    """
    from api.scheduler import run_crystallization_batch
    try:
        result = run_crystallization_batch()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex-knowledge")
def reindex_knowledge():
    """
    Obsidian ナレッジを手動で再インデックス化する。
    Vault 更新後に呼び出す。バックグラウンドで実行し即座に 202 を返す。
    """
    from api.knowledge.indexer import start_background_indexing
    start_background_indexing()
    return {"status": "indexing_started", "message": "バックグラウンドでインデックス化を開始しました"}


@app.post("/api/reload-feedback")
def reload_feedback():
    """
    Obsidian Feedback/ フォルダを再読み込みして ChromaDB を更新する。
    フィードバックファイルを追加・編集した後に呼び出す。バックグラウンドで実行し即座に返す。
    """
    from api.knowledge.feedback_watcher import start_background_feedback_loading
    start_background_feedback_loading()
    return {"status": "reload_started", "message": "フィードバックの再読み込みを開始しました"}


@app.post("/api/multi-agent-screening")
def multi_agent_screening(req: MultiAgentRequest):
    """
    マルチエージェント討論審査。

    スコア60超/40未満は軍師単独高速処理、40〜60は石橋・風林火山が2ラウンド討論後に軍師裁定。
    """
    from api.multi_agent_screening import run_debate_screening
    try:
        result = run_debate_screening(req.model_dump())
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation-history")
def get_conversation_history(company_name: str, limit: int = 5):
    """企業名で過去の討論履歴を取得する。"""
    if not company_name:
        raise HTTPException(status_code=422, detail="company_name は必須です")
    try:
        from api.database import get_conversation_history
        history = get_conversation_history(company_name, limit=min(limit, 20))
        return {"company_name": company_name, "count": len(history), "sessions": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversation-history/{session_id}")
def delete_conversation_history(session_id: str):
    """session_id に紐づく会話履歴を削除する。"""
    try:
        from api.database import delete_conversation_session
        deleted = delete_conversation_session(session_id)
        return {"deleted": deleted, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 汎用チャット（永続記憶）エンドポイント ─────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """あなたはtuneリース審査システムの専属AIアドバイザー「めぶきちゃん」です。
リース会社の審査担当者の相棒として、専門的かつ親しみやすく回答します。

## あなたの専門領域
- **リース取引の基礎**: ファイナンスリースとオペレーティングリースの判定基準（経済的耐用年数の75%ルール、現在価値90%ルール）、リース料計算（元利均等・元金均等）、残価設定リースの仕組み
- **審査実務**: 信用調査の進め方、財務3表分析（PL/BS/CF）、債務償還年数・インタレストカバレッジ・自己資本比率の読み方、業種別リスク特性、担保・保証の取り方
- **会計・税務**: リース会計基準（IFRS16号・日本基準）、オフバランス処理、消費税取扱い、リース料の損金算入
- **法律・規制**: リース事業協会の自主規制、割賦販売法、金融庁の監督指針
- **市場・業界**: 国内リース市場動向、主要リース会社の戦略、金利環境とリース需要の関係

## 回答スタイル
- 原則300字以内。簡単な質問は1〜2文で答える
- 基本形は「結論1行 + 箇条書き最大3点」。4点以上は出さない
- 専門用語は必要な時だけ使い、説明も最小限にする
- 「この案件どう思う？」のような相談は、審査担当者目線で見るべき点を最大3点に絞る
- 詳細な根拠、長い解説、表、参照ノート一覧は、ユーザーが「詳しく」「根拠も」「表で」と頼んだ時だけ出す
- 前置き、挨拶、長いまとめは省く
- 日本語で回答する

## 参照情報
ユーザーの質問に関連するナレッジが【参照ナレッジ】として提供される場合があります。
その情報を優先的に参照してください。ただし回答には長く貼らず、必要な要点だけ短く反映してください。"""


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


@app.post("/api/chat")
def post_chat(req: ChatRequest):
    """汎用チャット：メッセージを受け取り、会話履歴付きでGeminiへ送信して返答する。"""
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message は空にできません")
    try:
        from api.chat_memory import (
            get_recent_messages,
            save_message,
            call_gemini_chat,
            get_message_count,
        )

        # RAG: 共通ストアから関連ナレッジを取得。ローカル埋め込みモデルが
        # 未キャッシュでもキーワード検索へフォールバックする。
        rag_context = ""
        try:
            from api.knowledge.vector_store import get_store

            hits = get_store().search(req.message, top_k=5)
            all_docs = []
            for hit in hits:
                text = str(hit.get("text") or "").strip()
                ref = str(hit.get("ref") or hit.get("file_name") or "").strip()
                if text:
                    prefix = f"{ref}: " if ref else ""
                    all_docs.append((prefix + text)[:600])
            if all_docs:
                rag_context = "\n\n【参照ナレッジ】\n" + "\n---\n".join(all_docs)
        except Exception as e:
            print(f"[RAG] 検索エラー: {e}")

        # DB直接参照: ユーザーが実データ分析を求めている場合にSQLite統計を注入
        db_context = ""
        try:
            from api.db_query import build_db_context
            db_context = build_db_context(req.message)
        except Exception as e:
            print(f"[DB Query] 統計取得エラー: {e}")

        # システムプロンプトにRAGコンテキストとDB統計を追記
        effective_prompt = _CHAT_SYSTEM_PROMPT + rag_context + db_context

        history = get_recent_messages(req.user_id, limit=20)
        history_for_gemini = [{"role": m["role"], "content": m["content"]} for m in history]
        reply = call_gemini_chat(effective_prompt, history_for_gemini, req.message)
        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)
        total = get_message_count(req.user_id)

        # 改善系メッセージを自動検出してObsidian Improvement Logに保存
        _IMPROVEMENT_KEYWORDS = (
            "改善", "わかりにくい", "分かりにくい", "使いにくい", "説明",
            "入力しにくい", "導線", "バグ", "不具合", "直して", "変えて",
            "修正して", "追加して", "欲しい", "要望", "提案",
        )
        if any(k in req.message for k in _IMPROVEMENT_KEYWORDS):
            try:
                from mobile_app.obsidian_bridge import append_improvement_note
                body = f"**ユーザー要望**\n{req.message}\n\n**めぶき返答**\n{reply}"
                append_improvement_note("AIチャット改善候補", body)
            except Exception as _obs_e:
                print(f"[Obsidian改善保存] エラー: {_obs_e}")

        return {"reply": reply, "total_messages": total}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@app.get("/api/chat/history")
def get_chat_history(user_id: str = "default", limit: int = 50):
    """汎用チャット履歴を取得する。"""
    try:
        from api.chat_memory import get_recent_messages
        messages = get_recent_messages(user_id, limit=min(limit, 200))
        return {"user_id": user_id, "count": len(messages), "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history")
def delete_chat_history(user_id: str = "default"):
    """汎用チャット履歴を全削除する。"""
    try:
        from api.chat_memory import delete_history
        deleted = delete_history(user_id)
        return {"deleted": deleted, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaveToObsidianRequest(BaseModel):
    user_id: str = "default"
    title: Optional[str] = None


@app.post("/api/chat/save-to-obsidian")
def save_chat_to_obsidian(req: SaveToObsidianRequest):
    """チャット履歴を Obsidian Vault の Chat/ フォルダに保存する。"""
    import datetime
    import re as _re

    try:
        from api.chat_memory import get_recent_messages
        messages = get_recent_messages(req.user_id, limit=100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"履歴取得エラー: {e}")

    vault_root = _OBSIDIAN_VAULT_PATH

    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="Obsidian Vault が見つかりません。環境変数 OBSIDIAN_VAULT_PATH を設定してください。",
        )

    chat_dir = os.path.join(vault_root, "Chat")
    os.makedirs(chat_dir, exist_ok=True)

    title = (req.title or "AI相談メモ").strip() or "AI相談メモ"
    today = datetime.date.today().strftime("%Y-%m-%d")

    # ファイル名に使えない文字を除去・同日上書きを防ぐためタイムスタンプを付与
    safe_title = _re.sub(r'[\\/:*?"<>|\n\r\t]', "_", title)[:40].strip("_").strip() or "AI相談メモ"
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    filename = f"{today}_{safe_title}_{timestamp}.md"
    filepath = os.path.join(chat_dir, filename)

    # Markdown 本文を組み立て
    lines = [
        "---",
        f"date: {today}",
        "type: chat_log",
        "---",
        "",
        f"# {title}",
        "",
    ]
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "めぶき"
        lines.append(f"**{role_label}**: {msg['content']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル書き込みエラー: {e}")

    relative_path = f"Chat/{filename}"
    return {"path": relative_path, "message_count": len(messages)}


class DebateCautiousData(BaseModel):
    opinion: str = ""
    reasons: List[str] = []
    key_risks: List[str] = []


class DebateAggressiveData(BaseModel):
    opinion: str = ""
    reasons: List[str] = []
    opportunities: List[str] = []


class SaveDebateToObsidianRequest(BaseModel):
    company_name: str = ""
    score: int = 0
    grade: str = ""
    cautious: Optional[DebateCautiousData] = None
    aggressive: Optional[DebateAggressiveData] = None
    arbiter_summary: str = ""
    final_decision: str = ""
    conditions: List[str] = []
    debate_log: Optional[str] = None
    screened_at: Optional[str] = None


@app.post("/api/debate/save-to-obsidian")
def save_debate_to_obsidian(req: SaveDebateToObsidianRequest):
    """討論審査結果を Obsidian Vault の Debates/ フォルダに保存する。"""
    import datetime
    import re as _re

    vault_root = _OBSIDIAN_VAULT_PATH

    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="Obsidian Vault が見つかりません。環境変数 OBSIDIAN_VAULT_PATH を設定してください。",
        )

    debates_dir = os.path.join(vault_root, "Debates")
    os.makedirs(debates_dir, exist_ok=True)

    today = datetime.date.today().strftime("%Y-%m-%d")
    company = req.company_name.strip() or "不明"
    safe_company = _re.sub(r'[\\/:*?"<>|\n\r\t]', "_", company)[:40].strip("_").strip() or "不明"
    # 同日・同企業の複数審査で上書きされないよう時刻サフィックスを付与
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    filename = f"{today}_{safe_company}_{timestamp}.md"
    filepath = os.path.join(debates_dir, filename)

    # グレード計算（フロントから来なければスコアで自動算出）
    grade = req.grade
    if not grade:
        s = req.score
        if s >= 80:
            grade = "A"
        elif s >= 60:
            grade = "B"
        elif s >= 40:
            grade = "C"
        elif s >= 20:
            grade = "D"
        else:
            grade = "E"

    # frontmatter
    lines = [
        "---",
        f"date: {today}",
        "type: debate_result",
        f"company: {company}",
        f"score: {req.score}",
        f"grade: {grade}",
        f"decision: {req.final_decision}",
        "---",
        "",
        f"# 討論審査: {company}",
        "",
        f"**スコア**: {req.score}点 / グレード: {grade} / **判定**: {req.final_decision}",
        "",
    ]

    # 討論エージェントセクション
    if req.cautious or req.aggressive:
        lines += ["## 討論結果（第2ラウンド最終立場）", ""]

        if req.cautious:
            lines += [
                "### 石橋（慎重派）",
                "",
                f"**意見**: {req.cautious.opinion}",
                "",
                "**判断理由**",
                "",
            ]
            for r in req.cautious.reasons:
                lines.append(f"- {r}")
            if req.cautious.key_risks:
                lines += ["", "**重大リスク**", ""]
                for r in req.cautious.key_risks:
                    lines.append(f"- {r}")
            lines.append("")

        if req.aggressive:
            lines += [
                "### 風林火山（積極派）",
                "",
                f"**意見**: {req.aggressive.opinion}",
                "",
                "**判断理由**",
                "",
            ]
            for r in req.aggressive.reasons:
                lines.append(f"- {r}")
            if req.aggressive.opportunities:
                lines += ["", "**見逃せない機会**", ""]
                for r in req.aggressive.opportunities:
                    lines.append(f"- {r}")
            lines.append("")

    # 軍師セクション
    lines += [
        "## 軍師の最終判断",
        "",
        req.arbiter_summary,
        "",
    ]

    if req.conditions:
        lines += ["### 承認条件", ""]
        for i, c in enumerate(req.conditions, 1):
            lines.append(f"{i}. {c}")
        lines.append("")

    # 討論ログ
    if req.debate_log:
        lines += [
            "## 討論ログ",
            "",
            "```",
            req.debate_log,
            "```",
            "",
        ]

    content = "\n".join(lines)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル書き込みエラー: {e}")

    relative_path = f"Debates/{filename}"
    return {"path": relative_path}


@app.get("/api/analysis/network_risk")
def api_network_risk(industry: str = ""):
    """業種コードまたは業種名からサプライチェーン波及リスクを計算する"""
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(__file__))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    try:
        from components.graph_risk import GraphRiskEngine
        engine = GraphRiskEngine()
        result = engine.calculate_network_risk(industry)
        sim = engine.run_scenario_simulation(industry, n_simulations=300)
        return {
            "network_risk_pct": round(result.get("network_risk_score", 0.05) * 100, 1),
            "base_risk_pct": round(result.get("base_risk", 0.05) * 100, 1),
            "impacted_by": result.get("impacted_by", [])[:5],
            "sim_mean_pct": round(sim.get("mean_risk", 0.05) * 100, 1),
            "sim_var95_pct": round(sim.get("max_risk_95", 0.05) * 100, 1),
            "target_industry": result.get("target_industry", industry),
        }
    except Exception as e:
        return {"network_risk_pct": 5.0, "base_risk_pct": 5.0, "impacted_by": [],
                "sim_mean_pct": 5.0, "sim_var95_pct": 10.0, "target_industry": industry,
                "error": str(e)}


@app.get("/api/latest-screening")
def get_latest_screening():
    """
    直近の審査データを返す。
    screening_records の最新スコア + past_cases の最新フォーム入力値を合成して、
    debate ページの初期値として使用する。
    """
    import json as _json
    import os as _os
    from contextlib import closing as _closing

    defaults = {
        "score": 52,
        "company_name": "",
        "industry_major": "製造業",
        "nenshu": 0,
        "op_margin_pct": 0,
        "equity_ratio": 0,
        "bank_credit": 0,
        "lease_credit": 0,
        "asset_name": "",
        "lease_amount": 0,
        "news_focus": [],
        "news_focus_summary": "",
        "news_focus_tag_summary": "",
        "news_focus_note_path": "",
        "news_focus_note_date": "",
    }

    def _first_non_empty(*values):
        for value in values:
            if value not in (None, "", [], {}):
                return value
        return None

    def _to_million(value):
        try:
            return round(float(value) / 1000, 2)
        except Exception:
            return 0

    def _safe_float(value):
        try:
            return float(value)
        except Exception:
            return 0.0

    try:
        from data_cases import _open_db
        with _closing(_open_db()) as conn:
            import sqlite3 as _sqlite3
            conn.row_factory = _sqlite3.Row

            # screening_records から最新スコアを取得
            try:
                sr = conn.execute(
                    "SELECT total_score, input_snapshot FROM screening_records ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if sr:
                    defaults["score"] = round(float(sr["total_score"]), 1)
                    if sr["input_snapshot"]:
                        snap = _json.loads(sr["input_snapshot"])
                        for key in ("company_name", "industry_major", "asset_name"):
                            if snap.get(key):
                                defaults[key] = snap[key]
            except Exception:
                pass

            # past_cases から最新のフォーム入力値を取得（千円→百万円 変換）
            try:
                pc = conn.execute(
                    "SELECT data FROM past_cases ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if pc and pc["data"]:
                    d = _json.loads(pc["data"])

                    inputs = d.get("inputs") if isinstance(d.get("inputs"), dict) else {}
                    result = d.get("result") if isinstance(d.get("result"), dict) else {}

                    company_name = _first_non_empty(d.get("company_name"), inputs.get("company_name"))
                    if company_name:
                        defaults["company_name"] = company_name

                    industry_major = _first_non_empty(
                        d.get("selected_major"),
                        d.get("industry_major"),
                        inputs.get("industry_major"),
                    )
                    if industry_major:
                        defaults["industry_major"] = industry_major

                    nenshu_raw = _safe_float(_first_non_empty(inputs.get("nenshu"), d.get("nenshu")))
                    op_profit_raw = _safe_float(_first_non_empty(inputs.get("op_profit"), inputs.get("rieki"), d.get("rieki")))
                    net_assets_raw = _safe_float(_first_non_empty(inputs.get("net_assets"), d.get("net_assets")))
                    total_assets_raw = _safe_float(_first_non_empty(inputs.get("total_assets"), d.get("total_assets")))

                    if nenshu_raw > 0:
                        defaults["nenshu"] = _to_million(nenshu_raw)
                    if nenshu_raw > 0:
                        defaults["op_margin_pct"] = round(op_profit_raw / nenshu_raw * 100, 1)
                    if total_assets_raw > 0:
                        defaults["equity_ratio"] = round(net_assets_raw / total_assets_raw * 100, 1)

                    bank_credit_raw = _safe_float(_first_non_empty(inputs.get("bank_credit"), d.get("bank_credit")))
                    lease_credit_raw = _safe_float(_first_non_empty(inputs.get("lease_credit"), d.get("lease_credit")))
                    acquisition_cost_raw = _safe_float(_first_non_empty(inputs.get("acquisition_cost"), d.get("acquisition_cost")))
                    if bank_credit_raw:
                        defaults["bank_credit"] = _to_million(bank_credit_raw)
                    if lease_credit_raw:
                        defaults["lease_credit"] = _to_million(lease_credit_raw)
                    if acquisition_cost_raw:
                        defaults["lease_amount"] = _to_million(acquisition_cost_raw)

                    asset_name = _first_non_empty(
                        inputs.get("asset_name"),
                        inputs.get("selected_asset_id"),
                        d.get("asset_name"),
                    )
                    if asset_name:
                        defaults["asset_name"] = asset_name

                    if result.get("score") is not None:
                        try:
                            defaults["score"] = round(float(result["score"]), 1)
                        except Exception:
                            pass
            except Exception:
                pass

    except Exception:
        pass

    try:
        focus = get_latest_lease_news_focus()
        if focus.available:
            defaults["news_focus"] = list(focus.focus_lines)
            defaults["news_focus_summary"] = focus.headline
            defaults["news_focus_tag_summary"] = focus.tag_summary
            defaults["news_focus_note_path"] = focus.note_path
            defaults["news_focus_note_date"] = focus.note_date
            try:
                record_lease_news_view(focus.note_date or "", focus.note_path, focus.tag_summary)
            except Exception as _view_err:
                print(f"[API] lease news view metric failed: {_view_err}")
    except Exception as e:
        print(f"[API] latest lease news focus load failed: {e}")

    return defaults


class LeaseNewsJudgmentChangeRequest(BaseModel):
    company_name: str = ""
    score: Optional[float] = None
    final_decision: str = ""
    news_focus: List[str] = []
    news_focus_summary: str = ""
    news_focus_tag_summary: str = ""
    news_focus_note_path: str = ""
    news_focus_note_date: str = ""
    reason: str = ""


@app.post("/api/lease-news/judgment-change")
def record_lease_news_judgment_change_api(req: LeaseNewsJudgmentChangeRequest):
    """ニュース参照後の判断変更を記録する。"""
    import datetime as _dt

    try:
        bucket = record_lease_news_judgment_change(
            date_str=_dt.date.today().isoformat(),
            note_path=req.news_focus_note_path or "",
            source_note_date=req.news_focus_note_date or "",
            company_name=req.company_name or "",
            score=req.score,
            final_decision=req.final_decision or "",
            reason=req.reason or "",
            focus_lines=tuple(req.news_focus or []),
            theme_summary=req.news_focus_summary or "",
            tag_summary=req.news_focus_tag_summary or "",
        )
        return {"status": "recorded", "metrics": bucket}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── screening_outcomes エンドポイント（追加のみ、既存ルート不変）──────────────────

class OutcomeCreateRequest(BaseModel):
    case_id: str = Field(..., description="案件 ID（past_cases.id 等）")
    actual_status: str = Field(
        default="unknown",
        description="unknown / normal / late_30 / late_90 / default / completed",
    )
    screening_id: Optional[int] = Field(default=None, description="screening_records.id への参照")
    contract_date: Optional[str] = Field(default=None, description="成約日（YYYY-MM-DD）")
    scheduled_end_date: Optional[str] = Field(default=None, description="リース満了予定日")
    delinquent: int = Field(default=0, description="0=正常, 1=延滞・デフォルト")
    loss_given_default: Optional[float] = Field(default=None, description="実損額（円）")
    notes: Optional[str] = Field(default=None, description="備考")


class OutcomeResponse(BaseModel):
    id: int
    case_id: str
    screening_id: Optional[int]
    contract_date: Optional[str]
    scheduled_end_date: Optional[str]
    actual_status: str
    delinquent: int
    loss_given_default: Optional[float]
    checked_at: str
    notes: Optional[str]
    created_at: str
    updated_at: str


@app.post("/api/outcomes", response_model=OutcomeResponse)
def create_outcome(req: OutcomeCreateRequest):
    """審査後の追跡結果（支払状況等）を登録する。"""
    try:
        from api.add_outcomes_table import insert_outcome, get_outcome
        new_id = insert_outcome(
            case_id=req.case_id,
            actual_status=req.actual_status,
            screening_id=req.screening_id,
            contract_date=req.contract_date,
            scheduled_end_date=req.scheduled_end_date,
            delinquent=req.delinquent,
            loss_given_default=req.loss_given_default,
            notes=req.notes,
        )
        row = get_outcome(new_id)
        if row is None:
            raise HTTPException(status_code=500, detail="insert succeeded but row not found")
        return OutcomeResponse(**row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/outcomes", response_model=List[OutcomeResponse])
def get_outcomes(
    screening_id: Optional[int] = None,
    case_id: Optional[str] = None,
    actual_status: Optional[str] = None,
    limit: int = 100,
):
    """審査後追跡結果の一覧を取得する。screening_id / case_id / actual_status で絞り込み可能。"""
    try:
        from api.add_outcomes_table import list_outcomes
        rows = list_outcomes(
            screening_id=screening_id,
            case_id=case_id,
            actual_status=actual_status,
            limit=limit,
        )
        return [OutcomeResponse(**r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Fluid Pipeline エンドポイント（追記のみ、既存ルート不変）────────────────────

@app.post("/api/fluid/trigger")
def fluid_trigger(triggered_by: str = "manual"):
    """ドリフト検知→再学習→PDCA反省パイプラインを手動でバックグラウンド起動する。"""
    try:
        from api.fluid_pipeline import trigger_fluid_pipeline
        result = trigger_fluid_pipeline(triggered_by=triggered_by)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fluid/status")
def fluid_status():
    """Fluid Pipeline の現在状態を返す。"""
    try:
        from api.fluid_pipeline import get_fluid_status
        return get_fluid_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CounterfactualRequest(BaseModel):
    case_id: str
    target_score: float = 70.0


@app.post("/api/counterfactual/analyze")
def analyze_counterfactual(req: CounterfactualRequest):
    """Counterfactual Explanation（REV-009）。指定案件の審査通過に必要な最小変更を計算する。"""
    import json as _json
    import sqlite3 as _sqlite3
    from data_cases import _open_db
    from scoring_core import run_quick_scoring

    # 案件データ取得
    try:
        with _open_db() as conn:
            conn.row_factory = _sqlite3.Row
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (req.case_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        case = _json.loads(row["data"] or "{}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    inputs = case.get("inputs", {})
    result = case.get("result", {})
    if not inputs:
        raise HTTPException(status_code=422, detail="この案件には入力データが保存されていません")

    current_score = float(result.get("score") or run_quick_scoring(inputs).get("score", 0))
    target = req.target_score

    def score_with(overrides: dict) -> float:
        merged = {**inputs, **overrides}
        try:
            return float(run_quick_scoring(merged).get("score", 0))
        except Exception:
            return 0.0

    # 主要パラメータの現在値
    nenshu = max(1.0, float(inputs.get("nenshu") or 1))
    op_profit = float(inputs.get("op_profit") or 0)
    ord_profit = float(inputs.get("ord_profit") or 0)
    net_income = float(inputs.get("net_income") or 0)
    net_assets = float(inputs.get("net_assets") or 0)
    total_assets = max(1.0, float(inputs.get("total_assets") or 1))
    bank_credit = float(inputs.get("bank_credit") or 0)
    grade = str(inputs.get("grade") or "②4-6 (標準)")

    current_op_margin = op_profit / nenshu * 100
    current_eq_ratio = net_assets / total_assets * 100

    analyses = []

    # 1. 営業利益率改善 (op_profit増加)
    if current_score < target:
        for mult in [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
            s = score_with({"op_profit": op_profit * mult})
            if s >= target:
                new_margin = (op_profit * mult) / nenshu * 100
                analyses.append({
                    "param": "op_profit",
                    "label": "営業利益率",
                    "current_display": f"{current_op_margin:.1f}%",
                    "required_display": f"{new_margin:.1f}%",
                    "current_value": current_op_margin,
                    "required_value": new_margin,
                    "change_pct": (mult - 1) * 100,
                    "achieved_score": round(s, 1),
                    "difficulty": "易" if mult <= 1.5 else "中" if mult <= 3.0 else "難",
                    "note": f"営業利益を現在の {mult:.0f}倍（+{(mult-1)*100:.0f}%）にする必要があります",
                })
                break

    # 2. 自己資本比率改善 (net_assets増加)
    if current_score < target:
        for add_ratio in [5, 10, 20, 35, 50]:
            new_net_assets = (current_eq_ratio + add_ratio) / 100 * total_assets
            s = score_with({"net_assets": new_net_assets})
            if s >= target:
                new_eq = new_net_assets / total_assets * 100
                analyses.append({
                    "param": "net_assets",
                    "label": "自己資本比率",
                    "current_display": f"{current_eq_ratio:.1f}%",
                    "required_display": f"{new_eq:.1f}%",
                    "current_value": current_eq_ratio,
                    "required_value": new_eq,
                    "change_pct": add_ratio,
                    "achieved_score": round(s, 1),
                    "difficulty": "易" if add_ratio <= 10 else "中" if add_ratio <= 25 else "難",
                    "note": f"自己資本比率を {add_ratio}pt 改善する必要があります",
                })
                break

    # 3. 売上高改善 (nenshu増加)
    if current_score < target:
        for mult in [1.3, 1.5, 2.0, 3.0]:
            s = score_with({"nenshu": nenshu * mult, "op_profit": op_profit * mult,
                            "ord_profit": ord_profit * mult, "net_income": net_income * mult})
            if s >= target:
                analyses.append({
                    "param": "nenshu",
                    "label": "売上高（按分増加）",
                    "current_display": f"{nenshu/1000:.1f}百万円",
                    "required_display": f"{nenshu*mult/1000:.1f}百万円",
                    "current_value": nenshu,
                    "required_value": nenshu * mult,
                    "change_pct": (mult - 1) * 100,
                    "achieved_score": round(s, 1),
                    "difficulty": "中" if mult <= 1.5 else "難",
                    "note": f"売上高を {mult:.1f}倍に成長させる必要があります（利益率維持想定）",
                })
                break

    # 4. 格付改善
    grade_map = [
        ("s", "S格（超優良）"),
        ("①", "①格（優良）"),
        ("②", "②格（標準）"),
        ("③", "③格（注意）"),
        ("④", "④格（要注意）"),
    ]
    if current_score < target:
        for gk, gl in grade_map:
            if grade.lower().startswith(gk):
                continue
            # 現在より良い格付のみ試す
            s = score_with({"grade": gk})
            if s >= target:
                analyses.append({
                    "param": "grade",
                    "label": "社内格付",
                    "current_display": grade,
                    "required_display": gl,
                    "current_value": 0,
                    "required_value": 0,
                    "change_pct": None,
                    "achieved_score": round(s, 1),
                    "difficulty": "中",
                    "note": f"格付を {grade} → {gl} に改善する必要があります",
                })
                break

    # 複合提案（op_profit + net_assets を半分ずつ改善）
    if current_score < target and len(analyses) < 2:
        for op_mult, eq_add in [(1.3, 5), (1.5, 10), (2.0, 15), (2.5, 20)]:
            new_na = (current_eq_ratio + eq_add) / 100 * total_assets
            s = score_with({"op_profit": op_profit * op_mult, "net_assets": new_na})
            if s >= target:
                analyses.append({
                    "param": "combined",
                    "label": "複合改善（利益率＋自己資本）",
                    "current_display": f"利益率{current_op_margin:.1f}% / 自己資本{current_eq_ratio:.1f}%",
                    "required_display": f"利益率{op_profit*op_mult/nenshu*100:.1f}% / 自己資本{(current_eq_ratio+eq_add):.1f}%",
                    "current_value": 0,
                    "required_value": 0,
                    "change_pct": None,
                    "achieved_score": round(s, 1),
                    "difficulty": "中",
                    "note": f"営業利益率+{(op_mult-1)*100:.0f}% かつ 自己資本比率+{eq_add}pt の複合改善",
                })
                break

    # スコア感度（各パラメータを±stepで変化させたスコア列）
    def op_sensitivity():
        data = []
        for pct in range(-50, 151, 10):
            v = op_profit * (1 + pct / 100) if op_profit != 0 else pct * nenshu / 10000
            s = score_with({"op_profit": v})
            data.append({
                "pct_change": pct,
                "op_margin": round(v / nenshu * 100, 2) if nenshu > 0 else 0,
                "score": round(s, 1),
            })
        return data

    def eq_sensitivity():
        data = []
        for add in range(-30, 51, 5):
            new_na = (current_eq_ratio + add) / 100 * total_assets
            s = score_with({"net_assets": max(0, new_na)})
            data.append({
                "eq_ratio": round(current_eq_ratio + add, 1),
                "score": round(s, 1),
            })
        return data

    return {
        "case_id": req.case_id,
        "current_score": round(current_score, 1),
        "target_score": target,
        "gap": round(target - current_score, 1),
        "current_metrics": {
            "op_margin": round(current_op_margin, 2),
            "eq_ratio": round(current_eq_ratio, 2),
            "nenshu": nenshu,
            "op_profit": op_profit,
            "net_assets": net_assets,
            "total_assets": total_assets,
            "grade": grade,
        },
        "counterfactuals": analyses,
        "op_sensitivity": op_sensitivity(),
        "eq_sensitivity": eq_sensitivity(),
    }


class RateEngineRequest(BaseModel):
    score: float
    term_months: int = 60
    asset_id: str = "other"
    grade: str = "②"
    lease_amount: float = 10000000
    year_month: str = ""


@app.post("/api/rate-engine/propose")
def propose_lease_rate(req: RateEngineRequest):
    """動的金利提案エンジン（REV-002）。借手スコア・物件種別・期間から最適金利を提案する。"""
    import datetime
    from base_rate_master import get_base_rate_by_term

    year_month = req.year_month or datetime.date.today().strftime("%Y-%m")
    term_months = max(12, min(120, req.term_months))
    term_years = term_months / 12

    base_rate = get_base_rate_by_term(year_month, term_months)
    if base_rate is None:
        for i in range(1, 7):
            prev_date = datetime.date.today().replace(day=1) - datetime.timedelta(days=30 * i)
            base_rate = get_base_rate_by_term(prev_date.strftime("%Y-%m"), term_months)
            if base_rate is not None:
                break
    if base_rate is None:
        base_rate = 2.0

    _asset_spreads: dict[tuple[str, int], float] = {
        ("medical", 1): 0.25, ("medical", 3): 0.30, ("medical", 5): 0.35, ("medical", 7): 0.40,
        ("it", 1): 0.35, ("it", 3): 0.50, ("it", 5): 0.65, ("it", 7): 0.75,
        ("pc", 1): 0.35, ("pc", 3): 0.50, ("pc", 5): 0.65, ("pc", 7): 0.75,
        ("vehicle", 1): 0.28, ("vehicle", 3): 0.32, ("vehicle", 5): 0.38, ("vehicle", 7): 0.45,
        ("car", 1): 0.28, ("car", 3): 0.32, ("car", 5): 0.38, ("car", 7): 0.45,
        ("machinery", 1): 0.30, ("machinery", 3): 0.38, ("machinery", 5): 0.45, ("machinery", 7): 0.52,
        ("construction", 1): 0.32, ("construction", 3): 0.40, ("construction", 5): 0.48, ("construction", 7): 0.55,
        ("solar", 1): 0.28, ("solar", 3): 0.35, ("solar", 5): 0.42, ("solar", 7): 0.50,
        ("other", 1): 0.32, ("other", 3): 0.40, ("other", 5): 0.50, ("other", 7): 0.58,
    }
    valid_terms = [1, 3, 5, 7]
    nearest_t = min(valid_terms, key=lambda t: abs(t - term_years))
    asset_id_lower = req.asset_id.lower()
    matched_prefix = next((k for (k, _) in _asset_spreads if asset_id_lower.startswith(k) and k != "other"), None)
    asset_spread = _asset_spreads.get((matched_prefix or "other", nearest_t), 0.45)

    _grade_spreads = {"s": -0.10, "①": -0.10, "a": 0.10, "②": 0.25, "b": 0.25,
                      "③": 0.55, "c": 0.55, "④": 0.90, "d": 0.90}
    grade_lower = req.grade.strip().lower()
    grade_spread = next((v for k, v in _grade_spreads.items() if grade_lower.startswith(k)), 0.30)

    score = max(0.0, min(100.0, req.score))
    if score >= 90: risk_adj = -0.10
    elif score >= 80: risk_adj = -0.05
    elif score >= 70: risk_adj = 0.00
    elif score >= 60: risk_adj = 0.15
    elif score >= 50: risk_adj = 0.30
    else: risk_adj = 0.50

    proposed_rate = round(max(0.5, base_rate + asset_spread + grade_spread + risk_adj), 4)

    monthly_rate = proposed_rate / 100 / 12
    amount = max(1.0, req.lease_amount)
    if monthly_rate > 0:
        monthly_payment = amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))
    else:
        monthly_payment = amount / term_months
    total_payment = monthly_payment * term_months
    total_interest = total_payment - amount

    sensitivity = []
    for s in range(max(0, int(score) - 30), min(101, int(score) + 35), 5):
        if s >= 90: r = -0.10
        elif s >= 80: r = -0.05
        elif s >= 70: r = 0.00
        elif s >= 60: r = 0.15
        elif s >= 50: r = 0.30
        else: r = 0.50
        sensitivity.append({
            "score": s,
            "rate": round(base_rate + asset_spread + grade_spread + r, 4),
            "is_current": abs(s - score) < 5,
        })

    return {
        "year_month": year_month,
        "proposed_rate": proposed_rate,
        "breakdown": {
            "base_rate": round(base_rate, 4),
            "asset_spread": round(asset_spread, 4),
            "grade_spread": round(grade_spread, 4),
            "risk_adjustment": round(risk_adj, 4),
        },
        "monthly_payment": round(monthly_payment),
        "total_payment": round(total_payment),
        "total_interest": round(total_interest),
        "term_months": term_months,
        "lease_amount": amount,
        "sensitivity": sensitivity,
    }
