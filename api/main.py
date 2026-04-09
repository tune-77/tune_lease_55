from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# プロジェクトルートをPYTHONPATHに追加して、既存モジュール(scoring_core)をインポート可能にする
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# sys.path に同名モジュール(data_cases等)を持つ別ディレクトリが先に入っている場合があるため、
# _REPO_ROOT を最優先(position 0)に固定する
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# data_cases を正しいパスから強制ロード（clawd/data_cases.py が優先されないよう先読み）
import importlib.util as _ilu
_dc_spec = _ilu.spec_from_file_location("data_cases", os.path.join(_REPO_ROOT, "data_cases.py"))
_dc_mod = _ilu.module_from_spec(_dc_spec)
sys.modules["data_cases"] = _dc_mod
_dc_spec.loader.exec_module(_dc_mod)

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
from api.schemas import ScoringRequest, ScoringResponse, CaseRegisterRequest
from pydantic import BaseModel
from typing import List, Any, Dict

app = FastAPI(
    title="Lease Scoring API",
    description="リース審査ロジックのバックエンドAPI",
    version="1.0.0"
)

# モダンフロントエンド(Next.js)のReactローカルサーバー(例: 3000番ポート)からのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
            ai_completed_factors=[]
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
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases/pending")
def get_pending_cases():
    """全DB(lease_data.db, screening_db.sqlite)から未登録案件を統合して取得する"""
    import sqlite3
    import json
    from contextlib import closing
    
    _lease_db = os.path.join(_REPO_ROOT, "data", "lease_data.db")
    _screening_db = os.path.join(_REPO_ROOT, "data", "screening_db.sqlite")
    
    rows = []
    
    # 1. lease_data.db (past_cases)
    if os.path.exists(_lease_db):
        try:
            with closing(sqlite3.connect(_lease_db)) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("SELECT id, timestamp, industry_sub, score, data FROM past_cases WHERE final_status='未登録'").fetchall()
                for r in res:
                    try: d = json.loads(r["data"] or "{}")
                    except: d = {}
                    rows.append({
                        "id": str(r["id"]),
                        "company_no": d.get("company_no", ""),
                        "company_name": d.get("company_name", ""),
                        "timestamp": r["timestamp"],
                        "score": r["score"],
                        "industry": r["industry_sub"] or d.get("industry_major", ""),
                        "_source": "past_cases"
                    })
        except Exception: pass

    # 2. screening_db.sqlite (screening_records)
    if os.path.exists(_screening_db):
        try:
            with closing(sqlite3.connect(_screening_db)) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("""
                    SELECT id, created_at, industry_sub, industry_major, score, memo 
                    FROM screening_records 
                    WHERE (memo IS NULL OR memo='' OR (memo NOT LIKE '%final_status%' AND memo NOT LIKE '%"final_status":"成約"%' AND memo NOT LIKE '%"final_status":"失注"%'))
                """).fetchall()
                for r in res:
                    try: m = json.loads(r["memo"] or "{}")
                    except: m = {}
                    # 重複チェック用に、過去DB由来のIDがあればスキップ
                    orig_id = m.get("_original_past_case_id")
                    if orig_id and any(x["id"] == str(orig_id) for x in rows): continue
                    
                    rows.append({
                        "id": str(r["id"]),
                        "company_no": m.get("company_no") or m.get("corporate_number", ""),
                        "company_name": m.get("company_name", ""),
                        "timestamp": r["created_at"],
                        "score": r["score"],
                        "industry": r["industry_sub"] or r["industry_major"],
                        "_source": "screening_records"
                    })
        except Exception: pass

    # 日付順（降順）にソートして上位50件
    rows.sort(key=lambda x: x["timestamp"] or "", reverse=True)
    return rows[:50]

@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str):
    """案件を全DBから確実に削除する"""
    import sqlite3
    from contextlib import closing
    deleted = False

    paths = [
        (os.path.join(_REPO_ROOT, "data", "lease_data.db"), "past_cases"),
        (os.path.join(_REPO_ROOT, "data", "screening_db.sqlite"), "screening_records")
    ]

    for db_path, table in paths:
        if not os.path.exists(db_path): continue
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                # IDが数値の場合
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (case_id,))
                # IDが文字列の場合
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (str(case_id),))
                # screening_records の場合は memo 内の ID もチェック
                if table == "screening_records":
                    conn.execute(f"DELETE FROM {table} WHERE memo LIKE ?", (f'%"_original_past_case_id": "{case_id}"%',))
                    conn.execute(f"DELETE FROM {table} WHERE memo LIKE ?", (f'%"_original_past_case_id":"{case_id}"%',))
                conn.commit()
                deleted = True # 少なくとも試行はした
        except Exception: pass

    return {"message": "Deleted if existed", "case_id": case_id}

@app.delete("/api/cases/operation/clear-all")
def clear_all_pending_cases():
    """未登録案件をすべて削除する（一括クリア）"""
    import sqlite3
    from contextlib import closing
    try:
        # screening_db.sqlite
        _sdb = os.path.join(_REPO_ROOT, "data", "screening_db.sqlite")
        if os.path.exists(_sdb):
            with closing(sqlite3.connect(_sdb)) as conn:
                conn.execute("DELETE FROM screening_records WHERE (memo IS NULL OR memo NOT LIKE '%final_status%')")
                conn.commit()
        # lease_data.db
        _ldb = os.path.join(_REPO_ROOT, "data", "lease_data.db")
        if os.path.exists(_ldb):
            with closing(sqlite3.connect(_ldb)) as conn:
                conn.execute("DELETE FROM past_cases WHERE final_status='未登録'")
                conn.commit()
        return {"message": "Cleared all pending cases"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cases/register")
def register_case(req: CaseRegisterRequest):
    try:
        from data_cases import update_case
        patches = {
            "final_status": req.status,
            "final_rate": req.final_rate,
            "base_rate_at_time": req.base_rate_at_time,
            "loan_conditions": req.loan_conditions,
            "competitor_name": req.competitor_name,
            "competitor_rate": req.competitor_rate,
            "note": req.note
        }
        if req.status == "成約" and req.final_rate > 0:
            patches["winning_spread"] = req.final_rate - req.base_rate_at_time
        if req.status == "失注":
            patches["lost_reason"] = req.lost_reason

        success = update_case(req.case_id, patches)
        if not success:
            raise HTTPException(status_code=404, detail="Case not found or update failed")
        
        # 統計やAI学習のトリガー（Streamlit版と同等の処理）
        try:
            from components.shinsa_gunshi import refresh_evidence_weights
            refresh_evidence_weights()
        except Exception:
            pass
            
        return {"message": "Successfully registered", "case_id": req.case_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class AdviseRequest(BaseModel):
    score: float
    pd_percent: float
    industry_major: str

class AdviseResponseItem(BaseModel):
    id: str
    text: str
    score_boost: float

@app.post("/api/gunshi/advise", response_model=List[AdviseResponseItem])
def get_gunshi_advise(req: AdviseRequest):
    from components.shinsa_gunshi import PHRASES_100
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


class GunshiChatRequest(BaseModel):
    score: float
    pd_percent: float
    industry_major: str
    asset_name: str
    resale: str
    repeat_cnt: int
    subsidy: bool
    bank: bool
    intuition: int
    posterior: float

@app.post("/api/gunshi/chat")
def generate_gunshi_chat(req: GunshiChatRequest):
    from components.shinsa_gunshi import PHRASES_100, build_gunshi_prompt
    try:
        advices = PHRASES_100.get("逆転アドバイス", [])
        import random
        sampled = random.sample(advices, min(3, len(advices)))
        
        prompt = build_gunshi_prompt(
            industry=req.industry_major,
            score=req.score,
            pd_pct=req.pd_percent,
            resale=req.resale,
            repeat_cnt=req.repeat_cnt,
            subsidy=req.subsidy,
            bank=req.bank,
            intuition=req.intuition,
            posterior=req.posterior,
            success_patterns={"success_samples": [], "fail_samples": []},
            top_phrases=sampled,
            asset_name=req.asset_name
        )

        reply_text = ""
        try:
            from components.shinsa_gunshi import _get_gemini_key
            api_key = _get_gemini_key()
            if not api_key:
                import os
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

        return {"chat_text": reply_text}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    from components.analysis_regression import run_contract_driver_analysis
    from components.data_cases import load_all_cases
    try:
        analysis = run_contract_driver_analysis()
        all_cases = load_all_cases()
        recent_cases = list(reversed(all_cases[-15:])) if all_cases else []
        return {
            "analysis": analysis,
            "recent_cases": recent_cases
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visual/data")
def get_visual_data():
    from components.visual_insights import _build_dataframe
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
    from components.case_network import build_network_data
    try:
        data = build_network_data(None)
        return data
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

@app.post("/api/timesfm/financial_paths")
def api_tfm_financial_paths(req: TfmFinancialPathsRequest):
    from data_cases import load_all_cases
    from timesfm_engine import forecast_financial_paths, TIMESFM_AVAILABLE
    import numpy as np
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
            try: revenues.append(float(v))
            except: pass
            
    if not revenues: revenues = [10_000_000.0]
    
    # 200 paths, downsampled to 50 for frontend drawing to save bandwidth
    gbm_paths = forecast_financial_paths(revenues, req.n_periods, n_paths=200)[:50].tolist()
    gbm_median = np.median(forecast_financial_paths(revenues, req.n_periods, n_paths=200), axis=0).tolist()
    
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
@app.get("/api/analysis/quantitative")
def api_quantitative():
    from analysis_regression import run_quantitative_contract_analysis
    res = run_quantitative_contract_analysis()
    if res is None:
        raise HTTPException(status_code=400, detail="Not enough data (minimum 50 cases required).")
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
    rate: float
    note: str = ""

@app.post("/api/settings/interest")
def update_interest_rate(req: InterestRateUpdate):
    from base_rate_master import upsert_base_rate
    upsert_base_rate(req.month, req.rate, req.note)
    return {"status": "success"}

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
        
    patches = {
        "final_status": req.status,
        "final_rate": req.final_rate,
        "base_rate_at_time": req.base_rate_at_time,
        "loan_conditions": req.loan_conditions,
        "competitor_name": req.competitor_name,
        "competitor_rate": req.competitor_rate if req.competitor_rate > 0 else None,
        "final_note": req.note,
    }
    if req.status == "成約" and req.final_rate > 0:
        patches["winning_spread"] = req.final_rate - req.base_rate_at_time
    if req.status == "失注":
        patches["lost_reason"] = req.lost_reason

    if update_case(target_case_id, patches):
        # 自動最適化ロジックなどのトリガー（もしあれば）
        return {"status": "success", "message": f"Results updated for {target_case_id}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update DB")

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent_hub/novel/generate")
def generate_novel_api():
    """
    文豪AI「波乱丸」の小説生成エンドポイント。
    novelist_agent.py の streamlit依存部分を回避し、Gemini API を直接呼び出す。
    """
    from novelist_agent import (
        init_novel_db, get_latest_episode_no, _collect_recent_screenings,
        _collect_hub_events, _collect_math_discoveries, _collect_recent_crosstalk,
        _parse_and_save_civ_record, get_novel_system_prompt, _fallback_novel
    )
    import sqlite3 as _sqlite3
    import datetime as _dt
    import re as _re

    init_novel_db()

    # エピソード番号の決定
    _NOVEL_DB = os.path.join(_REPO_ROOT, "data", "novelist_agent.db")
    conn = _sqlite3.connect(_NOVEL_DB)
    last = conn.execute("SELECT MAX(episode_no) FROM novels").fetchone()[0]
    conn.close()
    episode_no = (last or 0) + 1
    now = _dt.datetime.now()
    week_label = now.strftime("第%Y年%m月%d日号")

    # ネタ収集
    screenings = _collect_recent_screenings(5)
    hub_events = _collect_hub_events(10)
    math_hits = _collect_math_discoveries(3)
    crosstalk = _collect_recent_crosstalk(3)

    # プロンプト構築
    neta_lines = [f"第{episode_no}話の執筆をお願いします。今週のネタ："]
    if screenings:
        neta_lines.append("\n【今週の審査案件（ネタ素材）】")
        for s in screenings:
            neta_lines.append(f"  - {s.get('company','?')} ({s.get('industry','?')}) スコア={s.get('score','?')}")
    if hub_events:
        neta_lines.append("\n【エージェントたちの出来事】")
        for e in hub_events[:5]:
            neta_lines.append(f"  - [{e.get('ts','')[:10]}] {e.get('agent','?')}: {e.get('detail','?')[:80]}")
    if math_hits:
        neta_lines.append("\n【Dr.Algoの発見】")
        for m in math_hits:
            neta_lines.append(f"  - {m[:100]}")
    if crosstalk:
        neta_lines.append("\n【最近のエージェント間の発言】")
        for c in crosstalk:
            neta_lines.append(f"  - {c.get('agent','?')}: {c.get('thought','?')[:80]}")

    prompt = "\n".join(neta_lines)

    # Gemini API 直接呼び出し
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEYが設定されていません。.streamlit/secrets.tomlを確認してください。")

    try:
        from ai_chat import _chat_for_thread
        system_prompt = get_novel_system_prompt("sf_drama")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = _chat_for_thread(
            "gemini", "", messages,
            timeout_seconds=180,
            api_key=api_key,
            gemini_model="gemini-2.0-flash",
            max_output_tokens=8192
        )
        text = (raw.get("message") or {}).get("content", "") or ""
        
        if not text or "APIキーが設定されていません" in text or "Gemini API エラー" in text[:50]:
            raise RuntimeError(f"Gemini 応答エラー: {text[:200]}")

        # タイトル抽出（「題名：〇〇」または「タイトル：〇〇」パターン）
        title = f"第{episode_no}話"
        for pattern in [r'(?:題名|タイトル|題)[：:]\s*(.+)', r'^#+\s+(.+)', r'^「(.+)」']:
            m = _re.search(pattern, text, _re.MULTILINE)
            if m:
                title = m.group(1).strip()[:60]
                break

        # DB保存
        conn = _sqlite3.connect(_NOVEL_DB)
        conn.execute(
            "INSERT INTO novels (ts, week_label, title, body, episode_no) VALUES (?,?,?,?,?)",
            (now.isoformat(), week_label, title, text, episode_no)
        )
        conn.commit()
        conn.close()

        # 文明記録の解析（バックグラウンドで）
        try:
            _parse_and_save_civ_record(text, episode_no)
        except Exception:
            pass

        return {
            "title": title,
            "body": text,
            "week_label": week_label,
            "episode_no": episode_no
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # フォールバック小説を返す
        fallback = _fallback_novel(episode_no, week_label)
        return {**fallback, "error": str(e)}

class AgentRunRequest(BaseModel):
    agent_id: str  # benchmark, market, gunshi, team, slack, anomaly, retrain
    params: Dict[str, Any] = {}

@app.post("/api/agent_hub/run_agent")
def run_agent_api(req: AgentRunRequest):
    from components.agent_hub import (
        _run_benchmark_agent, _run_market_agent, _run_retrain_trigger
    )
    
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

def _run_benchmark_agent_standalone(industry: str):
    from ai_chat import _chat_for_thread
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
    res_raw = _chat_for_thread("gemini", "", messages, timeout_seconds=60, api_key=api_key)
    content = (res_raw.get("message") or {}).get("content", "")
    try:
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match: return json.loads(match.group())
        return {"error": "JSON parse error", "raw": content}
    except: return {"error": "Failed to parse AI response", "raw": content}

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


