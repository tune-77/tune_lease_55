from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import json
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

_tfm_spec = _ilu.spec_from_file_location("timesfm_engine", os.path.join(_REPO_ROOT, "timesfm_engine.py"))
_tfm_mod = _ilu.module_from_spec(_tfm_spec)
sys.modules["timesfm_engine"] = _tfm_mod
_tfm_spec.loader.exec_module(_tfm_mod)

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
    # startup: Obsidian ナレッジのバックグラウンドインデックス化
    try:
        from api.knowledge.indexer import start_background_indexing
        start_background_indexing()
    except Exception as e:
        print(f"[API] knowledge indexing start failed (non-fatal): {e}")
    # startup: Obsidian フィードバックのバックグラウンド読み込み
    try:
        from api.knowledge.feedback_watcher import start_background_feedback_loading
        start_background_feedback_loading()
    except Exception as e:
        print(f"[API] feedback loading start failed (non-fatal): {e}")
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
    pd_percent: float
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
    score: float
    pd_pct: float
    resale_eval: str = "B"
    repeat_count: int = 0
    subsidy_flag: bool = False
    bank_support: bool = False
    intuition_score: float = 50.0
    company_name: str = ""
    asset_name: str = ""


@app.post("/api/gunshi/stream")
async def gunshi_stream(req: GunshiStreamRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set")

    async def event_generator():
        async for chunk in stream_gunshi_gemini(req.model_dump(), api_key):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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

@app.post("/api/gunshi/chat")
def generate_gunshi_chat(req: GunshiChatRequest):
    from shinsa_gunshi import PHRASES_100, build_gunshi_prompt
    try:
        _mode = (req.mode or "gunshi").lower()
        if _mode == "chat" and (req.message or "").strip():
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
                        "pd_percent": req.pd_percent,
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

        advices = PHRASES_100.get("逆転アドバイス", [])
        import random
        sampled = random.sample(advices, min(3, len(advices)))

        has_case_context = req.score != 0 or bool((req.industry_major or "").strip())
        if has_case_context:
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
            prompt += (
                "\n\n【追加方針】\n"
                "ユーザーの質問がこの案件の逆転戦略から外れ、業界動向・他社事例・一般的な与信相談であっても構いません。"
                "案件文脈を必要に応じて参照しつつ、質問そのものに丁寧かつ実務的に答えてください。"
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
    """文豪AI「波乱丸」の小説生成エンドポイント。novelist_agent.generate_novel() に委譲。"""
    from novelist_agent import generate_novel
    try:
        result = generate_novel()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
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

    return defaults
