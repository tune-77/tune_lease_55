"""システム・マスター・業種エンドポイントルーター (REV-234 Phase12)"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db_connection import current_backend, get_connection

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

router = APIRouter(tags=["system-misc"])


# ── DB helper (inline copy) ───────────────────────────────────────────────────

def _db_available() -> bool:
    """Cloud SQL ではローカル SQLite ファイルがなくても DB 利用可能とみなす。"""
    from runtime_paths import get_db_path  # type: ignore[import]
    return current_backend() == "postgresql" or os.path.exists(get_db_path())


# ── system/cloud-status helpers ─────────────────────────────────────────────

def _cloud_db_status() -> dict:
    backend = current_backend()
    status = {
        "backend": backend,
        "available": False,
        "database_url_configured": bool(os.environ.get("DATABASE_URL", "").strip()),
        "local_db_exists": os.path.exists(_LEASE_DB_PATH),
        "error": "",
    }
    if not _db_available():
        status["error"] = "DB is not configured or local SQLite file is missing"
        return status
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        status["available"] = True
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _cloud_gcs_vault_status() -> dict:
    vault_dir = Path(os.environ.get("GCS_VAULT_LOCAL_DIR", "/tmp/gcs_vault"))
    md_files = sorted(vault_dir.rglob("*.md")) if vault_dir.exists() else []
    latest_mtime = max((p.stat().st_mtime for p in md_files), default=None)
    return {
        "enabled": os.environ.get("USE_GCS_VAULT", "").lower() in ("1", "true"),
        "bucket": os.environ.get("GCS_BUCKET", "tune-lease-55-data"),
        "prefix": os.environ.get("GCS_VAULT_PREFIX", "vault/"),
        "local_dir": str(vault_dir),
        "local_dir_exists": vault_dir.exists(),
        "markdown_count": len(md_files),
        "latest_local_mtime": latest_mtime,
    }


def _cloud_chroma_status() -> dict:
    """ChromaDB（obsidian_knowledge コレクション）の接続状態を返す。

    encoder はここではロードしない（cloud-status 呼び出しで ~500MB のモデル
    読み込みを誘発しないため、現在の状態だけを覗く）。
    """
    status: dict = {
        "indexing_enabled": os.environ.get("ENABLE_OBSIDIAN_INDEXING", "false").lower() == "true",
        "connected": False,
        "document_count": 0,
        "chroma_dir": "",
        "encoder_loaded": False,
        "encoder_model_local": False,
    }
    try:
        from api.knowledge.vector_store import get_store

        store = get_store()
        status["chroma_dir"] = store._chroma_dir
        status["document_count"] = store.count()
        status["connected"] = status["document_count"] > 0
        status["encoder_loaded"] = store._encoder is not None
        status["encoder_model_local"] = os.path.isdir(store._model_name)
    except Exception as exc:
        status["error"] = str(exc)
    return status


# ── loop-proof helpers ──────────────────────────────────────────────────────

_loop_proof_mod = None


def _get_build_loop_proof():
    """scripts/build_loop_proof.py を遅延ロード（初回のみ）。"""
    global _loop_proof_mod
    if _loop_proof_mod is None:
        import importlib.util as _ilu2

        _p = os.path.join(_REPO_ROOT, "scripts", "build_loop_proof.py")
        _spec = _ilu2.spec_from_file_location("build_loop_proof", _p)
        _mod = _ilu2.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _loop_proof_mod = _mod
    return _loop_proof_mod



# ── endpoints ───────────────────────────────────────────────────────────────

@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/api/system/cloud-status")
def get_cloud_status():
    db = _cloud_db_status()
    gcs_vault = _cloud_gcs_vault_status()
    chroma = _cloud_chroma_status()
    ready = db["available"] and (
        not gcs_vault["enabled"] or gcs_vault["markdown_count"] > 0
    )
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "db": db,
        "gcs_vault": gcs_vault,
        "chroma": chroma,
        "cloud_run": {
            "service": os.environ.get("K_SERVICE", ""),
            "revision": os.environ.get("K_REVISION", ""),
            "configuration": os.environ.get("K_CONFIGURATION", ""),
        },
    }


@router.get("/api/loop-proof")
def get_loop_proof():
    """審査員向け「ループが閉じた証拠」の集計値。

    ledger 由来はライブ集計、reports 由来（Cloud Run では .dockerignore で欠落）は
    バンドル済み static_data スナップショットで補完して返す。
    """
    try:
        return _get_build_loop_proof().load_payload()
    except Exception as exc:  # noqa: BLE001
        logger.warning("loop-proof集計に失敗: %s", exc)
        raise HTTPException(status_code=500, detail="loop-proof metrics unavailable")



@router.get("/")
def read_root():
    return {"message": "Lease Scoring API is running."}


@router.get("/api/master/industries")
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

@router.get("/api/master/assets")
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

@router.get("/api/master/qualitative")
def get_qualitative_items():
    try:
        from constants import QUALITATIVE_SCORING_CORRECTION_ITEMS
        return {"items": QUALITATIVE_SCORING_CORRECTION_ITEMS}
    except Exception:
        return {"items": []}



class IndustrySuggestRequest(BaseModel):
    asset_name: str = ""
    industry_detail: str = ""
    company_name: str = ""


def _load_industry_master() -> dict:
    paths = [
        os.path.join(_REPO_ROOT, "static_data", "industry_trends_jsic.json"),
        os.path.join(_REPO_ROOT, "industry_trends_jsic.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _industry_subs_for_major(master: dict, major: str) -> list[str]:
    entry = master.get(major)
    if not entry:
        return []
    if isinstance(entry, list):
        return [str(item) for item in entry if item]
    if isinstance(entry, dict):
        sub = entry.get("sub")
        if isinstance(sub, dict):
            return list(sub.keys())
        return [str(k) for k in entry.keys() if k != "mapping"]
    return []


_INDUSTRY_SUGGESTION_RULES = [
    {
        "major": "E 製造業",
        "sub_terms": [("09 食料品製造業", ["食品", "惣菜", "弁当", "菓子", "パン"]), ("21 金属製品製造業", ["金属", "板金", "溶接"]), ("24 生産用機械器具製造業", ["工作機械", "加工機", "旋盤", "マシニング", "製造設備", "ロボット"])],
        "terms": ["製造", "工場", "加工", "工作機械", "生産設備", "プレス", "射出", "溶接", "切削"],
    },
    {
        "major": "D 建設業",
        "sub_terms": [("06 総合工事業", ["建設", "土木", "建築"]), ("07 職別工事業", ["内装", "足場", "電気工事", "管工事"]), ("08 設備工事業", ["設備工事", "空調", "配管"])],
        "terms": ["建機", "ショベル", "クレーン", "ダンプ", "土木", "建設", "工事", "足場"],
    },
    {
        "major": "H 運輸業・郵便業",
        "sub_terms": [("44 道路貨物運送業", ["トラック", "配送", "貨物", "運送"]), ("43 道路旅客運送業(バス・タクシー)", ["バス", "タクシー"])],
        "terms": ["車両", "トラック", "冷凍車", "配送", "運送", "物流", "フォークリフト"],
    },
    {
        "major": "P 医療・福祉",
        "sub_terms": [("83 医療業(病院・診療所)", ["医療", "クリニック", "歯科", "病院"]), ("85 社会保険・社会福祉・介護事業", ["介護", "福祉", "老人ホーム"])],
        "terms": ["医療", "検査機", "ct", "mri", "レントゲン", "超音波", "歯科", "介護"],
    },
    {
        "major": "G 情報通信業",
        "sub_terms": [("39 情報サービス業", ["システム", "ソフトウェア", "サーバ", "クラウド"]), ("40 インターネット附随サービス業", ["ec", "web", "アプリ"])],
        "terms": ["it", "oa", "pc", "サーバ", "ネットワーク", "システム", "ソフトウェア", "クラウド"],
    },
    {
        "major": "M 宿泊業・飲食サービス業",
        "sub_terms": [("76 飲食店", ["飲食", "厨房", "レストラン", "カフェ"]), ("75 宿泊業", ["ホテル", "旅館"])],
        "terms": ["厨房", "飲食", "店舗", "レストラン", "カフェ", "ホテル", "宿泊"],
    },
    {
        "major": "I 卸売業・小売業",
        "sub_terms": [("56-61 各種小売業", ["店舗", "小売", "食品販売", "スーパー"]), ("50-55 各種卸売業", ["卸売", "倉庫"])],
        "terms": ["小売", "店舗什器", "pos", "販売", "卸売", "倉庫"],
    },
    {
        "major": "R サービス業(他に分類されないもの)",
        "sub_terms": [("89 自動車整備業", ["整備", "車検"]), ("サービス業全般", ["派遣", "職業紹介", "清掃", "保守", "サービス"])],
        "terms": ["サービス", "整備", "清掃", "保守", "レンタル", "オフィス家具", "内装"],
    },
]


SERVICE_GENERAL_LABEL = "サービス業全般"
_SERVICE_GENERAL_ALIASES = {
    "91 職業紹介・労働者派遣業",
    "R サービス業(他に分類されないもの)",
}


def _normalize_industry_for_stats(industry: str) -> str:
    label = str(industry or "").strip()
    if label in _SERVICE_GENERAL_ALIASES:
        return SERVICE_GENERAL_LABEL
    return label


@router.post("/api/industry/suggest")
def suggest_industry(req: IndustrySuggestRequest):
    text = " ".join([req.asset_name, req.industry_detail, req.company_name]).lower()
    master = _load_industry_master()
    suggestions: list[dict] = []
    for rule in _INDUSTRY_SUGGESTION_RULES:
        matched_terms = [term for term in rule["terms"] if term.lower() in text]
        for sub_name, sub_terms in rule.get("sub_terms", []):
            matched_terms.extend([term for term in sub_terms if term.lower() in text])
        if not matched_terms:
            continue

        major = rule["major"]
        if master and major not in master:
            continue
        subs = _industry_subs_for_major(master, major)
        preferred_sub = next((sub_name for sub_name, sub_terms in rule.get("sub_terms", []) if any(term.lower() in text for term in sub_terms)), "")
        industry_sub = preferred_sub if preferred_sub in subs else (subs[0] if subs else preferred_sub)
        confidence = min(0.95, 0.55 + 0.12 * len(set(matched_terms)))
        suggestions.append({
            "industry_major": major,
            "industry_sub": industry_sub,
            "confidence": round(confidence, 2),
            "matched_terms": sorted(set(matched_terms))[:6],
            "reason": f"{', '.join(sorted(set(matched_terms))[:3])} から推測",
        })

    suggestions.sort(key=lambda item: item["confidence"], reverse=True)
    return {"suggestions": suggestions[:3]}



# ── industry/stats (separate location in main.py) ────────────────────────────

@router.get("/api/industry/stats")
def api_industry_stats():
    """業種別成約率・平均スコア集計（REV-055）"""
    import json
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT industry_sub, final_status, COUNT(*) as cnt, AVG(score) as avg_score
                FROM past_cases
                WHERE industry_sub IS NOT NULL AND industry_sub != '' AND industry_sub != '0'
                  AND final_status IN ('成約', '失注')
                GROUP BY industry_sub, final_status
            """)
            rows = cur.fetchall()
    except Exception as e:
        import logging as _lg; _lg.getLogger(__name__).error("api_industry_stats DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    industry_data: dict = {}
    for industry, status, cnt, avg_sc in rows:
        industry = _normalize_industry_for_stats(industry)
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

