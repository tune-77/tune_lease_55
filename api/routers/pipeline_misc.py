"""知識グラフ・支払アラート・補助金・耐用年数・バッチ審査ルーター (REV-234 Phase11)"""
from __future__ import annotations

import json, os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel

from api.db_connection import current_backend, get_connection

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

router = APIRouter(tags=["pipeline-misc"])


# ── DB helpers ──────────────────────────────────────────────────────────────

def _db_available() -> bool:
    """Cloud SQL ではローカル SQLite ファイルがなくても DB 利用可能とみなす。"""
    from runtime_paths import get_db_path  # type: ignore[import]
    return current_backend() == "postgresql" or os.path.exists(get_db_path())


def _table_exists(cur, table_name: str) -> bool:
    """現在のDBバックエンドでテーブル存在確認を行う。"""
    if current_backend() == "postgresql":
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        return bool(row and row[0])
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return bool(cur.fetchone())


# ── batch constants ──────────────────────────────────────────────────────────

BATCH_MAX_CSV_BYTES = 5 * 1024 * 1024
BATCH_MAX_ROWS = 1000
BATCH_TOKEN_TTL_SECONDS = 30 * 60
_batch_result_cache: dict[str, dict] = {}


# ── models ───────────────────────────────────────────────────────────────────

class BatchScoreRequest(BaseModel):
    csv_text: Optional[str] = None
    csv_base64: Optional[str] = None


class BatchSaveRequest(BatchScoreRequest):
    confirmed: bool = False
    batch_token: Optional[str] = None


# ── knowledge graph helpers ─────────────────────────────────────────────────

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
    if "projects/tune_lease_55/news/" in low or "research" in low or "clippings" in low or "業界リスクニュース/" in path or "リースニュース/" in path:
        return "research"
    if "daily/" in low:
        return "daily"
    if "wiki" in low or "検索語" in path:
        return "wiki"
    return "knowledge"


def _knowledge_graph_source(path: str) -> dict[str, str | bool]:
    low = path.lower()
    if "projects/tune_lease_55/cases/" in low:
        return {"kind": "case", "label": "過去案件", "highlight": True}
    if "projects/tune_lease_55/feedback/" in low or "improvement" in low:
        return {"kind": "feedback", "label": "改善ログ", "highlight": True}
    if "projects/tune_lease_55/news/" in low or "research" in low or "clippings" in low or "業界リスクニュース/" in path or "リースニュース/" in path:
        return {"kind": "research", "label": "調査・ニュース", "highlight": True}
    if "daily/" in low:
        return {"kind": "daily", "label": "日次メモ", "highlight": True}
    if "wiki" in low or "検索語" in path:
        return {"kind": "wiki", "label": "Wiki", "highlight": True}
    if "projects/tune_lease_55/asset" in low:
        return {"kind": "asset", "label": "物件・残価", "highlight": True}
    return {"kind": "knowledge", "label": "知識ノート", "highlight": False}

# ── pipeline / subsidies helpers (shared copies kept in main.py too) ────────

def _log_bigrams(s: str) -> set[str]:
    import re as _r
    s = _r.sub(r'\s+', '', s.lower())
    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def _is_implemented(title: str, impl_titles: set[str], threshold: float = 0.45) -> bool:
    tl = title.lower()
    for impl in impl_titles:
        il = impl.lower()
        if tl == il:
            return True
        if len(title) >= 6 and (tl in il or il in tl):
            return True
        sa, sb = _log_bigrams(title), _log_bigrams(impl)
        if sa and sb and len(sa & sb) / len(sa | sb) >= threshold:
            return True
    return False


def _dedup_by_similarity(items: list[dict], threshold: float = 0.50) -> list[dict]:
    """タイトルの類似度でまとめ、代表1件だけ残す。"""
    result: list[dict] = []
    for it in items:
        title = it.get("title", "")
        tl = title.lower()
        matched = False
        for kept in result:
            kt = kept.get("title", "").lower()
            if tl == kt or (len(title) >= 6 and (tl in kt or kt in tl)):
                matched = True
                break
            sa, sb = _log_bigrams(title), _log_bigrams(kept.get("title", ""))
            if sa and sb and len(sa & sb) / len(sa | sb) >= threshold:
                matched = True
                break
        if not matched:
            result.append(it)
    return result


def _load_applied_from_ledger() -> tuple[set[str], set[str]]:
    """ledger.jsonl から applied の key セットとタイトルセットを返す。"""
    import json as _j
    applied_keys: set[str] = set()
    applied_titles: set[str] = set()
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    if not os.path.exists(ledger_path):
        return applied_keys, applied_titles
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _j.loads(line)
                if obj.get("status") == "applied":
                    applied_keys.add(obj.get("key", ""))
                    t = obj.get("title", "")
                    if t:
                        applied_titles.add(t)
            except Exception:
                pass
    return applied_keys, applied_titles


def _find_similar_pipeline_items(text: str, threshold: float = 0.38) -> list[dict]:
    """テキストと類似するパイプライン改善候補（レポート＋ledger）を返す（上位5件）。"""
    import glob as _g, json as _j
    log_dir = os.path.expanduser("~/Library/Logs/tunelease")
    candidates: list[dict] = []
    seen_titles: set[str] = set()
    # レポートから収集
    reports = sorted(
        _g.glob(os.path.join(log_dir, "reports", "improvement_report_*.json")),
        reverse=True,
    )
    for rpath in reports[:3]:
        try:
            d = _j.load(open(rpath, encoding="utf-8"))
            for item in d.get("needs_review", []) + d.get("applied_improvements", []):
                t = item.get("title", "")
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    candidates.append({"id": item.get("id", ""), "title": t, "status": "needs_review"})
        except Exception:
            pass
    # ledger から収集（最新のステータスを優先）
    ledger_path = os.path.join(log_dir, "ledger.jsonl")
    if os.path.exists(ledger_path):
        ledger_latest: dict[str, dict] = {}
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = _j.loads(line.strip())
                    t = obj.get("title", "")
                    if t:
                        ledger_latest[t] = obj
                except Exception:
                    pass
        for t, obj in ledger_latest.items():
            if t not in seen_titles:
                seen_titles.add(t)
                candidates.append({"id": obj.get("id", ""), "title": t, "status": obj.get("status", "")})

    matches: list[dict] = []
    for c in candidates:
        if _is_implemented(text, {c["title"]}, threshold=threshold):
            matches.append(c)
        if len(matches) >= 5:
            break
    return matches

def _load_title_to_rev() -> dict[str, str]:
    """最新の improvement_report_*.json からタイトル→REV番号マップを返す。"""
    import glob as _g
    import json as _j
    reports = sorted(
        _g.glob(os.path.expanduser("~/Library/Logs/tunelease/reports/improvement_report_*.json")),
        reverse=True,
    )
    mapping: dict[str, str] = {}
    for rpath in reports[:5]:
        try:
            d = _j.load(open(rpath, encoding="utf-8"))
            for item in d.get("needs_review", []) + d.get("applied_improvements", []):
                t = item.get("title", "")
                rev = item.get("id", "")
                if t and rev and t not in mapping:
                    mapping[t] = rev
        except Exception:
            pass
    return mapping


def _load_obsidian_implemented_titles() -> set[str]:
    """Obsidian 実装済み改善一覧のタイトルを返す。"""
    from runtime_paths import resolve_obsidian_vault  # type: ignore[import]
    vault = resolve_obsidian_vault()
    index_file = vault / "tuneLease55/改善策インデックス_2026.md"
    if not index_file.exists():
        return set()
    implemented: set[str] = set()
    in_section = False
    for line in index_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "実装済み改善一覧" in stripped:
            in_section = True
            continue
        if in_section and stripped.startswith("#"):
            break
        if in_section and "✅ 実装済" in stripped:
            title = stripped.replace("✅ 実装済", "").split("<!--")[0].strip()
            if title:
                implemented.add(title)
    return implemented

# ── useful-life helpers ─────────────────────────────────────────────────────

_USEFUL_LIFE_TABLE: list[dict] | None = None

def _load_useful_life_table() -> list[dict]:
    global _USEFUL_LIFE_TABLE
    if _USEFUL_LIFE_TABLE is None:
        table_path = os.path.join(_REPO_ROOT, "api", "useful_life_table.json")
        with open(table_path, "r", encoding="utf-8") as f:
            _USEFUL_LIFE_TABLE = json.load(f)
    return _USEFUL_LIFE_TABLE

# ── batch helpers ───────────────────────────────────────────────────────────

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
    from api.prediction_snapshot import record_saved_case_prediction

    saved_count = 0
    with_result = 0
    excluded_saved_count = 0
    for db_data in db_results:
        case_id = save_case_log(db_data)
        if case_id:
            saved_count += 1
            record_saved_case_prediction(
                case_id=str(case_id),
                case_data=db_data,
                source="batch_save",
            )
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


# ── shared copy: _record_memory_usage_if_available ─────────────────────────

def _record_memory_usage_if_available(
    *,
    surface: str,
    question: str,
    response: str,
    knowledge_refs: list[str] | None = None,
    pdca_block: str = "",
    judgment_learning_used: bool = False,
    extra: dict | None = None,
) -> None:
    """Log which memory layers influenced a response for later audit."""
    try:
        import hashlib as _hashlib
        import json as _json
        from datetime import datetime as _dt

        log_path = Path(_REPO_ROOT) / "data" / "case_memory_usage_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": _dt.now().isoformat(timespec="seconds"),
            "surface": surface,
            "question_hash": _hashlib.sha256((question or "").encode("utf-8")).hexdigest(),
            "question_preview": str(question or "")[:160],
            "response_hash": _hashlib.sha256((response or "").encode("utf-8")).hexdigest(),
            "knowledge_refs": list(knowledge_refs or [])[:12],
            "pdca_applied": bool(str(pdca_block or "").strip()),
            "pdca_preview": str(pdca_block or "")[:500],
            "judgment_learning_used": bool(judgment_learning_used),
            **(extra or {}),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        print(f"[MemoryUsageLog] エラー: {_e}")


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

    _record_memory_usage_if_available(
        surface="batch_save" if save_to_db else "batch_score",
        question=f"batch_rows={total}",
        response=f"summary={summary}",
        knowledge_refs=["batch_scoring", "industry_normalizer", "scoring_core"],
        pdca_block="",
        judgment_learning_used=False,
        extra={
            "total": total,
            "saved_count": saved_count,
            "with_result": with_result,
            "save_to_db": bool(save_to_db),
        },
    )

    return _build_batch_response(df_in, df_out, summary, batch_token=batch_token)


async def _proxy_git_push_db() -> None:
    """batch/save バックグラウンドタスク用: main の _git_push_db を遅延呼び出し。"""
    from api.main import _git_push_db  # noqa: PLC0415  (lazy to avoid circular import)
    await _git_push_db()


# ── endpoints ───────────────────────────────────────────────────────────────

@router.get("/api/knowledge/graph")
def get_knowledge_graph(limit: int = 180):
    """インデックス済み Obsidian ナレッジをファイル単位の3Dグラフ用に返す。"""
    try:
        from api.knowledge.vector_store import get_store

        limit = max(30, min(int(limit or 180), 420))
        store = get_store()
        store._ensure_collection()  # collection only; does not force encoder/network
        collection = store._collection
        try:
            chunk_total = collection.count() if collection is not None else 0
            raw = collection.get(include=["metadatas"]) if chunk_total else None
        except Exception as stale_error:
            # 再インデックスでコレクションが作り直されると、キャッシュ済みハンドルが
            # "Collection [...] does not exist" で無効になる。取り直して1回だけリトライ。
            if "does not exist" not in str(stale_error):
                raise
            store._collection = None
            store._client = None
            store._ensure_collection()
            collection = store._collection
            chunk_total = collection.count() if collection is not None else 0
            raw = collection.get(include=["metadatas"]) if chunk_total else None
        if not raw:
            return {
                "nodes": [],
                "edges": [],
                "summary": {"indexed_chunks": 0, "notes": 0, "links": 0, "limit": limit},
                "legend": [],
            }
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
                "source": _knowledge_graph_source(path),
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
                "source_kind": note["source"]["kind"],
                "source_label": note["source"]["label"],
                "source_highlight": note["source"]["highlight"],
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



@router.get("/api/payment/alerts")
def get_payment_alerts():
    """延滞・デフォルト案件を検出してアラートリストを返す（REV-070）。"""
    if not _db_available():
        return {"alerts": [], "summary": {"normal": 0, "overdue": 0, "default": 0, "completed": 0}}
    with get_connection() as conn:
        cur = conn.cursor()
        if not _table_exists(cur, "payment_history"):
            return {"alerts": [], "summary": {"normal": 0, "overdue": 0, "default": 0, "completed": 0}}
        cur.execute("""
            SELECT ph.id, ph.contract_id, ph.check_date, ph.payment_status,
                   ph.overdue_amount, ph.screening_score, ph.notes,
                   pc.industry_sub, pc.score as original_score
            FROM payment_history ph
            LEFT JOIN past_cases pc ON ph.contract_id = pc.id
            ORDER BY ph.check_date DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
    summary = {"normal": 0, "overdue": 0, "default": 0, "completed": 0}
    alerts = []
    for row in rows:
        status = row.get("payment_status", "")
        if status == "正常":
            summary["normal"] += 1
        elif status == "延滞":
            summary["overdue"] += 1
            alerts.append({**row, "severity": "warning", "message": f"延滞発生 — 過延滞額: {row.get('overdue_amount', 0):,}円"})
        elif status == "デフォルト":
            summary["default"] += 1
            alerts.append({**row, "severity": "critical", "message": "デフォルト — 早急な対応が必要です"})
        elif status == "完済":
            summary["completed"] += 1
    return {"alerts": alerts, "summary": summary, "total": len(rows)}


@router.get("/api/subsidies")
def get_subsidies(q: str = ""):
    """補助金マスタ一覧を返す。q で asset_keywords/name を部分一致フィルタ（REV-022/047）。"""
    if not _db_available():
        return []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM subsidy_master WHERE active = 1 ORDER BY max_amount DESC")
        rows = [dict(r) for r in cur.fetchall()]
    if q.strip():
        q_l = q.lower()
        rows = [r for r in rows if q_l in (r.get("name") or "").lower() or q_l in (r.get("asset_keywords") or "").lower() or q_l in (r.get("notes") or "").lower()]
    return rows


@router.get("/api/asset/useful-life-all")
def get_useful_life_all():
    """法定耐用年数の全品目をカテゴリ付きで返す（REV-085/121）。"""
    import json as _json
    json_path = os.path.join(_REPO_ROOT, "static_data", "useful_life_equipment.json")
    if not os.path.exists(json_path):
        return {"categories": []}
    with open(json_path, encoding="utf-8") as f:
        return _json.load(f)


@router.get("/api/asset/useful-life-search")
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


@router.get("/api/batch/template")
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


@router.post("/api/batch/score")
def score_batch(req: BatchScoreRequest):
    """CSVを一括スコアリングする。DB保存は /api/batch/save で明示的に行う。"""
    return _run_batch_scoring(req, save_to_db=False)


@router.post("/api/batch/save")
def save_batch(req: BatchSaveRequest, background_tasks: BackgroundTasks):
    """確認済みCSVを一括スコアリングし、過去案件DBへ保存する。"""
    if not req.confirmed:
        raise HTTPException(status_code=422, detail="confirmed=true が必要です")
    if req.batch_token:
        result = _save_cached_batch(req.batch_token)
    else:
        result = _run_batch_scoring(req, save_to_db=True)
    background_tasks.add_task(_proxy_git_push_db)
    return result
