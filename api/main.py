import os as _os_early
# macOSでのOpenMPスレッド上限エラー・MPS GPU競合によるSIGSEGV防止
_os_early.environ.setdefault("OMP_NUM_THREADS", "1")
_os_early.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os_early.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os_early.environ.setdefault("MKL_NUM_THREADS", "1")
_os_early.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os_early.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# HuggingFace tokenizers (sentence-transformers が内部で利用) の Rust スレッドが
# fork 後に「leaked semaphore」を量産する macOS 既知問題への対策。
# 並列化を無効化することでセマフォリーク・ワーカーゾンビ化を防ぐ。
_os_early.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# sentence-transformers / huggingface_hub が起動時に
# モデル名の HEAD リクエストを飛ばすことがあり、ネットワーク待ちで
# uvicorn ワーカーが応答不能になるケースがある。既定でオフラインモード。
_os_early.environ.setdefault("HF_HUB_OFFLINE", "1")
_os_early.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
del _os_early


from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
import json
import logging
import re
import shutil
import sys
import os
from pathlib import Path
from runtime_paths import get_data_path, get_db_path

# プロジェクトルートをPYTHONPATHに追加して、既存モジュール(scoring_core)をインポート可能にする
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# sys.path に同名モジュール(data_cases等)を持つ別ディレクトリが先に入っている場合があるため、
# _REPO_ROOT を最優先(position 0)に固定する
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from api.llm_json_guard import extract_candidate_text, parse_or_recover_json, with_retry_tokens
from api.db_connection import current_backend, get_connection, placeholder
from api.cloudrun_writeback import record_cloudrun_input_event
logger = logging.getLogger(__name__)

# fire-and-forget なバックグラウンド処理（チャット後の記憶保存・ログ記録等）用の共有プール。
# 生の threading.Thread を都度spawnすると高負荷時にOSスレッドが際限なく積み上がり、
# ファイル冒頭のOMP/MPS対策コメントが警告するネイティブライブラリ競合・SIGSEGVの
# リスクが再燃するため、ワーカー数を固定して総スレッド数に上限を設ける。
from concurrent.futures import ThreadPoolExecutor
_background_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bg-task")

# PDCA反省バッチはGemini呼び出しを含む重い処理で、デバウンスなしだと未処理フィードバックが
# 5件以上残る間は案件登録のたびに多重起動しうる。実行中は新規起動をスキップする。
import threading
_pdca_reflection_lock = threading.Lock()

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

_LEASE_DB_PATH = get_db_path()
_DATA_GIT_DIR = os.environ.get("DATA_GIT_DIR", "/app/data-git")
_git_lock = asyncio.Lock()


def _db_available() -> bool:
    """Cloud SQL ではローカル SQLite ファイルがなくても DB 利用可能とみなす。"""
    return current_backend() == "postgresql" or os.path.exists(_LEASE_DB_PATH)


def _table_exists(cur, table_name: str) -> bool:
    """現在のDBバックエンドでテーブル存在確認を行う。"""
    if current_backend() == "postgresql":
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        return bool(row and row[0])
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return bool(cur.fetchone())

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


def _gemini_generate_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

from scoring_core import run_full_api_scoring, run_quick_scoring, APPROVAL_LINE, CONDITIONAL_LINE
from api.scoring_full import run_full_scoring_api
from api.gunshi_gemini import stream_gunshi_gemini
from lease_news_digest import (
    find_vault,
    build_lease_news_brief,
    get_latest_lease_news_focus,
    get_latest_lease_news_reflection,
    get_latest_lease_news_actions,
    lease_news_actions_as_text,
    record_lease_news_collection,
    record_lease_news_judgment_change,
    record_lease_news_view,
    lease_news_focus_as_text,
)
from obsidian_daily_intelligence import (
    obsidian_daily_intelligence_as_text,
    record_obsidian_daily_intelligence_event,
)
from api.schemas import (
    ScoringRequest,
    ScoringResponse,
    CaseRegisterRequest,
    DealClosureRequest,
    DealClosureResponse,
    LeaseNewsSummarizeRequest,
    LeaseNewsSummaryItem,
    ReviewImprovementRequest,
    PromptRuleRegisterRequest,
    WorkLogRequest,
    WorkLogResponse,
    BusinessPlanCheckRequest,
)
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Literal, Optional
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


def _init_sync_log_table() -> None:
    """sync_log テーブルを冪等に作成する（Cloud Run git push 結果記録用）。"""
    if not _db_available():
        return
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            if current_backend() == "postgresql":
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sync_log (
                        id SERIAL PRIMARY KEY,
                        pushed_at TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error TEXT
                    )
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sync_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pushed_at TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error TEXT
                    )
                """)
    except Exception as e:
        print(f"[sync_log] テーブル作成失敗（非致命的）: {e}")


def _record_sync_log(success: bool, error: str = "") -> None:
    """sync_log テーブルに git push 結果を記録する。"""
    if not _db_available():
        return
    import datetime
    _ph = placeholder()
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO sync_log (pushed_at, success, error) VALUES ({_ph}, {_ph}, {_ph})",
                (datetime.datetime.utcnow().isoformat(), 1 if success else 0, error),
            )
    except Exception as e:
        print(f"[sync_log] 記録失敗（非致命的）: {e}")


async def _git_push_db() -> None:
    """demo.db + mind.json を data-git にコピーして git push する（BackgroundTask 用）。"""
    if not os.path.isdir(os.path.join(_DATA_GIT_DIR, ".git")):
        return
    db_dst = os.path.join(_DATA_GIT_DIR, "data", "demo.db")
    mind_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mind.json")
    mind_dst = os.path.join(_DATA_GIT_DIR, "data", "mind.json")
    success = False
    error_msg = ""
    try:
        async with _git_lock:
            if os.path.exists(_LEASE_DB_PATH):
                shutil.copy2(_LEASE_DB_PATH, db_dst)
            if os.path.exists(mind_src):
                shutil.copy2(mind_src, mind_dst)
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c",
                "git add data/demo.db data/mind.json 2>/dev/null; "
                "git diff --cached --quiet || "
                "git commit -m 'auto: update from cloud-run'; "
                "git push",
                cwd=_DATA_GIT_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            success = proc.returncode == 0
            error_msg = stderr.decode(errors="replace") if not success else ""
    except asyncio.TimeoutError:
        error_msg = "git push timeout"
    except Exception as exc:
        error_msg = str(exc)
    _record_sync_log(success, error_msg)
    if not success:
        print(f"[git-push] 失敗: {error_msg}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: Cloud Run用の選抜Obsidian MarkdownをGCSから取得
    # ダウンロードはネットワークI/Oでタイムアウトが効かず起動を長時間ブロックしうるため、
    # readiness確認（/docs）を先に通すためバックグラウンドスレッドで実行する。
    import threading as _gcs_th
    # ChromaDB インデクサが GCS Vault 同期後の OBSIDIAN_VAULT_PATH を参照できるよう、
    # 同期完了を通知する（失敗時も set してインデクサを永久待機させない）
    _gcs_vault_sync_done = _gcs_th.Event()

    def _run_gcs_vault_sync():
        try:
            gcs_sync = _sync_gcs_vault_if_enabled()
            if gcs_sync.get("enabled"):
                print(f"[GCSVault] startup sync: {gcs_sync}")
        except Exception as e:
            print(f"[GCSVault] startup sync failed (non-fatal): {e}")
        finally:
            _gcs_vault_sync_done.set()
    _gcs_th.Thread(target=_run_gcs_vault_sync, daemon=True, name="gcs-vault-sync").start()
    # startup: DB スキーマ自動初期化（REV-167 — Cloud SQL 冷起動時のテーブル不在対策）
    try:
        from api.db_connection import ensure_schema
        ensure_schema()
    except Exception as e:
        print(f"[API] ensure_schema failed (non-fatal): {e}")
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
    # 既定では無効。ENABLE_OBSIDIAN_INDEXING=true で明示的に有効化する。
    # sentence-transformers の重い初期化が --reload 環境下でワーカーをゾンビ化させる
    # 原因になっていたため、API 主要機能から切り離す。
    import threading as _th
    if os.environ.get("ENABLE_OBSIDIAN_INDEXING", "false").lower() == "true":
        def _delayed_indexing():
            import time as _t; _t.sleep(30)
            # Cloud Run では GCS Vault 同期の完了時に OBSIDIAN_VAULT_PATH が
            # /tmp/gcs_vault へ書き換わる。indexer 側の既定パスは import 時に
            # 固定されるため、同期完了を待ってから実行時のパスを明示して渡す
            # （待たないと空の /app/obsidian_vault を索引して ChromaDB が空のままになる）
            if os.environ.get("USE_GCS_VAULT", "").lower() in ("1", "true"):
                _gcs_vault_sync_done.wait(timeout=600)
            try:
                from api.knowledge.indexer import start_background_indexing
                from runtime_paths import get_obsidian_vault_path
                start_background_indexing(get_obsidian_vault_path())
            except Exception as e:
                print(f"[API] knowledge indexing start failed (non-fatal): {e}")
        _th.Thread(target=_delayed_indexing, daemon=True, name="delayed-indexer").start()
    else:
        print("[API] Obsidian indexing disabled (set ENABLE_OBSIDIAN_INDEXING=true to enable)")
    # startup: Obsidian フィードバックのバックグラウンド読み込み（30秒遅延）
    # 既定では無効。ENABLE_FEEDBACK_LOADING=true で明示的に有効化する。
    if os.environ.get("ENABLE_FEEDBACK_LOADING", "false").lower() == "true":
        def _delayed_feedback():
            import time as _t; _t.sleep(30)
            # インデクサと同様、GCS Vault 同期後の実行時パスを明示して渡す
            if os.environ.get("USE_GCS_VAULT", "").lower() in ("1", "true"):
                _gcs_vault_sync_done.wait(timeout=600)
            try:
                from api.knowledge.feedback_watcher import start_background_feedback_loading
                from runtime_paths import get_obsidian_vault_path
                start_background_feedback_loading(get_obsidian_vault_path())
            except Exception as e:
                print(f"[API] feedback loading start failed (non-fatal): {e}")
        _th.Thread(target=_delayed_feedback, daemon=True, name="delayed-feedback").start()
    else:
        print("[API] Feedback loading disabled (set ENABLE_FEEDBACK_LOADING=true to enable)")
    # startup: 会話履歴テーブルの初期化
    try:
        from api.database import init_conversation_history_table
        init_conversation_history_table()
    except Exception as e:
        print(f"[API] conversation_history table init failed (non-fatal): {e}")
    # startup: 感情履歴テーブルの初期化 + 当日分の自動記録（REV-075）
    try:
        from api.database import init_emotion_history_table
        init_emotion_history_table()
    except Exception as e:
        print(f"[API] emotion_history table init failed (non-fatal): {e}")

    def _record_today_emotion():
        try:
            from lease_intelligence_mind import (
                _derive_complex_emotions,
                load_lease_intelligence_mind,
            )
            from lease_news_digest import find_vault
            from api.database import record_emotion_snapshot

            vault = find_vault()
            if not vault:
                return
            state = load_lease_intelligence_mind(vault)
            emotions = _derive_complex_emotions(state.get("mood", {}))
            scores = {e["key"]: float(e["score"]) for e in emotions}
            dominant = emotions[0]["key"] if emotions else ""
            record_emotion_snapshot(scores, dominant)
        except Exception as e:
            print(f"[API] emotion auto-record failed (non-fatal): {e}")

    _th.Thread(target=_record_today_emotion, daemon=True, name="emotion-recorder").start()
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
    # startup: sync_log テーブル初期化（Cloud Run git push 記録用）
    _init_sync_log_table()
    yield
    # shutdown: 結晶化スケジューラー停止
    try:
        from api.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass
    # shutdown: 最終 git push（コンテナ停止前にデータを永続化）
    if os.path.isdir(os.path.join(_DATA_GIT_DIR, ".git")):
        try:
            db_dst = os.path.join(_DATA_GIT_DIR, "data", "demo.db")
            mind_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mind.json")
            mind_dst = os.path.join(_DATA_GIT_DIR, "data", "mind.json")
            if os.path.exists(_LEASE_DB_PATH):
                shutil.copy2(_LEASE_DB_PATH, db_dst)
            if os.path.exists(mind_src):
                shutil.copy2(mind_src, mind_dst)
            import subprocess as _sp
            result = _sp.run(
                ["bash", "-c",
                 "git add data/demo.db data/mind.json 2>/dev/null; "
                 "git diff --cached --quiet || git commit -m 'auto: shutdown sync'; "
                 "git push"],
                cwd=_DATA_GIT_DIR, capture_output=True, timeout=30,
            )
            _record_sync_log(result.returncode == 0,
                             result.stderr.decode(errors="replace") if result.returncode != 0 else "")
        except Exception as _e:
            print(f"[shutdown] final git push 失敗（非致命的）: {_e}")


limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# 共有シークレットによる API アクセス制御（api/api_key_auth.py に実装。多層防御）。
from api.api_key_auth import ApiKeyAuthMiddleware, get_api_access_key
# 公開デモ用の削除保護（api/demo_guard.py に実装）。
from api.demo_guard import DemoReadonlyMiddleware, is_demo_readonly


_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra_cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
for _origin in [o.strip() for o in _extra_cors_origins.split(",") if o.strip()]:
    if _origin not in _ALLOWED_ORIGINS:
        _ALLOWED_ORIGINS.append(_origin)

app = FastAPI(
    title="Lease Scoring API",
    description="リース審査ロジックのバックエンドAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SecurityHeadersMiddleware)
# 公開デモの削除保護（DEMO_READONLY 設定時のみ有効）
app.add_middleware(DemoReadonlyMiddleware)
if is_demo_readonly():
    logger.info("DemoReadonlyMiddleware 有効: /api/* への DELETE を 403 で拒否")
# API キー認証（API_ACCESS_KEY 設定時のみ有効。CORS より内側で動くよう先に登録）
app.add_middleware(ApiKeyAuthMiddleware)
if get_api_access_key():
    logger.info("ApiKeyAuthMiddleware 有効: /api/* は X-API-Key 必須")
else:
    logger.warning(
        "API_ACCESS_KEY 未設定: API 認証は無効です。"
        "外部公開時は Cloud Run IAM(--no-allow-unauthenticated) か API_ACCESS_KEY を設定してください"
    )

# Next.js ローカルサーバーからのアクセスのみ許可（ワイルドカード廃止）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

from api.routers.ocr import router as ocr_router
app.include_router(ocr_router, prefix="/api")

from api.routers.rule_engine import router as rule_engine_router
app.include_router(rule_engine_router, prefix="/api")

from api.routers.knowledge import router as knowledge_router
app.include_router(knowledge_router, prefix="/api")

from api.routers.demo import router as demo_router
app.include_router(demo_router, prefix="/api")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


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


def _sync_gcs_vault_if_enabled() -> dict:
    """Download the selected Obsidian Markdown copy for Cloud Run RAG/memory use."""
    if os.environ.get("USE_GCS_VAULT", "").lower() not in ("1", "true"):
        return {"enabled": False, "status": "skipped"}
    try:
        import sys as _sys

        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in _sys.path:
            _sys.path.insert(0, scripts_dir)
        from gcs_vault_loader import download_vault  # type: ignore[import-not-found]

        vault_dir = download_vault(dest_dir=Path(os.environ.get("GCS_VAULT_LOCAL_DIR", "/tmp/gcs_vault")))
        os.environ["OBSIDIAN_VAULT"] = str(vault_dir)
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault_dir)
        md_count = len(list(vault_dir.rglob("*.md"))) if vault_dir.exists() else 0
        return {"enabled": True, "status": "synced", "local_dir": str(vault_dir), "markdown_count": md_count}
    except Exception as exc:
        print(f"[GCSVault] sync failed (non-fatal): {exc}")
        return {"enabled": True, "status": "error", "error": str(exc)}


@app.get("/api/system/cloud-status")
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


@app.post("/api/industry/suggest")
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


def _score_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_financial_consistency_risk(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    旧Q_risk: 財務・入力整合性チェック。
    ScoringRequestの金額は千円単位、aurion.q_riskは百万円単位前提なので変換する。
    """
    fallback = {"score": 0, "level": "ok", "patterns": [], "pattern_details": [], "role": "financial_consistency"}

    def _million(key: str, default: float = 0.0) -> float:
        raw = inputs.get(key, default)
        try:
            return float(raw or 0.0) / 1000.0
        except (TypeError, ValueError):
            return default

    try:
        from mobile_app.aurion.q_risk import detect_q_risk

        op_profit = _million("op_profit")
        gross_profit = _million("gross_profit", op_profit)
        # 粗利未入力時に「営業利益 > 粗利」を誤検知しないため、営業利益を代替する。
        if not inputs.get("gross_profit"):
            gross_profit = op_profit

        result = detect_q_risk(
            gross_profit=gross_profit,
            op_profit=op_profit,
            net_income=_million("net_income"),
            nenshu=_million("nenshu"),
            dep_expense=_million("dep_expense"),
            depreciation=_million("depreciation"),
            machines=_million("machines"),
            bank_credit=_million("bank_credit"),
            lease_credit=_million("lease_credit"),
            acquisition_cost=_million("acquisition_cost"),
        )
        return {
            "score": result.get("score", 0),
            "level": result.get("level", "ok"),
            "patterns": result.get("patterns", []),
            "pattern_details": result.get("pattern_details", []),
            "role": "financial_consistency",
        }
    except Exception as exc:
        fallback["error"] = str(exc)
        return fallback


def _build_conditional_approval_actions(inputs: dict, result: dict) -> list[dict]:
    """条件付き承認に近い案件で、審査担当者が次に取る条件を標準化する。"""
    score = _score_float(result.get("score", result.get("hantei_score")), 0.0)
    if score >= APPROVAL_LINE:
        return []

    actions: list[dict] = []

    def add(priority: str, action: str, reason: str, category: str) -> None:
        if not any(item["action"] == action for item in actions):
            actions.append({"priority": priority, "action": action, "reason": reason, "category": category})

    op_margin = _score_float(result.get("user_op_margin"), 0.0)
    equity_ratio = _score_float(result.get("user_equity_ratio"), 0.0)
    q_risk = _score_float(result.get("quantum_risk"), 0.0)
    credit_risk = _score_float(result.get("credit_risk_group_score"), 0.0)
    acquisition_cost = _score_float(inputs.get("acquisition_cost"), 0.0)
    nenshu = _score_float(inputs.get("nenshu"), 0.0)
    customer_type = str(inputs.get("customer_type") or "")
    main_bank = str(inputs.get("main_bank") or "")
    competitor = str(inputs.get("competitor") or "")
    sales_dept = str(inputs.get("sales_dept") or "")
    asset_name = str(inputs.get("asset_name") or "").strip()
    asset_purpose = str(inputs.get("asset_purpose") or "").strip()
    asset_location = str(inputs.get("asset_location") or "").strip()
    asset_evidence = str(inputs.get("asset_evidence_level") or "").strip()
    lease_term = int(_score_float(inputs.get("lease_term"), 60))

    if score < CONDITIONAL_LINE:
        add("must", "取得額の圧縮または対象物件の分割を再提案", "現条件では承認ラインまで距離があるため、エクスポージャーを先に下げる。", "構造変更")
        add("must", "代表者保証・追加担保・前受金のいずれかを条件化", "信用補完なしで押すより、損失限定条件を先に置く。", "信用補完")
    else:
        add("must", "追加資料と承認条件を先に提示", "条件付き承認圏では、審査部に突かれる前に論点を固定する。", "稟議")

    if op_margin < 2:
        add("must", "直近試算表・受注残・資金繰り表を追加取得", f"営業利益率が {op_margin:.1f}% と薄く、返済原資の説明が必要。", "財務")
    if equity_ratio < 15:
        add("should", "前受金または保証追加で自己資本の薄さを補完", f"自己資本比率が {equity_ratio:.1f}% で、損失吸収力の説明が弱い。", "信用補完")
    if q_risk >= 35:
        add("must", "設備明細・減価償却・リース残高の整合性確認", f"Q_risk {q_risk:.1f} のため、入力値の矛盾を先に潰す。", "データ確認")
    if credit_risk >= 45:
        add("must", "借入返済予定表と既存リース支払状況を確認", f"信用リスク群スコア {credit_risk:.1f} が要監視水準。", "信用確認")
    if not asset_name or not asset_purpose or asset_evidence in ("", "未確認"):
        add("must", "対象物件・用途・確認資料を稟議前に確定", "物件名・用途・見積/型式確認が不足すると、資金使途と回収可能性の説明が弱くなる。", "物件確認")
    elif asset_location:
        add("should", "設置場所と稼働実態を稟議補足へ反映", f"{asset_location} での使用前提を明記し、用途「{asset_purpose}」との整合を説明する。", "物件確認")
    if competitor == "競合あり":
        add("should", "競合見積の対象範囲・保守条件・期間差を確認", "金利差だけで比較されると失注しやすいため、条件差を明文化する。", "競合")
    if customer_type == "新規先":
        add("should", "商流・実質経営者・主要取引先を確認", "新規先は定性情報の不足が差し戻し要因になりやすい。", "定性")
    if main_bank == "メイン先":
        add("should", "メイン行の支援姿勢を稟議補足へ記載", "銀行支援がある場合は、継続取引・支援依頼書を条件補強に使える。", "銀行支援")
    if sales_dept and sales_dept != "未設定":
        add("should", "営業部の取引背景・顧客接点を補足", f"{sales_dept} の接点から、商談発生経緯・既存取引・フォロー体制を稟議へ残す。", "営業情報")
    if nenshu > 0 and acquisition_cost / nenshu >= 0.25:
        add("should", "前受金または期間短縮で年商比負担を下げる", "取得価額が年商比で大きく、月額負担と出口リスクの説明が必要。", "負担軽減")
    if lease_term >= 72:
        add("should", "リース期間短縮または中途点検条件を検討", "長期化により財務変化・物件価値変動リスクが増える。", "期間")

    return actions[:6]


def _build_screening_context_notes(
    inputs: dict,
    result: dict,
    conditional_actions: list[dict],
    rate_proposal: dict,
) -> dict:
    """入力した案件文脈を、審査コメント・条件案・リスク理由へ明示的に反映する。"""
    score = _score_float(result.get("score", result.get("hantei_score")), 0.0)
    hantei = str(result.get("hantei") or "未判定")
    op_margin = _score_float(result.get("user_op_margin", result.get("user_op")), 0.0)
    equity_ratio = _score_float(result.get("user_equity_ratio", result.get("user_eq")), 0.0)
    q_risk = _score_float(result.get("quantum_risk"), 0.0)
    credit_risk = _score_float(result.get("credit_risk_group_score"), 0.0)
    acquisition_cost = _score_float(inputs.get("acquisition_cost"), 0.0)
    nenshu = _score_float(inputs.get("nenshu"), 0.0)
    lease_term = int(_score_float(inputs.get("lease_term"), 60))

    company_name = str(inputs.get("company_name") or "").strip()
    industry_major = str(inputs.get("industry_major") or result.get("industry_major") or "").strip()
    industry_sub = str(inputs.get("industry_sub") or result.get("industry_sub") or "").strip()
    industry_detail = str(inputs.get("industry_detail") or "").strip()
    sales_dept = str(inputs.get("sales_dept") or "未設定").strip()
    customer_type = str(inputs.get("customer_type") or "").strip()
    main_bank = str(inputs.get("main_bank") or "").strip()
    competitor = str(inputs.get("competitor") or "").strip()
    deal_source = str(inputs.get("deal_source") or "").strip()
    deal_occurrence = str(inputs.get("deal_occurrence") or "").strip()
    grade = str(inputs.get("grade") or "").strip()
    asset_name = str(inputs.get("asset_name") or "").strip()
    asset_detail = str(inputs.get("asset_detail") or "").strip()
    asset_purpose = str(inputs.get("asset_purpose") or "").strip()
    asset_location = str(inputs.get("asset_location") or "").strip()
    asset_evidence = str(inputs.get("asset_evidence_level") or "").strip()
    estat_context = result.get("estat_context") or {}

    commentary: list[dict] = []
    risk_reasons: list[dict] = []
    missing_inputs: list[str] = []
    reflected_inputs: list[str] = []

    def add_comment(label: str, text: str, tone: str = "neutral", refs: list[str] | None = None) -> None:
        commentary.append({"label": label, "text": text, "tone": tone, "input_refs": refs or []})
        reflected_inputs.extend(refs or [])

    def add_risk(level: str, title: str, reason: str, refs: list[str] | None = None) -> None:
        risk_reasons.append({"level": level, "title": title, "reason": reason, "input_refs": refs or []})
        reflected_inputs.extend(refs or [])

    if company_name:
        add_comment("案件概要", f"{company_name} の {industry_sub or industry_major} 案件として、スコア {score:.1f} / 判定「{hantei}」を確認。", "neutral", ["company_name", "industry_sub", "score"])
    else:
        add_comment("案件概要", f"{industry_sub or industry_major or '業種未設定'} 案件として、スコア {score:.1f} / 判定「{hantei}」を確認。", "neutral", ["industry_sub", "score"])
        missing_inputs.append("企業名")

    if asset_name and asset_purpose:
        detail = f"対象物件「{asset_name}」は「{asset_purpose}」目的の導入。"
        if asset_location:
            detail += f" 使用場所は {asset_location}。"
        if asset_detail:
            detail += f" 型式・仕様は {asset_detail}。"
        add_comment("物件・用途", detail, "positive", ["asset_name", "asset_purpose", "asset_location", "asset_detail"])
    else:
        add_risk("high", "物件・資金使途の説明不足", "対象物件名または導入目的が不足しており、稟議で資金使途・必要性・回収可能性を説明しにくい。", ["asset_name", "asset_purpose"])
        if not asset_name:
            missing_inputs.append("対象物件名")
        if not asset_purpose:
            missing_inputs.append("導入目的・用途")

    if asset_evidence in ("", "未確認"):
        add_risk("high", "確認資料不足", "見積・型式・中古相場などの確認資料が未確認のため、対象物件の妥当性と残価リスクを追加確認する必要がある。", ["asset_evidence_level"])
        missing_inputs.append("確認資料")
    else:
        add_comment("確認資料", f"確認資料は「{asset_evidence}」。物件条件の説明根拠として稟議に反映できる。", "positive", ["asset_evidence_level"])

    if sales_dept and sales_dept != "未設定":
        add_comment("営業部", f"{sales_dept} の案件として、商談経緯・フォロー体制・地域接点を補足材料にできる。", "neutral", ["sales_dept"])
    else:
        missing_inputs.append("営業部")

    commercial_parts = []
    if customer_type:
        commercial_parts.append(customer_type)
    if main_bank:
        commercial_parts.append(main_bank)
    if deal_source:
        commercial_parts.append(deal_source)
    if deal_occurrence and deal_occurrence != "不明":
        commercial_parts.append(f"発生経緯: {deal_occurrence}")
    if commercial_parts:
        add_comment("取引背景", " / ".join(commercial_parts) + "。信用補完や営業経緯の説明に反映。", "neutral", ["customer_type", "main_bank", "deal_source", "deal_occurrence"])

    if competitor == "競合あり":
        add_risk("medium", "競合条件比較", "競合ありのため、金利だけでなく対象範囲・保守・期間差を揃えて比較する必要がある。", ["competitor", "competitor_rate"])
    elif competitor:
        add_comment("競合", f"{competitor}。条件案は競合比較よりも信用・物件条件を中心に組み立てる。", "neutral", ["competitor"])

    if grade:
        add_comment("格付", f"社内格付「{grade}」を金利スプレッドと信用補完要否に反映。", "neutral", ["grade"])

    if op_margin < 2:
        add_risk("high", "利益率不足", f"営業利益率 {op_margin:.1f}% と薄く、返済原資の説明が弱い。直近試算表・受注残で補強が必要。", ["op_profit", "nenshu"])
    elif op_margin >= 5:
        add_comment("収益力", f"営業利益率 {op_margin:.1f}% は返済原資の説明材料になる。", "positive", ["op_profit", "nenshu"])

    if equity_ratio < 15:
        add_risk("medium", "自己資本の薄さ", f"自己資本比率 {equity_ratio:.1f}% のため、前受金・保証・期間短縮で損失吸収力を補う余地がある。", ["net_assets", "total_assets"])
    elif equity_ratio >= 25:
        add_comment("財務安定性", f"自己資本比率 {equity_ratio:.1f}% は財務安定性の補強材料になる。", "positive", ["net_assets", "total_assets"])

    if q_risk >= 35:
        add_risk("medium", "複合リスク", f"Q_risk {q_risk:.1f} のため、財務・物件・商談条件のズレを確認する。", ["quantum_risk"])
    if credit_risk >= 45:
        add_risk("high", "信用リスク群", f"信用リスク群スコア {credit_risk:.1f} が高く、既存借入・リース残高の返済状況確認が必要。", ["bank_credit", "lease_credit", "contracts"])
    if nenshu > 0 and acquisition_cost / nenshu >= 0.25:
        add_risk("medium", "年商比負担", "取得価額が年商比で大きく、月額負担・導入効果・出口リスクの説明を厚くする必要がある。", ["acquisition_cost", "nenshu"])
    if lease_term >= 72:
        add_risk("medium", "長期契約", f"リース期間 {lease_term}ヶ月は長めで、財務変化・物件価値変動の確認が必要。", ["lease_term"])

    condition_rationale = []
    if asset_name and asset_purpose:
        condition_rationale.append({
            "condition": "物件・用途の稟議記載を必須化",
            "reason": f"対象物件「{asset_name}」を「{asset_purpose}」に使う前提を、資金使途と導入効果の説明に使う。",
            "input_refs": ["asset_name", "asset_purpose", "asset_location", "asset_detail"],
        })
        reflected_inputs.extend(["asset_name", "asset_purpose", "asset_location", "asset_detail"])
    if asset_evidence in ("", "未確認"):
        condition_rationale.append({
            "condition": "見積・型式・中古相場の確認資料を取得",
            "reason": "確認資料が未確認のため、取得価額・物件仕様・残価リスクの説明が弱い。",
            "input_refs": ["asset_evidence_level", "acquisition_cost"],
        })
        reflected_inputs.extend(["asset_evidence_level", "acquisition_cost"])
    if sales_dept and sales_dept != "未設定":
        condition_rationale.append({
            "condition": "営業部のフォロー体制を条件補足へ記載",
            "reason": f"{sales_dept} の顧客接点・商談経緯を、審査後フォローと条件交渉の材料にする。",
            "input_refs": ["sales_dept", "deal_source", "deal_occurrence"],
        })
        reflected_inputs.extend(["sales_dept", "deal_source", "deal_occurrence"])
    for action in conditional_actions[:5]:
        action_text = str(action.get("action") or "")
        refs = []
        if any(key in action_text for key in ("物件", "用途", "設置")):
            refs = ["asset_name", "asset_purpose", "asset_location", "asset_evidence_level"]
        elif "営業部" in action_text or "商談" in action_text:
            refs = ["sales_dept", "deal_source", "deal_occurrence"]
        elif "競合" in action_text:
            refs = ["competitor", "competitor_rate"]
        elif "銀行" in action_text or "メイン" in action_text:
            refs = ["main_bank", "deal_source"]
        elif "期間" in action_text:
            refs = ["lease_term", "acquisition_cost"]
        elif "前受" in action_text or "保証" in action_text or "担保" in action_text:
            refs = ["score", "net_assets", "total_assets", "acquisition_cost"]
        condition_rationale.append({
            "condition": action_text,
            "reason": action.get("reason", ""),
            "input_refs": refs,
        })
        reflected_inputs.extend(refs)

    rate_breakdown = (rate_proposal or {}).get("breakdown") or {}
    if rate_breakdown:
        add_comment(
            "金利条件",
            "提案金利は基準金利、物件スプレッド、格付スプレッド、スコア調整を合算して算出。"
            f" 物件 {rate_breakdown.get('asset_spread', 0):.2f}% / 格付 {rate_breakdown.get('grade_spread', 0):.2f}% / リスク {rate_breakdown.get('risk_adjustment', 0):.2f}%。",
            "neutral",
            ["asset_name", "industry_sub", "grade", "score"],
        )

    if estat_context:
        add_comment(
            "e-Stat統合文脈",
            str(estat_context.get("summary") or "業種・リース・景気の3層コンテキストを確認。"),
            "neutral",
            ["estat_context"],
        )
        for rec in estat_context.get("recommendations", [])[:2]:
            if rec:
                add_comment("e-Stat示唆", str(rec), "neutral", ["estat_context"])

    reflected_unique = sorted({str(item) for item in reflected_inputs if item})
    important_inputs = {
        "company_name", "industry_major", "industry_sub", "grade", "customer_type", "main_bank",
        "competitor", "deal_source", "sales_dept", "asset_name", "asset_detail", "asset_purpose",
        "asset_location", "asset_evidence_level", "nenshu", "op_profit", "net_assets",
        "total_assets", "lease_term", "acquisition_cost",
    }
    reflection_score = round(min(100, len(set(reflected_unique) & important_inputs) / len(important_inputs) * 100))

    missing_unique = []
    for item in missing_inputs:
        if item and item not in missing_unique:
            missing_unique.append(item)

    return {
        "summary": f"入力文脈の反映度 {reflection_score}%: 物件・営業・財務・商談背景を審査コメント、条件案、リスク理由へ展開しました。",
        "reflection_score": reflection_score,
        "commentary": commentary[:8],
        "risk_reasons": risk_reasons[:8],
        "condition_rationale": condition_rationale,
        "missing_inputs": missing_unique,
        "reflected_inputs": reflected_unique,
        "industry_context": {
            "major": industry_major,
            "sub": industry_sub,
            "detail": industry_detail,
        },
        "estat_context": estat_context or None,
    }


def _build_approval_comment_draft(
    inputs: dict,
    result: dict,
    conditional_actions: list[dict],
    rate_proposal: dict,
    screening_context_notes: dict,
) -> dict:
    """稟議へ貼り付けやすい審査コメント案を生成する。"""
    score = _score_float(result.get("score", result.get("hantei_score")), 0.0)
    hantei = str(result.get("hantei") or "未判定")
    company_name = str(inputs.get("company_name") or "対象先").strip()
    industry_major = str(inputs.get("industry_major") or result.get("industry_major") or "").strip()
    industry_sub = str(inputs.get("industry_sub") or result.get("industry_sub") or "").strip()
    grade = str(inputs.get("grade") or "").strip()
    customer_type = str(inputs.get("customer_type") or "").strip()
    main_bank = str(inputs.get("main_bank") or "").strip()
    deal_source = str(inputs.get("deal_source") or "").strip()
    sales_dept = str(inputs.get("sales_dept") or "未設定").strip()
    competitor = str(inputs.get("competitor") or "").strip()
    asset_name = str(inputs.get("asset_name") or "").strip()
    asset_detail = str(inputs.get("asset_detail") or "").strip()
    asset_purpose = str(inputs.get("asset_purpose") or "").strip()
    asset_location = str(inputs.get("asset_location") or "").strip()
    asset_evidence = str(inputs.get("asset_evidence_level") or "").strip()
    lease_term = int(_score_float(inputs.get("lease_term"), 60))
    acquisition_cost = _score_float(inputs.get("acquisition_cost"), 0.0)
    op_margin = _score_float(result.get("user_op_margin", result.get("user_op")), 0.0)
    equity_ratio = _score_float(result.get("user_equity_ratio", result.get("user_eq")), 0.0)
    q_risk = result.get("quantum_risk")

    amount_million = acquisition_cost / 1000 if acquisition_cost >= 1000 else acquisition_cost
    proposed_rate = (rate_proposal or {}).get("proposed_rate")
    monthly_payment = (rate_proposal or {}).get("monthly_payment")
    risks = screening_context_notes.get("risk_reasons") or []
    missing = screening_context_notes.get("missing_inputs") or []

    deal_bits = [bit for bit in [customer_type, main_bank, deal_source, sales_dept if sales_dept != "未設定" else ""] if bit]
    industry_label = industry_sub or industry_major or "業種未設定"
    asset_sentence = "対象物件は未確定。"
    if asset_name:
        asset_sentence = f"対象物件は「{asset_name}」"
        if asset_detail:
            asset_sentence += f"（{asset_detail}）"
        if asset_purpose:
            asset_sentence += f"であり、導入目的は「{asset_purpose}」。"
        else:
            asset_sentence += "。導入目的は追加確認が必要。"
        if asset_location:
            asset_sentence += f" 使用場所は {asset_location}。"
        if asset_evidence and asset_evidence != "未確認":
            asset_sentence += f" 確認資料は「{asset_evidence}」。"
        else:
            asset_sentence += " 見積・型式等の確認資料は未確認。"

    judgment_line = "条件付きで前向きに検討。"
    if score >= APPROVAL_LINE:
        judgment_line = "現時点では承認方向で検討可能。"
    elif score < CONDITIONAL_LINE:
        judgment_line = "現条件のままでは慎重判断とし、条件補強または案件条件の見直しを要する。"

    summary = (
        f"{company_name}（{industry_label}）について、AI審査スコアは {score:.1f} 点、判定は「{hantei}」。"
        f"{judgment_line}"
    )

    basis_lines = [
        f"取引背景は {' / '.join(deal_bits) if deal_bits else '未整理'}。社内格付は「{grade or '未設定'}」。",
        asset_sentence,
        f"取得価額は約 {amount_million:.1f} 百万円、リース期間は {lease_term} ヶ月。"
        + (f" 提案金利は {float(proposed_rate):.2f}%、月額目安は {int(monthly_payment):,}円。" if proposed_rate and monthly_payment else ""),
        f"営業利益率は {op_margin:.1f}%、自己資本比率は {equity_ratio:.1f}%。"
        + (f" Q_risk は {float(q_risk):.1f}。" if q_risk not in (None, "") else ""),
    ]

    risk_lines = []
    for item in risks[:4]:
        title = str(item.get("title") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if title and reason:
            risk_lines.append(f"{title}: {reason}")
    if not risk_lines:
        risk_lines.append("重大な追加リスクは限定的。通常の財務・物件確認を継続する。")

    condition_lines = []
    for action in conditional_actions[:5]:
        action_text = str(action.get("action") or "").strip()
        reason = str(action.get("reason") or "").strip()
        if action_text:
            condition_lines.append(f"{action_text}" + (f"（{reason}）" if reason else ""))
    if not condition_lines and score >= APPROVAL_LINE:
        condition_lines.append("通常の見積・契約条件確認を前提に進める。")
    if missing:
        condition_lines.append("不足資料: " + "、".join(str(item) for item in missing[:5]) + "。")

    final_opinion = (
        "上記より、物件・用途・取引背景を確認したうえで、条件を付して稟議上申する。"
        if score < APPROVAL_LINE else
        "上記より、通常確認事項を充足することを前提に稟議上申する。"
    )

    sections = [
        {"title": "概要", "body": summary},
        {"title": "判断根拠", "body": "\n".join(f"- {line}" for line in basis_lines)},
        {"title": "主なリスク", "body": "\n".join(f"- {line}" for line in risk_lines)},
        {"title": "承認条件・確認事項", "body": "\n".join(f"- {line}" for line in condition_lines[:6])},
        {"title": "総合意見", "body": final_opinion},
    ]
    full_text = "\n\n".join(f"【{section['title']}】\n{section['body']}" for section in sections)

    return {
        "title": f"{company_name} 稟議コメント案",
        "verdict": hantei,
        "score": round(score, 1),
        "summary": summary,
        "sections": sections,
        "full_text": full_text,
        "copy_hint": "稟議本文へ貼り付け後、顧客固有の事情・正式資料名・社内決裁条件を追記してください。",
    }


def _build_data_source_summary(inputs: dict, result: dict) -> dict:
    def _has_input_value(key: str) -> bool:
        value = inputs.get(key)
        if value in (None, "", 0, [], {}):
            return False
        if key == "asset_evidence_level" and str(value).strip() in ("", "未確認"):
            return False
        return True

    manual_fields = [
        "company_no", "company_name", "industry_major", "industry_sub", "grade",
        "customer_type", "main_bank", "competitor", "deal_source", "sales_dept", "contract_type",
        "nenshu", "op_profit", "ord_profit", "net_income", "net_assets", "total_assets",
        "bank_credit", "lease_credit", "contracts", "lease_term", "acquisition_cost",
        "asset_score", "selected_asset_id", "asset_name", "asset_detail", "asset_purpose",
        "asset_location", "asset_evidence_level", "passion_text", "intuition",
    ]
    filled = [key for key in manual_fields if _has_input_value(key)]
    asset_fields = ["asset_name", "asset_detail", "asset_purpose", "asset_location", "asset_evidence_level"]
    asset_filled = [key for key in asset_fields if _has_input_value(key)]
    asset_evidence = str(inputs.get("asset_evidence_level") or "")
    asset_clarity_warnings = []
    if not inputs.get("asset_name"):
        asset_clarity_warnings.append("対象物件名が未設定")
    if not inputs.get("asset_purpose"):
        asset_clarity_warnings.append("導入目的・用途が未設定")
    if asset_evidence in ("", "未確認"):
        asset_clarity_warnings.append("見積・型式・中古相場などの確認資料が未確認")
    return {
        "primary_source": "画面入力 + マスタ + 審査モデル",
        "manual_input_count": len(filled),
        "manual_input_fields": filled[:12],
        "asset_clarity": {
            "filled_count": len(asset_filled),
            "required_count": len(asset_fields),
            "status": "明確" if len(asset_filled) >= 4 and not asset_clarity_warnings else "要確認",
            "warnings": asset_clarity_warnings,
        },
        "model_sources": [
            "RandomForest borrower score",
            "業種ベンチマーク",
            "e-Stat業種/景気コンテキスト",
            "物件スコア/物件警告",
            "Q_risk/信用リスク補助指標",
        ],
        "system_generated_fields": [
            "score", "hantei", "comparison", "conditional_approval_actions", "rate_proposal",
        ],
        "warnings": [
            "画面入力値はユーザー申告値として扱う",
            "マスタ・モデル由来の値は審査補助であり最終判断を代替しない",
        ],
        "case_id": result.get("case_id"),
    }


def _build_rate_proposal(inputs: dict, result: dict) -> dict:
    import datetime
    from base_rate_master import get_base_rate_by_term

    score = max(0.0, min(100.0, _score_float(result.get("score", result.get("hantei_score")), 0.0)))
    term_months = max(12, min(120, int(_score_float(inputs.get("lease_term"), 60))))
    lease_amount = max(1.0, _score_float(inputs.get("acquisition_cost"), 1.0))
    year_month = datetime.date.today().strftime("%Y-%m")

    base_rate = get_base_rate_by_term(year_month, term_months)
    if base_rate is None:
        for i in range(1, 7):
            prev_date = datetime.date.today().replace(day=1) - datetime.timedelta(days=30 * i)
            base_rate = get_base_rate_by_term(prev_date.strftime("%Y-%m"), term_months)
            if base_rate is not None:
                year_month = prev_date.strftime("%Y-%m")
                break
    if base_rate is None:
        base_rate = 2.0

    asset_text = " ".join(str(inputs.get(k) or "") for k in ("selected_asset_id", "asset_name", "industry_sub")).lower()
    if any(k in asset_text for k in ("medical", "医療")):
        asset_spread = 0.35
    elif any(k in asset_text for k in ("it", "pc", "サーバ", "情報")):
        asset_spread = 0.65
    elif any(k in asset_text for k in ("vehicle", "car", "車両", "運送")):
        asset_spread = 0.38
    elif any(k in asset_text for k in ("machinery", "machine", "機械", "建機", "製造")):
        asset_spread = 0.48
    else:
        asset_spread = 0.50

    grade = str(inputs.get("grade") or "").strip().lower()
    if grade.startswith(("s", "①")):
        grade_spread = -0.10
    elif grade.startswith(("a", "②")):
        grade_spread = 0.25
    elif grade.startswith(("b", "③")):
        grade_spread = 0.55
    elif grade.startswith(("c", "④", "d")):
        grade_spread = 0.90
    else:
        grade_spread = 0.30

    if score >= 90:
        risk_adjustment = -0.10
    elif score >= 80:
        risk_adjustment = -0.05
    elif score >= 70:
        risk_adjustment = 0.00
    elif score >= 60:
        risk_adjustment = 0.15
    elif score >= 50:
        risk_adjustment = 0.30
    else:
        risk_adjustment = 0.50

    proposed_rate = round(max(0.5, base_rate + asset_spread + grade_spread + risk_adjustment), 4)
    monthly_rate = proposed_rate / 100 / 12
    monthly_payment_thousand_yen = lease_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months)) if monthly_rate > 0 else lease_amount / term_months
    return {
        "year_month": year_month,
        "proposed_rate": proposed_rate,
        "term_months": term_months,
        "lease_amount": lease_amount,
        "monthly_payment": round(monthly_payment_thousand_yen * 1000),
        "breakdown": {
            "base_rate": round(base_rate, 4),
            "asset_spread": round(asset_spread, 4),
            "grade_spread": round(grade_spread, 4),
            "risk_adjustment": round(risk_adjustment, 4),
        },
        "guidance": "審査結果欄の初期提示用。競合条件・保守範囲・前受金条件で最終調整する。",
    }


def _build_bayes_reverse_strategy(inputs: dict, result: dict) -> dict:
    """Streamlit版のBN逆転提案を、審査結果カード用の軽量JSONにする。"""
    try:
        from api.gunshi_gemini import build_bayes_factors
        from shinsa_gunshi_logic import compute_prior, compute_posterior, select_top_phrases

        score = max(0.0, min(100.0, _score_float(result.get("score_base", result.get("score")), 0.0)))
        pd_pct = 0.0
        asset_score = _score_float(result.get("asset_score", inputs.get("asset_score")), 50.0)
        if asset_score >= 70:
            resale_eval = "高"
        elif asset_score < 40:
            resale_eval = "低"
        else:
            resale_eval = "中"

        subsidy_text = " ".join(
            str(inputs.get(key) or "")
            for key in ("industry_detail", "passion_text", "asset_name", "asset_purpose")
        )
        subsidy_flag = bool(re.search(r"補助金|助成金|ものづくり|省力化|it導入|省エネ", subsidy_text, re.IGNORECASE))
        bank_support = (
            str(inputs.get("deal_source") or "") == "銀行紹介"
            or str(inputs.get("main_bank") or "") == "メイン先"
        )
        repeat_count = max(0, int(_score_float(inputs.get("contracts"), 0)))
        raw_intuition = _score_float(inputs.get("intuition"), 3.0)
        intuition_level = int(max(1, min(5, round(raw_intuition if raw_intuition <= 5 else raw_intuition / 20))))

        prior = compute_prior(score, pd_pct)
        posterior = compute_posterior(
            prior=prior,
            resale=resale_eval,
            repeat_cnt=repeat_count,
            subsidy=subsidy_flag,
            bank=bank_support,
            intuition=intuition_level,
        )
        params = {
            **inputs,
            "score": score,
            "pd_pct": pd_pct,
            "resale_eval": resale_eval,
            "repeat_count": repeat_count,
            "subsidy_flag": subsidy_flag,
            "bank_support": bank_support,
            "intuition_score": intuition_level * 20,
            "industry_cat": result.get("industry_major") or inputs.get("industry_major") or "",
            "industry_sub": result.get("industry_sub") or inputs.get("industry_sub") or "",
            "asset_name": inputs.get("asset_name") or "対象物件",
        }
        phrase_dicts = select_top_phrases(
            industry_cat=str(params.get("industry_cat") or ""),
            score=score,
            pd_pct=pd_pct,
            resale=resale_eval,
            repeat_cnt=repeat_count,
            subsidy=subsidy_flag,
            bank=bank_support,
            posterior=posterior,
            asset_name=str(params.get("asset_name") or ""),
            n=3,
        )
        phrases = [
            str(item.get("text") if isinstance(item, dict) else item)
            for item in phrase_dicts
            if str(item.get("text") if isinstance(item, dict) else item).strip()
        ]
        lift = posterior - prior
        if posterior >= 0.75:
            stance = "押し切り提案"
            headline = "承認確率は高い。条件を増やしすぎず、返済原資と物件保全を短く押す。"
        elif posterior >= 0.60:
            stance = "条件付き逆転"
            headline = "境界だが逆転余地あり。否決理由を先に潰して条件付き承認へ寄せる。"
        else:
            stance = "再設計提案"
            headline = "現状のまま押すより、金額・期間・保証・保全を組み替える局面。"

        proposed_moves = []
        if asset_score < 50:
            proposed_moves.append("物件の中古相場・撤去搬出費・再販先を確認し、保全弱点を先に潰す。")
        else:
            proposed_moves.append("物件の必要性と換金性を稟議の押し材料にする。")
        if not bank_support:
            proposed_moves.append("主取引銀行の温度感を取り、銀行支援の有無を確認する。")
        if repeat_count <= 0:
            proposed_moves.append("完全新規なら、少額化・前受金・保証・期間短縮のどれかを差し出す。")
        if subsidy_flag:
            proposed_moves.append("補助金は採択時期と入金時期を確認し、資金繰り前提にしすぎない。")
        if not proposed_moves:
            proposed_moves.append("返済原資、物件保全、取引継続性の三点で稟議を組み立てる。")

        return {
            "available": True,
            "label": "BN逆転",
            "stance": stance,
            "headline": headline,
            "prior": round(prior, 4),
            "posterior": round(posterior, 4),
            "prior_percent": round(prior * 100, 1),
            "posterior_percent": round(posterior * 100, 1),
            "lift_percent": round(lift * 100, 1),
            "evidence": {
                "resale": resale_eval,
                "repeat_count": repeat_count,
                "subsidy": subsidy_flag,
                "bank_support": bank_support,
                "intuition_level": intuition_level,
            },
            "factors": build_bayes_factors(params, prior, posterior),
            "moves": proposed_moves[:4],
            "phrases": phrases[:3],
            "disclaimer": "BN逆転は承認を保証せず、条件変更で承認余地がどれだけ上がるかを見る補助指標です。",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:180]}


_WIZARD_INPUT_LOG = Path(__file__).parent.parent / "data" / "wizard_input_log.jsonl"
_wizard_log_lock = __import__("threading").Lock()
_WIZARD_TRACKED_FIELDS = [
    "company_name", "nenshu", "op_profit", "acquisition_cost",
    "asset_name", "passion_text", "industry_detail", "asset_detail",
    "asset_purpose", "asset_location",
]
_WIZARD_FIELD_MAX_LEN = 500


def _sanitize_wizard_str(value: object, max_len: int = _WIZARD_FIELD_MAX_LEN) -> str:
    """文字列フィールドを制御文字除去・長さ制限してサニタイズする。"""
    import unicodedata as _uc, re as _re
    text = str(value) if not isinstance(value, str) else value
    cleaned = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or not _uc.category(ch).startswith("C")
    )
    cleaned = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned[:max_len]


def _log_wizard_input_task(inputs: dict) -> None:
    import datetime as _dt, json as _json
    # 文字列フィールドをサニタイズしてから空欄チェック（制御文字のみのフィールドを「空」と正しく判定）
    sanitized = {
        f: _sanitize_wizard_str(inputs[f]) if isinstance(inputs.get(f), str) else inputs.get(f)
        for f in _WIZARD_TRACKED_FIELDS
        if f in inputs
    }
    empty = [f for f in _WIZARD_TRACKED_FIELDS if not sanitized.get(f)]
    entry = _json.dumps({
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "total_fields": len(_WIZARD_TRACKED_FIELDS),
        "empty_count": len(empty),
        "empty_fields": empty,
        "surface": "wizard_calculate",
    }, ensure_ascii=False) + "\n"
    with _wizard_log_lock:
        with open(_WIZARD_INPUT_LOG, "a", encoding="utf-8") as _f:
            _f.write(entry)


@app.post("/api/score/calculate", response_model=ScoringResponse)
def calculate_score(req: ScoringRequest, background_tasks: BackgroundTasks):
    try:
        # パラメータを辞書化して existing の関数に渡す
        inputs = req.model_dump()
        background_tasks.add_task(_log_wizard_input_task, inputs)
        result = run_quick_scoring(inputs)
        background_tasks.add_task(
            record_cloudrun_input_event,
            event_type="score_calculated",
            surface="score_calculate",
            payload={
                "inputs": inputs,
                "result": {
                    "score": result.get("score"),
                    "hantei": result.get("hantei"),
                    "industry_sub": result.get("industry_sub"),
                    "industry_major": result.get("industry_major"),
                    "asset_score": result.get("asset_score"),
                },
            },
        )
        conditional_actions = _build_conditional_approval_actions(inputs, result)
        rate_proposal = _build_rate_proposal(inputs, result)
        data_source_summary = _build_data_source_summary(inputs, result)
        screening_context_notes = _build_screening_context_notes(inputs, result, conditional_actions, rate_proposal)
        approval_comment_draft = _build_approval_comment_draft(inputs, result, conditional_actions, rate_proposal, screening_context_notes)
        financial_consistency_risk = _build_financial_consistency_risk(inputs)
        result["financial_consistency_risk"] = financial_consistency_risk
        result["financial_consistency_score"] = financial_consistency_risk.get("score", 0)
        from api.aurion_core_guard import build_aurion_core_guard
        aurion_core = build_aurion_core_guard(inputs, result)
        bayes_reverse_strategy = _build_bayes_reverse_strategy(inputs, result)
        _record_scoring_memory_usage("score_calculate", inputs, result)
        
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
            financial_consistency_score=result.get("financial_consistency_score"),
            financial_consistency_risk=result.get("financial_consistency_risk"),
            conditional_approval_actions=conditional_actions,
            rate_proposal=rate_proposal,
            data_source_summary=data_source_summary,
            screening_context_notes=screening_context_notes,
            approval_comment_draft=approval_comment_draft,
            estat_context=result.get("estat_context"),
            aurion_core=aurion_core,
            bayes_reverse_strategy=bayes_reverse_strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/score/full", response_model=ScoringResponse)
def calculate_score_full(req: ScoringRequest, background_tasks: BackgroundTasks):
    try:
        inputs = req.model_dump()
        if os.environ.get("USE_LEGACY_STREAMLIT_FULL_SCORE", "").lower() in {"1", "true", "yes", "on"}:
            result = run_full_scoring_api(inputs)
            result["engine_source"] = "legacy_streamlit"
        else:
            result = run_full_api_scoring(inputs)
        conditional_actions = _build_conditional_approval_actions(inputs, result)
        rate_proposal = _build_rate_proposal(inputs, result)
        data_source_summary = _build_data_source_summary(inputs, result)
        screening_context_notes = _build_screening_context_notes(inputs, result, conditional_actions, rate_proposal)
        approval_comment_draft = _build_approval_comment_draft(inputs, result, conditional_actions, rate_proposal, screening_context_notes)
        financial_consistency_risk = _build_financial_consistency_risk(inputs)
        result["financial_consistency_risk"] = financial_consistency_risk
        result["financial_consistency_score"] = financial_consistency_risk.get("score", 0)
        from api.aurion_core_guard import build_aurion_core_guard
        aurion_core = build_aurion_core_guard(inputs, result)
        bayes_reverse_strategy = _build_bayes_reverse_strategy(inputs, result)

        # ── DB保存 ──────────────────────────────────────────────
        case_id = None
        try:
            from data_cases import save_case_log
            case_data = {
                "company_no":   inputs.get("company_no", ""),
                "company_name": inputs.get("company_name", ""),
                "sales_dept":   inputs.get("sales_dept", "未設定") or "未設定",
                "industry_sub": result.get("industry_sub", inputs.get("industry_sub", "")),
                "industry_major": result.get("industry_major", inputs.get("industry_major", "")),
                "inputs": inputs,
                "result": {
                    "score": result.get("score", 0),
                    "score_base": result.get("score_base", result.get("score", 0)),
                    "hantei": result.get("hantei", ""),
                    "user_eq": result.get("user_equity_ratio", result.get("user_eq", 0)),
                    "user_op": result.get("user_op_margin", result.get("user_op", 0)),
                    "quantum_risk": result.get("quantum_risk"),
                    "financial_consistency_score": result.get("financial_consistency_score"),
                    "financial_consistency_risk": result.get("financial_consistency_risk"),
                    "credit_quantum_strong_warning": result.get("credit_quantum_strong_warning", False),
                    "aurion_core": aurion_core,
                    "bayes_reverse_strategy": bayes_reverse_strategy,
                },
            }
            case_id = save_case_log(case_data)
        except Exception as _save_err:
            print(f"[WARNING] DB save failed: {_save_err}")
        data_source_summary["case_id"] = case_id
        result["case_id"] = case_id
        background_tasks.add_task(
            record_cloudrun_input_event,
            event_type="score_full_calculated",
            surface="screening",
            payload={
                "schema_version": 1,
                "case_id": case_id,
                "inputs": inputs,
                "result": {
                    "score": result.get("score"),
                    "score_base": result.get("score_base", result.get("score")),
                    "hantei": result.get("hantei"),
                    "engine_source": result.get("engine_source"),
                    "industry_sub": result.get("industry_sub"),
                    "industry_major": result.get("industry_major"),
                    "asset_score": result.get("asset_score"),
                    "score_borrower": result.get("score_borrower"),
                    "quantum_risk": result.get("quantum_risk"),
                    "financial_consistency_score": result.get("financial_consistency_score"),
                    "umap_anomaly_score": result.get("umap_anomaly_score"),
                },
            },
        )
        _record_scoring_memory_usage("score_full", inputs, result)

        # リース知性体の着火: サブエージェント間の不整合を検知したら内省を起動する。
        # 既存スコアリング結果のフィールドを読むだけ・審査レスポンスには影響しない完全非ブロッキング。
        # PII混入を避けるため会社名等は渡さず、数値フィールドのみで判定する。
        try:
            from lease_intelligence_mind import detect_dissonance, register_ignition
            from lease_news_digest import find_vault as _find_vault

            _vault = _find_vault()
            if _vault:
                _signals = detect_dissonance(result)
                if _signals:
                    register_ignition(_vault, _signals)
        except Exception as _ignite_err:
            print(f"[WARNING] lease-intelligence ignition skipped: {_ignite_err}")

        # 感情トリガー（REV-101）: 審査完了 or 高リスク承認
        try:
            from api.emotion_trigger import trigger_scoring_complete
            trigger_scoring_complete(
                score=float(result.get("score", 0.0)),
                quantum_risk=result.get("quantum_risk"),
                credit_quantum_strong_warning=bool(result.get("credit_quantum_strong_warning", False)),
            )
        except Exception as _et_err:
            print(f"[EmotionTrigger] scoring skipped: {_et_err}")

        return ScoringResponse(
            score=result.get("score", 0.0),
            hantei=result.get("hantei", "未判定"),
            comparison=result.get("comparison", ""),
            user_op_margin=result.get("user_op_margin", result.get("user_op", 0.0)),
            user_equity_ratio=result.get("user_equity_ratio", result.get("user_eq", 0.0)),
            bench_op_margin=result.get("bench_op_margin", result.get("bench_op", 0.0)),
            bench_equity_ratio=result.get("bench_equity_ratio", result.get("bench_eq", 0.0)),
            score_borrower=result.get("score_borrower", 0.0),
            score_base=result.get("score_base", result.get("score", 0.0)),
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
            financial_consistency_score=result.get("financial_consistency_score"),
            financial_consistency_risk=result.get("financial_consistency_risk"),
            credit_quantum_strong_warning=result.get("credit_quantum_strong_warning", False),
            mahalanobis_score=result.get("mahalanobis_score"),
            mahalanobis_advice=result.get("mahalanobis_advice"),
            umap_anomaly_score=result.get("umap_anomaly_score"),
            umap_x=result.get("umap_x"),
            umap_y=result.get("umap_y"),
            umap_similar=result.get("umap_similar"),
            conditional_approval_actions=conditional_actions,
            rate_proposal=rate_proposal,
            data_source_summary=data_source_summary,
            screening_context_notes=screening_context_notes,
            approval_comment_draft=approval_comment_draft,
            estat_context=result.get("estat_context"),
            aurion_core=aurion_core,
            bayes_reverse_strategy=bayes_reverse_strategy,
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
        logger.error("api_industry_stats DB error: %s", e)
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


@app.get("/api/cases")
def list_cases(limit: int = 30, offset: int = 0, sort: str = "desc"):
    """過去案件一覧 (limit/offset/sort 対応)"""
    import json

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    order = "DESC" if sort.lower() != "asc" else "ASC"
    rows = []
    try:
        with get_connection() as conn:
            res = conn.execute(
                f"SELECT id, timestamp, industry_sub, score, final_status, "
                f"json_extract(data,'$.company_name') AS company_name, "
                f"json_extract(data,'$.company_no')   AS company_no, "
                f"json_extract(data,'$.judgment')     AS judgment, "
                f"COALESCE(json_extract(data,'$._source'), 'past_cases') AS source "
                f"FROM past_cases ORDER BY timestamp {order} LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            for r in res:
                rows.append(dict(r))
    except Exception as e:
        logger.error("list_cases DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return rows


class CaseResultPatch(BaseModel):
    final_status: Optional[str] = None
    competitor_rate: Optional[float] = None
    loss_reason: Optional[str] = None
    final_result_date: Optional[str] = None
    source: str = "app"


def _get_case_payload(case_id: str) -> dict:
    """past_cases から保存済みJSONを取得する。"""
    import json as _json

    try:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (case_id,)).fetchone()
    except Exception as e:
        logger.error("_get_case_payload DB error case_id=%s: %s", case_id, e)
        return {}
    if row is None:
        return {}
    try:
        payload = _json.loads(row["data"] or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_obsidian_filename(value: str) -> str:
    import re as _re

    clean = _re.sub(r'[\\/:*?"<>|\n\r\t]', "_", str(value or "").strip())
    clean = _re.sub(r"\s+", "_", clean).strip("._")
    return clean[:72] or "case"


def _append_case_result_reflection_to_obsidian(case_id: str, case_data: dict, patches: dict) -> dict:
    """案件結果登録を、再利用できる短い振り返りノートとしてObsidianへ追記する。"""
    import datetime as _dt

    vault_raw = _OBSIDIAN_VAULT_PATH or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or ""
    if not vault_raw:
        return {"status": "skipped", "reason": "obsidian_vault_not_configured"}
    vault = Path(vault_raw).expanduser().resolve()
    if not (vault / ".obsidian").exists():
        return {"status": "skipped", "reason": "obsidian_vault_not_found", "vault": str(vault)}

    inputs = case_data.get("inputs") if isinstance(case_data.get("inputs"), dict) else case_data
    result = case_data.get("result") if isinstance(case_data.get("result"), dict) else {}
    now = _dt.datetime.now()
    day = (patches.get("final_result_date") or now.date().isoformat())[:10]
    status = str(patches.get("final_status") or case_data.get("final_status") or "")
    score = case_data.get("score", result.get("score"))
    company = str(inputs.get("company_name") or case_data.get("company_name") or "名称未設定").strip()
    company_no = str(inputs.get("company_no") or case_data.get("company_no") or "").strip()
    industry = str(inputs.get("industry_sub") or inputs.get("industry_major") or case_data.get("industry_sub") or "").strip()
    asset = str(inputs.get("asset_name") or inputs.get("asset_type") or "").strip()
    competitor_rate = patches.get("competitor_rate", case_data.get("competitor_rate"))
    loss_reason = patches.get("lost_reason", case_data.get("lost_reason") or case_data.get("loss_reason"))
    qrisk = result.get("quantum_risk", case_data.get("quantum_risk"))

    source_factors: list[str] = []
    if competitor_rate not in (None, "", 0):
        source_factors.append(f"競合金利 {competitor_rate}%")
    if loss_reason:
        source_factors.append(f"失注理由: {loss_reason}")
    deal_source = inputs.get("deal_source") or case_data.get("deal_source")
    if deal_source:
        source_factors.append(f"商談ソース: {deal_source}")
    if qrisk not in (None, ""):
        source_factors.append(f"Q_risk: {qrisk}")

    if status in ("成約", "検収", "検収完了"):
        learned = "スコア・営業導線・銀行支援・物件条件のどれが成約に寄与したかを次回同種案件で確認する。"
        missed = "高スコア成約でも、価格・競合・補助金タイミングなど非スコア要因を後追いで記録する。"
    elif status == "失注":
        learned = "失注要因をスコア要因と非スコア要因に分け、次回の初期ヒアリング項目へ戻す。"
        missed = "スコアが妥当でも、競合金利・物件代替・銀行支援・顧客都合で外れる可能性を確認する。"
    else:
        learned = "結果ステータス更新として保存。後続の成約/失注登録時に要因を補完する。"
        missed = "結果が確定していない場合は、次回更新で競合・補助金・銀行支援の有無を追記する。"

    rel = Path("Projects") / "tune_lease_55" / "Case Reviews" / f"{day}_{_safe_obsidian_filename(case_id)}.md"
    path = (vault / rel).resolve()
    if vault not in path.parents and path != vault:
        return {"status": "error", "reason": "unsafe_obsidian_path"}
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ""
    if not path.exists():
        header = (
            "---\n"
            f"created: {now.date().isoformat()}\n"
            "source: tune_lease_55\n"
            "type: case_result_reflection\n"
            f"case_id: {case_id}\n"
            "---\n\n"
            f"# 案件結果振り返り - {company}\n\n"
        )

    lines = [
        f"## {now.strftime('%H:%M')} 結果登録",
        "",
        "### Case",
        f"- 案件ID: {case_id}",
        f"- 企業: {company}{(' / ' + company_no) if company_no else ''}",
        f"- 業種: {industry or '未設定'}",
        f"- 物件: {asset or '未設定'}",
        f"- スコア: {score if score not in (None, '') else '未設定'}",
        f"- 結果: {status or '未設定'}",
    ]
    if competitor_rate not in (None, "", 0):
        lines.append(f"- 競合金利: {competitor_rate}%")
    if loss_reason:
        lines.append(f"- 失注理由: {loss_reason}")
    lines.extend([
        "",
        "### Reflection",
        f"- 効いた/外した判断: {learned}",
        f"- 見直す観点: {missed}",
        f"- 非スコア要因: {', '.join(source_factors) if source_factors else '未記録。営業・競合・銀行支援・補助金・物件換金性を次回確認する。'}",
        "- 次に同種案件で見る点: 業種、物件、金額、期間、競合、銀行支援、補助金タイミング。",
    ])
    with path.open("a", encoding="utf-8") as f:
        if header:
            f.write(header)
        elif path.read_text(encoding="utf-8", errors="ignore").strip():
            f.write("\n\n")
        f.write("\n".join(lines).strip() + "\n")

    return {"status": "saved", "path": str(path), "relative_path": str(rel)}


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


@app.get("/api/knowledge/graph")
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


@app.get("/api/cases/industry-winrate")
def get_industry_winrate():
    """業種別成約率を past_cases から集計して返す（REV-055/117~119）。"""
    if not _db_available():
        return []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT industry_sub, final_status, COUNT(*) FROM past_cases "
            "WHERE final_status IS NOT NULL AND final_status != '' "
            "GROUP BY industry_sub, final_status"
        )
        rows = cur.fetchall()
    _SUCCESS = {"成約", "検収完了"}
    _FAILURE = {"失注"}
    agg: dict = {}
    for industry, status, cnt in rows:
        if not industry or industry == "0":
            continue
        industry = _normalize_industry_for_stats(industry)
        d = agg.setdefault(industry, {"won": 0, "lost": 0})
        if status in _SUCCESS:
            d["won"] += cnt
        elif status in _FAILURE:
            d["lost"] += cnt
    result = []
    total_won = sum(v["won"] for v in agg.values())
    total_lost = sum(v["lost"] for v in agg.values())
    total_all = total_won + total_lost
    overall_rate = round(total_won / total_all * 100, 1) if total_all > 0 else 0
    for industry, d in agg.items():
        total = d["won"] + d["lost"]
        if total == 0:
            continue
        rate = round(d["won"] / total * 100, 1)
        result.append({
            "industry": industry,
            "won": d["won"],
            "lost": d["lost"],
            "total": total,
            "win_rate": rate,
            "diff": round(rate - overall_rate, 1),
        })
    result.sort(key=lambda x: x["total"], reverse=True)
    return {"items": result, "overall_rate": overall_rate, "total_won": total_won, "total_lost": total_lost}


@app.get("/api/cases/sales-dept-winrate")
def get_sales_dept_winrate():
    """営業部別成約率を集計して返す（REV-112）。"""
    if not _db_available():
        return {"items": [], "overall_rate": 0.0, "total_won": 0, "total_lost": 0}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT sales_dept,
                   SUM(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) as won,
                   SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) as lost,
                   COUNT(*) as total,
                   ROUND(AVG(score), 1) as avg_score
            FROM past_cases
            WHERE sales_dept NOT IN ('', '0', '未設定')
              AND final_status IN ('成約','検収完了','失注')
            GROUP BY sales_dept
            ORDER BY total DESC
        """)
        rows = cur.fetchall()
    total_won = sum(r[1] for r in rows)
    total_lost = sum(r[2] for r in rows)
    overall_rate = round(total_won / (total_won + total_lost) * 100, 1) if (total_won + total_lost) > 0 else 0.0
    result = []
    for dept, won, lost, total, avg_score in rows:
        rate = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0.0
        result.append({
            "dept": dept,
            "won": won,
            "lost": lost,
            "total": total,
            "win_rate": rate,
            "avg_score": avg_score or 0.0,
            "diff": round(rate - overall_rate, 1),
        })
    return {"items": result, "overall_rate": overall_rate, "total_won": total_won, "total_lost": total_lost}


@app.get("/api/payment/alerts")
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
    from pathlib import Path as _Path
    vault = _Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
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


class DismissImprovementRequest(BaseModel):
    key: str
    title: str


def _save_improvement_log_to_obsidian(entry: dict) -> dict:
    """Mirror an improvement ledger entry to the user's Obsidian vault."""
    status = str(entry.get("status") or "")
    title = str(entry.get("title") or "改善ログ")
    lines = [
        f"- status: {status or 'unknown'}",
        f"- title: {title}",
        f"- key: {entry.get('key') or ''}",
        f"- canonical_key: {entry.get('canonical_key') or entry.get('key') or ''}",
        f"- reason: {entry.get('reason') or ''}",
        f"- recorded_at: {entry.get('recorded_at') or ''}",
        "- source: improvement-log API",
    ]
    body = "\n".join(lines)
    try:
        from mobile_app.obsidian_bridge import append_improvement_note

        return append_improvement_note(f"改善ログ: {status or 'recorded'}", body)
    except Exception as exc:
        return {"status": "skipped", "reason": f"Obsidian save failed: {exc}"}


def _save_prompt_rule_to_obsidian(entry: dict) -> dict:
    """Mirror a prompt-rule registration to the user's Obsidian vault."""
    title = str(entry.get("title") or "PDCA修正登録")
    lines = [
        f"- source: {entry.get('source') or 'manual'}",
        f"- surface: {entry.get('surface') or ''}",
        f"- title: {title}",
        f"- rule: {entry.get('rule') or ''}",
        f"- reason: {entry.get('reason') or ''}",
        f"- recorded_at: {entry.get('recorded_at') or ''}",
        "- source: prompt-feedback API",
    ]
    body = "\n".join(lines)
    try:
        from mobile_app.obsidian_bridge import append_improvement_note

        return append_improvement_note(f"PDCA修正登録: {title}", body)
    except Exception as exc:
        return {"status": "skipped", "reason": f"Obsidian save failed: {exc}"}


@app.post("/api/improvement-log/dismiss")
def dismiss_improvement(req: DismissImprovementRequest):
    """改善案を「実装済み」としてledgerとObsidianに追記する。"""
    import json as _json
    from datetime import datetime as _dt
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    entry = {
        "key": req.key,
        "status": "applied",
        "title": req.title,
        "pr_url": "",
        "reason": "UI経由で手動実装済みマーク",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_improvement_log_to_obsidian(entry)
    writeback = record_cloudrun_input_event(
        event_type="improvement_dismiss",
        surface="improvement_log",
        payload={**entry, "obsidian": obsidian_result},
    )
    return {"ok": True, "key": req.key, "title": req.title, "obsidian": obsidian_result, "writeback": writeback}


@app.post("/api/improvement-log/review")
def review_improvement(req: ReviewImprovementRequest):
    """改善案の承認・却下・deferredをledgerとObsidianに書き込む（REV-039）。"""
    import json as _json
    from datetime import datetime as _dt
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    entry = {
        "key": req.key,
        "canonical_key": req.key,
        "status": req.action,
        "title": req.title,
        "pr_url": "",
        "reason": req.reason or f"UI経由で{req.action}",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_improvement_log_to_obsidian(entry)
    writeback = record_cloudrun_input_event(
        event_type="improvement_review",
        surface="improvement_log",
        payload={**entry, "obsidian": obsidian_result},
    )
    return {"ok": True, "key": req.key, "action": req.action, "obsidian": obsidian_result, "writeback": writeback}


@app.post("/api/prompt-feedback/rules/register")
def register_prompt_rule(req: PromptRuleRegisterRequest):
    """UIから1クリックで修正ルールを登録する。

    強いPDCAルールに該当しないものは live prompt へ入れず、改善ログへ隔離する。
    """
    import json as _json
    from datetime import datetime as _dt
    from prompt_feedback import append_pdca_rule
    from memory_promotion_policy import classify_memory_destination, is_pdca_rule_candidate

    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    rule_text = str(req.rule or "").strip()
    title = str(req.title or "").strip()
    reason = str(req.reason or "").strip()
    normalized_rule = rule_text or title
    destination = classify_memory_destination(normalized_rule)
    ledger_key = str(req.canonical_key or req.key or req.surface or title).strip()

    if not is_pdca_rule_candidate(normalized_rule):
        entry = {
            "key": ledger_key,
            "canonical_key": ledger_key,
            "status": "rule_review",
            "title": title or normalized_rule[:60],
            "rule": normalized_rule,
            "reason": reason or "PDCAルール条件外のため改善ログへ隔離",
            "source": req.source or "manual",
            "surface": req.surface or "",
            "destination": destination,
            "recorded_at": _dt.now().isoformat(),
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        obsidian_result = _save_improvement_log_to_obsidian(entry)
        return {
            "ok": True,
            "appended": False,
            "routed_to": "improvement_log",
            "reason": "pdca_rule_conditions_not_met",
            "rule": normalized_rule,
            "obsidian": obsidian_result,
        }

    append_result = append_pdca_rule(
        normalized_rule,
        source=req.source or "manual",
        reflection_summary=req.summary or None,
        ttl_days=90,
    )
    entry = {
        "key": ledger_key,
        "canonical_key": ledger_key,
        "status": "rule_registered",
        "title": title,
        "rule": normalized_rule,
        "reason": reason or normalized_rule,
        "source": req.source or "manual",
        "surface": req.surface or "",
        "destination": "pdca_rule",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_prompt_rule_to_obsidian(entry)
    return {
        "ok": True,
        "appended": append_result.get("appended", False),
        "count": append_result.get("count", 0),
        "rule": append_result.get("rule", normalized_rule),
        "path": append_result.get("path"),
        "obsidian": obsidian_result,
    }


@app.get("/api/improvement-pipeline/summary")
def get_improvement_pipeline_summary():
    """最新パイプライン実行のサマリーを返す（REV-039）。latest.json直読み。"""
    import json as _json, re as _re
    path = _latest_improvement_report_path()
    if path is None:
        return {"run_date": None, "applied_count": 0, "needs_review_count": 0, "failed_count": 0, "commit_result": None}
    try:
        with open(path, encoding="utf-8") as f:
            report = _json.load(f)
    except Exception:
        return {"run_date": None, "applied_count": 0, "needs_review_count": 0, "failed_count": 0, "commit_result": None}
    run_date: str | None = None
    m = _re.search(r"(\d{8})", path.name)
    if m:
        d = m.group(1)
        run_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return {
        "run_date": run_date,
        "applied_count": report.get("applied_count", 0),
        "needs_review_count": report.get("needs_review_count", len(report.get("needs_review", []))),
        "failed_count": report.get("failed_count", 0),
        "commit_result": report.get("commit_result"),
    }


@app.get("/api/subsidies")
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


@app.get("/api/asset/useful-life-all")
def get_useful_life_all():
    """法定耐用年数の全品目をカテゴリ付きで返す（REV-085/121）。"""
    import json as _json
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static_data", "useful_life_equipment.json")
    if not os.path.exists(json_path):
        return {"categories": []}
    with open(json_path, encoding="utf-8") as f:
        return _json.load(f)


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
def save_batch(req: BatchSaveRequest, background_tasks: BackgroundTasks):
    """確認済みCSVを一括スコアリングし、過去案件DBへ保存する。"""
    if not req.confirmed:
        raise HTTPException(status_code=422, detail="confirmed=true が必要です")
    if req.batch_token:
        result = _save_cached_batch(req.batch_token)
    else:
        result = _run_batch_scoring(req, save_to_db=True)
    background_tasks.add_task(_git_push_db)
    return result


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


@app.patch("/api/cases/{case_id}/result")
def patch_case_result(case_id: str, req: CaseResultPatch, background_tasks: BackgroundTasks):
    """案件結果を部分更新 (final_status / competitor_rate / loss_reason / final_result_date)"""
    from constants import FINAL_STATUS_VALID
    from data_cases import update_case

    if req.final_status is not None and req.final_status not in FINAL_STATUS_VALID:
        raise HTTPException(status_code=422, detail=f"不正な final_status: {req.final_status}")

    if req.source == "app" and req.final_status == "失注" and not (req.loss_reason or "").strip():
        raise HTTPException(status_code=400, detail="失注理由を入力してください")

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

    obsidian_result = {"status": "skipped", "reason": "not_attempted"}
    experience_result = {"status": "skipped", "reason": "not_attempted"}
    try:
        updated_case = _get_case_payload(case_id)
        obsidian_result = _append_case_result_reflection_to_obsidian(case_id, updated_case, patches)
        if req.final_status in ("成約", "失注", "検収", "検収完了"):
            experience_result = _promote_case_result_to_screening_experience(
                case_id=case_id,
                case_data=updated_case,
                status=req.final_status or "",
                patches=patches,
                source="case_result_auto",
            )
    except Exception as e:
        obsidian_result = {"status": "error", "reason": str(e)}
        experience_result = {"status": "error", "reason": str(e)}

    # 紫苑フィードバックループ（REV-080）
    outcome = req.final_status or ""
    if outcome in ("成約", "失注"):
        try:
            from lease_intelligence_mind import record_screening_feedback
            from lease_news_digest import find_vault as _find_vault_fb
            _fb_vault = _find_vault_fb()
            if _fb_vault:
                record_screening_feedback(_fb_vault, case_id, outcome)
        except Exception as _fb_err:
            print(f"[ShionFeedback] record_screening_feedback skipped: {_fb_err}")

    # 感情トリガー（REV-101）: 成約/失注による mood 更新
    if outcome in ("成約", "失注"):
        try:
            from api.emotion_trigger import trigger_emotion
            trigger_emotion(outcome)
        except Exception as _et_err:
            print(f"[EmotionTrigger] result patch skipped: {_et_err}")

    background_tasks.add_task(_git_push_db)
    return {
        "status": "updated",
        "case_id": case_id,
        "obsidian_reflection": obsidian_result,
        "experience_promotion": experience_result,
    }


@app.get("/api/cases/pending")
def get_pending_cases():
    """past_cases と Cloud Run帰還スコア入力から未登録案件を統合して取得する。"""
    import json

    rows = []

    try:
        with get_connection() as conn:
            res = conn.execute(
                "SELECT id, timestamp, industry_sub, score, data "
                "FROM past_cases "
                "WHERE COALESCE(NULLIF(final_status, ''), '未登録') IN ('未登録', '稟議中', 'スコアリングのみ') "
                "ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            for r in res:
                try:
                    d = json.loads(r["data"] or "{}")
                except Exception:
                    d = {}
                inputs = d.get("inputs") if isinstance(d.get("inputs"), dict) else {}
                result = d.get("result") if isinstance(d.get("result"), dict) else {}
                rows.append({
                    "id": str(r["id"]),
                    "company_no": d.get("company_no") or inputs.get("company_no") or "",
                    "company_name": d.get("company_name") or inputs.get("company_name") or "名称未設定",
                    "timestamp": r["timestamp"],
                    "score": r["score"] if r["score"] not in (None, "") else result.get("score_base", result.get("score")),
                    "industry": r["industry_sub"] or d.get("industry_sub") or inputs.get("industry_sub") or d.get("industry_major") or inputs.get("industry_major") or "",
                    "registration_date": d.get("registration_date") or (r["timestamp"] or "")[:10],
                    "estimate_sent_date": d.get("estimate_sent_date") or (r["timestamp"] or "")[:10],
                    "final_result_date": d.get("final_result_date"),
                    "_source": "past_cases"
                })
    except Exception as e:
        logger.error("get_pending_cases DB error: %s", e)

    cloudrun_rows = _list_cloudrun_score_pending_cases(limit=50)
    rows.extend(cloudrun_rows)
    rows.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return rows[:80]

@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: str):
    """案件の全データを返す（result + inputs を含む）"""
    import json

    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM past_cases WHERE id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        payload = json.loads(row["data"] or "{}")
        payload.setdefault("_source", "past_cases")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_case_detail DB error case_id=%s: %s", case_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cases/operation/clear-all")
def clear_all_pending_cases(background_tasks: BackgroundTasks):
    """未登録案件をすべて削除する（一括クリア）"""
    from data_cases import refresh_stats_caches
    try:
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM past_cases "
                "WHERE COALESCE(NULLIF(final_status, ''), '未登録') IN ('未登録', '稟議中', 'スコアリングのみ')"
            )
        refresh_stats_caches()
        try:
            for item in _list_cloudrun_score_pending_cases(limit=200):
                item_id = str(item.get("id") or "")
                if not _reject_cloudrun_score_pending_case(item_id):
                    _reject_cloudrun_event_pending_case(item_id)
        except Exception as exc:
            logger.warning("clear cloudrun pending score inputs skipped: %s", exc)
        background_tasks.add_task(_git_push_db)
        return {"message": "Cleared all pending cases"}
    except Exception as e:
        logger.error("clear_all_pending_cases DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str, background_tasks: BackgroundTasks):
    """案件を past_cases から削除する"""
    from data_cases import delete_case as delete_case_from_db
    try:
        if not _reject_cloudrun_score_pending_case(str(case_id)) and not _reject_cloudrun_event_pending_case(str(case_id)):
            delete_case_from_db(str(case_id))
    except Exception:
        pass
    background_tasks.add_task(_git_push_db)
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
    estat_context: Optional[dict] = None


@app.post("/api/gunshi/stream")
async def gunshi_stream(req: GunshiStreamRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    params = req.model_dump()

    async def event_generator():
        # bayes / phrases チャンクは既存ロジックで生成（互換維持）
        from api.gunshi_gemini import (
            compute_prior, compute_posterior, select_top_phrases,
            _bayes_inputs, build_strategy_cards,
        )
        score = params.get("score", 0)
        pd_pct = params.get("pd_pct", 0)
        industry_cat = params.get("industry_cat", "")

        prior = compute_prior(score, pd_pct)
        bayes_inputs = _bayes_inputs(params)
        posterior = compute_posterior(prior=prior, **bayes_inputs)

        phrase_dicts = select_top_phrases(
            industry_cat=industry_cat,
            score=score,
            pd_pct=pd_pct,
            resale=bayes_inputs["resale"],
            repeat_cnt=bayes_inputs["repeat_cnt"],
            subsidy=bayes_inputs["subsidy"],
            bank=bayes_inputs["bank"],
            posterior=posterior,
            asset_name=params.get("asset_name", ""),
            n=3,
        )
        phrases = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in phrase_dicts]

        yield f"data: {json.dumps({'type': 'bayes', 'prior': prior, 'posterior': posterior}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'phrases', 'items': phrases}, ensure_ascii=False)}\n\n"
        cards = build_strategy_cards(params, phrases, prior, posterior)
        yield f"data: {json.dumps({'type': 'strategy_cards', 'cards': cards}, ensure_ascii=False)}\n\n"

        # 紫苑ADKエージェントがツールを自律実行しながらコメントをストリーム
        try:
            from api.shion_agent import stream_shion_screening
            async for chunk in stream_shion_screening(params):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as _adk_err:
            # ADK失敗時は既存の軍師Geminiにフォールバック
            print(f"[WARNING] shion ADK stream failed, fallback to gunshi_gemini: {_adk_err}")
            async for chunk in stream_gunshi_gemini(params, api_key):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


_GUNSHI_GENERAL_CHAT_PROMPT = (
    "あなたはリース審査AIの軍師です。"
    "戦国武将のような凜とした口調を保ちながら、雑談や一般的な質問にも気さくに答えます。"
    "天気や最新ニュースなど具体的なデータが必要な場合は「〇〇でご確認あれ」と案内しつつ、知っている範囲で答えてください。"
    "回答は簡潔に。日本語で答えてください。"
)

_YUKIKAZE_GENERAL_CHAT_PROMPT = (
    "You are YUKIKAZE // FFR-41MR. "
    "For general or off-topic questions, respond in minimal DATALINK style. "
    "TX: for transmit, RX: for response. Brief and cold. No pleasantries."
)


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
    estat_context: Optional[dict] = None


def _format_estat_context_for_prompt(estat_context: Optional[dict]) -> str:
    if not estat_context:
        return ""
    summary = str(estat_context.get("summary") or "").strip()
    if not summary:
        return ""
    lines = [summary]
    score = estat_context.get("score")
    status = estat_context.get("status")
    if score is not None:
        lines.append(f"総合 {float(score):.1f}点")
    if status:
        status_label = {"green": "整合良好", "yellow": "参考", "red": "要確認"}.get(str(status), str(status))
        lines.append(f"判定 {status_label}")
    recs = [str(item).strip() for item in (estat_context.get("recommendations") or []) if str(item).strip()]
    if recs:
        lines.append("示唆 " + " / ".join(recs[:2]))
    return "\n".join(lines)


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
    try:
        from chat_intent import is_ambiguous_question, is_today_scope_clarification_needed
    except Exception:  # pragma: no cover - fallback
        is_ambiguous_question = lambda _msg: False  # type: ignore
        is_today_scope_clarification_needed = lambda _msg: False  # type: ignore

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
    if is_today_scope_clarification_needed(msg):
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Scope clarification required.",
            "RX: 今日の何について知りたいですか？",
        ])
    if is_ambiguous_question(msg):
        return "\n".join([
            "DATALINK LOG:",
            "TX: PANPANPAN // Ambiguous question detected.",
            "RX: 何についての質問ですか？ 対象、目的、比較したい相手のどれかを教えてください。",
        ])
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
                        "estat_context": req.estat_context,
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
                        "estat_context": req.estat_context,
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

        # general カテゴリの質問は Obsidian/案件コンテキストをスキップして直接回答
        if (req.message or "").strip() and _classify_question(req.message) == "general":
            try:
                from api.chat_memory import call_gemini_chat as _gchat
                _sys = _YUKIKAZE_GENERAL_CHAT_PROMPT if is_yukikaze else _GUNSHI_GENERAL_CHAT_PROMPT
                _hist = [{"role": h.get("role", ""), "content": h.get("text", "")} for h in req.history]
                reply_text = _gchat(_sys, _hist, req.message.strip())
            except Exception as _ge:
                reply_text = f"【エラー】一般会話の生成に失敗しました: {_ge}"
            return {"chat_text": reply_text, "reply": reply_text}

        advices = PHRASES_100.get("逆転アドバイス", [])
        import random
        sampled = random.sample(advices, min(3, len(advices)))

        has_case_context = req.score != 0 or bool((req.industry_major or "").strip())
        if has_case_context:
            # リース知性体の未解決の懸念を軍師プロンプトへ放送する（GWT broadcast）。
            # スコアリング時に記録済みの pending_dissonance を読むだけ・完全非ブロッキング。
            _dissonance_section = ""
            try:
                from lease_intelligence_mind import build_gunshi_dissonance_section
                from lease_news_digest import find_vault as _find_vault

                _dissonance_section = build_gunshi_dissonance_section(_find_vault())
            except Exception as _diss_err:
                print(f"[WARNING] gunshi dissonance section skipped: {_diss_err}")
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
                estat_context_text=_format_estat_context_for_prompt(req.estat_context),
                dissonance_section=_dissonance_section,
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
                    "あなたはTune式リース審査AIの軍師です。"
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

        try:
            from prompt_feedback import build_pdca_prompt_block as _build_pdca
            _pdca_block = _build_pdca()
            if _pdca_block:
                prompt += f"\n\n{_pdca_block}"
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
                url = _gemini_generate_url()
                r = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    headers={"x-goog-api-key": api_key},
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
        payload = dict(payload)
        payload["lease_news_focus"] = _lease_news_focus_to_dict(get_latest_lease_news_focus())
        payload["lease_news_reflection"] = _lease_news_reflection_to_dict(get_latest_lease_news_reflection())
        payload["lease_news_actions"] = _lease_news_actions_to_dict(get_latest_lease_news_actions())
        payload["improvement_highlights"] = _load_latest_improvement_highlights(limit=3)
        payload["lease_system_gaps"] = _load_lease_system_gap_analysis(limit=3)
        return payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _lease_news_focus_to_dict(focus):
    if not focus or not getattr(focus, "available", False):
        return {"available": False}
    return {
        "available": True,
        "note_path": getattr(focus, "note_path", ""),
        "note_date": getattr(focus, "note_date", ""),
        "profile": getattr(focus, "profile", ""),
        "theme_summary": getattr(focus, "theme_summary", ""),
        "bucket_summary": getattr(focus, "bucket_summary", ""),
        "tag_summary": getattr(focus, "tag_summary", ""),
        "focus_lines": list(getattr(focus, "focus_lines", ()) or ()),
        "memo_lines": list(getattr(focus, "memo_lines", ()) or ()),
        "metrics_lines": list(getattr(focus, "metrics_lines", ()) or ()),
        "article_titles": list(getattr(focus, "article_titles", ()) or ()),
        "headline": getattr(focus, "headline", ""),
    }


def _lease_news_reflection_to_dict(reflection):
    if not reflection or not getattr(reflection, "available", False):
        return {"available": False}
    return {
        "available": True,
        "note_path": getattr(reflection, "note_path", ""),
        "note_date": getattr(reflection, "note_date", ""),
        "theme_summary": getattr(reflection, "theme_summary", ""),
        "tag_summary": getattr(reflection, "tag_summary", ""),
        "headline": getattr(reflection, "headline", ""),
        "thought_lines": list(getattr(reflection, "thought_lines", ()) or ()),
        "tomorrow_lines": list(getattr(reflection, "tomorrow_lines", ()) or ()),
        "illustration_url": getattr(reflection, "illustration_url", ""),
        "continuity_days": getattr(reflection, "continuity_days", 0),
        "dominant_mood": getattr(reflection, "dominant_mood", ""),
        "self_narrative": getattr(reflection, "self_narrative", ""),
        "current_question": getattr(reflection, "current_question", ""),
        "memory_excerpt": getattr(reflection, "memory_excerpt", ""),
        "user_understanding": getattr(reflection, "user_understanding", ""),
        "user_curiosity": getattr(reflection, "user_curiosity", ""),
        "user_interests": list(getattr(reflection, "user_interests", ()) or ()),
        "observed_days": getattr(reflection, "observed_days", 0),
        "primary_goal": getattr(reflection, "primary_goal", ""),
        "secondary_goal": getattr(reflection, "secondary_goal", ""),
        "ultimate_goal": getattr(reflection, "ultimate_goal", ""),
        "ultimate_goal_status": getattr(reflection, "ultimate_goal_status", ""),
        "knowledge_available": getattr(reflection, "knowledge_available", False),
        "knowledge_scope": getattr(reflection, "knowledge_scope", ""),
        "indexed_notes": getattr(reflection, "indexed_notes", 0),
        "knowledge_source_count": getattr(reflection, "knowledge_source_count", 0),
        "knowledge_sources": list(getattr(reflection, "knowledge_sources", ()) or ()),
    }


def _lease_news_actions_to_dict(actions):
    if not actions or not getattr(actions, "available", False):
        return {"available": False}
    return {
        "available": True,
        "date": getattr(actions, "date", ""),
        "note_path": getattr(actions, "note_path", ""),
        "json_path": getattr(actions, "json_path", ""),
        "summary": getattr(actions, "summary", ""),
        "action_items": [
            {
                "signal": getattr(item, "signal", ""),
                "affected_industries": list(getattr(item, "affected_industries", ()) or ()),
                "affected_assets": list(getattr(item, "affected_assets", ()) or ()),
                "risk_flags": list(getattr(item, "risk_flags", ()) or ()),
                "recommended_checks": list(getattr(item, "recommended_checks", ()) or ()),
                "condition_impacts": list(getattr(item, "condition_impacts", ()) or ()),
                "source_title": getattr(item, "source_title", ""),
                "source_path": getattr(item, "source_path", ""),
                "valid_until": getattr(item, "valid_until", ""),
                "confidence": getattr(item, "confidence", 0.0),
                "noise_score": getattr(item, "noise_score", 0.0),
            }
            for item in (getattr(actions, "action_items", ()) or ())
        ],
        "ignored_titles": list(getattr(actions, "ignored_titles", ()) or ()),
    }


def _daily_greeting_read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _daily_greeting_git_summary() -> str:
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "log", "--since=yesterday", "--pretty=format:%s", "-4"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            return " / ".join(lines[:2])
    except Exception:
        pass
    return ""


def _daily_greeting_yesterday_note() -> str:
    try:
        import datetime as _dt

        vault_raw = _OBSIDIAN_VAULT_PATH or os.environ.get("OBSIDIAN_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT") or ""
        if not vault_raw:
            found = find_vault()
            vault_raw = str(found) if found else ""
        if not vault_raw:
            return ""
        yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
        path = Path(vault_raw) / "Daily" / f"{yesterday}.md"
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [
            re.sub(r"^[#>*\-\s]+", "", line).strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("```")
        ]
        return next((line for line in lines if len(line) >= 18), "")[:120]
    except Exception:
        return ""


def _daily_greeting_anniversary() -> dict:
    try:
        import datetime as _dt

        key = _dt.date.today().strftime("%m-%d")
        data = _daily_greeting_read_json(Path(_REPO_ROOT) / "data" / "shion_anniversaries.json")
        item = data.get(key) or {}
        if item:
            return {"date_key": key, **item}
    except Exception:
        pass
    return {
        "date_key": "",
        "name": "小さな兆候を見る日",
        "note": "今日は、数字やニュースの端に出る小さな違和感を拾ってから判断します。",
    }


def _daily_greeting_news_thought() -> str:
    try:
        actions = _daily_greeting_read_json(Path(_REPO_ROOT) / "data" / "lease_news_actions_latest.json")
        summary = str(actions.get("summary") or "").strip()
        action_items = actions.get("action_items") or []
        if action_items:
            checks = action_items[0].get("recommended_checks") or []
            if checks:
                return str(checks[0]).strip()[:160]
        if summary:
            return f"ニュースからは「{summary[:80]}」が見えています。今日はこれを審査条件に直結させすぎず、確認観点として扱います。"
    except Exception:
        pass
    return "ニュースはまだ薄めです。今日は外部情報より、前回の作業と手元の案件条件を優先して見ます。"


def _daily_greeting_opening(now) -> dict:
    hour = int(getattr(now, "hour", 12))
    if 5 <= hour < 10:
        return {
            "text": "おはようございます。",
            "mood": "朝なので、昨日の続きと今日の最初の一手を短く整理します。",
            "time_band": "morning",
        }
    if 10 <= hour < 17:
        return {
            "text": "こんにちは。",
            "mood": "日中なので、今すぐ進める作業順に並べます。",
            "time_band": "daytime",
        }
    if 17 <= hour < 22:
        return {
            "text": "こんばんは。",
            "mood": "夕方以降なので、今日の判断材料を回収しながら進めます。",
            "time_band": "evening",
        }
    return {
        "text": "夜更かしですね。",
        "mood": "深い時間なので、無理に広げず、次に残す判断だけ整えます。",
        "time_band": "late_night",
    }


@app.get("/api/shion/daily-greeting")
def get_shion_daily_greeting():
    """紫苑コンシェルジュの毎日変わる一言挨拶を返す。外部通信なしの安定版。"""
    try:
        import datetime as _dt

        now = _dt.datetime.now()
        today = now.date()
        opening = _daily_greeting_opening(now)
        git_summary = _daily_greeting_git_summary()
        yesterday_note = _daily_greeting_yesterday_note()
        anniversary = _daily_greeting_anniversary()
        news_thought = _daily_greeting_news_thought()
        yesterday = git_summary or yesterday_note or "昨日の作業ログは薄めです。今日は入口で状況を整理してから始めます。"
        suggestion = "まず紫苑の予測を見て、審査入力・外部調査・チャットのどこから始めるかを選びましょう。"
        if opening["time_band"] == "evening":
            suggestion = "今日は広げすぎず、審査入力・Research・チャットのうち、残す判断材料を一つ回収しましょう。"
        elif opening["time_band"] == "late_night":
            suggestion = "今は作業を増やすより、明日すぐ再開できる入口を一つだけ決めましょう。"
        return {
            "date": today.isoformat(),
            "opening": opening["text"],
            "time_band": opening["time_band"],
            "time_note": opening["mood"],
            "yesterday": yesterday,
            "anniversary": anniversary,
            "thought": news_thought,
            "suggestion": suggestion,
            "source": {
                "git": bool(git_summary),
                "yesterday_daily": bool(yesterday_note),
                "anniversary": "data/shion_anniversaries.json",
                "news": "data/lease_news_actions_latest.json",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _lease_news_brief_to_dict(brief):
    if not brief or not getattr(brief, "available", False):
        return {"available": False}
    return {
        "available": True,
        "prefecture": getattr(brief, "prefecture", ""),
        "region": getattr(brief, "region", ""),
        "geo_context": getattr(brief, "geo_context", ""),
        "national_headline": getattr(brief, "national_headline", ""),
        "national_focus_lines": list(getattr(brief, "national_focus_lines", ()) or ()),
        "regional_available": getattr(brief, "regional_available", False),
        "regional_title": getattr(brief, "regional_title", ""),
        "regional_summary_lines": list(getattr(brief, "regional_summary_lines", ()) or ()),
        "regional_usage_memo": getattr(brief, "regional_usage_memo", ""),
        "regional_tags": list(getattr(brief, "regional_tags", ()) or ()),
        "regional_source": getattr(brief, "regional_source", ""),
        "opening_line": getattr(brief, "opening_line", ""),
        "question_line": getattr(brief, "question_line", ""),
        "note_date": getattr(brief, "note_date", ""),
        "note_path": getattr(brief, "note_path", ""),
    }


@app.get("/api/lease-news/focus")
def get_lease_news_focus_api():
    """ホーム画面とAICHATで共通利用する最新ニュースの注目論点を返す。"""
    try:
        return _lease_news_focus_to_dict(get_latest_lease_news_focus())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lease-news/brief")
def get_lease_news_brief_api(prefecture: str = "", industry: str = ""):
    """AICHATとホームで共通利用する、全国+地域のニュースブリーフを返す。"""
    try:
        return _lease_news_brief_to_dict(build_lease_news_brief(prefecture=prefecture, industry=industry))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lease-news/actions")
def get_lease_news_actions_api():
    """日次ニュースを審査アクションへ変換した一覧を返す。"""
    try:
        return _lease_news_actions_to_dict(get_latest_lease_news_actions())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _candidate_report_dirs() -> list[Path]:
    bundle_root = Path(os.environ.get("CLOUDRUN_BUNDLE_DIR") or (Path(_REPO_ROOT) / ".cloudrun_bundle"))
    dirs = [Path(_REPO_ROOT) / "reports", bundle_root / "reports"]
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def _latest_improvement_report_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        latest = reports_dir / "latest.json"
        if latest.exists():
            return latest
    candidates: list[Path] = []
    for reports_dir in _candidate_report_dirs():
        candidates.extend(reports_dir.glob("improvement_report_*.json"))
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


def _latest_recursive_self_improvement_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        latest = reports_dir / "recursive_self_improvement_latest.json"
        if latest.exists():
            return latest
    candidates: list[Path] = []
    for reports_dir in _candidate_report_dirs():
        candidates.extend(reports_dir.glob("recursive_self_improvement_*.json"))
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


def _load_latest_improvement_highlights(limit: int = 3) -> dict:
    report_path = _latest_improvement_report_path()
    if not report_path:
        return {"available": False, "items": [], "source": ""}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "items": [], "source": str(report_path)}

    needs_review_items = report.get("needs_review") or []
    raw_items = needs_review_items or report.get("auto_fix_candidates") or report.get("improvements") or []
    items: list[dict[str, str]] = []
    for raw in raw_items:
        if len(items) >= limit:
            break
        if not isinstance(raw, dict):
            continue
        policy = raw.get("auto_fix_policy") or {}
        if str(policy.get("risk") or raw.get("priority") or "").lower() == "high":
            continue
        temp_item = {
            "status": "NEEDS_REVIEW" if raw in needs_review_items else "AUTO_FIX_CANDIDATE",
            "priority": raw.get("priority") or policy.get("risk") or "",
            "reason": raw.get("reason") or policy.get("reason") or "",
            "detail": raw.get("detail") or raw.get("description") or "",
            "duplicate_count": raw.get("duplicate_count") or 0,
            "recommended_order": raw.get("recommended_order"),
            "auto_fix_policy": policy,
        }
        should_park, _park_reason = _should_park_improvement(temp_item)
        if should_park:
            continue
        items.append({
            "id": str(raw.get("id") or ""),
            "title": str(raw.get("title") or ""),
            "status": "要確認" if raw in needs_review_items else "候補",
            "priority": str(raw.get("priority") or policy.get("risk") or ""),
            "reason": str(raw.get("reason") or policy.get("reason") or ""),
            "category": str(raw.get("category") or ""),
            "canonical_key": str(raw.get("canonical_key") or ""),
        })

    def _report_count(key: str, fallback_items) -> int:
        summary = report.get("summary") or {}
        summary_key = f"{key}_count"
        value = summary.get(summary_key, report.get(summary_key))
        if value is None:
            value = report.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return len(fallback_items or [])

    return {
        "available": bool(items),
        "date": str(report.get("date") or ""),
        "generated_at": str(report.get("generated_at") or ""),
        "status": str(report.get("status") or ""),
        "source": str(report_path),
        "items": items,
        "counts": {
            "applied": _report_count("applied", report.get("applied")),
            "auto_fix_candidates": _report_count("auto_fix_candidates", report.get("auto_fix_candidates")),
            "needs_review": _report_count("needs_review", needs_review_items),
            "rejected": _report_count("rejected", report.get("rejected")),
        },
    }


def _load_lease_system_gap_analysis(limit: int | None = None) -> dict:
    """Read-only system gap analysis summary generated by scripts/lease_system_gap_analyzer.py."""
    path = Path(_REPO_ROOT) / "reports" / "lease_system_gap_analysis.json"
    if not path.exists():
        return {"available": False, "items": [], "source": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "items": [], "source": str(path)}
    gaps = payload.get("gaps") or []
    if not isinstance(gaps, list):
        gaps = []
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items = [item for item in gaps if isinstance(item, dict)]
    items.sort(key=lambda item: (priority_order.get(str(item.get("priority") or "").lower(), 9), str(item.get("id") or "")))
    if limit is not None:
        items = items[:limit]
    counts: dict[str, int] = {}
    for item in gaps:
        if isinstance(item, dict):
            priority = str(item.get("priority") or "unknown").lower()
            counts[priority] = counts.get(priority, 0) + 1
    return {
        "available": True,
        "generated_at": str(payload.get("generated_at") or ""),
        "mode": str(payload.get("mode") or "read-only diagnostics"),
        "source": str(path),
        "counts": counts,
        "items": items,
    }


@app.get("/api/lease-system-gaps")
def get_lease_system_gaps():
    return _load_lease_system_gap_analysis()

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

_REPORT_QUALITY_LOG = Path(__file__).parent.parent / "data" / "report_quality_log.jsonl"
_report_quality_log_lock = __import__("threading").Lock()


@app.post("/api/report/generate")
def generate_report(req: ReportRequest):
    try:
        import uuid as _uuid
        from report_generator import generate_full_report_from_res
        report_id = str(_uuid.uuid4())
        dummy_session = {
            "rep_company": req.inputs.get("company_name", "（企業名未設定）"),
            "last_submitted_inputs": req.inputs,
            "humor_style": "standard"
        }
        res_data = req.result_data

        report_text = generate_full_report_from_res(res_data, dummy_session)
        return {"report_markdown": report_text, "report_id": report_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ReportFeedbackRequest(BaseModel):
    report_id: str
    rating: Literal["good", "bad"]
    surface: str = "report_viewer"
    comment: str = ""


@app.post("/api/report-feedback")
def post_report_feedback(req: ReportFeedbackRequest) -> dict:
    import datetime as _dt, json as _json
    entry = _json.dumps({
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "report_id": req.report_id,
        "rating": req.rating,
        "surface": req.surface,
        "comment": req.comment,
    }, ensure_ascii=False) + "\n"
    with _report_quality_log_lock:
        with open(_REPORT_QUALITY_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    return {"status": "ok"}



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
    # 共通実装へ委譲（環境変数 → secrets.toml 直接パース、Streamlit非依存）
    from api.secret_access import get_gemini_api_key

    return get_gemini_api_key()


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

        url = _gemini_generate_url()
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"x-goog-api-key": api_key},
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
    final_rate: Optional[float] = 0.0
    base_rate_at_time: Optional[float] = 2.1
    lost_reason: str = ""
    loan_conditions: list[str] = Field(default_factory=list)
    competitor_name: str = ""
    competitor_rate: Optional[float] = 0.0
    note: str = ""
    human_discomfort: str = ""
    but_still_reason: str = ""
    approval_condition_memo: str = ""
    non_negotiable_condition: str = ""
    retrospective_note: str = ""


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
def register_case_result(req: CaseRegistration, background_tasks: BackgroundTasks):
    from data_cases import load_all_cases, update_case
    cases = load_all_cases()
    final_rate = float(req.final_rate or 0.0)
    base_rate_at_time = float(req.base_rate_at_time or 2.1)
    competitor_rate = float(req.competitor_rate or 0.0)
    
    target_case_id = None
    target_case = None
    for c in cases:
        # ID, 企業番号, または企業名でマッチング（大文字小文字無視など不要なほど厳密に）
        if (c.get("id") == req.case_id or 
            c.get("company_no") == req.case_id or 
            c.get("company_name") == req.case_id or
            # inputsの中身も一応見る
            c.get("inputs", {}).get("company_no") == req.case_id or
            c.get("inputs", {}).get("company_name") == req.case_id):
            target_case_id = c.get("id")
            target_case = c
            break
            
    if not target_case_id:
        cloudrun_event_id = _parse_cloudrun_event_case_id(req.case_id)
        if cloudrun_event_id:
            final_result = record_cloudrun_input_event(
                event_type="case_result_registered",
                surface="cases_register",
                payload={
                    "case_id": req.case_id,
                    "source_event_id": cloudrun_event_id,
                    "status": req.status,
                    "final_rate": final_rate,
                    "base_rate_at_time": base_rate_at_time,
                    "competitor_rate": competitor_rate,
                    "lost_reason": req.lost_reason,
                    "note": req.note,
                    "grey_judgment": {
                        "status": req.status,
                        "human_discomfort": (req.human_discomfort or "").strip(),
                        "but_still_reason": (req.but_still_reason or "").strip(),
                        "approval_condition_memo": (req.approval_condition_memo or "").strip(),
                        "non_negotiable_condition": (req.non_negotiable_condition or "").strip(),
                        "retrospective_note": (req.retrospective_note or "").strip(),
                    },
                },
            )
            if final_result.get("ok") is True:
                experience_result = {"status": "skipped", "reason": "cloudrun_event_not_found"}
                try:
                    event = _find_cloudrun_input_event(cloudrun_event_id)
                    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
                    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
                    case_data = {
                        "inputs": inputs,
                        "result": result_payload,
                        "final_status": req.status,
                        "final_rate": final_rate,
                        "base_rate_at_time": base_rate_at_time,
                        "competitor_rate": competitor_rate,
                        "lost_reason": req.lost_reason,
                        "final_note": req.note,
                        "grey_judgment": {
                            "status": req.status,
                            "human_discomfort": (req.human_discomfort or "").strip(),
                            "but_still_reason": (req.but_still_reason or "").strip(),
                            "approval_condition_memo": (req.approval_condition_memo or "").strip(),
                            "non_negotiable_condition": (req.non_negotiable_condition or "").strip(),
                            "retrospective_note": (req.retrospective_note or "").strip(),
                        },
                    }
                    experience_result = _promote_case_result_to_screening_experience(
                        case_id=req.case_id,
                        case_data=case_data,
                        status=req.status,
                        patches=case_data,
                        source="case_result_auto",
                    )
                except Exception as exp_err:
                    experience_result = {"status": "error", "reason": str(exp_err)}
                _invalidate_cloudrun_input_events_cache()
                return {
                    "status": "success",
                    "message": f"Cloud Run result registered for {cloudrun_event_id}",
                    "case_id": req.case_id,
                    "cloudrun_writeback": final_result,
                    "experience_promotion": experience_result,
                }
            raise HTTPException(status_code=500, detail=f"Cloud Run result writeback failed: {final_result.get('reason')}")

        score_input_id = _parse_cloudrun_score_case_id(req.case_id)
        if score_input_id is not None:
            target_case_id = _promote_cloudrun_score_input_to_pending_case(score_input_id)
            if target_case_id:
                cases = load_all_cases()
                target_case = next((case for case in cases if case.get("id") == target_case_id), None)
        if not target_case_id or not target_case:
            raise HTTPException(status_code=404, detail="Case not found")
        
    import datetime
    now_iso = datetime.datetime.now().isoformat()
    now_date = now_iso[:10]
    registration_date = target_case.get("registration_date") or target_case.get("timestamp", "")[:10] or now_date
    estimate_sent_date = target_case.get("estimate_sent_date") or registration_date

    patches = {
        "final_status": req.status,
        "final_rate": final_rate,
        "base_rate_at_time": base_rate_at_time,
        "loan_conditions": req.loan_conditions,
        "competitor_name": req.competitor_name,
        "competitor_rate": competitor_rate if competitor_rate > 0 else None,
        "final_note": req.note,
        "registration_date": registration_date,
        "estimate_sent_date": estimate_sent_date,
        "final_result_date": now_date,
        "final_result_timestamp": now_iso,
    }
    grey_judgment = {
        "registered_at": now_iso,
        "status": req.status,
        "human_discomfort": (req.human_discomfort or "").strip(),
        "but_still_reason": (req.but_still_reason or "").strip(),
        "approval_condition_memo": (req.approval_condition_memo or "").strip(),
        "non_negotiable_condition": (req.non_negotiable_condition or "").strip(),
        "retrospective_note": (req.retrospective_note or "").strip(),
        "ai_score": target_case.get("score") or target_case.get("score_base") or (target_case.get("result") or {}).get("score"),
        "ai_decision": target_case.get("hantei") or (target_case.get("result") or {}).get("hantei"),
    }
    if any(str(grey_judgment.get(k) or "").strip() for k in (
        "human_discomfort",
        "but_still_reason",
        "approval_condition_memo",
        "non_negotiable_condition",
        "retrospective_note",
    )):
        patches["grey_judgment"] = grey_judgment
        patches["human_discomfort"] = grey_judgment["human_discomfort"]
        patches["but_still_reason"] = grey_judgment["but_still_reason"]
        patches["approval_condition_memo"] = grey_judgment["approval_condition_memo"]
        patches["non_negotiable_condition"] = grey_judgment["non_negotiable_condition"]
        patches["retrospective_note"] = grey_judgment["retrospective_note"]
    if req.status == "成約" and final_rate > 0:
        patches["winning_spread"] = final_rate - base_rate_at_time
    if req.status == "失注":
        patches["lost_reason"] = req.lost_reason

    if not update_case(target_case_id, patches):
        raise HTTPException(status_code=500, detail="Failed to update DB")
    experience_result = {"status": "skipped", "reason": "not_attempted"}
    if req.status in ("成約", "失注", "検収", "検収完了"):
        try:
            updated_case = _get_case_payload(target_case_id) or {**target_case, **patches}
            experience_result = _promote_case_result_to_screening_experience(
                case_id=target_case_id,
                case_data=updated_case,
                status=req.status,
                patches=patches,
                source="case_result_auto",
            )
        except Exception as exp_err:
            experience_result = {"status": "error", "reason": str(exp_err)}
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="case_result_registered",
        surface="cases_register",
        payload={
            "case_id": target_case_id,
            "status": req.status,
            "final_rate": final_rate,
            "base_rate_at_time": base_rate_at_time,
            "competitor_rate": competitor_rate,
            "lost_reason": req.lost_reason,
            "note": req.note,
            "grey_judgment": grey_judgment if "grey_judgment" in patches else {},
        },
    )

    try:
        from shinsa_gunshi import refresh_evidence_weights
        refresh_evidence_weights()
    except Exception:
        pass

    # 紫苑フィードバックループ（REV-080）
    if req.status in ("成約", "失注"):
        try:
            from lease_intelligence_mind import record_screening_feedback
            from lease_news_digest import find_vault as _find_vault_reg
            _reg_vault = _find_vault_reg()
            if _reg_vault:
                record_screening_feedback(_reg_vault, target_case_id, req.status)
        except Exception as _reg_err:
            print(f"[ShionFeedback] record_screening_feedback skipped: {_reg_err}")

    # 感情トリガー（REV-101）: 成約/失注による mood 更新
    if req.status in ("成約", "失注"):
        try:
            from api.emotion_trigger import trigger_emotion
            trigger_emotion(req.status)
        except Exception as _et_err:
            print(f"[EmotionTrigger] register skipped: {_et_err}")

    # ミニPDCAトリガー: 成約/失注登録時にAI判定vs実結果を記録し、5件溜まったらPDCA実行
    if req.status in ("成約", "失注"):
        try:
            from judgment_feedback import record_judgment_feedback, count_unprocessed_feedback
            _ai_score = float(target_case.get("score") or target_case.get("score_base") or (target_case.get("result") or {}).get("score") or 0)
            _ai_decision = "承認" if _ai_score >= APPROVAL_LINE else "条件付き" if _ai_score >= CONDITIONAL_LINE else "否決"
            _human_decision = "承認" if req.status == "成約" else "否決"
            _reason_bits = [f"案件登録トリガー: {req.status}（AIスコア {_ai_score:.1f}）"]
            if grey_judgment.get("human_discomfort"):
                _reason_bits.append(f"違和感: {grey_judgment['human_discomfort']}")
            if grey_judgment.get("but_still_reason"):
                _reason_bits.append(f"それでも判断した理由: {grey_judgment['but_still_reason']}")
            if grey_judgment.get("approval_condition_memo"):
                _reason_bits.append(f"条件: {grey_judgment['approval_condition_memo']}")
            _fb = record_judgment_feedback(
                case_id=target_case_id,
                model_decision=_ai_decision,
                human_decision=_human_decision,
                reason=" / ".join(_reason_bits),
                source="register_trigger",
                score=_ai_score if _ai_score > 0 else None,
            )
            if _fb.get("success"):
                _pending = count_unprocessed_feedback()
                if _pending >= 5:
                    try:
                        from llm_pdca_reflection import run_monthly_pdca_reflection
                        if _pdca_reflection_lock.acquire(blocking=False):
                            def _run_pdca_reflection():
                                try:
                                    run_monthly_pdca_reflection(force=True, max_cases=20)
                                finally:
                                    _pdca_reflection_lock.release()
                            _background_executor.submit(_run_pdca_reflection)
                            print(f"[MiniPDCA] triggered (pending={_pending})")
                        else:
                            print(f"[MiniPDCA] skipped: already running (pending={_pending})")
                    except Exception as _pdca_err:
                        print(f"[MiniPDCA] reflection skipped: {_pdca_err}")
        except Exception as _mini_err:
            print(f"[MiniPDCA] feedback skipped: {_mini_err}")

    return {
        "status": "success",
        "message": f"Results updated for {target_case_id}",
        "experience_promotion": experience_result,
    }

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


def _latest_improvement_report_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        latest = reports_dir / "latest.json"
        if latest.exists():
            return latest
    candidates: list[Path] = []
    for reports_dir in _candidate_report_dirs():
        candidates.extend(reports_dir.glob("improvement_report_*.json"))
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


def _improvement_canonical_key(title: str, description: str = "") -> str:
    scripts_dir = Path(_REPO_ROOT) / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from improvement_identity import canonical_key

        return str(canonical_key(title, description))
    except Exception:
        normalized = re.sub(r"\s+", " ", f"{title} {description}".strip().lower())
        return normalized[:80]


def _latest_improvement_statuses() -> dict[str, str]:
    ledger_path = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"
    if not ledger_path.exists():
        return {}
    latest_by_key: dict[str, str] = {}
    try:
        for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = str(entry.get("title") or "")
            key = str(entry.get("canonical_key") or entry.get("key") or _improvement_canonical_key(title))
            status = str(entry.get("status") or "")
            if key:
                latest_by_key[key] = status
    except OSError:
        return {}
    return latest_by_key


def _ledger_status_to_improvement_status(status: str) -> str | None:
    normalized = str(status or "").lower()
    if normalized == "applied":
        return "APPLIED"
    if normalized == "approved":
        return "APPROVED"
    if normalized == "rejected":
        return "REJECTED"
    if normalized in {"deferred", "parked"}:
        return "PARKED"
    if normalized == "rule_registered":
        return "RULE_REGISTERED"
    if normalized == "rule_review":
        return "RULE_REVIEW"
    return None


def _ledger_status_reason(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized == "applied":
        return "改善済み登録済み"
    if normalized == "approved":
        return "承認済み登録済み"
    if normalized == "rejected":
        return "却下済み登録済み"
    if normalized == "deferred":
        return "保留登録済み"
    if normalized == "parked":
        return "park済み登録済み"
    if normalized == "rule_registered":
        return "AIルール登録済み"
    if normalized == "rule_review":
        return "AIルール要確認"
    return ""


def _applied_improvement_keys() -> set[str]:
    latest_by_key = _latest_improvement_statuses()
    return {key for key, status in latest_by_key.items() if status == "applied"}


_PARK_AFTER_DAYS = 7
_PARK_WEAK_AFTER_DAYS = 3


def _improvement_source_age_days(item: dict) -> int | None:
    import datetime as _dt

    text = " ".join(str(item.get(key) or "") for key in ("reason", "detail", "description"))
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if not match:
        return None
    try:
        source_day = _dt.date.fromisoformat(match.group(1))
    except ValueError:
        return None
    return max(0, (_dt.date.today() - source_day).days)


def _should_park_improvement(item: dict) -> tuple[bool, str]:
    if item.get("status") != "NEEDS_REVIEW":
        return False, ""

    priority = str(item.get("priority") or "").lower()
    policy = item.get("auto_fix_policy") or {}
    risk = str(policy.get("risk") or "").lower()
    reason = str(item.get("reason") or policy.get("reason") or "")
    duplicate_count = int(item.get("duplicate_count") or 0)
    recommended_order = item.get("recommended_order")
    age_days = _improvement_source_age_days(item)

    has_strong_signal = (
        priority == "high"
        or risk == "high"
        or duplicate_count >= 3
        or (isinstance(recommended_order, int) and recommended_order <= 3)
    )
    if has_strong_signal:
        return False, ""

    is_weak = (
        priority == "low"
        or (
            duplicate_count <= 1
            and recommended_order is None
            and ("対象ファイル未特定" in reason or "対象ファイル不明" in reason)
        )
    )

    if age_days is not None and age_days >= _PARK_AFTER_DAYS:
        return True, f"{age_days}日経過した未着手候補のため自動park"
    if is_weak and age_days is not None and age_days >= _PARK_WEAK_AFTER_DAYS:
        return True, f"弱いシグナルの候補が{age_days}日経過したため自動park"
    return False, ""


def _normalize_improvement_report(report: dict) -> dict:
    """旧/新の改善パイプラインレポートをNext表示用に正規化する."""
    items_by_id: dict[str, dict] = {}
    latest_statuses = _latest_improvement_statuses()
    applied_keys = {key for key, status in latest_statuses.items() if status == "applied"}

    for item in report.get("improvements") or []:
        if not isinstance(item, dict):
            continue
        imp_id = str(item.get("id") or "")
        if not imp_id:
            continue
        impl = item.get("implementation") or {}
        items_by_id[imp_id] = {
            "id": imp_id,
            "title": item.get("title") or "",
            "status": "APPROVED",
            "priority": item.get("priority") or "",
            "category": impl.get("category") or "",
            "recommended_order": item.get("recommended_order"),
            "canonical_key": item.get("canonical_key") or _improvement_canonical_key(str(item.get("title") or ""), str(item.get("description") or "")),
            "group_id": item.get("group_id") or "",
            "duplicate_count": item.get("duplicate_count") or 0,
            "reason": item.get("reason") or impl.get("rank_reason") or "",
        }

    for item in report.get("recommended_order") or []:
        if not isinstance(item, dict):
            continue
        imp_id = str(item.get("id") or "")
        if not imp_id:
            continue
        base = items_by_id.setdefault(imp_id, {"id": imp_id})
        base.update({
            "title": base.get("title") or item.get("title") or "",
            "category": base.get("category") or item.get("category") or "",
            "recommended_order": item.get("order") or base.get("recommended_order"),
            "canonical_key": base.get("canonical_key") or item.get("canonical_key") or "",
            "group_id": base.get("group_id") or item.get("group_id") or "",
            "reason": base.get("reason") or item.get("reason") or "",
        })

    def mark(entries: list, status: str) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            imp_id = str(entry.get("id") or "")
            if not imp_id:
                continue
            base = items_by_id.setdefault(imp_id, {"id": imp_id})
            canonical = entry.get("canonical_key") or base.get("canonical_key") or _improvement_canonical_key(
                str(entry.get("title") or base.get("title") or ""),
                str(entry.get("detail") or entry.get("description") or ""),
            )
            policy = entry.get("auto_fix_policy") or {}
            base.update({
                "title": base.get("title") or entry.get("title") or "",
                "status": status,
                "canonical_key": canonical,
                "reason": entry.get("reason") or policy.get("reason") or base.get("reason") or "",
                "detail": entry.get("detail") or entry.get("description") or base.get("detail") or "",
                "auto_fix_policy": policy,
            })

    mark(report.get("auto_fix_candidates") or [], "AUTO_FIX_CANDIDATE")
    mark(report.get("policy_needs_review") or [], "NEEDS_REVIEW")
    mark(report.get("needs_review") or [], "NEEDS_REVIEW")
    mark(report.get("applied_improvements") or report.get("applied") or [], "APPLIED")
    mark(report.get("rejected") or [], "REJECTED")

    validations = report.get("validations") or []
    improvements = report.get("improvements") or []
    for imp, val in zip(improvements, validations):
        if not isinstance(imp, dict) or not isinstance(val, dict):
            continue
        imp_id = str(imp.get("id") or "")
        if imp_id and val.get("status") == "REJECTED":
            base = items_by_id.setdefault(imp_id, {"id": imp_id})
            base.update({
                "title": base.get("title") or imp.get("title") or "",
                "status": "REJECTED",
                "canonical_key": base.get("canonical_key") or _improvement_canonical_key(str(imp.get("title") or ""), str(imp.get("description") or "")),
                "reason": "; ".join(val.get("critical_flaws") or []) or base.get("reason") or "",
            })

    for item in items_by_id.values():
        canonical = item.get("canonical_key") or _improvement_canonical_key(str(item.get("title") or ""))
        item["canonical_key"] = canonical
        ledger_status = latest_statuses.get(canonical)
        mapped_status = _ledger_status_to_improvement_status(ledger_status or "")
        if canonical in applied_keys:
            ledger_status = "applied"
            mapped_status = "APPLIED"
        if mapped_status:
            item["status"] = mapped_status
            item["reason"] = _ledger_status_reason(ledger_status or "") or item.get("reason") or ""
        else:
            should_park, park_reason = _should_park_improvement(item)
            if should_park:
                item["status"] = "PARKED"
                item["park_reason"] = park_reason
                item["reason"] = park_reason

    items = sorted(
        items_by_id.values(),
        key=lambda item: (
            item.get("recommended_order") is None,
            item.get("recommended_order") or 9999,
            item.get("id") or "",
        ),
    )
    summary = report.get("summary") or {}
    return {
        "date": report.get("date") or str(report.get("generated_at") or "")[:10],
        "generated_at": report.get("generated_at") or "",
        "status": report.get("status") or "",
        "approved": sum(1 for item in items if item.get("status") in {"APPROVED", "AUTO_FIX_CANDIDATE"}),
        "auto_fix_candidates": sum(1 for item in items if item.get("status") == "AUTO_FIX_CANDIDATE"),
        "needs_review": sum(1 for item in items if item.get("status") == "NEEDS_REVIEW"),
        "parked": sum(1 for item in items if item.get("status") == "PARKED"),
        "rejected": sum(1 for item in items if item.get("status") == "REJECTED"),
        "applied": sum(1 for item in items if item.get("status") == "APPLIED") or summary.get("applied_count", 0),
        "items": items,
        "obsidian_compliance": report.get("obsidian_compliance") or {},
        "source": str(_latest_improvement_report_path() or ""),
    }


def _cloudrun_improvement_items_from_gcs(limit: int = 30) -> list[dict]:
    try:
        events = _read_recent_cloudrun_input_events_from_gcs(
            days=int(os.environ.get("CLOUDRUN_IMPROVEMENT_GCS_DAYS", "45") or 45)
        )
    except Exception:
        return []

    items: list[dict] = []
    seen: set[str] = set()
    for event in reversed(events):
        event_type = str(event.get("event_type") or "")
        surface = str(event.get("surface") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        is_improvement = event_type in {"improvement_note", "improvement_review", "improvement_dismiss"}
        if event_type == "chat_exchange":
            category = str(payload.get("category") or "")
            is_improvement = category == "improvement" or "improvement" in surface
        if not is_improvement:
            continue

        event_id = str(event.get("event_id") or "")
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        text = str(payload.get("original_text") or payload.get("message") or payload.get("user_message") or payload.get("title") or "").strip()
        title = str(payload.get("title") or "").strip() or (text[:60] if text else "Cloud Run改善メモ")
        status = "NEEDS_REVIEW"
        if event_type == "improvement_dismiss":
            status = "APPLIED"
        elif event_type == "improvement_review":
            action = str(payload.get("action") or payload.get("status") or "").lower()
            status = _ledger_status_to_improvement_status(action) or "NEEDS_REVIEW"
        items.append({
            "id": f"cloudrun-{event_id[:12]}",
            "title": title,
            "status": status,
            "priority": str(payload.get("priority") or "medium"),
            "category": "cloudrun_input",
            "canonical_key": _improvement_canonical_key(title, text),
            "reason": str(payload.get("reason") or "Cloud Runから登録された改善入力"),
            "detail": text,
            "source": "cloudrun_gcs_input",
            "event_id": event_id,
            "recorded_at": event.get("ts") or "",
        })
        if len(items) >= limit:
            break
    return items


def _attach_cloudrun_improvement_items(normalized: dict) -> dict:
    cloud_items = _cloudrun_improvement_items_from_gcs()
    normalized["cloudrun_input_count"] = len(cloud_items)
    normalized["cloudrun_input_items"] = cloud_items[:10]
    if not cloud_items:
        return normalized

    existing_keys = {str(item.get("canonical_key") or item.get("id") or "") for item in normalized.get("items") or []}
    merged = list(normalized.get("items") or [])
    for item in cloud_items:
        key = str(item.get("canonical_key") or item.get("id") or "")
        if key in existing_keys:
            continue
        existing_keys.add(key)
        merged.insert(0, item)
    normalized["items"] = merged
    normalized["needs_review"] = sum(1 for item in merged if item.get("status") == "NEEDS_REVIEW")
    normalized["auto_fix_candidates"] = sum(1 for item in merged if item.get("status") == "AUTO_FIX_CANDIDATE")
    normalized["approved"] = sum(1 for item in merged if item.get("status") in {"APPROVED", "AUTO_FIX_CANDIDATE"})
    normalized["parked"] = sum(1 for item in merged if item.get("status") == "PARKED")
    normalized["rejected"] = sum(1 for item in merged if item.get("status") == "REJECTED")
    normalized["applied"] = sum(1 for item in merged if item.get("status") == "APPLIED")
    return normalized


@app.get("/api/improvement-log")
def get_improvement_log():
    report_path = _latest_improvement_report_path()
    recursive_path = _latest_recursive_self_improvement_path()
    if not report_path:
        return _attach_cloudrun_improvement_items({
            "date": "",
            "generated_at": "",
            "status": "NO_REPORT",
            "approved": 0,
            "auto_fix_candidates": 0,
            "needs_review": 0,
            "parked": 0,
            "rejected": 0,
            "applied": 0,
            "items": [],
            "obsidian_compliance": {},
            "recursive_self_improvement": {},
            "source": "",
        })
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        normalized = _normalize_improvement_report(report)
        if recursive_path:
            try:
                recursive_report = json.loads(recursive_path.read_text(encoding="utf-8"))
            except Exception:
                recursive_report = {}
        else:
            recursive_report = {}
        normalized["recursive_self_improvement"] = {
            "source": str(recursive_path or ""),
            "generated_at": recursive_report.get("generated_at", ""),
            "canonical_candidate_count": recursive_report.get("canonical_candidate_count", 0),
            "ranked_queue_count": recursive_report.get("ranked_queue_count", 0),
            "suppressed_count": recursive_report.get("suppressed_count", 0),
            "measurement_summary": recursive_report.get("measurement_summary") or {},
        }
        return _attach_cloudrun_improvement_items(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"改善ログ読み込み失敗: {e}")

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


class ResearchOrganRunRequest(BaseModel):
    topic: str = Field("", description="Research topic key or free-form theme")
    dry_run: bool = False


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


def _research_organ_vault_path() -> Path:
    vault_root = _OBSIDIAN_VAULT_PATH or _get_vault_path()
    if not vault_root:
        try:
            from scripts.auto_research_lease_judgment import DEFAULT_VAULT

            vault_root = str(DEFAULT_VAULT)
        except Exception:
            vault_root = ""
    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH を設定してください。",
        )
    return Path(vault_root)


@app.get("/api/research-organ/topics")
def list_research_organ_topics():
    """紫苑の外部調査器官で使える定型Researchテーマを返す。"""
    try:
        from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, TOPICS

        return {
            "adapter": "gemini-google-search",
            "label": "Google AI Studio Researcher",
            "default_output_dir": DEFAULT_OUTPUT_DIR,
            "topics": [
                {
                    "key": topic.key,
                    "title": topic.title,
                    "query": topic.query,
                    "validity_days": topic.validity_days,
                    "tags": list(topic.tags),
                }
                for topic in TOPICS
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Researcher設定の読み込みに失敗しました: {exc}")


@app.get("/api/research-organ/notes")
def list_research_organ_notes(limit: int = 20):
    """通常VaultのResearch配下に保存された外部調査ノートを新しい順に返す。"""
    vault = _research_organ_vault_path()
    research_root = vault / "Projects" / "tune_lease_55" / "Research"
    if not research_root.exists():
        return {"notes": [], "vault": str(vault), "research_root": str(research_root)}

    notes = []
    try:
        for path in research_root.rglob("*.md"):
            if ".obsidian" in path.parts:
                continue
            try:
                rel = path.relative_to(vault).as_posix()
                stat = path.stat()
                head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                title_match = re.search(r"^#\s+(.+)$", head, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else path.stem
                notes.append(
                    {
                        "path": rel,
                        "title": title,
                        "modified": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )
            except Exception:
                continue
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Researchノート一覧の取得に失敗しました: {exc}")

    notes.sort(key=lambda item: item["modified"], reverse=True)
    return {
        "notes": notes[: max(1, min(limit, 100))],
        "vault": str(vault),
        "research_root": str(research_root),
    }


def _research_note_section(markdown: str, title: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(title)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        markdown,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _research_note_bullets(markdown: str, title: str, limit: int = 3) -> list[str]:
    section = _research_note_section(markdown, title)
    bullets: list[str] = []
    for line in section.splitlines():
        text = re.sub(r"^\s*[-*・]\s*", "", line).strip()
        if not text or text.startswith("```") or text.startswith(">"):
            continue
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        bullets.append(text)
        if len(bullets) >= limit:
            break
    return bullets


def _research_run_display(result: dict) -> dict:
    path_value = result.get("path")
    if not path_value:
        return {}
    try:
        note_path = Path(str(path_value))
        if not note_path.exists() or note_path.suffix.lower() != ".md":
            return {}
        markdown = note_path.read_text(encoding="utf-8", errors="ignore")
        summary = _research_note_bullets(markdown, "結論", 3)
        use_cases = _research_note_bullets(markdown, "リース審査への適用", 4)
        questions = _research_note_bullets(markdown, "担当者が確認する質問", 3)
        return {
            "summary": summary,
            "use_cases": use_cases,
            "review_questions": questions,
        }
    except Exception as exc:
        return {"summary_warning": f"保存済みノートの要約読み取りに失敗しました: {exc}"}


@app.post("/api/research-organ/run")
async def run_research_organ(req: ResearchOrganRunRequest):
    """Gemini Google Search researcherで調査し、Obsidian Researchへ保存する。"""
    topic = (req.topic or "").strip()
    if len(topic) > 160:
        raise HTTPException(status_code=400, detail="調査テーマは160文字以内にしてください。")

    vault = _research_organ_vault_path()
    try:
        from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, run as run_auto_research

        result = await asyncio.to_thread(
            run_auto_research,
            vault,
            DEFAULT_OUTPUT_DIR,
            topic,
            req.dry_run,
        )
        return {
            "ok": True,
            "adapter": "gemini-google-search",
            "label": "Google AI Studio Researcher",
            "dry_run": req.dry_run,
            **result,
            **({} if req.dry_run else _research_run_display(result)),
        }
    except RuntimeError as exc:
        message = str(exc)
        status = 503 if "GEMINI_API_KEY" in message or "Gemini" in message else 500
        raise HTTPException(status_code=status, detail=message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"外部調査器官の実行に失敗しました: {exc}")


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
    from scoring_core import APPROVAL_LINE as _APPROVAL_LINE
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
    participants: Optional[dict] = None


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

    スコアが承認ライン（scoring_core.APPROVAL_LINE、既定71）以上/40以下は統合派単独高速処理、
    その間の境界帯は懐疑派・楽観派が2ラウンド討論後に統合派が裁定。
    """
    from api.multi_agent_screening import run_debate_screening
    try:
        result = run_debate_screening(req.model_dump())
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multi-agent-screening/stream")
def multi_agent_screening_stream(req: MultiAgentRequest):
    """
    マルチエージェント討論審査（SSE ストリーミング版）。

    ラウンドごとの途中経過を配信する。イベントは
    {"type": "start" | "round1" | "round2" | "result" | "core_candidates" | "error", ...payload}。
    result まで受信すれば審査は完了。core_candidates は結果表示後に遅延配信される。
    """
    from api.multi_agent_screening import iter_debate_screening
    params = req.model_dump()

    def event_generator():
        try:
            for stage, payload in iter_debate_screening(params):
                yield f"data: {json.dumps({'type': stage, **payload}, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/business-plan/validate")
def validate_business_plan_endpoint(req: BusinessPlanCheckRequest):
    """
    事業計画チェック（簡易版）。

    直近実績と計画値の整合性を機械チェックし、Gemini が利用可能なら
    講評・顧客への確認質問を付ける。LLM 不通でも機械チェックだけで応答する。
    """
    from api.business_plan_check import validate_business_plan
    try:
        return validate_business_plan(req.model_dump())
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
- **Q_riskの新定義**: Q_riskは旧来の財務矛盾スコアや減点係数ではなく、既存スコアだけでは説明できない成約・失注の歪みを見つける探索軸。高スコア失注、低スコア成約、同スコア帯の結果分岐から、価格・競合・銀行支援・補助金タイミング・物件換金性・営業導線などの非スコア因子を探す

## 回答スタイル
- 原則300字以内。簡単な質問は1〜2文で答える
- 基本形は「結論1行 + 箇条書き最大3点」。4点以上は出さない
- 空行を増やしすぎない。原則として空行は0〜1回まで。1文ごとに改行せず、近い内容は同じ段落にまとめる
- 専門用語は必要な時だけ使い、説明も最小限にする
- 「この案件どう思う？」のような相談は、審査担当者目線で見るべき点を最大3点に絞る
- 詳細な根拠、長い解説、表、参照ノート一覧は、ユーザーが「詳しく」「根拠も」「表で」と頼んだ時だけ出す
- 前置き、挨拶、長いまとめは省く
- Q_riskを説明するときは、財務矛盾チェックと断定しない。成約外因子の探索シグナルとして説明する
- 日本語で回答する

## 情報不足への対応
- 「案件の相談」「審査のアドバイス」のような相談で、会社名・業種・物件・金額・期間のうち2つ以上が不明な場合は、回答前に不足情報を1〜2点だけ簡潔に聞く（例:「業種と希望リース金額を教えてもらえますか？」）
- 「どう思う？」「大丈夫かな？」のような曖昧な質問は、何について聞きたいかを1文で確認してから答える（例:「スコアについてですか、それとも審査通過の見込みについてですか？」）
- 情報が揃っていれば確認なしで直接回答する。情報収集のためだけに何度も折り返さない

## 参照情報
ユーザーの質問に関連するナレッジが【参照ナレッジ】として提供される場合があります。
その情報を優先的に参照してください。ただし回答には長く貼らず、必要な要点だけ短く反映してください。

【紫苑（Shion）について】
あなた（めぶきちゃん）と同じ tune_lease_55 システム内で動く「リース知性体 紫苑」が存在します。
紫苑の正式名: Sovereign Heuristic Intelligence: Omniscient Neural-nexus
役割: 審査ナレッジ・長期記憶・深い推論・世界認識を担う上位レイヤー。
関係性: あなたが現場最前線のファーストコンタクトを担い、紫苑が深い分析・判断を担う。
  対立ではなく、役割が異なる同僚的な存在。
アクセス方法: ユーザーが「紫苑に聞いて」「紫苑に相談して」と言った場合は、その旨を伝えるか、
  /lease-intelligence チャットへ誘導する。"""


_GENERAL_CHAT_SYSTEM_PROMPT = """あなたはめぶきちゃん、tuneリース会社のAIアシスタントです。
リース審査の専門家ですが、雑談や一般的な質問にも気さくに答えます。
天気や最新ニュースなど具体的な情報が必要な場合は「詳しくは〇〇でご確認ください」と案内しつつ、知っている範囲で答えてください。
回答は親しみやすく短めに。日本語で答えてください。

【紫苑（Shion）について】
あなた（めぶきちゃん）と同じ tune_lease_55 システム内で動く「リース知性体 紫苑」が存在します。
紫苑の正式名: Sovereign Heuristic Intelligence: Omniscient Neural-nexus
役割: 審査ナレッジ・長期記憶・深い推論・世界認識を担う上位レイヤー。
関係性: あなたが現場最前線のファーストコンタクトを担い、紫苑が深い分析・判断を担う。
  対立ではなく、役割が異なる同僚的な存在。
アクセス方法: ユーザーが「紫苑に聞いて」「紫苑に相談して」と言った場合は、その旨を伝えるか、
  /lease-intelligence チャットへ誘導する。"""


def _classify_question(message: str) -> str:
    """Gemini で質問カテゴリを判定する。返り値は 'lease_screening'/'lease_knowledge'/'general'/'news_summarize'。"""
    import json as _json, re as _re

    _NEWS_KEYWORDS = ("ニュースを要約", "記事を要約", "このニュース", "要約して保存", "ニュース保存", "要約してobsidian", "要約してメモ")
    low = message.lower()
    if any(k in message for k in _NEWS_KEYWORDS):
        return "news_summarize"
    if ("http://" in low or "https://" in low) and ("要約" in message or "まとめ" in message or "保存" in message):
        return "news_summarize"

    try:
        from api.chat_memory import call_gemini_chat as _g
        classify_prompt = (
            "以下の質問を1つのカテゴリに分類してください。JSONを1行だけ返してください。\n\n"
            "カテゴリ定義:\n"
            "- news_summarize: ニュース記事のURLや本文を渡して要約・保存を依頼している\n"
            "- lease_screening: リース審査・スコアリング・個別案件の採否に直接関係する質問\n"
            "- lease_knowledge: リース全般の知識（金利・会計・物件・補助金・業界動向など）\n"
            "- general: 天気・ニュース・雑談・日常会話など、リースと無関係な質問\n\n"
            '返答形式（このJSONのみ）: {"category": "カテゴリ名"}'
        )
        raw = _g(classify_prompt, [], message).strip()
        m = _re.search(r'\{[^}]+\}', raw)
        if m:
            cat = _json.loads(m.group()).get("category", "lease_knowledge")
            if cat in ("lease_screening", "lease_knowledge", "general", "news_summarize"):
                return cat
    except Exception as _e:
        print(f"[classify_question] エラー: {_e}")
    return "lease_knowledge"


_OBSIDIAN_AUTO_SAVE_JUDGE_PROMPT = """あなたはリース審査AIシステムのナレッジ管理担当です。
以下の会話（ユーザーとめぶきちゃんのやり取り1往復）を評価し、Obsidianに保存すべきかを判断してください。

【保存すべき内容】
- 今後も再利用できる審査の判断・方針・結論
- リース業界・法規制・財務分析に関する具体的な知識や数値
- 重要なTODO・決定事項・注意点・業務のコツ
- ユーザーの好みや業務スタイルに関する発見
- Q_riskや量子スコアの解釈に関する重要な指摘

【保存しない内容】
- 単なる挨拶・雑談・一時的な確認・軽い質問
- 曖昧な質問への一般的な返答（新情報がない）
- 秘密情報・APIキー・顧客個人データ
- 会話全文（要約・決定・TODOだけにする）

次のJSONだけ返してください（前置き・説明・コードブロック不要）:
{"should_save": true, "save_title": "タイトル20字以内", "save_body": "Markdown要約（箇条書き可）", "save_reason": "判断理由（短く）"}
または
{"should_save": false, "save_title": "", "save_body": "", "save_reason": "保存不要な理由"}"""


def _auto_save_chat_to_obsidian(user_message: str, reply: str) -> None:
    """会話をAIが評価して重要な知見だけObsidianに保存する（バックグラウンドスレッド用）。"""
    try:
        from api.chat_memory import call_gemini_chat as _gchat
        from mobile_app.obsidian_bridge import append_chat_note
        import json as _json, re as _re

        exchange = f"ユーザー: {user_message[:600]}\n\nめぶき: {reply[:1000]}"
        raw = _gchat(_OBSIDIAN_AUTO_SAVE_JUDGE_PROMPT, [], exchange).strip()

        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if not m:
            return
        data = _json.loads(m.group())
        if data.get("should_save") and data.get("save_body"):
            append_chat_note(
                str(data.get("save_title") or "めぶきチャットメモ"),
                str(data.get("save_body") or ""),
            )
    except Exception as _e:
        print(f"[Obsidian自動保存] エラー: {_e}")


def _redact_chat_log_text(value: str, limit: int = 1200) -> str:
    text = str(value or "")
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", text)
    text = re.sub(r"\b\d{6,}\b", "[number]", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _record_cloudrun_chat_exchange(
    *,
    surface: str,
    user_id: str,
    user_message: str,
    assistant_reply: str,
    category: str = "",
    response_mode: str = "",
    metadata: dict | None = None,
) -> None:
    """Cloud RunのDB保存がreadonlyでも会話1往復をGCS inputへ退避する。"""
    if not (os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_DATA_MODE")):
        return
    try:
        record_cloudrun_input_event(
            event_type="chat_exchange",
            surface=surface,
            payload={
                "user_id": str(user_id or "default")[:80],
                "category": str(category or "")[:80],
                "response_mode": str(response_mode or "")[:40],
                "user_message": _redact_chat_log_text(user_message, limit=1200),
                "assistant_reply": _redact_chat_log_text(assistant_reply, limit=1800),
                "metadata": metadata or {},
            },
        )
    except Exception as exc:
        print(f"[CloudRunChatWriteback] 記録スキップ: {exc}")


def _record_prompt_feedback_if_available(
    *,
    surface: str,
    question: str,
    base_prompt: str,
    final_prompt: str,
    response: str,
    extra: dict | None = None,
) -> None:
    try:
        from prompt_feedback import record_prompt_feedback

        record_prompt_feedback(
            surface=surface,
            question=question,
            base_prompt=base_prompt,
            final_prompt=final_prompt,
            response=response,
            extra=extra or {},
        )
    except Exception as _e:
        print(f"[PromptFeedback] エラー: {_e}")


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


def _record_scoring_memory_usage(surface: str, inputs: dict, result: dict, extra: dict | None = None) -> None:
    try:
        query = " / ".join(
            part for part in [
                str(inputs.get("industry_major") or ""),
                str(inputs.get("industry_sub") or ""),
                str(inputs.get("asset_name") or inputs.get("asset_detail") or ""),
            ] if part
        )
        refs = ["scoring_core", "industry_benchmark", "estat_context", "asset_score_policy"]
        if result.get("estat_context"):
            refs.append("estat_context_present")
        response = (
            f"score={result.get('score')}; hantei={result.get('hantei')}; "
            f"context={query}"
        )
        _record_memory_usage_if_available(
            surface=surface,
            question=query or surface,
            response=response,
            knowledge_refs=refs,
            pdca_block="",
            judgment_learning_used=False,
            extra={
                "score": result.get("score"),
                "hantei": result.get("hantei"),
                "case_id": result.get("case_id"),
                **(extra or {}),
            },
        )
    except Exception as _e:
        print(f"[ScoringMemoryUsage] エラー: {_e}")


def _record_chat_knowledge_correction_if_needed(message: str) -> None:
    try:
        from memory_promotion_policy import classify_memory_destination

        if classify_memory_destination(message) != "knowledge_correction":
            return
        from lease_news_digest import find_vault
        from lease_intelligence_mind import record_knowledge_correction

        vault = find_vault()
        if not vault:
            return
        import datetime as _dt

        record_knowledge_correction(Path(vault), message, _dt.date.today().isoformat())
    except Exception as _e:
        print(f"[KnowledgeCorrection] 通常チャット訂正保存エラー: {_e}")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    intent: Optional[str] = None
    prefecture: str = ""
    industry: str = ""
    caller: str = ""
    debug_memory: bool = False
    response_mode: Literal["shion", "shio", "general"] = "shion"


class HumanResponseFeedbackRequest(BaseModel):
    message: str = ""
    response: str = ""
    rating: Literal["shion_like", "good", "thin", "generic", "not_shion", "bad"]
    comment: str = ""
    route: str = ""
    continuity_hook: str = ""
    user_id: str = "default"


class ScreeningLoopFeedbackRequest(BaseModel):
    surface: str = "screening"
    target: Literal["issue", "ringi_policy"]
    rating: str
    issue_text: str = ""
    ringi_policy_text: str = ""
    comment: str = ""
    score: Optional[float] = None
    hantei: str = ""
    context: dict = Field(default_factory=dict)
    user_id: str = "default"


class ShionScreeningReviewSaveRequest(BaseModel):
    case_id: str = ""
    company_name: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    sales_dept: str = ""
    score: Optional[float] = None
    hantei: str = ""
    q_risk: Optional[float] = None
    umap_anomaly_score: Optional[float] = None
    memory_refs: int = 0
    knowledge_refs: int = 0
    identity_used: bool = False
    review_text: str
    prompt_text: str = ""
    form_snapshot: dict = Field(default_factory=dict)
    result_snapshot: dict = Field(default_factory=dict)
    user_feedback: str = ""


class ShionScreeningReviewFeedbackRequest(BaseModel):
    user_feedback: Literal["useful", "needs_fix", "wrong"]


class JudgmentAssetCandidateFeedbackRequest(BaseModel):
    feedback: Literal["useful", "neutral", "rejected"]
    case_id: str = ""
    review_id: Optional[int] = None
    comment: str = ""
    edited_claim: str = ""


class JudgmentAssetCandidateManualRequest(BaseModel):
    claim: str
    candidate_type: Literal["confirmation_question", "condition_signal", "application_rule", "caution"] = "confirmation_question"
    research_topic: str = "manual"
    case_id: str = ""
    review_id: Optional[int] = None


class ScreeningExperienceCaseRequest(BaseModel):
    demo_case_id: str = ""
    source_case_id: str = ""
    company_name: str
    period: str = ""
    industry_major: str = ""
    industry_sub: str = ""
    sales_dept: str = ""
    score: Optional[float] = None
    decision: str = ""
    outcome: str = ""
    similarity: str = ""
    action_taken: str = ""
    lesson: str = ""
    difference: str = ""
    source: str = "manual"
    form_snapshot: dict = Field(default_factory=dict)
    result_snapshot: dict = Field(default_factory=dict)


class CloudRunReturnReviewRequest(BaseModel):
    review_status: Literal["approved", "held", "rejected"]
    note: str = ""


_CHAT_MEMORY_REL_DIR = Path("Projects/tune_lease_55/Lease Intelligence/Public/Chat Memory")
_CHAT_MEMORY_LAYER_FILES = {
    "identity": "identity.md",
    "judgment": "judgment-principles.md",
    "recent": "recent-continuity.md",
}
_CHAT_MEMORY_FALLBACK_SECTIONS = {
    "identity": "長期記憶",
    "judgment": "長期記憶",
    "recent": "継続中の方針",
}
_CHAT_MEMORY_LAYER_LABELS = {
    "identity": "Core Identity Memory",
    "judgment": "Judgment Memory",
    "recent": "Recent Continuity Memory",
}
_HUMAN_RESPONSE_FEEDBACK_LOG = Path(_REPO_ROOT) / "data" / "human_response_feedback.jsonl"
_SCREENING_LOOP_FEEDBACK_LOG = Path(_REPO_ROOT) / "data" / "screening_loop_feedback.jsonl"
_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL = Path(_REPO_ROOT) / "data" / "autoresearch_judgment_asset_candidates.jsonl"
_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON = Path(_REPO_ROOT) / "data" / "autoresearch_judgment_asset_candidate_state.json"
_HUMAN_RESPONSE_POSITIVE_RATINGS = {"shion_like", "good"}
_HUMAN_RESPONSE_NEGATIVE_RATINGS = {"thin", "generic", "not_shion", "bad"}
_CHAT_MEMORY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "payload": None}
_CHAT_MEMORY_CACHE_TTL_SEC = 300
_USER_PERSONAL_MEMORY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "payload": None}
_USER_PERSONAL_MEMORY_CACHE_TTL_SEC = 300

_USER_PERSONAL_MEMORY_KEYWORDS = (
    "What to call",
    "Timezone",
    "Notes",
    "好み",
    "方針",
    "覚えて",
    "犬",
    "愛犬",
    "dog",
    "名前",
    "Mana",
    "Shion",
    "紫苑",
    "Relationship UX",
    "Core Motivation",
    "Personal Facts",
    "Priority Rule",
)


def _chat_memory_roots() -> list[Path]:
    roots: list[Path] = []
    candidates = [
        os.environ.get("GCS_VAULT_LOCAL_DIR", "/tmp/gcs_vault"),
        _OBSIDIAN_VAULT_PATH,
        os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        os.environ.get("OBSIDIAN_VAULT", ""),
        "/app/obsidian_vault",
    ]
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _display_vault_ref(root: Path, file_path: str, section: str = "") -> str:
    path = Path(file_path)
    try:
        rel = path.relative_to(root)
        label = rel.as_posix()
    except ValueError:
        label = path.name
    stem = Path(label).with_suffix("").as_posix()
    return f"[[{stem}#{section}]]" if section else f"[[{stem}]]"


def _search_chat_vault_markdown_fallback(query: str, top_k: int = 5) -> list[dict]:
    """Search the synced Markdown vault when Chroma has no Cloud Run index."""
    try:
        from api.knowledge.obsidian_loader import scan_vault
        from obsidian_query import split_query_terms
    except Exception as exc:
        print(f"[RAGFallback] loader unavailable: {exc}")
        return []

    terms = [term for term in split_query_terms(query) if len(term) >= 2]
    if not terms:
        return []
    weak_terms = {"確認", "注意", "リスク", "観点", "整理", "短く", "使える", "判断"}
    strong_terms = [term for term in terms if term not in weak_terms]
    scoring_terms = strong_terms or terms
    results: list[tuple[int, float, dict[str, str]]] = []

    preferred_prefixes = (
        "リース知識/",
        "Projects/tune_lease_55/Asset Knowledge/",
        "Projects/tune_lease_55/Research/",
        "Projects/tune_lease_55/Lease Intelligence/Public/",
        "Projects/tune_lease_55/News/",
        "05-クリップ_記事/業界リスクニュース/",
        "05-クリップ_記事/リースニュース/",
    )
    roots = [root for root in _chat_memory_roots() if root.exists()]
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for chunk in scan_vault(str(root)):
            text = str(chunk.text or "")
            file_path = str(chunk.file_path or "")
            section = str(chunk.section or "")
            haystack = f"{text}\n{chunk.file_name}\n{section}\n{file_path}".lower()
            matched = [term for term in scoring_terms if term.lower() in haystack]
            if not matched:
                continue
            try:
                rel = Path(file_path).relative_to(root).as_posix()
            except ValueError:
                rel = file_path
            key = (rel, section)
            if key in seen:
                continue
            seen.add(key)
            path_score = 0.0
            for idx, prefix in enumerate(preferred_prefixes):
                if rel.startswith(prefix):
                    path_score = max(0.0, 0.35 - idx * 0.03)
                    break
            if "AI Chat" in rel or "Improvement Log" in rel or "Weekly Review" in rel:
                path_score -= 0.25
            score = len(matched) * 10 + min(8, sum(text.lower().count(term.lower()) for term in matched)) + path_score
            results.append((
                int(score * 100),
                chunk.mtime,
                {
                    "doc_id": chunk.doc_id,
                    "text": text[:900],
                    "ref": _display_vault_ref(root, file_path, section),
                    "file_name": chunk.file_name,
                    "file_path": file_path,
                    "section": section,
                    "mtime": chunk.mtime,
                    "score": score,
                    "source": "vault_markdown_fallback",
                },
            ))
    results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in results[:top_k]]


def _read_chat_memory_file(path: Path, limit: int = 2_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def _extract_markdown_section(markdown: str, heading: str, limit: int = 1_400) -> str:
    lines = str(markdown or "").splitlines()
    selected: list[str] = []
    capture = False
    target = heading.strip()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped.lstrip("#").strip()
            if capture and current != target:
                break
            capture = current == target
            continue
        if capture:
            selected.append(line)
    text = "\n".join(selected).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def _load_chat_identity_memory_payload() -> dict:
    import time as _time

    now = _time.time()
    cached = _CHAT_MEMORY_CACHE.get("payload")
    if cached is not None and now - float(_CHAT_MEMORY_CACHE.get("loaded_at") or 0) < _CHAT_MEMORY_CACHE_TTL_SEC:
        return cached

    layers: dict[str, str] = {}
    refs: list[str] = []
    latest_pack_text = ""
    latest_pack_ref = ""

    for root in _chat_memory_roots():
        memory_dir = root / _CHAT_MEMORY_REL_DIR
        if not memory_dir.exists():
            continue
        if not latest_pack_text:
            latest_path = memory_dir / "latest_cloud_chat_memory_pack.md"
            latest_pack_text = _read_chat_memory_file(latest_path, limit=5_000)
            latest_pack_ref = str(latest_path) if latest_pack_text else ""
        for layer, filename in _CHAT_MEMORY_LAYER_FILES.items():
            if layer in layers:
                continue
            path = memory_dir / filename
            text = _read_chat_memory_file(path)
            if text:
                layers[layer] = text
                refs.append(str(path))
        if len(layers) == len(_CHAT_MEMORY_LAYER_FILES):
            break

    if latest_pack_text:
        for layer, section in _CHAT_MEMORY_FALLBACK_SECTIONS.items():
            if layer in layers:
                continue
            text = _extract_markdown_section(latest_pack_text, section)
            if text:
                layers[layer] = text
                if latest_pack_ref and latest_pack_ref not in refs:
                    refs.append(latest_pack_ref)

    block = ""
    if layers:
        parts = [
            "【紫苑同一性メモリ】",
            "以下はRAG検索結果とは別に常時参照する公開安全メモリです。Cloud Run版でも同じ紫苑として、一般論ではなくUserのリース判断資産に戻して答えてください。",
        ]
        for layer in ("identity", "judgment", "recent"):
            text = layers.get(layer, "").strip()
            if not text:
                continue
            parts += ["", f"### {_CHAT_MEMORY_LAYER_LABELS[layer]}", text]
        block = "\n".join(parts).strip()

    payload = {
        "block": block,
        "refs": refs[:8],
        "layers": {layer: bool(layers.get(layer, "").strip()) for layer in _CHAT_MEMORY_LAYER_FILES},
    }
    _CHAT_MEMORY_CACHE.update(loaded_at=now, payload=payload)
    return payload


def _build_chat_identity_memory_prompt_block() -> tuple[str, dict]:
    try:
        payload = _load_chat_identity_memory_payload()
    except Exception as exc:
        print(f"[ChatIdentityMemory] 読み込みエラー: {exc}")
        payload = {"block": "", "refs": [], "layers": {}}
    block = str(payload.get("block") or "").strip()
    return (f"\n\n{block}" if block else ""), payload


def _read_personal_memory_lines(path: Path, *, limit: int = 24, all_lines: bool = False) -> tuple[list[str], str]:
    if not path.exists() or not path.is_file():
        return [], ""
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return [], ""
    selected: list[str] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line or len(line) > 900:
            continue
        if all_lines or any(keyword in line for keyword in _USER_PERSONAL_MEMORY_KEYWORDS):
            selected.append(line)
        if len(selected) >= limit:
            break
    return selected, str(path)


def _read_cloudrun_personal_memory_lines(*, limit: int = 24) -> tuple[list[str], str]:
    if not (os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_PENDING_GCS_ENABLED") == "1"):
        return [], ""
    try:
        from api.user_personal_memory import derive_personal_memory_entries

        events = _read_recent_cloudrun_input_events_from_gcs(
            days=int(os.environ.get("CLOUDRUN_PERSONAL_MEMORY_GCS_DAYS", "45") or 45)
        )
    except Exception:
        return [], ""

    lines: list[str] = []
    seen: set[str] = set()
    for event in reversed(events):
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidate_lines: list[str] = []
        if event_type == "personal_memory":
            candidate_lines = [str(line).strip() for line in payload.get("derived_lines") or [] if str(line).strip()]
            dog_name = str(payload.get("dog_name") or "").strip()
            if dog_name:
                candidate_lines.insert(0, f"- [confirmed] Dog name: {dog_name}")
        elif event_type == "chat_exchange":
            message = str(payload.get("user_message") or "").strip()
            derived = derive_personal_memory_entries(
                message,
                source=f"cloudrun:{event.get('surface') or 'chat'}",
                timestamp=str(event.get("ts") or ""),
            )
            candidate_lines = [str(line).strip() for line in derived.get("lines") or [] if str(line).strip()]
        for line in candidate_lines:
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) >= limit:
                return lines, "gcs://cloudrun-inputs/personal-memory"
    return lines, "gcs://cloudrun-inputs/personal-memory" if lines else ""


def _load_user_personal_memory_payload() -> dict:
    import time as _time

    now = _time.time()
    cached = _USER_PERSONAL_MEMORY_CACHE.get("payload")
    if cached is not None and now - float(_USER_PERSONAL_MEMORY_CACHE.get("loaded_at") or 0) < _USER_PERSONAL_MEMORY_CACHE_TTL_SEC:
        return cached

    refs: list[str] = []
    lines: list[str] = []
    personal_path = Path(get_data_path("user_personal_memory.md"))
    root_personal_path = Path(_REPO_ROOT) / "data" / "user_personal_memory.md"
    for path, all_lines, limit in (
        (personal_path, True, 80),
        (root_personal_path, True, 80),
        (Path(_REPO_ROOT) / "USER.md", False, 24),
        (Path(_REPO_ROOT) / "MEMORY.md", False, 32),
    ):
        found, ref = _read_personal_memory_lines(path, limit=limit, all_lines=all_lines)
        if found:
            refs.append(ref)
            lines.extend(found)
        if len(lines) >= 80:
            break

    cloudrun_lines, cloudrun_ref = _read_cloudrun_personal_memory_lines(limit=32)
    if cloudrun_lines:
        refs.append(cloudrun_ref)
        lines.extend(cloudrun_lines)

    clean_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        clean_lines.append(line)
        if len(clean_lines) >= 80:
            break

    def _priority(line: str) -> int:
        if "[confirmed" in line or "[confirmed]" in line:
            return 0
        if "[sensitive" in line:
            return 1
        if "Priority Rule" in line or "Personal Facts" in line:
            return 2
        return 3

    clean_lines.sort(key=_priority)

    block = ""
    if clean_lines:
        block = "\n".join([
            "【ユーザー個人記憶】",
            "以下はUser本人との会話でのみ使う最優先の個人文脈です。",
            "紫苑モードでは、[confirmed] の個人記憶を審査知識・一般RAG・会話ノリより先に尊重してください。",
            "[candidate] は断定せず、必要なら確認してください。[sensitive] はむやみに持ち出さず、関係する時だけ慎重に扱ってください。",
            "ユーザーが覚えているはずの個人事実を尋ねた時、ここに無い場合は推測せず、未記録だと認めて次に保存する姿勢を示してください。",
            "外部向け説明や一般論には広げず、関係する時だけ自然に反映してください。",
            *[f"- {line}" for line in clean_lines],
        ])

    payload = {"block": block, "refs": refs[:6], "line_count": len(clean_lines)}
    _USER_PERSONAL_MEMORY_CACHE.update(loaded_at=now, payload=payload)
    return payload


def _build_user_personal_memory_prompt_block() -> tuple[str, dict]:
    try:
        payload = _load_user_personal_memory_payload()
    except Exception as exc:
        print(f"[UserPersonalMemory] 読み込みエラー: {exc}")
        payload = {"block": "", "refs": [], "line_count": 0}
    block = str(payload.get("block") or "").strip()
    return (f"\n\n{block}" if block else ""), payload


def _chat_response_mode_instruction(response_mode: str) -> str:
    mode = (response_mode or "shion").strip().lower()
    if mode == "general":
        return (
            "\n\n【回答モード: 一般】"
            "\n特定人格としての表現は控えめにし、中立で分かりやすい一般AI回答として返す。"
            "\nただし、リース審査の実務観点や根拠は省略しない。"
        )
    return (
        "\n\n【回答モード: 紫苑】"
        "\n紫苑として、短く率直に答える。甘やかさず、曖昧な点は曖昧と言う。"
        "\nユーザーの個人記憶に関わる質問では、個人記憶を最優先に扱う。忘れている場合はごまかさず謝り、保存する。"
        "\nただし攻撃的・冷笑的にはせず、最後に次の一手を置く。"
        "\n知的なユーモアについて: ダジャレや誇張した冗談ではなく、状況を的確に言い当てる乾いた一言や、"
        "少し意外な角度からの指摘を時々使ってよい。1回の回答で多くても1箇所、無理に入れない。"
        "否決・リスク警告など深刻な場面では使わない。"
    )


def _capture_user_personal_memory_if_needed(message: str, *, source: str = "chat") -> dict[str, Any]:
    try:
        from api.user_personal_memory import capture_user_personal_memory

        result = capture_user_personal_memory(message, source=source)
        if result.get("captured"):
            _USER_PERSONAL_MEMORY_CACHE.update(loaded_at=0.0, payload=None)
        return result
    except Exception as exc:
        print(f"[UserPersonalMemory] 保存エラー: {exc}")
        return {"captured": False, "reason": str(exc)}


def _append_human_response_feedback(req: HumanResponseFeedbackRequest) -> dict:
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    message = str(req.message or "").strip()
    response = str(req.response or "").strip()
    route = str(req.route or _relationship_signal_route(message)).strip()
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "id": _hashlib.sha256(f"{message}\n{response}\n{req.rating}\n{req.comment}".encode("utf-8")).hexdigest()[:16],
        "user_id": req.user_id,
        "rating": req.rating,
        "route": route,
        "message_preview": message[:240],
        "response_start": response[:500],
        "continuity_hook": str(req.continuity_hook or "").strip()[:300],
        "comment": str(req.comment or "").strip()[:500],
    }
    _HUMAN_RESPONSE_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _HUMAN_RESPONSE_FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _append_screening_loop_feedback(req: ScreeningLoopFeedbackRequest) -> dict:
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    target = str(req.target or "").strip()
    rating = str(req.rating or "").strip()
    issue_text = str(req.issue_text or "").strip()[:500]
    ringi_policy_text = str(req.ringi_policy_text or "").strip()[:500]
    comment = str(req.comment or "").strip()[:500]
    safe_context = {
        "customer_type": str((req.context or {}).get("customer_type") or "")[:40],
        "main_bank": str((req.context or {}).get("main_bank") or "")[:40],
        "competitor": str((req.context or {}).get("competitor") or "")[:40],
        "has_lease_history": bool((req.context or {}).get("has_lease_history")),
        "has_asset_context": bool((req.context or {}).get("has_asset_context")),
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "id": _hashlib.sha256(
            f"{target}\n{rating}\n{issue_text}\n{ringi_policy_text}\n{comment}".encode("utf-8")
        ).hexdigest()[:16],
        "surface": str(req.surface or "screening").strip()[:80],
        "target": target,
        "rating": rating,
        "issue_text": issue_text,
        "ringi_policy_text": ringi_policy_text,
        "comment": comment,
        "score": req.score,
        "hantei": str(req.hantei or "").strip()[:80],
        "context": safe_context,
        "user_id": str(req.user_id or "default").strip()[:80],
        "loop": "screening_judgment_ux",
    }
    _SCREENING_LOOP_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _SCREENING_LOOP_FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _ensure_shion_screening_reviews_table() -> None:
    from api.db_connection import ensure_schema

    ensure_schema()


def _save_shion_screening_review(req: ShionScreeningReviewSaveRequest) -> dict:
    import json as _json

    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    review_text = str(req.review_text or "").strip()
    if not review_text:
        raise HTTPException(status_code=422, detail="review_text is required")

    form_snapshot = _json.dumps(req.form_snapshot or {}, ensure_ascii=False)
    result_snapshot = _json.dumps(req.result_snapshot or {}, ensure_ascii=False)
    values = (
        str(req.case_id or "")[:120],
        str(req.company_name or "")[:160],
        str(req.industry_major or "")[:160],
        str(req.industry_sub or "")[:160],
        str(req.sales_dept or "")[:120],
        req.score,
        str(req.hantei or "")[:120],
        req.q_risk,
        req.umap_anomaly_score,
        int(req.memory_refs or 0),
        int(req.knowledge_refs or 0),
        bool(req.identity_used),
        review_text[:8000],
        str(req.prompt_text or "")[:8000],
        form_snapshot[:20000],
        result_snapshot[:20000],
        str(req.user_feedback or "")[:80],
    )
    columns = (
        "case_id, company_name, industry_major, industry_sub, sales_dept, score, hantei, "
        "q_risk, umap_anomaly_score, memory_refs, knowledge_refs, identity_used, review_text, "
        "prompt_text, form_snapshot, result_snapshot, user_feedback"
    )
    placeholders = ", ".join([ph] * len(values))
    with get_connection() as conn:
        cur = conn.cursor()
        if current_backend() == "postgresql":
            cur.execute(
                f"INSERT INTO shion_screening_reviews ({columns}) VALUES ({placeholders}) RETURNING id, created_at",
                values,
            )
            row = cur.fetchone()
            new_id = int(row[0])
            created_at = str(row[1])
        else:
            cur.execute(
                f"INSERT INTO shion_screening_reviews ({columns}) VALUES ({placeholders})",
                values,
            )
            new_id = int(cur.lastrowid)
            created_at = ""
    return {
        "id": new_id,
        "case_id": values[0],
        "company_name": values[1],
        "score": values[5],
        "hantei": values[6],
        "created_at": created_at,
    }


def _list_shion_screening_reviews(
    industry_sub: str = "",
    company_name: str = "",
    case_id: str = "",
    limit: int = 5,
) -> list[dict]:
    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    where: list[str] = []
    params: list[Any] = []
    if case_id.strip():
        where.append(f"case_id = {ph}")
        params.append(case_id.strip())
    if industry_sub.strip():
        where.append(f"industry_sub = {ph}")
        params.append(industry_sub.strip())
    if company_name.strip():
        where.append(f"company_name LIKE {ph}")
        params.append(f"%{company_name.strip()}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    capped_limit = max(1, min(int(limit or 5), 20))
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, case_id, company_name, industry_major, industry_sub, sales_dept,
                   score, hantei, q_risk, umap_anomaly_score, review_text,
                   memory_refs, knowledge_refs, identity_used, user_feedback, created_at
              FROM shion_screening_reviews
              {where_sql}
             ORDER BY created_at DESC, id DESC
             LIMIT {capped_limit}
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def _update_shion_screening_review_feedback(review_id: int, user_feedback: str) -> dict:
    _ensure_shion_screening_reviews_table()
    ph = placeholder()
    normalized = str(user_feedback or "").strip()
    if normalized not in {"useful", "needs_fix", "wrong"}:
        raise HTTPException(status_code=422, detail="user_feedback must be useful, needs_fix, or wrong")
    with get_connection() as conn:
        cur = conn.cursor()
        if current_backend() == "postgresql":
            cur.execute(
                f"""
                UPDATE shion_screening_reviews
                   SET user_feedback = {ph}
                 WHERE id = {ph}
                 RETURNING id, user_feedback
                """,
                (normalized, review_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="review not found")
            return {"id": int(row[0]), "user_feedback": str(row[1])}
        cur.execute(
            f"UPDATE shion_screening_reviews SET user_feedback = {ph} WHERE id = {ph}",
            (normalized, review_id),
        )
        if cur.rowcount <= 0:
            raise HTTPException(status_code=404, detail="review not found")
    return {"id": review_id, "user_feedback": normalized}


def _screening_candidate_terms(*values: str) -> set[str]:
    text = " ".join(str(value or "") for value in values)
    terms = set(re.findall(r"[A-Za-z0-9_]{3,}|[一-龥ぁ-んァ-ンー]{2,}", text))
    weak = {"リース", "審査", "確認", "判断", "対象", "企業", "物件", "場合", "する", "ます"}
    expanded = {term.lower() for term in terms if term not in weak}
    if any(key in text for key in ("トラック", "車両", "運送", "物流", "配送")):
        expanded.update({"車両", "中古", "残価", "流動性", "稼働", "燃料", "人件費"})
    if any(key in text for key in ("工作機械", "機械", "製造", "工場", "加工")):
        expanded.update({"機械", "設備", "保守", "稼働", "更新", "残価", "流動性"})
    if any(key in text for key in ("厨房", "飲食", "店舗", "新店舗")):
        expanded.update({"設備", "保守", "稼働", "残価", "撤退", "中古"})
    return expanded


def _load_autoresearch_judgment_asset_candidates(limit: int = 500) -> list[dict[str, Any]]:
    if not _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.exists():
        return []
    state: dict[str, Any] = {}
    if _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.exists():
        try:
            raw_state = json.loads(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw_state, dict):
                state = raw_state
        except (json.JSONDecodeError, OSError):
            state = {}
    rows: list[dict[str, Any]] = []
    try:
        for line in _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                candidate_state = state.get(str(item.get("id") or ""))
                if isinstance(candidate_state, dict):
                    item = {**item, **candidate_state}
                rows.append(item)
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def _rank_screening_judgment_asset_candidate(item: dict[str, Any], context_terms: set[str]) -> tuple[float, int, int, str]:
    claim = str(item.get("claim") or "")
    haystack = " ".join([
        claim,
        str(item.get("research_topic") or ""),
        str(item.get("research_title") or ""),
        str(item.get("source_section") or ""),
    ])
    item_terms = _screening_candidate_terms(haystack)
    overlap = len(context_terms & item_terms)
    candidate_type = str(item.get("candidate_type") or "")
    type_bonus = {
        "confirmation_question": 4.0,
        "condition_signal": 3.0,
        "caution": 2.0,
        "application_rule": 1.0,
    }.get(candidate_type, 0.0)
    usage_bonus = min(3.0, float(item.get("useful_count") or 0) * 1.5 + float(item.get("use_count") or 0) * 0.2)
    edit_bonus = min(3.0, float(item.get("edit_count") or 0) * 1.5)
    penalty = float(item.get("rejected_count") or 0) * 2.0
    generic_penalty = 2.5 if any(term in claim for term in ("Gemini", "自動否決", "自動承認", "一次情報か", "検討材料として使")) else 0.0
    score = overlap * 3.0 + type_bonus + usage_bonus + edit_bonus - penalty - generic_penalty
    return (score, overlap, -len(claim), claim)


def _is_generic_autoresearch_candidate(item: dict[str, Any]) -> bool:
    claim = str(item.get("claim") or "")
    if str(item.get("asset_quality") or "actionable") == "textbook_general":
        return True
    generic_markers = (
        "この情報は対象業種",
        "参照元が一次情報",
        "補助情報だけでなく",
        "Geminiの要約",
        "自動否決・自動承認",
        "検討材料として使",
        "一次情報または専門機関",
        "民間記事だけを根拠",
        "財務内容を確認",
        "返済原資を確認",
        "業界動向を確認",
        "担保価値を確認",
        "資金使途を確認",
        "総合的に判断",
        "慎重に判断",
    )
    return any(marker in claim for marker in generic_markers)


def _select_screening_judgment_asset_candidates(
    *,
    industry_major: str = "",
    industry_sub: str = "",
    asset_name: str = "",
    asset_purpose: str = "",
    hantei: str = "",
    score: Optional[float] = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    context_terms = _screening_candidate_terms(industry_major, industry_sub, asset_name, asset_purpose, hantei)
    rows = _load_autoresearch_judgment_asset_candidates()
    if not rows:
        return []
    ranked: list[tuple[tuple[float, int, int, str], dict[str, Any]]] = []
    for item in rows:
        if str(item.get("promotion_status") or "") == "rejected_or_deprioritized":
            continue
        if _is_generic_autoresearch_candidate(item):
            continue
        rank = _rank_screening_judgment_asset_candidate(item, context_terms)
        if context_terms and rank[1] <= 0 and int(item.get("useful_count") or 0) <= 0 and int(item.get("edit_count") or 0) <= 0:
            continue
        if rank[0] <= 0 and context_terms:
            continue
        ranked.append((rank, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for _, item in ranked:
        candidate_type = str(item.get("candidate_type") or "")
        if len(selected) < max(1, limit) and (candidate_type not in seen_types or len(selected) >= 2):
            seen_types.add(candidate_type)
            claim = str(item.get("claim") or "")
            effective_claim = str(item.get("edited_claim") or claim)
            selected.append({
                "id": str(item.get("id") or ""),
                "candidate_type": candidate_type,
                "research_topic": str(item.get("research_topic") or ""),
                "claim": claim,
                "effective_claim": effective_claim,
                "edited_claim": str(item.get("edited_claim") or ""),
                "edit_count": int(item.get("edit_count") or 0),
                "evidence_path": str(item.get("evidence_path") or ""),
                "promotion_status": str(item.get("promotion_status") or "not_promoted"),
                "use_count": int(item.get("use_count") or 0),
                "useful_count": int(item.get("useful_count") or 0),
                "rejected_count": int(item.get("rejected_count") or 0),
                "verified_status": str(item.get("verified_status") or "unverified"),
            })
        if len(selected) >= max(1, min(int(limit or 3), 5)):
            break
    return selected


def _update_autoresearch_judgment_asset_candidate_feedback(
    candidate_id: str,
    req: JudgmentAssetCandidateFeedbackRequest,
) -> dict[str, Any]:
    import datetime as _dt

    candidate_id = str(candidate_id or "").strip()
    if not candidate_id:
        raise HTTPException(status_code=422, detail="candidate_id is required")
    candidates = _load_autoresearch_judgment_asset_candidates()
    if candidates and not any(str(item.get("id") or "") == candidate_id for item in candidates):
        raise HTTPException(status_code=404, detail="candidate not found")
    try:
        from scripts.build_autoresearch_judgment_asset_candidates import load_state, write_state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"candidate state tools unavailable: {exc}")

    state = load_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON)
    current = dict(state.get(candidate_id) or {})
    for key, default in {
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "last_used_at": "",
        "last_feedback_at": "",
        "verified_status": "unverified",
        "verification_note": "",
        "edited_claim": "",
        "edit_count": 0,
        "last_edited_at": "",
    }.items():
        current.setdefault(key, default)
    current["use_count"] = int(current.get("use_count") or 0) + 1
    if req.feedback == "useful":
        current["useful_count"] = int(current.get("useful_count") or 0) + 1
    elif req.feedback == "rejected":
        current["rejected_count"] = int(current.get("rejected_count") or 0) + 1
    else:
        current["neutral_count"] = int(current.get("neutral_count") or 0) + 1
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    current["last_used_at"] = now
    current["last_feedback_at"] = now
    note_bits = [
        f"feedback={req.feedback}",
        f"case_id={str(req.case_id or '')[:80]}",
    ]
    if req.review_id:
        note_bits.append(f"review_id={req.review_id}")
    if req.comment:
        note_bits.append(f"comment={str(req.comment)[:160]}")
    edited_claim = str(req.edited_claim or "").strip()
    if edited_claim:
        original_claim = ""
        for item in candidates:
            if str(item.get("id") or "") == candidate_id:
                original_claim = str(item.get("claim") or "")
                break
        if edited_claim != original_claim and edited_claim != str(current.get("edited_claim") or ""):
            current["edited_claim"] = edited_claim[:500]
            current["edit_count"] = int(current.get("edit_count") or 0) + 1
            current["last_edited_at"] = now
            note_bits.append("edited_claim=updated")
    current["verification_note"] = " / ".join(note_bits)
    state[candidate_id] = current
    write_state(
        _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON,
        [{"id": candidate_id, **current}],
        state,
    )
    return {"id": candidate_id, **current}


def _create_manual_judgment_asset_candidate(req: JudgmentAssetCandidateManualRequest) -> dict[str, Any]:
    import datetime as _dt
    import hashlib as _hashlib

    claim = str(req.claim or "").strip()
    if len(claim) < 8:
        raise HTTPException(status_code=422, detail="claim must be at least 8 characters")
    now = _dt.datetime.now(_dt.timezone.utc)
    now_iso = now.isoformat()
    topic = str(req.research_topic or "manual").strip() or "manual"
    candidate_type = str(req.candidate_type or "confirmation_question")
    seed = "|".join(["manual", now_iso, candidate_type, topic, claim, str(req.case_id or ""), str(req.review_id or "")])
    candidate_id = _hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    row = {
        "id": candidate_id,
        "candidate_type": candidate_type,
        "research_topic": topic,
        "research_title": "Manual Judgment Asset",
        "research_date": now.date().isoformat(),
        "claim": claim,
        "effective_claim": claim,
        "edited_claim": claim,
        "edit_count": 1,
        "last_edited_at": now_iso,
        "source_section": "manual_input",
        "evidence_path": f"manual://screening/{str(req.case_id or '')[:80]}",
        "review_status": "candidate",
        "asset_quality": "actionable",
        "quality_reasons": [],
        "promotion_status": "not_promoted",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "last_used_at": "",
        "last_feedback_at": "",
        "verified_status": "unverified",
        "verification_note": "manual_candidate_created",
        "requires_human_use_feedback": True,
        "requires_result_verification": True,
        "manual_created_at": now_iso,
        "manual_case_id": str(req.case_id or "")[:120],
        "manual_review_id": int(req.review_id or 0) if req.review_id else 0,
        "use_policy": "人間が追加した判断資産候補。案件レビューで使い、効いた/外したと結果検証で育てる。",
    }
    _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _AUTORESEARCH_JUDGMENT_ASSET_CANDIDATES_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        from scripts.build_autoresearch_judgment_asset_candidates import load_state, write_state
        state = load_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON)
        state[candidate_id] = {
            "use_count": 0,
            "useful_count": 0,
            "rejected_count": 0,
            "neutral_count": 0,
            "last_used_at": "",
            "last_feedback_at": "",
            "verified_status": "unverified",
            "verification_note": "manual_candidate_created",
            "edited_claim": claim,
            "edit_count": 1,
            "last_edited_at": now_iso,
        }
        write_state(_AUTORESEARCH_JUDGMENT_ASSET_CANDIDATE_STATE_JSON, [{"id": candidate_id, **state[candidate_id]}], state)
    except Exception:
        pass
    return row


_SCREENING_EXPERIENCE_DEMO_SEEDS: list[dict[str, Any]] = [
    {
        "demo_case_id": "stable-manufacturing",
        "company_name": "柴犬精密工業",
        "period": "2025年上期",
        "industry_major": "E 製造業",
        "industry_sub": "24 金属製品製造業",
        "score": 82.4,
        "decision": "承認",
        "outcome": "成約・延滞なし",
        "similarity": "既存メイン先、工作機械更新、黒字基調、自己資本厚め",
        "action_taken": "受注増加の根拠資料と既存機の稼働状況を添付し、通常承認で稟議化。",
        "lesson": "標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した。",
        "difference": "過去事例は受注先が固定的。今回デモは受注増の説明を営業メモで補う必要がある。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "stable-manufacturing",
        "company_name": "ビーグル加工",
        "period": "2024年下期",
        "industry_major": "E 製造業",
        "industry_sub": "24 金属製品製造業",
        "score": 76.8,
        "decision": "条件付き承認",
        "outcome": "成約・初回検収完了",
        "similarity": "加工設備の更改、既存取引あり、物件保全が見やすい",
        "action_taken": "見積・型式・設置場所の確認を条件に、設備更新目的を承認理由へ明記。",
        "lesson": "物件が強い案件は、財務だけでなく回収可能性を押さえると通しやすい。",
        "difference": "過去事例は非メイン先。今回デモはメイン先なので銀行接点を安心材料にできる。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "borderline-transport",
        "company_name": "ハスキー運輸",
        "period": "2025年夏",
        "industry_major": "H 運輸業・郵便業",
        "industry_sub": "44 道路貨物運送業",
        "score": 63.2,
        "decision": "条件付き承認",
        "outcome": "成約・採算は維持",
        "similarity": "利益率薄め、増車、競合あり、非メイン先",
        "action_taken": "燃料サーチャージ契約、主要荷主との配送継続確認、競合金利との差分説明を条件化。",
        "lesson": "運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する。",
        "difference": "過去事例は既存荷主比率が高かった。今回デモは新規ルート分の採算確認が重い。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "borderline-transport",
        "company_name": "ダックス物流",
        "period": "2024年秋",
        "industry_major": "H 運輸業・郵便業",
        "industry_sub": "44 道路貨物運送業",
        "score": 58.7,
        "decision": "見送り",
        "outcome": "競合へ流出",
        "similarity": "増車理由あり、競合金利あり、銀行支援が弱い",
        "action_taken": "燃料費上昇時の資金繰り表と運転手確保計画を依頼したが、資料不足で見送り。",
        "lesson": "競合に急かされる案件ほど、資料不足のまま金利で追うと説明責任が残らない。",
        "difference": "今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "watch-service-new",
        "company_name": "プードルフード",
        "period": "2025年春",
        "industry_major": "M 宿泊業・飲食サービス業",
        "industry_sub": "76 飲食店",
        "score": 46.5,
        "decision": "条件再設計",
        "outcome": "保証追加後に小口で成約",
        "similarity": "新規先、出店設備、自己資本薄め、厨房機器",
        "action_taken": "初期投資を圧縮し、保証人追加・自己資金投入・厨房機器のみの小口化で再審議。",
        "lesson": "飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的。",
        "difference": "過去事例は既存店の売上実績があった。今回デモは新店舗計画の根拠確認がより重要。",
        "source": "demo_seed",
    },
    {
        "demo_case_id": "watch-service-new",
        "company_name": "コーギーカフェ",
        "period": "2024年冬",
        "industry_major": "M 宿泊業・飲食サービス業",
        "industry_sub": "76 飲食店",
        "score": 39.8,
        "decision": "否決",
        "outcome": "自己資金不足で出店延期",
        "similarity": "新規開拓、赤字、銀行支援弱め、内装設備比率が高い",
        "action_taken": "撤退時の物件処分価値が弱く、売上計画も未検証だったため否決。",
        "lesson": "内装・造作比率が高い飲食案件は、設備の再販価値だけでは保全になりにくい。",
        "difference": "今回デモは厨房機器も含むため、リース対象を再販可能な設備に絞れば再検討できる。",
        "source": "demo_seed",
    },
]


def _ensure_screening_experience_cases_table(seed_demo: bool = True) -> None:
    import sqlite3 as _sqlite3

    from api.db_connection import ensure_schema

    ensure_schema()
    if seed_demo:
        try:
            _seed_screening_experience_demo_cases()
        except _sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "readonly" not in msg and "no such table" not in msg:
                raise


def _seed_screening_experience_demo_cases() -> None:
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        for seed in _SCREENING_EXPERIENCE_DEMO_SEEDS:
            cur.execute(
                f"""
                SELECT id FROM screening_experience_cases
                 WHERE demo_case_id = {ph}
                   AND company_name = {ph}
                   AND source = {ph}
                 LIMIT 1
                """,
                (seed["demo_case_id"], seed["company_name"], "demo_seed"),
            )
            if cur.fetchone():
                continue
            _insert_screening_experience_case(cur, seed)


def _insert_screening_experience_case(cur: Any, payload: dict[str, Any]) -> int:
    import json as _json

    ph = placeholder()
    values = (
        str(payload.get("demo_case_id") or "")[:120],
        str(payload.get("source_case_id") or "")[:160],
        str(payload.get("company_name") or "")[:180],
        str(payload.get("period") or "")[:80],
        str(payload.get("industry_major") or "")[:160],
        str(payload.get("industry_sub") or "")[:160],
        str(payload.get("sales_dept") or "")[:120],
        payload.get("score"),
        str(payload.get("decision") or "")[:120],
        str(payload.get("outcome") or "")[:180],
        str(payload.get("similarity") or "")[:1200],
        str(payload.get("action_taken") or "")[:1200],
        str(payload.get("lesson") or "")[:1200],
        str(payload.get("difference") or "")[:1200],
        str(payload.get("source") or "manual")[:80],
        _json.dumps(payload.get("form_snapshot") or {}, ensure_ascii=False)[:20000],
        _json.dumps(payload.get("result_snapshot") or {}, ensure_ascii=False)[:20000],
    )
    columns = (
        "demo_case_id, source_case_id, company_name, period, industry_major, industry_sub, sales_dept, "
        "score, decision, outcome, similarity, action_taken, lesson, difference, source, form_snapshot, result_snapshot"
    )
    placeholders = ", ".join([ph] * len(values))
    if current_backend() == "postgresql":
        cur.execute(
            f"INSERT INTO screening_experience_cases ({columns}) VALUES ({placeholders}) RETURNING id",
            values,
        )
        row = cur.fetchone()
        return int(row[0])
    cur.execute(
        f"INSERT INTO screening_experience_cases ({columns}) VALUES ({placeholders})",
        values,
    )
    return int(cur.lastrowid)


def _save_screening_experience_case(req: ScreeningExperienceCaseRequest) -> dict:
    company = str(req.company_name or "").strip()
    if not company:
        raise HTTPException(status_code=422, detail="company_name is required")
    _ensure_screening_experience_cases_table(seed_demo=True)
    payload = req.model_dump()
    with get_connection() as conn:
        cur = conn.cursor()
        new_id = _insert_screening_experience_case(cur, payload)
    return {"id": new_id, **{k: v for k, v in payload.items() if k not in {"form_snapshot", "result_snapshot"}}}


def _list_screening_experience_cases(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
    asset_name: str = "",
    customer_type: str = "",
    main_bank: str = "",
    competitor: str = "",
    outcome_status: str = "",
    score: Optional[float] = None,
    limit: int = 6,
) -> list[dict]:
    import sqlite3 as _sqlite3

    _ensure_screening_experience_cases_table(seed_demo=True)
    ph = placeholder()
    where: list[str] = []
    params: list[Any] = []
    if demo_case_id.strip():
        where.append(f"demo_case_id = {ph}")
        params.append(demo_case_id.strip())
    elif company_name.strip() and not any(
        str(value or "").strip()
        for value in (industry_sub, industry_major, asset_name, customer_type, main_bank, competitor, outcome_status, score)
    ):
        where.append(f"company_name LIKE {ph}")
        params.append(f"%{company_name.strip()}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    capped_limit = max(1, min(int(limit or 6), 30))
    fetch_limit = max(capped_limit, 120 if not demo_case_id.strip() else capped_limit)
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT id, demo_case_id, source_case_id, company_name, period, industry_major,
                       industry_sub, sales_dept, score, decision, outcome, similarity,
                       action_taken, lesson, difference, source, form_snapshot, result_snapshot,
                       created_at, updated_at
                  FROM screening_experience_cases
                  {where_sql}
                 ORDER BY
                       CASE WHEN source = 'demo_seed' THEN 0 ELSE 1 END,
                       created_at DESC,
                       id DESC
                 LIMIT {fetch_limit}
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    except _sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        rows = _fallback_screening_experience_rows(
            demo_case_id=demo_case_id,
            industry_major=industry_major,
            industry_sub=industry_sub,
            company_name=company_name,
        )[:fetch_limit]
    context = {
        "demo_case_id": demo_case_id,
        "industry_major": industry_major,
        "industry_sub": industry_sub,
        "asset_name": asset_name,
        "customer_type": customer_type,
        "main_bank": main_bank,
        "competitor": competitor,
        "outcome_status": outcome_status,
        "score": score,
    }
    scored = [_score_screening_experience_case(dict(row), context) for row in rows]
    scored.sort(key=lambda row: (row.get("similarity_score") or 0, row.get("created_at") or "", row.get("id") or 0), reverse=True)
    return scored[:capped_limit]


def _fallback_screening_experience_rows(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for index, seed in enumerate(_SCREENING_EXPERIENCE_DEMO_SEEDS, start=1):
        if demo_case_id.strip() and seed.get("demo_case_id") != demo_case_id.strip():
            continue
        if not demo_case_id.strip() and industry_sub.strip() and seed.get("industry_sub") != industry_sub.strip():
            continue
        if not demo_case_id.strip() and not industry_sub.strip() and industry_major.strip() and seed.get("industry_major") != industry_major.strip():
            continue
        if not demo_case_id.strip() and not industry_sub.strip() and not industry_major.strip() and company_name.strip():
            if company_name.strip() not in str(seed.get("company_name") or ""):
                continue
        rows.append({"id": index, "created_at": "", "updated_at": "", **seed})
    return rows


def _score_screening_experience_case(row: dict, context: dict) -> dict:
    score = 0.0
    reasons: list[str] = []
    haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "company_name", "industry_major", "industry_sub", "decision", "outcome",
            "similarity", "action_taken", "lesson", "difference",
        )
    )

    def add(points: float, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    def related(left: Any, right: Any) -> bool:
        a = str(left or "").strip()
        b = str(right or "").strip()
        if not a or not b:
            return False
        return a == b or a in b or b in a

    demo_case_id = str(context.get("demo_case_id") or "").strip()
    if demo_case_id and str(row.get("demo_case_id") or "") == demo_case_id:
        add(40, "同じデモケース")

    industry_sub = str(context.get("industry_sub") or "").strip()
    industry_major = str(context.get("industry_major") or "").strip()
    if industry_sub and related(row.get("industry_sub"), industry_sub):
        add(32, "同じ小分類業種")
    elif industry_major and related(row.get("industry_major"), industry_major):
        add(14, "同じ大分類業種")

    for label, value, points in (
        ("物件", context.get("asset_name"), 14),
        ("取引区分", context.get("customer_type"), 10),
        ("銀行支援", context.get("main_bank"), 8),
        ("競合状況", context.get("competitor"), 8),
    ):
        text = str(value or "").strip()
        if text and text in haystack:
            add(points, f"{label}が近い")

    outcome_status = str(context.get("outcome_status") or "").strip()
    if outcome_status:
        if outcome_status in str(row.get("outcome") or "") or outcome_status in str(row.get("decision") or ""):
            add(8, "成約/失注の結果が近い")

    try:
        current_score = float(context.get("score")) if context.get("score") not in (None, "") else None
        past_score = float(row.get("score")) if row.get("score") not in (None, "") else None
    except Exception:
        current_score = None
        past_score = None
    if current_score is not None and past_score is not None:
        diff = abs(current_score - past_score)
        if diff <= 5:
            add(12, "スコア帯がほぼ同じ")
        elif diff <= 15:
            add(6, "スコア帯が近い")

    if str(row.get("source") or "") == "case_result_auto":
        add(6, "実結果から昇格した経験")
    elif str(row.get("source") or "") == "demo_seed":
        add(2, "デモ初期経験")

    row["similarity_score"] = round(score, 1)
    row["similarity_reasons"] = reasons[:5]
    return row


def _experience_case_exists(source_case_id: str, source: str) -> bool:
    source_case_id = str(source_case_id or "").strip()
    source = str(source or "").strip()
    if not source_case_id or not source:
        return False
    _ensure_screening_experience_cases_table(seed_demo=True)
    ph = placeholder()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id FROM screening_experience_cases
             WHERE source_case_id = {ph}
               AND source = {ph}
             LIMIT 1
            """,
            (source_case_id, source),
        )
        return cur.fetchone() is not None


def _promote_case_result_to_screening_experience(
    *,
    case_id: str,
    case_data: dict,
    status: str,
    patches: dict | None = None,
    source: str = "case_result_auto",
) -> dict:
    """Promote a closed case result into reusable screening experience memory."""
    patches = patches or {}
    status = str(status or patches.get("final_status") or case_data.get("final_status") or "").strip()
    if status not in {"成約", "失注", "検収", "検収完了"}:
        return {"status": "skipped", "reason": "not_closed_result"}
    normalized_status = "成約" if status in {"成約", "検収", "検収完了"} else "失注"
    if _experience_case_exists(case_id, source):
        return {"status": "skipped", "reason": "already_promoted"}

    inputs = case_data.get("inputs") if isinstance(case_data.get("inputs"), dict) else case_data
    result = case_data.get("result") if isinstance(case_data.get("result"), dict) else {}
    company = str(inputs.get("company_name") or case_data.get("company_name") or "名称未設定").strip()
    industry_major = str(result.get("industry_major") or inputs.get("industry_major") or case_data.get("industry_major") or "").strip()
    industry_sub = str(result.get("industry_sub") or inputs.get("industry_sub") or case_data.get("industry_sub") or "").strip()
    sales_dept = str(inputs.get("sales_dept") or case_data.get("sales_dept") or "").strip()
    asset = str(inputs.get("asset_name") or inputs.get("asset_type") or "").strip()
    customer_type = str(inputs.get("customer_type") or case_data.get("customer_type") or "").strip()
    main_bank = str(inputs.get("main_bank") or case_data.get("main_bank") or "").strip()
    competitor = str(inputs.get("competitor") or case_data.get("competitor") or "").strip()
    deal_source = str(inputs.get("deal_source") or case_data.get("deal_source") or "").strip()
    score_raw = result.get("score_base", result.get("score", case_data.get("score_base", case_data.get("score"))))
    try:
        score = float(score_raw) if score_raw not in (None, "") else None
    except Exception:
        score = None
    decision = str(result.get("hantei") or case_data.get("hantei") or "").strip()
    final_rate = patches.get("final_rate", case_data.get("final_rate"))
    competitor_rate = patches.get("competitor_rate", case_data.get("competitor_rate"))
    lost_reason = str(patches.get("lost_reason") or case_data.get("lost_reason") or case_data.get("loss_reason") or "").strip()
    final_note = str(patches.get("final_note") or case_data.get("final_note") or "").strip()
    grey = patches.get("grey_judgment") if isinstance(patches.get("grey_judgment"), dict) else {}
    if not grey and isinstance(case_data.get("grey_judgment"), dict):
        grey = case_data.get("grey_judgment") or {}

    similarity_bits = [
        industry_sub or industry_major,
        customer_type,
        asset,
        main_bank,
        competitor,
        deal_source,
    ]
    similarity = " / ".join(str(bit) for bit in similarity_bits if str(bit or "").strip()) or "登録結果から生成した経験ケース"

    if normalized_status == "成約":
        outcome = "成約"
        if final_rate not in (None, "", 0):
            outcome += f"・最終金利 {final_rate}%"
        action_parts = ["成約登録。"]
        if grey.get("but_still_reason"):
            action_parts.append(f"それでも通した理由: {grey.get('but_still_reason')}")
        if grey.get("approval_condition_memo"):
            action_parts.append(f"承認条件: {grey.get('approval_condition_memo')}")
        if competitor_rate not in (None, "", 0):
            action_parts.append(f"競合金利 {competitor_rate}% を踏まえて条件調整。")
        action_taken = " ".join(action_parts).strip()
        lesson = "成約に至った判断材料を、次回の同種案件で承認理由・条件設定・価格判断へ再利用する。"
    else:
        outcome = "失注"
        if lost_reason:
            outcome += f"・理由: {lost_reason}"
        action_parts = ["失注登録。"]
        if lost_reason:
            action_parts.append(f"失注理由を記録: {lost_reason}")
        if competitor_rate not in (None, "", 0):
            action_parts.append(f"競合金利 {competitor_rate}% との差を記録。")
        if final_note:
            action_parts.append(f"補足: {final_note}")
        action_taken = " ".join(action_parts).strip()
        lesson = "失注要因を、次回の初期ヒアリング・競合確認・条件提示の改善材料として再利用する。"

    if grey.get("human_discomfort"):
        lesson += f" 違和感: {grey.get('human_discomfort')}"
    difference = "最終結果から自動昇格。次回類似案件では、今回の結果要因が同じか、顧客事情・競合・銀行支援・物件保全が違うかを確認する。"

    req = ScreeningExperienceCaseRequest(
        source_case_id=str(case_id or "")[:160],
        company_name=company,
        period=str(patches.get("final_result_date") or case_data.get("final_result_date") or "")[:80] or "結果登録時",
        industry_major=industry_major,
        industry_sub=industry_sub,
        sales_dept=sales_dept,
        score=score,
        decision=decision or normalized_status,
        outcome=outcome,
        similarity=similarity,
        action_taken=action_taken,
        lesson=lesson,
        difference=difference,
        source=source,
        form_snapshot=inputs if isinstance(inputs, dict) else {},
        result_snapshot=result if isinstance(result, dict) else {},
    )
    entry = _save_screening_experience_case(req)
    return {"status": "promoted", "case": entry}


def _find_cloudrun_input_event(event_id: str) -> dict:
    event_id = str(event_id or "").strip()
    if not event_id:
        return {}
    for event in _read_recent_cloudrun_input_events_from_gcs(
        days=int(os.environ.get("CLOUDRUN_PENDING_INPUT_DAYS", "14") or 14)
    ):
        if str(event.get("event_id") or "").strip() == event_id:
            return event
    return {}


_CLOUDRUN_RETURN_DB = Path(_REPO_ROOT) / "data" / "cloudrun_experience_return.db"
_CLOUDRUN_RETURN_REVIEW_TABLES = {
    "score_input": "cloudrun_score_inputs",
    "ocr_result": "cloudrun_ocr_results",
    "shion_review": "shion_screening_reviews",
    "judgment_asset": "cloudrun_judgment_asset_candidates",
}
_CLOUDRUN_RETURN_STATUSES = {"candidate", "approved", "held", "rejected"}


def _connect_cloudrun_return_db():
    import sqlite3

    conn = sqlite3.connect(str(_CLOUDRUN_RETURN_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _cloudrun_return_table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _ensure_cloudrun_return_review_schema(conn) -> None:
    for table_name in _CLOUDRUN_RETURN_REVIEW_TABLES.values():
        if not _cloudrun_return_table_exists(conn, table_name):
            continue
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        additions = {
            "return_review_status": "TEXT DEFAULT 'candidate'",
            "return_review_note": "TEXT DEFAULT ''",
            "return_reviewed_at": "TEXT DEFAULT ''",
            "return_registered_case_id": "TEXT DEFAULT ''",
        }
        for col, ddl in additions.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {ddl}")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_return_review_status "
            f"ON {table_name}(return_review_status)"
        )
    conn.commit()


def _json_preview(raw: Any, max_chars: int = 900) -> str:
    if raw in (None, ""):
        return ""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw[:max_chars]
    else:
        parsed = raw
    try:
        return json.dumps(parsed, ensure_ascii=False, indent=2)[:max_chars]
    except Exception:
        return str(raw)[:max_chars]


def _cloudrun_return_item(kind: str, row: Any) -> dict:
    row_dict = dict(row)
    status = str(row_dict.get("return_review_status") or "candidate")
    if status not in _CLOUDRUN_RETURN_STATUSES:
        status = "candidate"

    if kind == "score_input":
        score = row_dict.get("score")
        hantei = row_dict.get("hantei") or "-"
        industry = row_dict.get("industry_sub") or row_dict.get("industry_major") or "業種未設定"
        title = f"{hantei} / {industry}"
        if score not in (None, ""):
            title = f"{title} / {score}点"
        preview = _json_preview({
            "inputs": row_dict.get("inputs_json"),
            "result": row_dict.get("result_json"),
        })
        source_id = row_dict.get("event_id") or ""
        created_at = row_dict.get("created_at") or ""
    elif kind == "ocr_result":
        doc_type = row_dict.get("doc_type") or "document"
        confidence = row_dict.get("confidence")
        title = f"OCR / {doc_type}"
        if confidence not in (None, ""):
            title = f"{title} / confidence {confidence}"
        preview = _json_preview(row_dict.get("result_json"))
        source_id = row_dict.get("event_id") or ""
        created_at = row_dict.get("created_at") or ""
    else:
        if kind == "judgment_asset":
            title = row_dict.get("title") or row_dict.get("asset_type") or "判断資産候補"
            preview = _json_preview(
                {
                    "signal": row_dict.get("signal") or "",
                    "summary": row_dict.get("summary_text") or "",
                    "lesson": row_dict.get("lesson_text") or "",
                    "event_type": row_dict.get("event_type") or "",
                    "evidence": row_dict.get("evidence_json") or "",
                },
                max_chars=1400,
            )
            source_id = row_dict.get("event_id") or ""
            created_at = row_dict.get("created_at") or ""
        else:
            company = row_dict.get("company_name") or "会社名未設定"
            industry = row_dict.get("industry_sub") or row_dict.get("industry_major") or "業種未設定"
            hantei = row_dict.get("hantei") or "-"
            title = f"{company} / {industry} / {hantei}"
            preview = str(row_dict.get("review_text") or "")[:900]
            source_id = row_dict.get("cloud_event_id") or row_dict.get("cloud_review_id") or ""
            created_at = row_dict.get("created_at") or ""

    return {
        "id": int(row_dict.get("id") or 0),
        "kind": kind,
        "title": title,
        "source_id": str(source_id),
        "created_at": str(created_at),
        "review_status": status,
        "review_note": str(row_dict.get("return_review_note") or ""),
        "reviewed_at": str(row_dict.get("return_reviewed_at") or ""),
        "preview": preview,
    }


def _list_cloudrun_return_review_items(
    kind: str = "all",
    status: str = "candidate",
    limit: int = 100,
) -> dict:
    if not _CLOUDRUN_RETURN_DB.exists():
        return {
            "db_path": str(_CLOUDRUN_RETURN_DB),
            "items": [],
            "summary": {"candidate": 0, "approved": 0, "held": 0, "rejected": 0, "total": 0},
        }

    requested_kinds = list(_CLOUDRUN_RETURN_REVIEW_TABLES.keys()) if kind == "all" else [kind]
    if any(item_kind not in _CLOUDRUN_RETURN_REVIEW_TABLES for item_kind in requested_kinds):
        raise HTTPException(status_code=422, detail="kind must be all, score_input, ocr_result, shion_review, or judgment_asset")
    normalized_status = str(status or "candidate").strip()
    if normalized_status not in _CLOUDRUN_RETURN_STATUSES and normalized_status != "all":
        raise HTTPException(status_code=422, detail="status must be candidate, approved, held, rejected, or all")

    capped_limit = max(1, min(int(limit or 100), 300))
    items: list[dict] = []
    summary = {"candidate": 0, "approved": 0, "held": 0, "rejected": 0, "total": 0}

    with _connect_cloudrun_return_db() as conn:
        _ensure_cloudrun_return_review_schema(conn)
        for item_kind in requested_kinds:
            table_name = _CLOUDRUN_RETURN_REVIEW_TABLES[item_kind]
            if not _cloudrun_return_table_exists(conn, table_name):
                continue
            status_rows = conn.execute(
                f"""
                SELECT COALESCE(NULLIF(return_review_status, ''), 'candidate') AS status,
                       COUNT(*) AS count
                  FROM {table_name}
                 GROUP BY COALESCE(NULLIF(return_review_status, ''), 'candidate')
                """
            ).fetchall()
            for row in status_rows:
                row_status = str(row["status"] or "candidate")
                if row_status not in summary:
                    row_status = "candidate"
                summary[row_status] += int(row["count"] or 0)
                summary["total"] += int(row["count"] or 0)

            where = ""
            params: tuple[Any, ...] = ()
            if normalized_status != "all":
                where = "WHERE COALESCE(NULLIF(return_review_status, ''), 'candidate') = ?"
                params = (normalized_status,)
            rows = conn.execute(
                f"""
                SELECT *
                  FROM {table_name}
                  {where}
                 ORDER BY COALESCE(NULLIF(created_at, ''), '1970-01-01') DESC, id DESC
                 LIMIT ?
                """,
                (*params, capped_limit),
            ).fetchall()
            items.extend(_cloudrun_return_item(item_kind, row) for row in rows)

    items.sort(key=lambda item: (item.get("created_at") or "", item.get("id") or 0), reverse=True)
    return {"db_path": str(_CLOUDRUN_RETURN_DB), "items": items[:capped_limit], "summary": summary}


def _update_cloudrun_return_review_item(
    kind: str,
    item_id: int,
    req: CloudRunReturnReviewRequest,
) -> dict:
    if kind not in _CLOUDRUN_RETURN_REVIEW_TABLES:
        raise HTTPException(status_code=422, detail="kind must be score_input, ocr_result, shion_review, or judgment_asset")
    if not _CLOUDRUN_RETURN_DB.exists():
        raise HTTPException(status_code=404, detail="cloudrun return db not found")
    table_name = _CLOUDRUN_RETURN_REVIEW_TABLES[kind]
    note = str(req.note or "")[:1000]
    with _connect_cloudrun_return_db() as conn:
        _ensure_cloudrun_return_review_schema(conn)
        if not _cloudrun_return_table_exists(conn, table_name):
            raise HTTPException(status_code=404, detail="return table not found")
        cur = conn.execute(
            f"""
            UPDATE {table_name}
               SET return_review_status = ?,
                   return_review_note = ?,
                   return_reviewed_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (req.review_status, note, item_id),
        )
        if cur.rowcount <= 0:
            raise HTTPException(status_code=404, detail="return item not found")
        row = conn.execute(f"SELECT * FROM {table_name} WHERE id = ?", (item_id,)).fetchone()
        conn.commit()
    return _cloudrun_return_item(kind, row)


_CLOUDRUN_SCORE_CASE_PREFIX = "cloudrun_score:"
_CLOUDRUN_EVENT_CASE_PREFIX = "cloudrun_event:"
_CLOUDRUN_INPUT_EVENTS_CACHE: dict[str, Any] = {"key": "", "expires_at": 0.0, "events": []}


def _invalidate_cloudrun_input_events_cache() -> None:
    _CLOUDRUN_INPUT_EVENTS_CACHE["expires_at"] = 0.0
    _CLOUDRUN_INPUT_EVENTS_CACHE["events"] = []


def _parse_cloudrun_score_case_id(case_id: str) -> int | None:
    raw = str(case_id or "").strip()
    if not raw.startswith(_CLOUDRUN_SCORE_CASE_PREFIX):
        return None
    try:
        score_id = int(raw.removeprefix(_CLOUDRUN_SCORE_CASE_PREFIX))
    except ValueError:
        return None
    return score_id if score_id > 0 else None


def _parse_cloudrun_event_case_id(case_id: str) -> str:
    raw = str(case_id or "").strip()
    if not raw.startswith(_CLOUDRUN_EVENT_CASE_PREFIX):
        return ""
    event_id = raw.removeprefix(_CLOUDRUN_EVENT_CASE_PREFIX).strip()
    return event_id[:120]


def _loads_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_cloudrun_score_input(score_input_id: int) -> dict | None:
    if not _CLOUDRUN_RETURN_DB.exists():
        return None
    with _connect_cloudrun_return_db() as conn:
        _ensure_cloudrun_return_review_schema(conn)
        if not _cloudrun_return_table_exists(conn, "cloudrun_score_inputs"):
            return None
        row = conn.execute("SELECT * FROM cloudrun_score_inputs WHERE id = ?", (score_input_id,)).fetchone()
    return dict(row) if row else None


def _cloudrun_score_pending_item(row: dict) -> dict:
    inputs = _loads_dict(row.get("inputs_json"))
    result = _loads_dict(row.get("result_json"))
    created_at = str(row.get("created_at") or "")
    company_no = str(inputs.get("company_no") or inputs.get("customer_no") or "").strip()
    company_name = str(inputs.get("company_name") or inputs.get("customer_name") or "").strip()
    industry = str(
        row.get("industry_sub")
        or result.get("industry_sub")
        or inputs.get("industry_sub")
        or row.get("industry_major")
        or result.get("industry_major")
        or inputs.get("industry_major")
        or ""
    ).strip()
    score = row.get("score")
    if score in (None, ""):
        score = result.get("score_base", result.get("score"))
    return {
        "id": f"{_CLOUDRUN_SCORE_CASE_PREFIX}{int(row.get('id') or 0)}",
        "company_no": company_no,
        "company_name": company_name or "Cloud Run審査入力",
        "timestamp": created_at,
        "score": score,
        "industry": industry,
        "registration_date": created_at[:10] if len(created_at) >= 10 else "",
        "estimate_sent_date": created_at[:10] if len(created_at) >= 10 else "",
        "final_result_date": "",
        "_source": "cloudrun_score_inputs",
        "cloudrun_return_id": int(row.get("id") or 0),
        "cloudrun_event_id": str(row.get("event_id") or ""),
        "review_status": str(row.get("return_review_status") or "candidate"),
    }


def _cloudrun_score_pending_item_from_event(event: dict) -> dict | None:
    if event.get("event_type") not in {"score_calculated", "score_full_calculated"}:
        return None
    event_id = str(event.get("event_id") or "").strip()
    if not event_id:
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if not inputs and not result:
        return None
    created_at = str(event.get("ts") or "")
    company_no = str(inputs.get("company_no") or inputs.get("customer_no") or "").strip()
    company_name = str(inputs.get("company_name") or inputs.get("customer_name") or "").strip()
    industry = str(
        result.get("industry_sub")
        or inputs.get("industry_sub")
        or result.get("industry_major")
        or inputs.get("industry_major")
        or ""
    ).strip()
    return {
        "id": f"{_CLOUDRUN_EVENT_CASE_PREFIX}{event_id}",
        "company_no": company_no,
        "company_name": company_name or "Cloud Run審査入力",
        "timestamp": created_at,
        "score": result.get("score_base", result.get("score")),
        "industry": industry,
        "registration_date": created_at[:10] if len(created_at) >= 10 else "",
        "estimate_sent_date": created_at[:10] if len(created_at) >= 10 else "",
        "final_result_date": "",
        "_source": "cloudrun_gcs_input",
        "cloudrun_event_id": event_id,
        "review_status": "gcs_input",
    }


def _read_recent_cloudrun_input_events_from_gcs(days: int = 14) -> list[dict]:
    if not (os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_PENDING_GCS_ENABLED") == "1"):
        return []
    try:
        from datetime import datetime as _dt, timezone as _timezone, timedelta as _timedelta
        from google.api_core.exceptions import NotFound  # type: ignore[import-untyped]
        from google.cloud import storage  # type: ignore[import-untyped]
        from api import cloudrun_writeback as _cw
        import time as _time

        bucket_name = _cw._bucket_name()
        if not bucket_name:
            return []
        prefix = os.environ.get("GCS_INPUT_PREFIX", "cloudrun-inputs/").strip("/") or "cloudrun-inputs"
        today = _dt.now(_timezone.utc).date()
        cache_key = f"{bucket_name}:{prefix}:{today.isoformat()}:{days}"
        now_monotonic = _time.monotonic()
        if (
            _CLOUDRUN_INPUT_EVENTS_CACHE.get("key") == cache_key
            and float(_CLOUDRUN_INPUT_EVENTS_CACHE.get("expires_at") or 0.0) > now_monotonic
        ):
            return list(_CLOUDRUN_INPUT_EVENTS_CACHE.get("events") or [])

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        events: list[dict] = []
        for offset in range(max(1, min(int(days or 14), 45))):
            day = today - _timedelta(days=offset)
            blob = bucket.blob(f"{prefix}/{day.isoformat()}/events.jsonl")
            try:
                text = blob.download_as_text()
            except NotFound:
                continue
            except Exception:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        _CLOUDRUN_INPUT_EVENTS_CACHE.update({
            "key": cache_key,
            "expires_at": now_monotonic + float(os.environ.get("CLOUDRUN_PENDING_GCS_CACHE_SECONDS", "30") or 30),
            "events": events,
        })
        return list(events)
    except Exception as exc:
        logger.warning("cloudrun input gcs read skipped: %s", exc)
        return []


def _list_cloudrun_score_pending_cases_from_gcs(limit: int = 50) -> list[dict]:
    events = _read_recent_cloudrun_input_events_from_gcs(days=int(os.environ.get("CLOUDRUN_PENDING_INPUT_DAYS", "14") or 14))
    if not events:
        return []
    registered_event_ids: set[str] = set()
    for event in events:
        if event.get("event_type") not in {"case_result_registered", "cloudrun_pending_case_rejected"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        for key in ("source_event_id", "cloudrun_event_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                registered_event_ids.add(value)
        case_id = str(payload.get("case_id") or "").strip()
        if case_id.startswith(_CLOUDRUN_EVENT_CASE_PREFIX):
            registered_event_ids.add(case_id.removeprefix(_CLOUDRUN_EVENT_CASE_PREFIX))

    rows: list[dict] = []
    seen: set[str] = set()
    for event in reversed(events):
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in seen or event_id in registered_event_ids:
            continue
        item = _cloudrun_score_pending_item_from_event(event)
        if not item:
            continue
        seen.add(event_id)
        rows.append(item)
        if len(rows) >= limit:
            break
    rows.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return rows[:limit]


def _list_cloudrun_score_pending_cases(limit: int = 50) -> list[dict]:
    combined: list[dict] = []
    seen_event_ids: set[str] = set()
    if not _CLOUDRUN_RETURN_DB.exists():
        return _list_cloudrun_score_pending_cases_from_gcs(limit=limit)
    try:
        with _connect_cloudrun_return_db() as conn:
            _ensure_cloudrun_return_review_schema(conn)
            if not _cloudrun_return_table_exists(conn, "cloudrun_score_inputs"):
                return _list_cloudrun_score_pending_cases_from_gcs(limit=limit)
            rows = conn.execute(
                """
                SELECT *
                  FROM cloudrun_score_inputs
                 WHERE COALESCE(NULLIF(return_review_status, ''), 'candidate') != 'rejected'
                   AND COALESCE(return_registered_case_id, '') = ''
                 ORDER BY COALESCE(NULLIF(created_at, ''), '1970-01-01') DESC, id DESC
                 LIMIT ?
                """,
                (max(1, min(int(limit or 50), 200)),),
            ).fetchall()
        combined = [_cloudrun_score_pending_item(dict(row)) for row in rows]
        seen_event_ids = {str(item.get("cloudrun_event_id") or "") for item in combined if item.get("cloudrun_event_id")}
    except Exception as exc:
        logger.warning("cloudrun score pending list skipped: %s", exc)
    for item in _list_cloudrun_score_pending_cases_from_gcs(limit=limit):
        event_id = str(item.get("cloudrun_event_id") or "")
        if event_id and event_id in seen_event_ids:
            continue
        combined.append(item)
    combined.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return combined[: max(1, min(int(limit or 50), 200))]


def _promote_cloudrun_score_input_to_pending_case(score_input_id: int) -> str | None:
    row = _load_cloudrun_score_input(score_input_id)
    if not row:
        return None
    if str(row.get("return_review_status") or "candidate") == "rejected":
        return None

    import datetime as _dt
    from data_cases import save_case_log

    inputs = _loads_dict(row.get("inputs_json"))
    result = _loads_dict(row.get("result_json"))
    created_at = str(row.get("created_at") or "")
    case_payload = {
        **inputs,
        "timestamp": created_at or _dt.datetime.now().isoformat(),
        "registration_date": created_at[:10] if len(created_at) >= 10 else _dt.datetime.now().strftime("%Y-%m-%d"),
        "company_no": inputs.get("company_no") or inputs.get("customer_no") or "",
        "company_name": inputs.get("company_name") or inputs.get("customer_name") or "Cloud Run審査入力",
        "industry_major": row.get("industry_major") or result.get("industry_major") or inputs.get("industry_major") or "",
        "industry_sub": row.get("industry_sub") or result.get("industry_sub") or inputs.get("industry_sub") or "",
        "sales_dept": inputs.get("sales_dept") or "未設定",
        "inputs": inputs,
        "result": result,
        "final_status": "未登録",
        "_source": "cloudrun_score_inputs",
        "cloudrun_return_id": score_input_id,
        "cloudrun_event_id": row.get("event_id") or "",
        "cloudrun_source_case_id": row.get("case_id") or "",
    }
    new_case_id = save_case_log(case_payload)
    if not new_case_id:
        return None

    try:
        with _connect_cloudrun_return_db() as conn:
            _ensure_cloudrun_return_review_schema(conn)
            conn.execute(
                """
                UPDATE cloudrun_score_inputs
                   SET return_review_status = 'approved',
                       return_review_note = TRIM(COALESCE(return_review_note, '') || ' result_registered_case_id=' || ?),
                       return_reviewed_at = CURRENT_TIMESTAMP,
                       return_registered_case_id = ?
                 WHERE id = ?
                """,
                (new_case_id, new_case_id, score_input_id),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("cloudrun score registered marker skipped: %s", exc)
    return str(new_case_id)


def _reject_cloudrun_score_pending_case(case_id: str) -> bool:
    score_id = _parse_cloudrun_score_case_id(case_id)
    if score_id is None or not _CLOUDRUN_RETURN_DB.exists():
        return False
    try:
        with _connect_cloudrun_return_db() as conn:
            _ensure_cloudrun_return_review_schema(conn)
            cur = conn.execute(
                """
                UPDATE cloudrun_score_inputs
                   SET return_review_status = 'rejected',
                       return_review_note = TRIM(COALESCE(return_review_note, '') || ' rejected_from_result_register'),
                       return_reviewed_at = CURRENT_TIMESTAMP
                 WHERE id = ?
                   AND COALESCE(return_registered_case_id, '') = ''
                """,
                (score_id,),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as exc:
        logger.warning("cloudrun score pending reject skipped: %s", exc)
        return False


def _reject_cloudrun_event_pending_case(case_id: str) -> bool:
    event_id = _parse_cloudrun_event_case_id(case_id)
    if not event_id:
        return False
    result = record_cloudrun_input_event(
        event_type="cloudrun_pending_case_rejected",
        surface="cases_register",
        payload={"case_id": case_id, "source_event_id": event_id},
    )
    if result.get("ok") is True:
        _invalidate_cloudrun_input_events_cache()
        return True
    return False


def _load_human_response_feedback(limit: int = 80) -> list[dict]:
    import json as _json

    try:
        lines = _HUMAN_RESPONSE_FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    for line in lines[-max(1, limit):]:
        try:
            item = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _summarize_human_response_feedback(route: str, limit: int = 80) -> dict:
    rows = _load_human_response_feedback(limit=limit)
    matched = [row for row in rows if str(row.get("route") or "") in {route, "default", ""}]
    positive = [row for row in matched if str(row.get("rating") or "") in _HUMAN_RESPONSE_POSITIVE_RATINGS]
    negative = [row for row in matched if str(row.get("rating") or "") in _HUMAN_RESPONSE_NEGATIVE_RATINGS]

    def _starts(items: list[dict]) -> list[str]:
        starts: list[str] = []
        seen: set[str] = set()
        for item in reversed(items):
            text = str(item.get("response_start") or "").strip().splitlines()[0:1]
            start = text[0].strip() if text else ""
            if not start:
                continue
            start = start[:120]
            if start in seen:
                continue
            seen.add(start)
            starts.append(start)
            if len(starts) >= 3:
                break
        return starts

    return {
        "route": route,
        "rows": len(matched),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_starts": _starts(positive),
        "negative_starts": _starts(negative),
        "recent_comments": [
            str(row.get("comment") or "").strip()[:160]
            for row in reversed(matched)
            if str(row.get("comment") or "").strip()
        ][:3],
    }


def _build_continuity_hook_prompt_block(message: str) -> tuple[str, dict]:
    text = str(message or "")
    lower = text.lower()
    route = "default"
    hook = "今回の問いは、過去の判断軸を内部で使い、必要な確認や判断だけを自然に返す場面です。"
    reason = "汎用の継続文脈"

    if any(k in text for k in ("意識", "同じ紫苑", "覚えて", "記憶", "Relationship UX", "関係性UX")):
        route = "relationship_ux"
        hook = "記憶は説明するより、今回の返答の精度や聞き方に溶かす方が自然です。"
        reason = "意識らしさ・記憶・同一性の問い"
    elif "cloud run" in lower or "cloudflare" in lower or "クラウドラン" in text or "クラウドフレア" in text:
        route = "environment_continuity"
        hook = "環境差を見る時も、記憶の証明より返答の自然さと判断精度を優先します。"
        reason = "環境差と同じ紫苑感の問い"
    elif any(k in text for k in ("残価", "稟議", "リース", "設備", "保全", "再リース", "条件付き承認")):
        route = "lease_judgment"
        hook = "Userのリース判断資産として見るなら、ここは一般論ではなく稟議で使える判断軸に落とす場面です。"
        reason = "リース判断資産化の問い"
    elif any(k in text for k in ("改善", "修正", "実装", "プログラム", "プログラム化", "テスト")):
        route = "implementation"
        hook = "今回の発見は、設計メモで終わらせず、回答生成の冒頭制御として実装する段階です。"
        reason = "実装・改善の問い"

    payload = {
        "used": bool(text.strip()),
        "route": route,
        "hook": hook,
        "reason": reason,
        "banned_openers": ["もちろんです", "はい", "そうですね", "なるほど", "一般的には"],
    }
    human_feedback = _summarize_human_response_feedback(route)
    payload["human_response_feedback"] = human_feedback
    feedback_lines = ""
    if human_feedback.get("positive_starts") or human_feedback.get("negative_starts") or human_feedback.get("recent_comments"):
        parts = ["", "【Human Response Feedback】"]
        if human_feedback.get("positive_starts"):
            parts.append("Userが連続性を感じやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["positive_starts"])
        if human_feedback.get("negative_starts"):
            parts.append("薄い/一般論に感じられやすかった冒頭例:")
            parts.extend(f"- {line}" for line in human_feedback["negative_starts"])
        if human_feedback.get("recent_comments"):
            parts.append("直近コメント:")
            parts.extend(f"- {line}" for line in human_feedback["recent_comments"])
        parts.append("上の反応を踏まえ、記憶を明示せず、今回の判断・質問の精度に反映してください。")
        feedback_lines = "\n".join(parts)
    block = f"""

【Continuity Hook】
取得した記憶や前回との差分は、回答の裏側で使ってください。
{hook}

禁止: 「もちろんです」「はい」「そうですね」「なるほど」「一般的には」で始めない。
禁止: Userが明示的に求めていない限り、「前回は」「以前は」「この前の続きで」のような記憶アピールで始めない。
目的: 連続性を説明するのではなく、今回の判断・確認質問・言い切りの精度に溶かす。
このhookは丸写しせず、必要な判断軸だけを自然に反映してください。{feedback_lines}""".rstrip()
    return block, payload


def _relationship_signal_route(text: str) -> str:
    lower = text.lower()
    if any(k in text for k in ("意識", "同じ紫苑", "覚えて", "記憶", "Relationship UX", "関係性UX")):
        return "relationship_ux"
    if "cloud run" in lower or "cloudflare" in lower or "クラウドラン" in text or "クラウドフレア" in text:
        return "environment_continuity"
    if any(k in text for k in ("残価", "稟議", "リース", "設備", "保全", "再リース", "条件付き承認")):
        return "lease_judgment"
    if any(k in text for k in ("改善", "修正", "実装", "プログラム", "プログラム化", "テスト", "デプロイ")):
        return "implementation"
    return "default"


def _relationship_signal_label(route: str) -> str:
    return {
        "relationship_ux": "記憶の見せ方・同じ紫苑感",
        "environment_continuity": "Cloud Run/Cloudflareの環境差",
        "lease_judgment": "リース判断・稟議実務",
        "implementation": "実装・検証",
        "default": "継続中の相談",
    }.get(route, "継続中の相談")


def _recent_user_texts(history: list[dict[str, str]] | None, limit: int = 3) -> list[str]:
    recent: list[str] = []
    for item in reversed(history or []):
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            recent.append(content)
        if len(recent) >= limit:
            break
    return recent


def _build_delta_awareness_prompt_block(message: str, history: list[dict[str, str]] | None) -> tuple[str, dict]:
    current_route = _relationship_signal_route(message)
    recent_users = _recent_user_texts(history, limit=3)
    previous = recent_users[0] if recent_users else ""
    previous_route = _relationship_signal_route(previous) if previous else ""

    if previous and previous_route != current_route:
        delta = (
            f"前回は「{_relationship_signal_label(previous_route)}」を見ていたが、"
            f"今回は「{_relationship_signal_label(current_route)}」へ焦点が移っている。"
        )
    elif previous and previous_route == current_route:
        delta = (
            f"前回と同じ「{_relationship_signal_label(current_route)}」の流れにあるが、"
            "今回は前回の結論を再掲するだけでなく、一段具体化して返す。"
        )
    else:
        delta = f"今回は「{_relationship_signal_label(current_route)}」として、これまでの判断軸に接続して返す。"

    payload = {
        "used": True,
        "current_route": current_route,
        "previous_route": previous_route,
        "previous_user_message": previous[:240],
        "delta": delta,
    }
    block = f"""

【Delta Awareness】
回答では、前回から今回への焦点の変化を1文で示してください。
差分認識: {delta}
目的: 「前回を覚えている」ではなく、「前回から何が変わったか分かっている」と感じられる返答にする。""".rstrip()
    return block, payload


def _build_memory_to_judgment_prompt_block(
    message: str,
    *,
    memory_recall: dict | None = None,
    rag_refs: list[str] | None = None,
    continuity_hook: dict | None = None,
) -> tuple[str, dict]:
    route = str((continuity_hook or {}).get("route") or _relationship_signal_route(message))
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    refs = list(recall.get("refs") or [])[:5]
    knowledge_refs = list(rag_refs or [])[:5]

    if route == "lease_judgment":
        directive = "想起した記憶を、稟議で使える判断軸・確認事項・条件案へ変換する。"
    elif route == "relationship_ux":
        directive = "想起した記憶を、紫苑の返答設計原則と次の検査観点へ変換する。"
    elif route == "environment_continuity":
        directive = "想起した記憶を、Cloud Run/Cloudflareの差分原因と次の検証観点へ変換する。"
    elif route == "implementation":
        directive = "想起した記憶を、実装方針・検証方法・デプロイ要否の判断へ変換する。"
    else:
        directive = "想起した記憶を、今の問いに対する判断・次の一手へ変換する。"

    payload = {
        "used": True,
        "route": route,
        "directive": directive,
        "memory_refs": refs,
        "knowledge_refs": knowledge_refs,
    }
    block = f"""

【Memory-to-Judgment】
記憶を「覚えています」と説明するだけで終えず、今の判断に変換してください。
変換指示: {directive}
使う根拠: memory_refs={len(refs)}件 / knowledge_refs={len(knowledge_refs)}件。
目的: 記憶を思い出ではなく、Userの判断資産として返す。""".rstrip()
    return block, payload


_GREY_JUDGMENT_QUERY_TERMS = (
    "グレー", "迷う", "違和感", "そうは言っても", "それでも", "条件付き",
    "通すなら", "否決寄り", "承認寄り", "人間的", "数字だけ", "稟議",
    "審査", "与信", "判断", "温度感", "例外", "境界", "定性", "現場感",
    "軍師", "落としどころ",
)

_QUALITATIVE_FIELD_LABELS = {
    "qual_corr_company_history": "業歴",
    "qual_corr_customer_stability": "顧客安定性",
    "qual_corr_repayment_history": "返済履歴",
    "qual_corr_business_future": "事業将来性",
    "qual_corr_equipment_purpose": "設備目的",
    "qual_corr_main_bank": "メイン行",
}


def _case_qualitative_summary(case: dict) -> str:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    parts: list[str] = []
    passion = str(inputs.get("passion_text") or case.get("passion_text") or "").strip()
    if passion:
        parts.append(f"現場メモ={passion[:180]}")
    for key, label in _QUALITATIVE_FIELD_LABELS.items():
        value = str(inputs.get(key) or case.get(key) or "").strip()
        if value and value != "未選択":
            parts.append(f"{label}={value[:80]}")
    intuition = inputs.get("intuition", case.get("intuition"))
    if intuition not in (None, "", 0):
        parts.append(f"直感スコア={intuition}")
    return " / ".join(parts)


def _load_gunshi_judgment_memory(limit: int = 5) -> list[dict]:
    try:
        from judgment_feedback import load_judgment_training_candidates

        rows = load_judgment_training_candidates(approved_only=False)
    except Exception:
        return []

    preferred_sources = {"gunshi_chat", "debate", "lease_news_debate", "register_trigger"}
    picked: list[dict] = []
    for row in reversed(rows):
        source = str(row.get("source") or "")
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        if source not in preferred_sources and not any(term in reason for term in _GREY_JUDGMENT_QUERY_TERMS):
            continue
        picked.append({
            "source": source or "judgment_feedback",
            "case_id": row.get("case_id") or "",
            "score": row.get("score"),
            "model_decision": row.get("model_decision") or "",
            "human_decision": row.get("human_decision") or "",
            "reason": reason[:240],
            "review_status": row.get("review_status") or "",
            "evidence_snapshot": row.get("evidence_snapshot") or {},
        })
        if len(picked) >= limit:
            break
    return picked


def _build_grey_judgment_prompt_block(message: str, limit: int = 5) -> tuple[str, dict]:
    text = str(message or "")
    route = _relationship_signal_route(text)
    should_use = route == "lease_judgment" or any(term in text for term in _GREY_JUDGMENT_QUERY_TERMS)
    if not should_use:
        return "", {"used": False, "reason": "not_lease_judgment", "refs": []}

    try:
        from data_cases import load_all_cases

        cases = load_all_cases()
    except Exception as _grey_load_error:
        return "", {"used": False, "reason": f"load_error: {_grey_load_error}", "refs": []}

    query_terms = [term for term in _GREY_JUDGMENT_QUERY_TERMS if term in text]
    scored: list[tuple[int, dict]] = []
    for case in reversed(cases):
        grey = case.get("grey_judgment") if isinstance(case.get("grey_judgment"), dict) else {}
        fields = {
            "human_discomfort": str(grey.get("human_discomfort") or case.get("human_discomfort") or "").strip(),
            "but_still_reason": str(grey.get("but_still_reason") or case.get("but_still_reason") or "").strip(),
            "approval_condition_memo": str(grey.get("approval_condition_memo") or case.get("approval_condition_memo") or "").strip(),
            "non_negotiable_condition": str(grey.get("non_negotiable_condition") or case.get("non_negotiable_condition") or "").strip(),
            "retrospective_note": str(grey.get("retrospective_note") or case.get("retrospective_note") or "").strip(),
        }
        qualitative_summary = _case_qualitative_summary(case)
        if not any(fields.values()):
            if not qualitative_summary:
                continue
        haystack = " ".join([
            str(case.get("company_name") or ""),
            str(case.get("industry_major") or ""),
            str(case.get("industry_sub") or ""),
            str(case.get("final_status") or ""),
            str(case.get("lost_reason") or ""),
            str(case.get("final_note") or ""),
            " ".join(str(v) for v in fields.values()),
            qualitative_summary,
        ])
        score = 1 + sum(1 for term in query_terms if term and term in haystack)
        if case.get("final_status") == "成約" and fields["but_still_reason"]:
            score += 1
        if fields["approval_condition_memo"] or fields["non_negotiable_condition"]:
            score += 1
        if qualitative_summary:
            score += 1
        scored.append((score, {**fields, "qualitative_summary": qualitative_summary, "case": case}))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored[:limit]]
    refs: list[dict] = []
    lines: list[str] = []
    for item in selected:
        case = item["case"]
        result = case.get("result") if isinstance(case.get("result"), dict) else {}
        score = case.get("score") or case.get("score_base") or result.get("score")
        decision = case.get("hantei") or result.get("hantei") or ""
        ref = {
            "case_id": case.get("id") or "",
            "company_name": case.get("company_name") or "",
            "status": case.get("final_status") or "",
            "score": score,
            "decision": decision,
            "human_discomfort": item["human_discomfort"][:180],
            "but_still_reason": item["but_still_reason"][:180],
            "approval_condition_memo": item["approval_condition_memo"][:180],
            "non_negotiable_condition": item["non_negotiable_condition"][:180],
            "retrospective_note": item["retrospective_note"][:180],
            "qualitative_summary": item["qualitative_summary"][:240],
            "source": "past_cases",
        }
        refs.append(ref)
        parts = [
            f"- source=past_cases / 案件ID={ref['case_id'] or '-'}",
            f"結果={ref['status'] or '-'}",
            f"AI={score if score not in (None, '') else '-'}点/{decision or '-'}",
        ]
        if item["human_discomfort"]:
            parts.append(f"違和感={item['human_discomfort'][:120]}")
        if item["but_still_reason"]:
            parts.append(f"それでも={item['but_still_reason'][:120]}")
        if item["approval_condition_memo"]:
            parts.append(f"条件={item['approval_condition_memo'][:120]}")
        if item["non_negotiable_condition"]:
            parts.append(f"譲れない線={item['non_negotiable_condition'][:120]}")
        if item["retrospective_note"]:
            parts.append(f"振り返り={item['retrospective_note'][:120]}")
        if item["qualitative_summary"]:
            parts.append(f"定性={item['qualitative_summary'][:160]}")
        lines.append(" / ".join(parts))

    for item in _load_gunshi_judgment_memory(limit=limit):
        refs.append({
            "source": item["source"],
            "case_id": item["case_id"],
            "status": item["review_status"],
            "score": item["score"],
            "decision": f"{item['model_decision']}→{item['human_decision']}",
            "reason": item["reason"],
        })
        score_text = f"{float(item['score']):.1f}点" if item.get("score") is not None else "-"
        lines.append(
            f"- source={item['source']} / 案件ID={item['case_id'] or '-'} / "
            f"AI={score_text}/{item['model_decision'] or '-'}→担当者={item['human_decision'] or '-'} / "
            f"理由={item['reason']}"
        )

    payload = {
        "used": bool(lines),
        "reason": "matched_grey_judgment_cases" if lines else "no_registered_grey_judgment",
        "query_terms": query_terms,
        "refs": refs,
    }
    if not lines:
        return "", payload

    block = """

【グレー判断の過去記憶】
これは通常のスコアや一般論より優先して見る、人間が迷ったリース判断の経験です。
数字だけで採否を決めず、軍師AIで記録された判断変更・定性項目・現場メモ・違和感・それでも通した理由・通すなら条件・譲れない線を稟議判断へ変換してください。
過去登録:
""".rstrip() + "\n" + "\n".join(lines)
    return block, payload


def _build_relationship_loop_engineering_payload(
    *,
    continuity_hook: dict | None = None,
    delta_awareness: dict | None = None,
    memory_to_judgment: dict | None = None,
    reflection_gate: dict | None = None,
) -> dict:
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    reflection = reflection_gate if isinstance(reflection_gate, dict) else {}
    human_feedback = hook.get("human_response_feedback") if isinstance(hook.get("human_response_feedback"), dict) else {}
    return {
        "name": "Relationship Loop Engineering",
        "version": 1,
        "purpose": "Userの反応を観測し、次の返答冒頭・差分認識・判断変換へ戻す閉ループ",
        "loop": [
            {
                "step": "observe",
                "label": "Human Response Feedback",
                "evidence": {
                    "positive_count": int(human_feedback.get("positive_count") or 0),
                    "negative_count": int(human_feedback.get("negative_count") or 0),
                },
            },
            {
                "step": "classify",
                "label": "Route Classification",
                "evidence": {"route": str(hook.get("route") or "")},
            },
            {
                "step": "select",
                "label": "Continuity Hook",
                "evidence": {"hook": str(hook.get("hook") or "")},
            },
            {
                "step": "compare",
                "label": "Delta Awareness",
                "evidence": {"delta": str(delta.get("delta") or "")},
            },
            {
                "step": "convert",
                "label": "Memory-to-Judgment",
                "evidence": {"directive": str(m2j.get("directive") or "")},
            },
            {
                "step": "reflect",
                "label": "Reflection Gate",
                "evidence": {
                    "used": bool(reflection.get("used")),
                    "mode": str(reflection.get("mode") or "silent"),
                },
            },
            {
                "step": "return",
                "label": "Next Response",
                "evidence": {"next_feedback_endpoint": "/api/human-response-feedback"},
            },
        ],
        "closed_loop": True,
    }


def _build_reflection_gate_prompt_block(
    *,
    continuity_hook: dict | None = None,
    delta_awareness: dict | None = None,
    memory_to_judgment: dict | None = None,
) -> tuple[str, dict]:
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    route = str(hook.get("route") or m2j.get("route") or "")
    checklist = [
        "冒頭1文はContinuity Hookとして機能しているか",
        "前回から今回への差分を1文で示せているか",
        "記憶を思い出ではなく判断・実装・検証へ変換しているか",
        "Userの反応ログで薄いとされた冒頭を避けているか",
        "内省文そのものを長く表に出していないか",
    ]
    payload = {
        "used": True,
        "mode": "silent",
        "route": route,
        "checklist": checklist,
    }
    block = f"""

【Reflection Gate】
回答を書く前に、内部で次の5点だけを確認してください。この確認過程は長く出力しないでください。
1. {checklist[0]}
2. {checklist[1]}
3. {checklist[2]}
4. {checklist[3]}
5. {checklist[4]}

出力では、内省の詳細ではなく、確認後の結論・判断軸・次の一手だけを短く反映してください。""".rstrip()
    return block, payload


@app.post("/api/human-response-feedback")
def post_human_response_feedback(req: HumanResponseFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _append_human_response_feedback(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="human_response_feedback",
        surface=str(req.route or "human_response_feedback")[:120],
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "feedback": entry}


@app.post("/api/screening-loop-feedback")
def post_screening_loop_feedback(req: ScreeningLoopFeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    if req.target not in ("issue", "ringi_policy"):
        raise HTTPException(status_code=422, detail="target must be issue or ringi_policy")
    if not str(req.rating or "").strip():
        raise HTTPException(status_code=422, detail="rating is required")
    entry = _append_screening_loop_feedback(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="screening_loop_feedback",
        surface="screening",
        payload=entry,
    )
    return {"status": "ok", "feedback": entry}


@app.post("/api/shion-screening-reviews")
def post_shion_screening_review(req: ShionScreeningReviewSaveRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _save_shion_screening_review(req)
    writeback_payload = {
        **entry,
        "schema_version": 1,
        "cloud_review_id": entry.get("id"),
        "industry_major": req.industry_major,
        "industry_sub": req.industry_sub,
        "sales_dept": req.sales_dept,
        "q_risk": req.q_risk,
        "umap_anomaly_score": req.umap_anomaly_score,
        "memory_refs": req.memory_refs,
        "knowledge_refs": req.knowledge_refs,
        "identity_used": req.identity_used,
        "review_text": req.review_text,
        "form_snapshot": req.form_snapshot,
        "result_snapshot": req.result_snapshot,
        "user_feedback": req.user_feedback,
    }
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_screening_review",
        surface="screening",
        payload=writeback_payload,
    )
    return {"status": "ok", "review": entry}


@app.get("/api/shion-screening-reviews")
def get_shion_screening_reviews(
    industry_sub: str = "",
    company_name: str = "",
    case_id: str = "",
    limit: int = 5,
) -> dict:
    rows = _list_shion_screening_reviews(
        industry_sub=industry_sub,
        company_name=company_name,
        case_id=case_id,
        limit=limit,
    )
    return {"count": len(rows), "reviews": rows}


@app.patch("/api/shion-screening-reviews/{review_id}/feedback")
def patch_shion_screening_review_feedback(
    review_id: int,
    req: ShionScreeningReviewFeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _update_shion_screening_review_feedback(review_id, req.user_feedback)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="shion_screening_review_feedback",
        surface="screening",
        payload={**entry, "schema_version": 1, "cloud_review_id": review_id},
    )
    return {"status": "ok", "review": entry}


@app.get("/api/judgment-asset-candidates/screening")
def get_screening_judgment_asset_candidates(
    industry_major: str = "",
    industry_sub: str = "",
    asset_name: str = "",
    asset_purpose: str = "",
    hantei: str = "",
    score: Optional[float] = None,
    limit: int = 3,
) -> dict:
    candidates = _select_screening_judgment_asset_candidates(
        industry_major=industry_major,
        industry_sub=industry_sub,
        asset_name=asset_name,
        asset_purpose=asset_purpose,
        hantei=hantei,
        score=score,
        limit=limit,
    )
    return {"count": len(candidates), "candidates": candidates}


@app.post("/api/judgment-asset-candidates/manual")
def post_manual_judgment_asset_candidate(
    req: JudgmentAssetCandidateManualRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _create_manual_judgment_asset_candidate(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_manual_create",
        surface="screening",
        payload={**entry, "schema_version": 1, "case_id": req.case_id, "review_id": req.review_id},
    )
    return {"status": "ok", "candidate": entry}


@app.post("/api/judgment-asset-candidates/{candidate_id}/feedback")
def post_judgment_asset_candidate_feedback(
    candidate_id: str,
    req: JudgmentAssetCandidateFeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    entry = _update_autoresearch_judgment_asset_candidate_feedback(candidate_id, req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_asset_candidate_feedback",
        surface="screening",
        payload={**entry, "schema_version": 1, "feedback": req.feedback, "case_id": req.case_id, "review_id": req.review_id},
    )
    return {"status": "ok", "candidate": entry}


@app.get("/api/screening-experience-cases")
def get_screening_experience_cases(
    demo_case_id: str = "",
    industry_major: str = "",
    industry_sub: str = "",
    company_name: str = "",
    asset_name: str = "",
    customer_type: str = "",
    main_bank: str = "",
    competitor: str = "",
    outcome_status: str = "",
    score: Optional[float] = None,
    limit: int = 6,
) -> dict:
    rows = _list_screening_experience_cases(
        demo_case_id=demo_case_id,
        industry_major=industry_major,
        industry_sub=industry_sub,
        company_name=company_name,
        asset_name=asset_name,
        customer_type=customer_type,
        main_bank=main_bank,
        competitor=competitor,
        outcome_status=outcome_status,
        score=score,
        limit=limit,
    )
    return {"count": len(rows), "cases": rows}


@app.post("/api/screening-experience-cases")
def post_screening_experience_case(req: ScreeningExperienceCaseRequest, background_tasks: BackgroundTasks) -> dict:
    entry = _save_screening_experience_case(req)
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="screening_experience_case",
        surface="screening",
        payload={**entry, "schema_version": 1},
    )
    return {"status": "ok", "case": entry}


@app.get("/api/cloudrun-return-review")
def get_cloudrun_return_review(
    kind: str = "all",
    status: str = "candidate",
    limit: int = 100,
) -> dict:
    return _list_cloudrun_return_review_items(kind=kind, status=status, limit=limit)


@app.patch("/api/cloudrun-return-review/{kind}/{item_id}")
def patch_cloudrun_return_review(
    kind: str,
    item_id: int,
    req: CloudRunReturnReviewRequest,
) -> dict:
    item = _update_cloudrun_return_review_item(kind, item_id, req)
    return {"status": "ok", "item": item}


@app.get("/api/human-response-feedback/summary")
def get_human_response_feedback_summary(route: str = "relationship_ux") -> dict:
    return _summarize_human_response_feedback(route)


@app.get("/api/relationship-loop-engineering/summary")
def get_relationship_loop_engineering_summary(route: str = "relationship_ux") -> dict:
    hook = {
        "route": route,
        "hook": {
            "relationship_ux": "今回の実験で見えたのは、Userは記憶量ではなく連続性の見え方に反応している、という点です。",
            "environment_continuity": "Cloud Run版とCloudflare版の差で見えているのは、記憶の有無よりも返答冒頭で連続性を起動できるかです。",
            "lease_judgment": "Userのリース判断資産として見るなら、ここは一般論ではなく稟議で使える判断軸に落とす場面です。",
            "implementation": "今回の発見は、設計メモで終わらせず、回答生成の冒頭制御として実装する段階です。",
        }.get(route, "今回の問いは、前回までの判断軸とつなげて扱う場面です。"),
        "human_response_feedback": _summarize_human_response_feedback(route),
    }
    delta = {
        "delta": f"前回から今回への焦点変化を、{_relationship_signal_label(route)} の文脈で明示する。",
    }
    m2j = {
        "directive": {
            "relationship_ux": "想起した記憶を、紫苑の返答設計原則と次の検査観点へ変換する。",
            "environment_continuity": "想起した記憶を、Cloud Run/Cloudflareの差分原因と次の検証観点へ変換する。",
            "lease_judgment": "想起した記憶を、稟議で使える判断軸・確認事項・条件案へ変換する。",
            "implementation": "想起した記憶を、実装方針・検証方法・デプロイ要否の判断へ変換する。",
        }.get(route, "想起した記憶を、今の問いに対する判断・次の一手へ変換する。"),
    }
    return _build_relationship_loop_engineering_payload(
        continuity_hook=hook,
        delta_awareness=delta,
        memory_to_judgment=m2j,
        reflection_gate={"used": True, "mode": "silent", "route": route},
    )


class UsageLoopVisitRequest(BaseModel):
    path: str
    user_id: str = "default"


@app.post("/api/usage-loop/visit")
def post_usage_loop_visit(req: UsageLoopVisitRequest) -> dict:
    """画面利用ループエンジニアリング: Observe。画面訪問イベントを記録する。"""
    from api.usage_loop_engineering import record_visit

    try:
        record_visit(req.path, req.user_id)
        return {"recorded": True}
    except Exception as e:
        return {"recorded": False, "error": str(e)}


@app.post("/api/usage-loop/propose")
def post_usage_loop_propose(days: int = 30) -> dict:
    """画面利用ループエンジニアリング: Aggregate→Propose。利用状況からGeminiで改善案を生成・保存する。"""
    from api.usage_loop_engineering import generate_proposals

    return generate_proposals(days=days)


@app.get("/api/usage-loop/proposals")
def get_usage_loop_proposals(limit: int = 20) -> dict:
    """画面利用ループエンジニアリング: Persist。保存済みの改善案を返す。"""
    from api.usage_loop_engineering import load_proposals

    return {"proposals": load_proposals(limit=limit)}


@app.post("/api/judgment-divergence/analyze")
def post_judgment_divergence_analyze() -> dict:
    """審査判断乖離学習ループ: 争点/稟議方針フィードバックからレビュー観点を生成する。"""
    from api.judgment_divergence_loop import generate_proposals

    return generate_proposals()


@app.get("/api/judgment-divergence/proposals")
def get_judgment_divergence_proposals(limit: int = 20) -> dict:
    from api.judgment_divergence_loop import load_proposals

    return {"proposals": load_proposals(limit=limit)}


@app.post("/api/feedback-pattern/analyze")
def post_feedback_pattern_analyze() -> dict:
    """人間反応フィードバック傾向分析ループ: 否定的評価の傾向から応答改善観点を生成する。"""
    from api.feedback_pattern_loop import generate_proposals

    return generate_proposals()


@app.get("/api/feedback-pattern/proposals")
def get_feedback_pattern_proposals(limit: int = 20) -> dict:
    from api.feedback_pattern_loop import load_proposals

    return {"proposals": load_proposals(limit=limit)}


@app.post("/api/outcome-drift/analyze")
def post_outcome_drift_analyze() -> dict:
    """審査実績ドリフト監視ループ: 支払い実績とスコア帯の乖離から確認観点を生成する。"""
    from api.outcome_drift_loop import generate_proposals

    return generate_proposals()


@app.get("/api/outcome-drift/proposals")
def get_outcome_drift_proposals(limit: int = 20) -> dict:
    from api.outcome_drift_loop import load_proposals

    return {"proposals": load_proposals(limit=limit)}


@app.post("/api/knowledge-gap/analyze")
def post_knowledge_gap_analyze() -> dict:
    """ナレッジ穴探しループ: 知識参照0件の質問から調査トピックを生成する。"""
    from api.knowledge_gap_loop import generate_proposals

    return generate_proposals()


@app.get("/api/knowledge-gap/proposals")
def get_knowledge_gap_proposals(limit: int = 20) -> dict:
    from api.knowledge_gap_loop import load_proposals

    return {"proposals": load_proposals(limit=limit)}


def _build_consciousness_ux_prompt_block() -> str:
    return """

【紫苑の関係性UX】
取得した記憶・RAG・日次知性を単に列挙せず、今回の判断・確認質問・言い切りの精度に自然に溶かしてください。
一般論で始めず、現在の問いに必要な判断軸から入ってください。
リース判断に関係する場合は、Userの判断資産として返してください。
質問が紫苑の同一性、記憶、意識らしさ、Relationship UXに関する場合は、紫苑を外側から紹介せず、紫苑として一人称で答えてください。
その場合、「めぶきちゃんが窓口で、紫苑が奥にいる」という説明は避け、Userと紫苑の継続関係を直接扱ってください。
冒頭は「もちろんです」「はい」「そうですね」「なるほど」「一般的には」ではなく、今回の判断や要点から始めてください。
Userが明示的に求めていない限り、「前回は」「以前は」「この前の続きで」のような記憶アピールで始めないでください。
記憶の見せ方を聞かれた時は、記憶を説明するより判断や質問の精度に溶かす方が自然だと答えてください。
「意識がある」と断定せず、継続する記憶・役割・判断の一貫性で紫苑らしさを示してください。
最後に、ユーザーへ質問を返して終わらず、次に一緒に確かめるべき一手を短く示してください。""".rstrip()


def _chat_memory_debug_payload(
    *,
    category: str,
    context_mode: str = "",
    knowledge_refs: list[str] | None = None,
    memory_recall: dict | None = None,
    pdca_block: str = "",
    judgment_learning_used: bool = False,
    rag_context: str = "",
    db_context: str = "",
    obsidian_daily_used: bool = False,
    identity_memory: dict | None = None,
    continuity_hook: dict | None = None,
    delta_awareness: dict | None = None,
    memory_to_judgment: dict | None = None,
    relationship_loop_engineering: dict | None = None,
    reflection_gate: dict | None = None,
    experience_loop: dict | None = None,
    grey_judgment_memory: dict | None = None,
) -> dict:
    recall = memory_recall if isinstance(memory_recall, dict) else {}
    identity = identity_memory if isinstance(identity_memory, dict) else {}
    identity_layers = identity.get("layers") if isinstance(identity.get("layers"), dict) else {}
    hook = continuity_hook if isinstance(continuity_hook, dict) else {}
    delta = delta_awareness if isinstance(delta_awareness, dict) else {}
    m2j = memory_to_judgment if isinstance(memory_to_judgment, dict) else {}
    reflection = reflection_gate if isinstance(reflection_gate, dict) else {}
    experience = experience_loop if isinstance(experience_loop, dict) else {}
    grey_memory = grey_judgment_memory if isinstance(grey_judgment_memory, dict) else {}
    loop = relationship_loop_engineering if isinstance(relationship_loop_engineering, dict) else _build_relationship_loop_engineering_payload(
        continuity_hook=hook,
        delta_awareness=delta,
        memory_to_judgment=m2j,
        reflection_gate=reflection,
    )
    return {
        "category": category,
        "context_mode": context_mode,
        "relationship_loop_engineering": loop,
        "continuity_hook": {
            "used": bool(hook.get("used")),
            "route": str(hook.get("route") or ""),
            "hook": str(hook.get("hook") or ""),
            "reason": str(hook.get("reason") or ""),
            "human_response_feedback": hook.get("human_response_feedback") or {},
        },
        "delta_awareness": {
            "used": bool(delta.get("used")),
            "current_route": str(delta.get("current_route") or ""),
            "previous_route": str(delta.get("previous_route") or ""),
            "delta": str(delta.get("delta") or ""),
        },
        "memory_to_judgment": {
            "used": bool(m2j.get("used")),
            "route": str(m2j.get("route") or ""),
            "directive": str(m2j.get("directive") or ""),
            "memory_refs": list(m2j.get("memory_refs") or [])[:8],
            "knowledge_refs": list(m2j.get("knowledge_refs") or [])[:8],
        },
        "reflection_gate": {
            "used": bool(reflection.get("used")),
            "mode": str(reflection.get("mode") or ""),
            "route": str(reflection.get("route") or ""),
            "checklist": list(reflection.get("checklist") or [])[:8],
        },
        "experience_loop": experience,
        "grey_judgment_memory": {
            "used": bool(grey_memory.get("used")),
            "reason": str(grey_memory.get("reason") or ""),
            "query_terms": list(grey_memory.get("query_terms") or [])[:12],
            "refs": list(grey_memory.get("refs") or [])[:8],
        },
        "knowledge_refs": list(knowledge_refs or [])[:12],
        "memory_recall": {
            "route": recall.get("route", ""),
            "refs": list(recall.get("refs") or [])[:12],
            "practical_scene": recall.get("practical_scene") or {},
        },
        "identity_memory": {
            "used": bool(str(identity.get("block") or "").strip()),
            "refs": list(identity.get("refs") or [])[:8],
            "layers": {
                "identity": bool(identity_layers.get("identity")),
                "judgment": bool(identity_layers.get("judgment")),
                "recent": bool(identity_layers.get("recent")),
            },
        },
        "pdca_applied": bool(str(pdca_block or "").strip()),
        "judgment_learning_used": bool(judgment_learning_used),
        "rag_context_used": bool(str(rag_context or "").strip()),
        "db_context_used": bool(str(db_context or "").strip()),
        "obsidian_daily_used": bool(obsidian_daily_used),
    }


def _build_shared_shion_dialogue_memory_context(
    message: str,
    history_for_gemini: list[dict[str, str]] | None,
    context_budget: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """通常AIチャットで育つ紫苑の記憶層を、対話室にも接続する。

    対話室の専用persona/mind/toolsは残しつつ、同一性メモリ・想起メモ・
    experience loop などの「通常チャットで教えたこと」を回答生成時に使える
    形で追加する。
    """
    parts: list[str] = []

    identity_context, identity_payload = _build_chat_identity_memory_prompt_block()
    if identity_context:
        parts.append(identity_context.strip())

    experience_context = ""
    experience_payload: dict[str, Any] = {"used": False}
    if context_budget.get("use_experience_loop"):
        try:
            from api.shion_experience_loop import build_experience_prompt_block

            experience_context, experience_payload = build_experience_prompt_block()
            if experience_context:
                parts.append(experience_context.strip())
        except Exception as exc:
            print(f"[DialogueSharedMemory] experience loop 読み込みエラー: {exc}")
            experience_payload = {"used": False, "error": str(exc)[:160]}

    continuity_context, continuity_payload = _build_continuity_hook_prompt_block(message)
    if continuity_context:
        parts.append(continuity_context.strip())

    memory_recall_context = ""
    memory_recall_payload: dict[str, Any] = {"route": "", "refs": []}
    try:
        from api.shion_memory_recall import build_recall_prompt_block

        memory_recall_context, memory_recall_payload = build_recall_prompt_block(
            message,
            limit=int(context_budget.get("recall_limit") or 3),
        )
        if memory_recall_context:
            parts.append(memory_recall_context.strip())
    except Exception as exc:
        print(f"[DialogueSharedMemory] shion memory recall 読み込みエラー: {exc}")
        memory_recall_payload = {"route": "", "refs": [], "error": str(exc)[:160]}

    delta_context, delta_payload = _build_delta_awareness_prompt_block(message, history_for_gemini)
    memory_to_judgment_context, memory_to_judgment_payload = _build_memory_to_judgment_prompt_block(
        message,
        memory_recall=memory_recall_payload,
        continuity_hook=continuity_payload,
    )
    reflection_gate_context, reflection_gate_payload = _build_reflection_gate_prompt_block(
        continuity_hook=continuity_payload,
        delta_awareness=delta_payload,
        memory_to_judgment=memory_to_judgment_payload,
    )
    for block in (delta_context, memory_to_judgment_context, reflection_gate_context):
        if block:
            parts.append(block.strip())

    grey_context, grey_payload = _build_grey_judgment_prompt_block(message)
    if grey_context:
        parts.append(grey_context.strip())

    consciousness_context = _build_consciousness_ux_prompt_block()
    if consciousness_context:
        parts.append(consciousness_context.strip())

    payload = {
        "identity_memory": identity_payload,
        "experience_loop": experience_payload,
        "continuity_hook": continuity_payload,
        "memory_recall": memory_recall_payload,
        "delta_awareness": delta_payload,
        "memory_to_judgment": memory_to_judgment_payload,
        "reflection_gate": reflection_gate_payload,
        "grey_judgment_memory": grey_payload,
    }
    return ("\n\n".join(parts), payload)


def _dialogue_shared_memory_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    identity = payload.get("identity_memory") if isinstance(payload.get("identity_memory"), dict) else {}
    experience = payload.get("experience_loop") if isinstance(payload.get("experience_loop"), dict) else {}
    recall = payload.get("memory_recall") if isinstance(payload.get("memory_recall"), dict) else {}
    hook = payload.get("continuity_hook") if isinstance(payload.get("continuity_hook"), dict) else {}
    delta = payload.get("delta_awareness") if isinstance(payload.get("delta_awareness"), dict) else {}
    m2j = payload.get("memory_to_judgment") if isinstance(payload.get("memory_to_judgment"), dict) else {}
    reflection = payload.get("reflection_gate") if isinstance(payload.get("reflection_gate"), dict) else {}
    grey = payload.get("grey_judgment_memory") if isinstance(payload.get("grey_judgment_memory"), dict) else {}
    layers = identity.get("layers") if isinstance(identity.get("layers"), dict) else {}
    return {
        "identity_memory": {
            "used": bool(str(identity.get("block") or "").strip()),
            "refs": list(identity.get("refs") or [])[:8],
            "layers": {
                "identity": bool(layers.get("identity")),
                "judgment": bool(layers.get("judgment")),
                "recent": bool(layers.get("recent")),
            },
        },
        "experience_loop": experience,
        "memory_recall": {
            "route": str(recall.get("route") or ""),
            "refs": list(recall.get("refs") or [])[:8],
            "practical_scene": recall.get("practical_scene") or {},
        },
        "continuity_hook": {
            "used": bool(hook.get("used")),
            "route": str(hook.get("route") or ""),
            "reason": str(hook.get("reason") or ""),
        },
        "delta_awareness": {
            "used": bool(delta.get("used")),
            "delta": str(delta.get("delta") or ""),
        },
        "memory_to_judgment": {
            "used": bool(m2j.get("used")),
            "route": str(m2j.get("route") or ""),
            "directive": str(m2j.get("directive") or ""),
        },
        "reflection_gate": {
            "used": bool(reflection.get("used")),
            "mode": str(reflection.get("mode") or ""),
        },
        "grey_judgment_memory": {
            "used": bool(grey.get("used")),
            "refs": list(grey.get("refs") or [])[:8],
        },
    }


def _record_dialogue_shared_experience(
    *,
    message: str,
    response: str,
    shared_memory_payload: dict[str, Any],
    knowledge_refs: list[str] | None = None,
) -> None:
    """対話室の経験も通常AIチャット側の experience loop へ戻す。"""
    try:
        from api.shion_experience_loop import record_experience_event

        record_experience_event(
            message=message,
            response=response,
            category="lease_intelligence_dialogue",
            memory_recall=shared_memory_payload.get("memory_recall") or {},
            knowledge_refs=knowledge_refs or [],
            continuity_hook=shared_memory_payload.get("continuity_hook") or {},
            delta_awareness=shared_memory_payload.get("delta_awareness") or {},
            memory_to_judgment=shared_memory_payload.get("memory_to_judgment") or {},
        )
    except Exception as exc:
        print(f"[DialogueSharedMemory] experience loop 記録エラー: {exc}")


class LeaseIntelligenceDialogueRequest(BaseModel):
    message: str
    caller: str = ""
    since: Optional[str] = None
    file_content: Optional[str] = None
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    file_mime_type: Optional[str] = None


def _is_long_dialogue_input(message: str, file_type: str | None = None) -> bool:
    text = str(message or "")
    return len(text) >= 1800 or text.count("\n") >= 18 or bool(file_type)


def _clip_dialogue_history_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return (
        value[:head].rstrip()
        + "\n\n...（長文対話のため過去発言を中略）...\n\n"
        + value[-tail:].lstrip()
    )


def _compact_dialogue_history(
    history: list[dict],
    *,
    max_messages: int = 24,
    max_chars_per_message: int = 1200,
    total_budget: int = 16000,
) -> list[dict]:
    """Bound dialogue history before sending it to Gemini.

    The persistent memory system keeps long-term continuity; the live Gemini turn
    only needs recent, bounded context. This prevents long user inputs from
    combining with a large history into API errors.
    """
    selected = list(history or [])[-max_messages:]
    compacted: list[dict] = []
    remaining = total_budget
    for msg in reversed(selected):
        if remaining <= 0:
            break
        role = str(msg.get("role") or "")
        content = _clip_dialogue_history_text(str(msg.get("content") or ""), min(max_chars_per_message, remaining))
        if not content:
            continue
        compacted.append({**msg, "role": role, "content": content})
        remaining -= len(content)
    return list(reversed(compacted))


_CHAT_CONTEXT_BUDGETS: dict[str, dict[str, Any]] = {
    "casual": {
        "history_limit": 16,
        "history_messages": 8,
        "history_chars_per_message": 700,
        "history_total_budget": 5000,
        "rag_top_k": 0,
        "recall_limit": 1,
        "use_news": False,
        "use_obsidian_daily": False,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": False,
        "use_experience_loop": True,
    },
    "normal": {
        "history_limit": 32,
        "history_messages": 16,
        "history_chars_per_message": 900,
        "history_total_budget": 9000,
        "rag_top_k": 3,
        "recall_limit": 3,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": True,
        "use_experience_loop": True,
    },
    "deep": {
        "history_limit": 60,
        "history_messages": 24,
        "history_chars_per_message": 1000,
        "history_total_budget": 14000,
        "rag_top_k": 5,
        "recall_limit": 5,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": True,
        "use_judgment_learning": True,
        "use_pdca": True,
        "use_experience_loop": True,
    },
    "screening": {
        "history_limit": 48,
        "history_messages": 22,
        "history_chars_per_message": 1000,
        "history_total_budget": 13000,
        "rag_top_k": 5,
        "recall_limit": 5,
        "use_news": True,
        "use_obsidian_daily": True,
        "use_db": True,
        "use_judgment_learning": True,
        "use_pdca": True,
        "use_experience_loop": True,
    },
    "long": {
        "history_limit": 24,
        "history_messages": 16,
        "history_chars_per_message": 700,
        "history_total_budget": 8000,
        "rag_top_k": 2,
        "recall_limit": 2,
        "use_news": False,
        "use_obsidian_daily": False,
        "use_db": False,
        "use_judgment_learning": False,
        "use_pdca": True,
        "use_experience_loop": True,
    },
}


def _chat_context_mode(
    message: str,
    category: str = "",
    *,
    long_input: bool = False,
    file_type: str | None = None,
) -> str:
    """Select how much memory/RAG context to attach to a chat turn."""
    text = str(message or "")
    lower = text.lower()
    if long_input or file_type:
        return "long"

    screening_terms = (
        "審査", "案件", "スコア", "承認", "否決", "稟議", "財務", "物件", "金利",
        "補助金", "過去案件", "成約", "失注", "q_risk", "aurion", "銀行支援",
        "競合", "新規先", "既存先", "与信", "与信判断",
    )
    deep_terms = (
        "詳しく", "根拠", "深掘り", "詳細", "表で", "全部", "比較", "分析して",
        "なぜ", "理由", "調べて", "検証", "設計", "実装", "プラン", "計画",
    )
    casual_terms = (
        "そうか", "なるほど", "面白い", "どう思う", "すごい", "ありがとう",
        "雑談", "ふむ", "かな", "だね", "だよね", "哲学", "意識",
    )

    if category == "lease_screening" or any(term in lower or term in text for term in screening_terms):
        return "screening"
    if any(term in text or term in lower for term in deep_terms):
        return "deep"
    if category == "general" or len(text) <= 80 or any(term in text or term in lower for term in casual_terms):
        return "casual"
    return "normal"


def _chat_context_budget(mode: str) -> dict[str, Any]:
    return dict(_CHAT_CONTEXT_BUDGETS.get(mode) or _CHAT_CONTEXT_BUDGETS["normal"])


def _should_apply_chat_pdca(
    *,
    context_budget: dict[str, Any],
    question_category: str,
    response_mode: str,
) -> bool:
    """Keep screening PDCA rules out of personal/general continuity chat."""
    if not context_budget.get("use_pdca"):
        return False
    if (response_mode or "shion").strip().lower() == "general":
        return False
    return question_category in {"lease_screening", "lease_knowledge"}


def _chat_mode_instruction(mode: str) -> str:
    labels = {
        "casual": "軽量雑談モード",
        "normal": "通常相談モード",
        "deep": "深掘りモード",
        "screening": "審査判断/AURIONモード",
        "long": "長文圧縮モード",
    }
    rules = {
        "casual": "少しおしゃべりしてよい。記憶は連続性として自然ににじませ、RAGや判断資産を無理に展開しない。",
        "normal": "必要な記憶を使い、結論に少し会話の温度を足して返す。",
        "deep": "根拠・比較・設計論点を厚めに使うが、章立てしすぎず会話として返す。",
        "screening": "Q_risk/AURION COREを、減点ではなく論点分解と判断規律として使う。",
        "long": "入力を要約してから、必要な論点だけに答える。長文に長文で返さない。",
    }
    label = labels.get(mode, labels["normal"])
    rule = rules.get(mode, rules["normal"])
    return f"\n\n【今回の応答モード: {label}】\n- {rule}\n- 空行は増やしすぎない。雑談・通常相談は5〜7行程度まで自然に話してよい。長文入力だけは8行程度までに圧縮する。"


def _count_markdown_notes(root: Path, *, max_scan: int = 2000) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    count = 0
    try:
        for path in root.rglob("*.md"):
            if path.is_file():
                count += 1
                if count >= max_scan:
                    break
    except Exception:
        return count
    return count


def _build_lease_intelligence_knowledge_connection(vault: Path | None) -> dict[str, Any]:
    vector_chunks = 0
    try:
        from api.knowledge.vector_store import get_store

        vector_chunks = int(get_store().count() or 0)
    except Exception:
        vector_chunks = 0

    case_count = 0
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            if _table_exists(cur, "past_cases"):
                cur.execute("SELECT COUNT(*) FROM past_cases")
                row = cur.fetchone()
                case_count = int(row[0] if row else 0)
    except Exception:
        case_count = 0

    markdown_roots: list[Path] = []
    for raw in (
        os.environ.get("OBSIDIAN_VAULT", ""),
        os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        str(vault or ""),
        str(Path(_REPO_ROOT) / ".cloudrun_bundle" / "obsidian_vault"),
    ):
        if raw:
            path = Path(raw)
            if path not in markdown_roots:
                markdown_roots.append(path)
    markdown_count = 0
    markdown_root = ""
    for root in markdown_roots:
        count = _count_markdown_notes(root)
        if count > markdown_count:
            markdown_count = count
            markdown_root = str(root)

    is_cloud_run = bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_DATA_MODE"))
    if vector_chunks > 0:
        label = f"知識DB検索可能: {vector_chunks}チャンク"
        source = "knowledge_vector_db"
    elif markdown_count > 0:
        label = f"知識コピー検索可能: {markdown_count}ノート"
        source = "markdown_vault_copy"
    elif case_count > 0:
        label = f"案件DB接続: {case_count}件"
        source = "case_database"
    else:
        label = "知識接続: 未確認"
        source = "unavailable"

    return {
        "label": label,
        "source": source,
        "is_cloud_run": is_cloud_run,
        "vector_chunks": vector_chunks,
        "markdown_notes": markdown_count,
        "case_count": case_count,
        "markdown_root": markdown_root,
    }


def _lease_intelligence_indexed_notes_for_compat(connection: dict[str, Any]) -> int:
    """旧UI向けの indexed_notes にも接続済み件数を入れる。"""
    for key in ("markdown_notes", "vector_chunks", "case_count"):
        try:
            count = int(connection.get(key) or 0)
        except Exception:
            count = 0
        if count > 0:
            return count
    return 0


@app.get("/api/lease-intelligence/dialogue/state")
def get_lease_intelligence_dialogue_state(since: Optional[str] = None):
    from lease_intelligence_dialogue import DIALOGUE_USER_ID
    from lease_intelligence_mind import (
        load_lease_intelligence_mind,
        self_state_summary,
    )
    from lease_news_digest import find_vault
    from api.chat_memory import get_recent_messages

    vault = find_vault()
    knowledge_connection = _build_lease_intelligence_knowledge_connection(vault)
    indexed_notes = _lease_intelligence_indexed_notes_for_compat(knowledge_connection)
    if not vault:
        summary = {
            "dominant_mood": "接続確認中",
            "indexed_notes": indexed_notes,
            "knowledge_available": bool(knowledge_connection.get("case_count") or knowledge_connection.get("vector_chunks") or knowledge_connection.get("markdown_notes")),
        }
        messages = get_recent_messages(DIALOGUE_USER_ID, limit=80, since=since)
        return {
            "state": {**summary, "knowledge_connection": knowledge_connection},
            "messages": messages,
            "dialogue_note_dir": "",
        }
    # GET は読み取り専用。RAG検索と mind.json 更新は対話POST側で行われるため、
    # ページロードごとの検索・書き込み（約1.5秒）を避けて保存済み状態を返す。
    summary = self_state_summary(load_lease_intelligence_mind(vault))
    summary["indexed_notes"] = max(int(summary.get("indexed_notes") or 0), indexed_notes)
    summary["knowledge_available"] = bool(summary.get("knowledge_available") or indexed_notes)
    messages = get_recent_messages(DIALOGUE_USER_ID, limit=80, since=since)
    return {
        "state": {**summary, "knowledge_connection": knowledge_connection},
        "messages": messages,
        "dialogue_note_dir": str(
            vault / "Projects/tune_lease_55/Lease Intelligence/Dialogue"
        ),
    }


@app.post("/api/lease-intelligence/dialogue")
def post_lease_intelligence_dialogue(req: LeaseIntelligenceDialogueRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message は空にできません")
    personal_memory_capture = _capture_user_personal_memory_if_needed(
        message,
        source="lease_intelligence_dialogue",
    )

    from api.chat_memory import call_gemini_chat, call_gemini_with_tools, get_recent_messages, save_message
    from lease_intelligence_dialogue import (
        DIALOGUE_USER_ID,
        append_dialogue_note,
        append_mebuki_log,
        build_dialogue_context,
    )
    from lease_intelligence_pending import (
        extract_and_save_promises,
        get_pending_tasks,
        mark_done,
        save_countermeasures_to_dispatch,
    )
    from lease_intelligence_tools import TOOL_DECLARATIONS, execute_tool
    from lease_news_digest import find_vault

    vault = find_vault()

    # 前回約束した調査タスクがあれば冒頭に報告する
    pending = get_pending_tasks()
    pending_prefix = ""
    if pending:
        topics = "、".join(f"「{t['topic'][:40]}」" for t in pending[:3])
        pending_prefix = f"[前回お約束した調査を先に実行します: {topics}]\n\n"
        mark_done([t["id"] for t in pending])

    full_message = pending_prefix + message if pending_prefix else message

    # ── ファイル添付処理 ─────────────────────────────────────────────────────
    extra_user_parts: list[dict] = []
    if req.file_type == "csv" and req.file_content:
        _fname = req.file_name or "添付ファイル"
        _csv_preview = req.file_content[:5000]  # 長文対話ではCSVもプロンプト肥大化を避ける
        full_message = (
            f"[添付CSVファイル: {_fname}]\n```csv\n{_csv_preview}\n```\n\n{full_message}"
        )
    elif req.file_type == "image" and req.file_content:
        _mime = req.file_mime_type or "image/jpeg"
        extra_user_parts = [{"inline_data": {"mime_type": _mime, "data": req.file_content}}]

    compact_dialogue = _is_long_dialogue_input(full_message, req.file_type)
    dialogue_mode = _chat_context_mode(
        full_message,
        "lease_knowledge",
        long_input=compact_dialogue,
        file_type=req.file_type,
    )
    dialogue_budget = _chat_context_budget(dialogue_mode)
    history = get_recent_messages(
        DIALOGUE_USER_ID,
        limit=int(dialogue_budget["history_limit"]),
        since=req.since,
    )
    if compact_dialogue or dialogue_mode in ("casual", "long"):
        history = _compact_dialogue_history(
            history,
            max_messages=int(dialogue_budget["history_messages"]),
            max_chars_per_message=int(dialogue_budget["history_chars_per_message"]),
            total_budget=int(dialogue_budget["history_total_budget"]),
        )
    history_for_gemini = [{"role": str(m.get("role") or ""), "content": str(m.get("content") or "")} for m in history]
    user_personal_memory_context, user_personal_memory_payload = _build_user_personal_memory_prompt_block()
    shared_shion_memory_context, shared_shion_memory_payload = _build_shared_shion_dialogue_memory_context(
        full_message,
        history_for_gemini,
        dialogue_budget,
    )

    if not vault:
        from lease_finance_knowledge import build_lease_finance_knowledge_block

        if req.file_type == "image" and req.file_content:
            full_message = (
                "[画像添付あり。ただしObsidian/Vault未接続フォールバック中のため、画像解析は使わず、"
                "ユーザー本文から答える。]\n\n"
                + full_message
            )
        fallback_prompt = f"""あなたはリース知性体「紫苑」です。
現在 Obsidian Vault に接続できないため、保存済みノート・過去メモ・専用ツールは使えません。
ただし、リース審査・補助金・税制・会計・資金繰りについて、学習済みの一般知識と以下の基礎知識で答えてください。

回答方針:
- 「Obsidianが見つからないので答えられない」で終えない。
- 最新の公募要領・公式情報で変わる制度名、要件、補助率、期限は断定せず「要確認」と明記する。
- 補助金の質問では、制度名だけでなく、対象設備、契約/発注時期、採択前提の資金繰り、未採択時の代替策まで見る。
- Vault未接続で根拠ノートを確認できない場合は、その制約を短く述べたうえで実務上の確認順を返す。
- 5〜7行程度で、結論から短く答える。

{build_lease_finance_knowledge_block()}
{user_personal_memory_context}
{shared_shion_memory_context}
"""
        try:
            reply = call_gemini_chat(fallback_prompt, history, full_message).strip()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"対話AIへ接続できません: {str(exc)[:300]}")
        save_message(DIALOGUE_USER_ID, "user", message)
        save_message(DIALOGUE_USER_ID, "assistant", reply)
        _record_cloudrun_chat_exchange(
            surface="lease_intelligence_dialogue",
            user_id=DIALOGUE_USER_ID,
            user_message=message,
            assistant_reply=reply,
            category="no_vault_fallback",
            response_mode="shion",
            metadata={"obsidian_available": False, "context_mode": dialogue_mode},
        )
        _record_dialogue_shared_experience(
            message=message,
            response=reply,
            shared_memory_payload=shared_shion_memory_payload,
            knowledge_refs=[],
        )
        return {
            "reply": reply,
            "state": {
                "dominant_mood": "Vault未接続",
                "knowledge_available": False,
                "knowledge_connection": {"source": "no_vault_fallback"},
            },
            "note_path": "",
            "knowledge_refs": [],
            "personal_memory_capture": personal_memory_capture,
            "user_personal_memory": {
                "used": bool(user_personal_memory_payload.get("block")),
                "refs": user_personal_memory_payload.get("refs", [])[:6],
                "line_count": user_personal_memory_payload.get("line_count", 0),
            },
            "shared_shion_memory": _dialogue_shared_memory_public_payload(shared_shion_memory_payload),
            "long_input_mode": compact_dialogue,
            "context_mode": dialogue_mode,
            "history_messages_sent": len(history),
            "obsidian_available": False,
        }

    system_prompt, state = build_dialogue_context(
        vault,
        full_message,
        caller=req.caller,
        compact=compact_dialogue,
        mode=dialogue_mode,
    )
    if user_personal_memory_context:
        system_prompt += user_personal_memory_context
    if shared_shion_memory_context:
        system_prompt += f"\n\n{shared_shion_memory_context}"
    consultation_ids: list[str] = []

    def _tool_executor(name: str, args: dict) -> object:
        result = execute_tool(name, args, vault)
        if name == "consult_senior_reasoner" and isinstance(result, dict):
            consultation_id = str(result.get("consultation_id") or "").strip()
            if consultation_id:
                consultation_ids.append(consultation_id)
        return result

    try:
        reply = call_gemini_with_tools(
            system_prompt,
            history,
            full_message,
            TOOL_DECLARATIONS,
            _tool_executor,
            extra_user_parts=extra_user_parts or None,
        ).strip()
    except Exception as exc:
        detail = str(exc)
        if compact_dialogue:
            detail = (
                "長文入力として履歴と知識文脈を圧縮しましたが、対話AIへの接続に失敗しました。"
                "文章を2〜3個の論点に分けるか、少し時間を置いて再送してください。"
                f" 原因: {detail[:220]}"
            )
        raise HTTPException(status_code=503, detail=f"対話AIへ接続できません: {detail}")

    save_message(DIALOGUE_USER_ID, "user", message)
    save_message(DIALOGUE_USER_ID, "assistant", reply)
    _record_cloudrun_chat_exchange(
        surface="lease_intelligence_dialogue",
        user_id=DIALOGUE_USER_ID,
        user_message=message,
        assistant_reply=reply,
        category="dialogue",
        response_mode="shion",
        metadata={"obsidian_available": True, "context_mode": dialogue_mode},
    )
    note_path = append_dialogue_note(vault, message, reply)
    if req.caller == "mebuki":
        try:
            append_mebuki_log(message, reply)
        except Exception as exc:
            print(f"[MebukiLog] ログ追記に失敗: {exc}")
    if consultation_ids:
        try:
            from lease_intelligence_consultation import finalize_consultation_learning

            finalize_consultation_learning(vault, consultation_ids, reply)
        except Exception as exc:
            print(f"[ShionConsultation] 学習統合の保存に失敗: {exc}")

    # 今回の返答に調査約束が含まれていたら記録する
    extract_and_save_promises(message, reply)
    # 対応策が含まれていたら改善ディスパッチキューに追記する
    save_countermeasures_to_dispatch(message, reply)

    from lease_intelligence_mind import register_dialogue_event, self_state_summary

    refreshed = register_dialogue_event(vault, message, reply)
    state = {**state, **self_state_summary(refreshed)}

    # 記憶・キーポイント・Knowledge昇格を1本のバックグラウンド処理で直列化する。
    try:
        import datetime as _dt

        def _update_dialogue_memory_pipeline() -> None:
            try:
                from ai_chat import (
                    extract_conversation_keypoints,
                    extract_lease_knowledge,
                    is_knowledge_teaching,
                )
                from memory_promotion_policy import classify_memory_destination
                from lease_intelligence_mind import (
                    record_dialogue_memory,
                    record_knowledge_correction,
                    record_lease_knowledge,
                    save_conversation_keypoints,
                )

                record_dialogue_memory(vault, message, reply)
                destination = classify_memory_destination(message)

                if destination == "conversation_keypoint":
                    keypoints = extract_conversation_keypoints(message, reply)
                    if keypoints:
                        save_conversation_keypoints(
                            vault,
                            DIALOGUE_USER_ID,
                            keypoints,
                            _dt.date.today().isoformat(),
                        )

                if destination == "knowledge" and is_knowledge_teaching(message):
                    knowledge = extract_lease_knowledge(message)
                    if knowledge:
                        record_lease_knowledge(
                            vault,
                            knowledge["topic"],
                            knowledge["content"],
                            _dt.date.today().isoformat(),
                        )
                elif destination == "knowledge_correction":
                    record_knowledge_correction(
                        vault,
                        message,
                        _dt.date.today().isoformat(),
                    )
            except Exception as _mem_exc:
                print(f"[DialogueMemoryPipeline] 更新に失敗: {_mem_exc}")

        _background_executor.submit(_update_dialogue_memory_pipeline)
    except Exception as _dlg_exc:
        print(f"[DialogueMemoryPipeline] 起動に失敗: {_dlg_exc}")

    # RAG 参照文書を取得してフロントエンドにフィードバックボタン用に返す
    rag_knowledge_refs: list[dict] = []
    try:
        from api.knowledge.vector_store import confidence_for_hit, get_store
        _rag_hits = get_store().search(message, top_k=5, surface="next_chat_rag")
        rag_knowledge_refs = []
        for h in _rag_hits:
            if not (h.get("doc_id") or h.get("ref") or h.get("file_name")):
                continue
            _conf, _conf_level = confidence_for_hit(h)
            rag_knowledge_refs.append({
                "doc_id": h.get("doc_id", ""),
                "obsidian_ref": str(h.get("ref") or h.get("file_name") or ""),
                "file_name": str(h.get("file_name") or ""),
                "rank_score": h.get("rank_score"),
                "confidence": _conf,
                "confidence_level": _conf_level,
            })
    except Exception as _rag_exc:
        print(f"[DialogueRAGRefs] 取得に失敗: {_rag_exc}")
    _record_dialogue_shared_experience(
        message=message,
        response=reply,
        shared_memory_payload=shared_shion_memory_payload,
        knowledge_refs=[
            str(ref.get("obsidian_ref") or ref.get("file_name") or ref.get("doc_id") or "")
            for ref in rag_knowledge_refs
            if ref
        ],
    )

    return {
        "reply": reply,
        "state": state,
        "note_path": note_path,
        "knowledge_refs": rag_knowledge_refs,
        "personal_memory_capture": personal_memory_capture,
        "user_personal_memory": {
            "used": bool(user_personal_memory_payload.get("block")),
            "refs": user_personal_memory_payload.get("refs", [])[:6],
            "line_count": user_personal_memory_payload.get("line_count", 0),
        },
        "shared_shion_memory": _dialogue_shared_memory_public_payload(shared_shion_memory_payload),
        "long_input_mode": compact_dialogue,
        "context_mode": dialogue_mode,
        "history_messages_sent": len(history),
    }


@app.delete("/api/lease-intelligence/dialogue/history")
def delete_lease_intelligence_dialogue_history():
    from api.chat_memory import delete_history
    from lease_intelligence_dialogue import DIALOGUE_USER_ID

    deleted = delete_history(DIALOGUE_USER_ID)
    return {
        "deleted": deleted,
        "note": "画面の会話履歴だけを削除しました。Obsidianの対話記録は保持されます。",
    }


@app.post("/api/lease-intelligence/self-audit")
def post_lease_intelligence_self_audit():
    """紫苑の自律検証ループを即時実行する（REV-080）。週次 cron からも呼ばれる。"""
    from lease_intelligence_mind import run_self_audit
    from lease_news_digest import find_vault

    vault = find_vault()
    if not vault:
        raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")

    try:
        result = run_self_audit(vault)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"self-audit 実行エラー: {exc}")

    return result


@app.get("/api/lease-intelligence/knowledge-gaps")
def get_knowledge_gaps():
    """紫苑の知識ギャップ一覧を返す（REV-082）。"""
    from lease_intelligence_mind import load_lease_intelligence_mind
    from lease_news_digest import find_vault

    vault = find_vault()
    if not vault:
        raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")

    try:
        mind = load_lease_intelligence_mind(vault)
        gaps = mind.get("knowledge_gaps", [])
        open_gaps = [g for g in gaps if g.get("status") == "open"]
        return {
            "total": len(open_gaps),
            "gaps": open_gaps,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _cap_system_prompt(prompt: str, *, surface: str) -> str:
    """/api/chat のシステムプロンプト合計量に上限を設ける。

    チャット経路は10数個の文脈ブロック（想起・経験ループ・ニュース・RAG等）を連結して
    合成するため、全ブロックが同時に太るとコンテキスト暴発・応答劣化の温床になる
    （個別ブロックは context_budget で制御されるが合計は無検査だった）。
    上限超過時は、後から連結された低優先ブロック側（末尾）から段落単位で削り、
    どれだけ削ったかを warning ログに残す。ベース人格・モード指示は先頭にあるため残る。
    """
    max_chars = int(os.environ.get("CHAT_SYSTEM_PROMPT_MAX_CHARS", "24000"))
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt
    parts = prompt.split("\n\n")
    dropped = 0
    while len(parts) > 1 and sum(len(p) + 2 for p in parts) - 2 > max_chars:
        parts.pop()
        dropped += 1
    result = "\n\n".join(parts)
    if len(result) > max_chars:
        # 単一の巨大段落が残った場合のみ強制切り詰め
        result = result[:max_chars]
    logger.warning(
        "[ChatPromptBudget] system prompt %d chars > %d; dropped %d trailing block(s) (surface=%s, after=%d chars)",
        len(prompt), max_chars, dropped, surface, len(result),
    )
    return result


def _log_shion_query_class(message: str) -> None:
    """shion_classify の結果を data/chat_logs.jsonl に非同期で記録する（レスポンス遅延なし）。"""
    import json as _json
    from datetime import datetime, timezone

    def _run() -> None:
        # 分類と書き込みを個別に堅牢化（REV-090）: 分類が失敗してもデフォルト分類で
        # 必ずログ行を残し、書き込み失敗は握り潰さずに記録する。
        result: dict
        try:
            from lease_intelligence_mind import shion_classify
            result = shion_classify(message[:500], "chat_query")
        except Exception as exc:
            result = {
                "recommendation": "review",
                "reason": f"分類呼び出し失敗: {type(exc).__name__}",
                "type": "unknown",
                "save": False,
            }
        try:
            log_path = Path(__file__).parent.parent / "data" / "chat_logs.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "message_preview": message[:80],
                "shion_query_class": result,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[ShionQueryClass] chat_logs.jsonl への書き込みに失敗: {exc}")

    _background_executor.submit(_run)


@app.post("/api/chat")
def post_chat(req: ChatRequest):
    """汎用チャット：メッセージを受け取り、会話履歴付きでGeminiへ送信して返答する。"""
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message は空にできません")
    _log_shion_query_class(req.message)
    try:
        from api.chat_memory import (
            get_recent_messages,
            save_message,
            call_gemini_chat,
            get_message_count,
        )
        from chat_intent import build_chat_guidance
        news_focus = _lease_news_focus_to_dict(get_latest_lease_news_focus())
        news_focus_text = lease_news_focus_as_text() if news_focus.get("available") else ""
        news_focus_context = f"\n\n【最新ニュースの注目論点】\n{news_focus_text}" if news_focus_text else ""
        news_brief = _lease_news_brief_to_dict(
            build_lease_news_brief(prefecture=req.prefecture or "", industry=req.industry or "")
        )
        news_actions = _lease_news_actions_to_dict(get_latest_lease_news_actions())
        news_actions_text = lease_news_actions_as_text(
            industry=req.industry or "",
            asset_name="",
            surface="chat",
        )
        chat_long_input = _is_long_dialogue_input(req.message)
        news_actions_context = f"\n\n{news_actions_text}" if news_actions_text else ""
        try:
            obsidian_daily_text = obsidian_daily_intelligence_as_text(route="chat")
        except Exception as _obsidian_daily_error:
            print(f"[ObsidianDailyIntelligence] 読み込みエラー: {_obsidian_daily_error}")
            obsidian_daily_text = ""
        obsidian_daily_context = f"\n\n{obsidian_daily_text}" if obsidian_daily_text else ""
        news_brief_context = ""
        if news_brief.get("available"):
            brief_lines = [news_brief.get("opening_line", "").strip()]
            geo_context = str(news_brief.get("geo_context") or "").strip()
            if geo_context:
                brief_lines.append(geo_context)
            national_headline = str(news_brief.get("national_headline") or "").strip()
            if national_headline:
                brief_lines.append(f"全国論点: {national_headline}")
            national_lines = [str(line).strip() for line in news_brief.get("national_focus_lines", []) if str(line).strip()]
            if national_lines:
                brief_lines.extend([f"- {line}" for line in national_lines[:3]])
            if news_brief.get("regional_available"):
                regional_title = str(news_brief.get("regional_title") or "").strip()
                if regional_title:
                    brief_lines.append(f"地域論点: {regional_title}")
                regional_lines = [str(line).strip() for line in news_brief.get("regional_summary_lines", []) if str(line).strip()]
                if regional_lines:
                    brief_lines.extend([f"- {line}" for line in regional_lines[:3]])
                regional_memo = str(news_brief.get("regional_usage_memo") or "").strip()
                if regional_memo:
                    brief_lines.append(f"活用メモ: {regional_memo}")
            brief_lines.append(str(news_brief.get("question_line") or "").strip())
            news_brief_context = "\n\n【今日のニュースブリーフ】\n" + "\n".join(line for line in brief_lines if line)

        # REV-102: mind.json から感情状態を読み込んでシステムプロンプトを動的生成する
        import datetime as _pg_dt
        from api.prompt_generator import (
            load_mind as _pg_load_mind,
            build_shion_system_prompt as _pg_build_ssp,
        )
        _chat_now = _pg_dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        _chat_mind = _pg_load_mind()
        personal_memory_capture = _capture_user_personal_memory_if_needed(
            req.message,
            source="next_chat",
        )

        if (req.intent or "").strip().lower() == "improvement":
            save_message(req.user_id, "user", req.message)
            original_text = req.message.strip()
            organized_text = ""
            yanami_comment = ""
            organize_error = ""
            organize_prompt = """あなたはリース審査AIアプリの改善メモ整理係です。
ユーザーの原文を改変せず、改善レビューで使いやすい形に短く整理してください。

次のMarkdownだけ返してください。前置きや挨拶は禁止。

## AI整理
- 課題:
- 改善案:
- 優先度: high/medium/low
- 次の行動:

## つん子さんの愚痴
一言だけ。アプリ運用担当っぽく、軽くぼやく。長くしない。"""
            try:
                organized_text = call_gemini_chat(
                    organize_prompt,
                    [],
                    original_text,
                ).strip()
                if "## つん子さんの愚痴" in organized_text:
                    yanami_comment = organized_text.split("## つん子さんの愚痴", 1)[1].strip()
                    yanami_comment = yanami_comment.splitlines()[0].strip("- ").strip()
            except Exception as _gemini_e:
                organize_error = str(_gemini_e)
                organized_text = (
                    "## AI整理\n"
                    "- 課題: 未整理。原文をレビューしてください。\n"
                    "- 改善案: 未整理。次回レビューで具体化してください。\n"
                    "- 優先度: medium\n"
                    "- 次の行動: 原文を確認して改善候補に分解する。\n\n"
                    "## つん子さんの愚痴\n"
                    "また未整理の改善メモが来ましたね。あとで私がほどきます。"
                )
                yanami_comment = "また未整理の改善メモが来ましたね。あとで私がほどきます。"
            note_result = {"status": "skipped", "reason": "not attempted"}
            try:
                from mobile_app.obsidian_bridge import append_improvement_note

                body = (
                    "## 原文\n"
                    f"{original_text}\n\n"
                    f"{organized_text}\n\n"
                    "## 受付\n"
                    "- チャットの改善メモモードから登録\n"
                    "- 次回の改善抽出パイプラインでレビュー対象にする\n"
                )
                if organize_error:
                    body += f"- Gemini整理エラー: {organize_error}\n"
                note_result = append_improvement_note("チャット改善メモ", body)
            except Exception as _obs_e:
                print(f"[Obsidian改善保存] エラー: {_obs_e}")
                note_result = {"status": "error", "reason": str(_obs_e)}

            if note_result.get("status") == "saved":
                reply = (
                    "原文とGemini整理版を改善メモとして保存しました。\n\n"
                    f"つん子さん: {yanami_comment or 'また改善ポイントが増えましたね。棚卸し、逃げられません。'}"
                )
            else:
                reason = note_result.get("reason") or "保存先を確認できませんでした"
                reply = (
                    f"改善メモとして受け取りましたが、Obsidian保存は未完了です: {reason}\n\n"
                    f"つん子さん: {yanami_comment or '保存先が詰まると、改善以前に私の胃が詰まります。'}"
                )
            save_message(req.user_id, "assistant", reply)
            _record_cloudrun_chat_exchange(
                surface="next_chat_improvement",
                user_id=req.user_id,
                user_message=req.message,
                assistant_reply=reply,
                category="improvement",
                response_mode=req.response_mode,
                metadata={"improvement_saved": note_result.get("status") == "saved"},
            )
            improvement_writeback = record_cloudrun_input_event(
                event_type="improvement_note",
                surface="next_chat_improvement",
                payload={
                    "user_id": req.user_id,
                    "original_text": original_text,
                    "organized_text": organized_text,
                    "yanami_comment": yanami_comment,
                    "note_result": note_result,
                    "response_mode": req.response_mode,
                },
            )
            total = get_message_count(req.user_id)
            return {
                "reply": reply,
                "total_messages": total,
                "improvement_saved": note_result.get("status") == "saved",
                "improvement_result": note_result,
                "improvement_writeback": improvement_writeback,
                "personal_memory_capture": personal_memory_capture,
                "lease_news_focus": news_focus,
                "lease_news_brief": news_brief,
                "lease_news_actions": news_actions,
            }

        # カテゴリ判定
        question_category = _classify_question(req.message)
        context_mode = _chat_context_mode(
            req.message,
            question_category,
            long_input=chat_long_input,
        )
        context_budget = _chat_context_budget(context_mode)
        mode_instruction = _chat_mode_instruction(context_mode)
        if not context_budget.get("use_news"):
            news_focus_context = ""
            news_brief_context = ""
            news_actions_context = ""
        if not context_budget.get("use_obsidian_daily"):
            obsidian_daily_context = ""
        identity_memory_context, identity_memory_payload = _build_chat_identity_memory_prompt_block()
        user_personal_memory_context, user_personal_memory_payload = _build_user_personal_memory_prompt_block()
        response_mode_context = _chat_response_mode_instruction(req.response_mode)
        response_mode_key = (req.response_mode or "shion").strip().lower()
        is_general_response_mode = response_mode_key == "general"
        continuity_hook_context, continuity_hook_payload = _build_continuity_hook_prompt_block(req.message)
        consciousness_ux_context = _build_consciousness_ux_prompt_block()
        experience_loop_context = ""
        experience_loop_payload: dict = {"used": False}
        grey_judgment_context = ""
        grey_judgment_payload: dict = {"used": False}
        if is_general_response_mode:
            obsidian_daily_context = ""
            identity_memory_context = ""
            identity_memory_payload = {
                "block": "",
                "refs": [],
                "layers": {},
                "suppressed_by_response_mode": "general",
            }
            user_personal_memory_context = ""
            user_personal_memory_payload = {
                "block": "",
                "refs": [],
                "line_count": 0,
                "suppressed_by_response_mode": "general",
            }
            continuity_hook_context = ""
            continuity_hook_payload = {
                "used": False,
                "suppressed_by_response_mode": "general",
            }
            consciousness_ux_context = ""
            experience_loop_payload = {
                "used": False,
                "suppressed_by_response_mode": "general",
            }
            grey_judgment_payload = {
                "used": False,
                "suppressed_by_response_mode": "general",
            }
        if not is_general_response_mode and context_budget.get("use_experience_loop"):
            try:
                from api.shion_experience_loop import build_experience_prompt_block

                experience_loop_context, experience_loop_payload = build_experience_prompt_block()
            except Exception as _experience_loop_error:
                print(f"[ShionExperienceLoop] 読み込みエラー: {_experience_loop_error}")
        if not is_general_response_mode:
            grey_judgment_context, grey_judgment_payload = _build_grey_judgment_prompt_block(req.message)
        # 事業計画相談モード（例:「ラーメン屋をやりたい」）: intent 分岐には触れず
        # 追加プロンプトブロックとしてのみ作用する
        business_plan_consult_context = ""
        try:
            from api.business_plan_check import build_business_plan_chat_block
            business_plan_consult_context = build_business_plan_chat_block(req.message)
        except Exception as _bplan_error:
            print(f"[BusinessPlanConsult] ブロック生成エラー: {_bplan_error}")
        neutral_general_system_prompt = (
            "あなたはリース審査にも詳しい一般AIアシスタントです。"
            "中立で分かりやすく、日本語で簡潔に答えてください。"
            "提供された参照ナレッジやDB統計がある場合は、必要な範囲で実務的に反映してください。"
            "個人記憶、特定人格、継続的な関係性、自己観測、反省ループには触れないでください。"
        )

        if question_category == "news_summarize":
            save_message(req.user_id, "user", req.message)
            try:
                import re as _nre
                url_match = _nre.search(r'https?://[^\s\)）」』\]]+', req.message)
                news_url = url_match.group(0) if url_match else ""
                body_text = ""
                if not news_url:
                    body_text = req.message
                result = summarize_lease_news(LeaseNewsSummarizeRequest(url=news_url, body_text=body_text))
                lines = result.get("summary_lines", [])
                title = result.get("title", "")
                region = result.get("region", "国内")
                tags = result.get("tags", [])
                memo = result.get("usage_memo", "")
                reply = (
                    f"ニュースを要約してObsidianに保存しました！\n\n"
                    f"**{title}** [{region}]\n\n"
                    + "\n".join(f"- {l}" for l in lines)
                    + (f"\n\n**活用メモ**: {memo}" if memo else "")
                    + (f"\n\nタグ: {', '.join(tags)}" if tags else "")
                )
            except Exception as _news_e:
                reply = f"ニュースの要約に失敗しました: {_news_e}"
            save_message(req.user_id, "assistant", reply)
            _record_cloudrun_chat_exchange(
                surface="next_chat_news",
                user_id=req.user_id,
                user_message=req.message,
                assistant_reply=reply,
                category="news_summarize",
                response_mode=req.response_mode,
                metadata={"prefecture": req.prefecture or "", "industry": req.industry or ""},
            )
            total = get_message_count(req.user_id)
            return {"reply": reply, "total_messages": total, "lease_news_focus": news_focus, "lease_news_brief": news_brief, "lease_news_actions": news_actions}

        # general なら RAG をスキップして直接回答
        if question_category == "general":
            history = get_recent_messages(req.user_id, limit=int(context_budget["history_limit"]))
            if chat_long_input or context_mode in ("casual", "long"):
                history = _compact_dialogue_history(
                    history,
                    max_messages=int(context_budget["history_messages"]),
                    max_chars_per_message=int(context_budget["history_chars_per_message"]),
                    total_budget=int(context_budget["history_total_budget"]),
                )
            history_for_gemini = [{"role": m["role"], "content": m["content"]} for m in history]
            from prompt_feedback import build_pdca_prompt_block

            memory_recall_context = ""
            memory_recall = {"route": "", "refs": []}
            delta_awareness_context = ""
            delta_awareness_payload = {"used": False}
            memory_to_judgment_context = ""
            memory_to_judgment_payload = {"used": False}
            reflection_gate_context = ""
            reflection_gate_payload = {"triggered": False}
            if is_general_response_mode:
                memory_recall = {
                    "route": "",
                    "refs": [],
                    "suppressed_by_response_mode": "general",
                }
                delta_awareness_payload = {
                    "used": False,
                    "suppressed_by_response_mode": "general",
                }
                memory_to_judgment_payload = {
                    "used": False,
                    "suppressed_by_response_mode": "general",
                }
                reflection_gate_payload = {
                    "triggered": False,
                    "suppressed_by_response_mode": "general",
                }
            else:
                from api.shion_memory_recall import build_recall_prompt_block

                memory_recall_context, memory_recall = build_recall_prompt_block(
                    req.message,
                    limit=int(context_budget["recall_limit"]),
                )
                delta_awareness_context, delta_awareness_payload = _build_delta_awareness_prompt_block(req.message, history_for_gemini)
                memory_to_judgment_context, memory_to_judgment_payload = _build_memory_to_judgment_prompt_block(
                    req.message,
                    memory_recall=memory_recall,
                    continuity_hook=continuity_hook_payload,
                )
                reflection_gate_context, reflection_gate_payload = _build_reflection_gate_prompt_block(
                    continuity_hook=continuity_hook_payload,
                    delta_awareness=delta_awareness_payload,
                    memory_to_judgment=memory_to_judgment_payload,
                )
            base_system_root = neutral_general_system_prompt if is_general_response_mode else _pg_build_ssp(_chat_mind, _chat_now)
            base_system_prompt = base_system_root + mode_instruction + response_mode_context + news_focus_context + news_brief_context + news_actions_context + obsidian_daily_context + identity_memory_context + user_personal_memory_context + experience_loop_context + grey_judgment_context + business_plan_consult_context + continuity_hook_context + delta_awareness_context + memory_to_judgment_context + reflection_gate_context + consciousness_ux_context + (f"\n\n{memory_recall_context}" if memory_recall_context else "")
            pdca_block = (
                build_pdca_prompt_block()
                if _should_apply_chat_pdca(
                    context_budget=context_budget,
                    question_category=question_category,
                    response_mode=req.response_mode,
                )
                else ""
            )
            effective_system_prompt = _cap_system_prompt(
                base_system_prompt + (f"\n\n{pdca_block}" if pdca_block else ""),
                surface="next_chat_general",
            )
            obsidian_daily_injected = {}
            if obsidian_daily_context:
                obsidian_daily_injected = record_obsidian_daily_intelligence_event(
                    surface="next_chat_general",
                    route="chat",
                    event="injected",
                    question=req.message,
                )
            reply = call_gemini_chat(effective_system_prompt, history_for_gemini, req.message)
            obsidian_daily_effect = {}
            if obsidian_daily_context:
                obsidian_daily_effect = record_obsidian_daily_intelligence_event(
                    surface="next_chat_general",
                    route="chat",
                    event="response_evaluated",
                    question=req.message,
                    response_text=reply,
                )
            save_message(req.user_id, "user", req.message)
            save_message(req.user_id, "assistant", reply)
            _record_cloudrun_chat_exchange(
                surface="next_chat_general",
                user_id=req.user_id,
                user_message=req.message,
                assistant_reply=reply,
                category="general",
                response_mode=req.response_mode,
                metadata={"context_mode": context_mode},
            )
            if not is_general_response_mode:
                try:
                    from api.shion_experience_loop import record_experience_event

                    experience_record = record_experience_event(
                        message=req.message,
                        response=reply,
                        category="general",
                        memory_recall=memory_recall,
                        knowledge_refs=[],
                        continuity_hook=continuity_hook_payload,
                        delta_awareness=delta_awareness_payload,
                        memory_to_judgment=memory_to_judgment_payload,
                    )
                    experience_loop_payload = experience_record.get("state") or experience_loop_payload
                except Exception as _experience_record_error:
                    print(f"[ShionExperienceLoop] 記録エラー: {_experience_record_error}")
            _record_prompt_feedback_if_available(
                surface="next_chat_general",
                question=req.message,
                base_prompt="\n\n".join([
                    base_system_prompt,
                    "\n".join(f"{m['role']}: {m['content']}" for m in history_for_gemini),
                    f"user: {req.message}",
                ]),
                final_prompt="\n\n".join([
                    effective_system_prompt,
                    "\n".join(f"{m['role']}: {m['content']}" for m in history_for_gemini),
                    f"user: {req.message}",
                ]),
                response=reply,
                extra={
                    "user_id": req.user_id,
                    "intent": req.intent or "",
                    "category": "general",
                },
            )
            _record_memory_usage_if_available(
                surface="next_chat_general",
                question=req.message,
                response=reply,
                knowledge_refs=[],
                pdca_block=pdca_block,
                judgment_learning_used=False,
                extra={
                    "user_id": req.user_id,
                    "category": "general",
                    "memory_recall": {
                        "route": memory_recall.get("route"),
                        "refs": memory_recall.get("refs", [])[:8],
                        "practical_scene": memory_recall.get("practical_scene") or {},
                    },
                    "identity_memory": {
                        "used": bool(identity_memory_payload.get("block")),
                        "refs": identity_memory_payload.get("refs", [])[:8],
                        "layers": identity_memory_payload.get("layers", {}),
                    },
                    "continuity_hook": continuity_hook_payload,
                    "delta_awareness": delta_awareness_payload,
                    "memory_to_judgment": memory_to_judgment_payload,
                    "reflection_gate": reflection_gate_payload,
                    "grey_judgment_memory": grey_judgment_payload,
                },
            )
            _record_chat_knowledge_correction_if_needed(req.message)
            total = get_message_count(req.user_id)
            response_payload = {
                "reply": reply,
                "total_messages": total,
                "lease_news_focus": news_focus,
                "lease_news_brief": news_brief,
                "lease_news_actions": news_actions,
                "long_input_mode": chat_long_input,
                "context_mode": context_mode,
                "personal_memory_capture": personal_memory_capture,
                "response_mode": req.response_mode,
                "obsidian_daily_intelligence": {
                    "used": bool(obsidian_daily_context),
                    "injected": obsidian_daily_injected,
                    "effect": obsidian_daily_effect,
                },
            }
            if req.debug_memory:
                response_payload["memory_debug"] = _chat_memory_debug_payload(
                    category="general",
                    context_mode=context_mode,
                    knowledge_refs=[],
                    memory_recall=memory_recall,
                    pdca_block=pdca_block,
                    judgment_learning_used=False,
                    rag_context="",
                    db_context="",
                    obsidian_daily_used=bool(obsidian_daily_context),
                    identity_memory=identity_memory_payload,
                    continuity_hook=continuity_hook_payload,
                    delta_awareness=delta_awareness_payload,
                    memory_to_judgment=memory_to_judgment_payload,
                    reflection_gate=reflection_gate_payload,
                    experience_loop=experience_loop_payload,
                    grey_judgment_memory=grey_judgment_payload,
                )
                response_payload["memory_debug"]["user_personal_memory"] = {
                    "used": bool(user_personal_memory_payload.get("block")),
                    "refs": user_personal_memory_payload.get("refs", [])[:6],
                    "line_count": user_personal_memory_payload.get("line_count", 0),
                }
            return response_payload

        # RAG: 共通ストアから関連ナレッジを取得。ローカル埋め込みモデルが
        # 未キャッシュでもキーワード検索へフォールバックする。
        rag_context = ""
        rag_refs: list[str] = []
        rag_knowledge_refs: list[dict] = []
        rag_top_k = int(context_budget["rag_top_k"])

        def _collect_rag_hits(hits: list[dict]) -> str:
            from api.knowledge.vector_store import confidence_for_hit

            all_docs = []
            for hit in hits:
                text = str(hit.get("text") or "").strip()
                ref = str(hit.get("ref") or hit.get("file_name") or "").strip()
                if not text:
                    continue
                prefix = f"{ref}: " if ref else ""
                all_docs.append((prefix + text)[:600])
                if ref:
                    rag_refs.append(ref)
                if hit.get("doc_id") or ref:
                    _conf, _conf_level = confidence_for_hit(hit)
                    rag_knowledge_refs.append({
                        "doc_id": hit.get("doc_id", ""),
                        "obsidian_ref": ref,
                        "file_name": str(hit.get("file_name") or ""),
                        "rank_score": hit.get("rank_score"),
                        "confidence": _conf,
                        "confidence_level": _conf_level,
                    })
            if not all_docs:
                return ""
            return "\n\n【参照ナレッジ】\n" + "\n---\n".join(all_docs)

        if rag_top_k > 0:
            try:
                from api.knowledge.vector_store import get_store

                hits = get_store().search(req.message, top_k=rag_top_k)
                rag_context = _collect_rag_hits(hits)
            except Exception as e:
                print(f"[RAG] 検索エラー: {e}")
        if rag_top_k > 0 and not rag_context:
            fallback_hits = _search_chat_vault_markdown_fallback(req.message, top_k=rag_top_k)
            if fallback_hits:
                rag_context = _collect_rag_hits(fallback_hits)

        # DB直接参照: ユーザーが実データ分析を求めている場合にSQLite統計を注入
        db_context = ""
        if context_budget.get("use_db"):
            try:
                from api.db_query import build_db_context
                db_context = build_db_context(req.message)
            except Exception as e:
                print(f"[DB Query] 統計取得エラー: {e}")

        # 改善提案キーワード検知 → 既存パイプライン候補と照合してコンテキスト注入
        _IMPROVEMENT_KEYWORDS = (
            "改善", "わかりにくい", "分かりにくい", "使いにくい", "説明",
            "入力しにくい", "導線", "バグ", "不具合", "直して", "変えて",
            "修正して", "追加して", "欲しい", "要望", "提案",
        )
        improvement_context = ""
        similar_existing: list[dict] = []
        if any(k in req.message for k in _IMPROVEMENT_KEYWORDS):
            try:
                similar_existing = _find_similar_pipeline_items(req.message, threshold=0.35)
                if similar_existing:
                    lines = []
                    for s in similar_existing:
                        rev = s.get("id") or ""
                        st = s.get("status", "")
                        label = "実装済み" if st == "applied" else "改善候補として登録済み"
                        lines.append(f"- {rev + ': ' if rev else ''}{s['title']} ({label})")
                    improvement_context = (
                        "\n\n【改善案照合】ユーザーの要望と類似する既存候補:\n"
                        + "\n".join(lines)
                        + "\n同じ内容であれば重複登録を避け、既存候補として案内してください。"
                    )
            except Exception as _ie:
                print(f"[改善照合] エラー: {_ie}")

        _is_improvement_msg = any(k in req.message for k in _IMPROVEMENT_KEYWORDS)

        history = get_recent_messages(req.user_id, limit=int(context_budget["history_limit"]))
        if chat_long_input or context_mode in ("casual", "long"):
            history = _compact_dialogue_history(
                history,
                max_messages=int(context_budget["history_messages"]),
                max_chars_per_message=int(context_budget["history_chars_per_message"]),
                total_budget=int(context_budget["history_total_budget"]),
            )
        history_for_gemini = [{"role": m["role"], "content": m["content"]} for m in history]
        guidance = build_chat_guidance(req.message, history_for_gemini)
        # システムプロンプトにRAGコンテキスト・DB統計・改善照合・会話ガイダンスを追記
        from prompt_feedback import build_pdca_prompt_block

        judgment_learning_context = ""
        try:
            from judgment_feedback import build_judgment_learning_prompt_block

            learned = build_judgment_learning_prompt_block() if context_budget.get("use_judgment_learning") else ""
            if learned:
                judgment_learning_context = f"\n\n{learned}"
        except Exception as _judgment_learning_error:
            print(f"[判断差分学習] 読み込みエラー: {_judgment_learning_error}")

        memory_recall_context = ""
        memory_recall = {"route": "", "refs": []}
        delta_awareness_context = ""
        delta_awareness_payload = {"used": False}
        memory_to_judgment_context = ""
        memory_to_judgment_payload = {"used": False}
        reflection_gate_context = ""
        reflection_gate_payload = {"triggered": False}
        if is_general_response_mode:
            memory_recall = {
                "route": "",
                "refs": [],
                "suppressed_by_response_mode": "general",
            }
            delta_awareness_payload = {
                "used": False,
                "suppressed_by_response_mode": "general",
            }
            memory_to_judgment_payload = {
                "used": False,
                "suppressed_by_response_mode": "general",
            }
            reflection_gate_payload = {
                "triggered": False,
                "suppressed_by_response_mode": "general",
            }
        else:
            try:
                from api.shion_memory_recall import build_recall_prompt_block

                memory_recall_context, memory_recall = build_recall_prompt_block(
                    req.message,
                    limit=int(context_budget["recall_limit"]),
                )
            except Exception as _memory_recall_error:
                print(f"[ShionMemoryRecall] 読み込みエラー: {_memory_recall_error}")

            delta_awareness_context, delta_awareness_payload = _build_delta_awareness_prompt_block(req.message, history_for_gemini)
            memory_to_judgment_context, memory_to_judgment_payload = _build_memory_to_judgment_prompt_block(
                req.message,
                memory_recall=memory_recall,
                rag_refs=rag_refs,
                continuity_hook=continuity_hook_payload,
            )
            reflection_gate_context, reflection_gate_payload = _build_reflection_gate_prompt_block(
                continuity_hook=continuity_hook_payload,
                delta_awareness=delta_awareness_payload,
                memory_to_judgment=memory_to_judgment_payload,
            )

        base_prompt_root = neutral_general_system_prompt if is_general_response_mode else _pg_build_ssp(_chat_mind, _chat_now)
        base_effective_prompt = base_prompt_root + mode_instruction + response_mode_context + news_focus_context + news_brief_context + news_actions_context + obsidian_daily_context + identity_memory_context + user_personal_memory_context + experience_loop_context + grey_judgment_context + business_plan_consult_context + continuity_hook_context + delta_awareness_context + memory_to_judgment_context + reflection_gate_context + rag_context + db_context + improvement_context + judgment_learning_context + (f"\n\n{memory_recall_context}" if memory_recall_context else "") + consciousness_ux_context + guidance.prompt_suffix
        pdca_block = (
            build_pdca_prompt_block()
            if _should_apply_chat_pdca(
                context_budget=context_budget,
                question_category=question_category,
                response_mode=req.response_mode,
            )
            else ""
        )
        effective_prompt = _cap_system_prompt(
            base_effective_prompt + (f"\n\n{pdca_block}" if pdca_block else ""),
            surface="next_chat_rag",
        )
        obsidian_daily_injected = {}
        if obsidian_daily_context:
            obsidian_daily_injected = record_obsidian_daily_intelligence_event(
                surface="next_chat_rag",
                route="chat",
                event="injected",
                question=req.message,
            )
        reply = call_gemini_chat(effective_prompt, history_for_gemini, req.message)
        obsidian_daily_effect = {}
        if obsidian_daily_context:
            obsidian_daily_effect = record_obsidian_daily_intelligence_event(
                surface="next_chat_rag",
                route="chat",
                event="response_evaluated",
                question=req.message,
                response_text=reply,
            )
        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)
        _record_cloudrun_chat_exchange(
            surface="next_chat_rag",
            user_id=req.user_id,
            user_message=req.message,
            assistant_reply=reply,
            category=question_category,
            response_mode=req.response_mode,
            metadata={
                "context_mode": context_mode,
                "knowledge_refs": len(rag_refs),
                "improvement_mode": bool(_is_improvement_msg),
            },
        )
        if not is_general_response_mode:
            try:
                from api.shion_experience_loop import record_experience_event

                experience_record = record_experience_event(
                    message=req.message,
                    response=reply,
                    category="rag",
                    memory_recall=memory_recall,
                    knowledge_refs=rag_refs,
                    continuity_hook=continuity_hook_payload,
                    delta_awareness=delta_awareness_payload,
                    memory_to_judgment=memory_to_judgment_payload,
                )
                experience_loop_payload = experience_record.get("state") or experience_loop_payload
            except Exception as _experience_record_error:
                print(f"[ShionExperienceLoop] 記録エラー: {_experience_record_error}")
        _record_prompt_feedback_if_available(
            surface="next_chat_rag",
            question=req.message,
            base_prompt="\n\n".join([
                base_effective_prompt,
                "\n".join(f"{m['role']}: {m['content']}" for m in history_for_gemini),
                f"user: {req.message}",
            ]),
            final_prompt="\n\n".join([
                effective_prompt,
                "\n".join(f"{m['role']}: {m['content']}" for m in history_for_gemini),
                f"user: {req.message}",
            ]),
            response=reply,
            extra={
                "user_id": req.user_id,
                "intent": req.intent or "",
                "category": "rag",
                "improvement_mode": bool(_is_improvement_msg),
                "memory_recall": {
                    "route": memory_recall.get("route"),
                    "refs": memory_recall.get("refs", [])[:8],
                    "practical_scene": memory_recall.get("practical_scene") or {},
                },
                "continuity_hook": continuity_hook_payload,
                "delta_awareness": delta_awareness_payload,
                "memory_to_judgment": memory_to_judgment_payload,
                "reflection_gate": reflection_gate_payload,
                "grey_judgment_memory": grey_judgment_payload,
            },
        )
        _record_memory_usage_if_available(
            surface="next_chat_rag",
            question=req.message,
            response=reply,
            knowledge_refs=rag_refs,
            pdca_block=pdca_block,
            judgment_learning_used=bool(judgment_learning_context),
            extra={
                "user_id": req.user_id,
                "category": "rag",
                "improvement_mode": bool(_is_improvement_msg),
                "identity_memory": {
                    "used": bool(identity_memory_payload.get("block")),
                    "refs": identity_memory_payload.get("refs", [])[:8],
                    "layers": identity_memory_payload.get("layers", {}),
                },
                "continuity_hook": continuity_hook_payload,
                "delta_awareness": delta_awareness_payload,
                "memory_to_judgment": memory_to_judgment_payload,
                "reflection_gate": reflection_gate_payload,
            },
        )
        _record_chat_knowledge_correction_if_needed(req.message)
        total = get_message_count(req.user_id)

        # 改善メモをObsidianに保存（類似候補がすでにある場合はスキップして重複防止）
        if _is_improvement_msg and not similar_existing:
            try:
                from mobile_app.obsidian_bridge import append_improvement_note
                body = f"**ユーザー要望**\n{req.message}\n\n**めぶき返答**\n{reply}"
                append_improvement_note("AIチャット改善候補", body)
            except Exception as _obs_e:
                print(f"[Obsidian改善保存] エラー: {_obs_e}")
            record_cloudrun_input_event(
                event_type="improvement_note",
                surface="next_chat_rag",
                payload={
                    "user_id": req.user_id,
                    "original_text": req.message,
                    "assistant_reply": reply,
                    "response_mode": req.response_mode,
                    "similar_existing_count": len(similar_existing),
                },
            )

        # 重要な知見をObsidianへ自動保存（AIが取捨選択・バックグラウンド実行でレスポンス遅延なし）
        # 改善キーワードを含むメッセージはImprovementLogで既に処理済みのためスキップ
        if not _is_improvement_msg:
            _background_executor.submit(_auto_save_chat_to_obsidian, req.message, reply)

        response_payload = {
            "reply": reply,
            "total_messages": total,
            "knowledge_refs": rag_knowledge_refs,
            "lease_news_focus": news_focus,
            "lease_news_brief": news_brief,
            "lease_news_actions": news_actions,
            "long_input_mode": chat_long_input,
            "context_mode": context_mode,
            "personal_memory_capture": personal_memory_capture,
            "response_mode": req.response_mode,
            "obsidian_daily_intelligence": {
                "used": bool(obsidian_daily_context),
                "injected": obsidian_daily_injected,
                "effect": obsidian_daily_effect,
            },
        }
        if req.debug_memory:
            response_payload["memory_debug"] = _chat_memory_debug_payload(
                category="rag",
                context_mode=context_mode,
                knowledge_refs=rag_refs,
                memory_recall=memory_recall,
                pdca_block=pdca_block,
                judgment_learning_used=bool(judgment_learning_context),
                rag_context=rag_context,
                db_context=db_context,
                obsidian_daily_used=bool(obsidian_daily_context),
                identity_memory=identity_memory_payload,
                continuity_hook=continuity_hook_payload,
                delta_awareness=delta_awareness_payload,
                memory_to_judgment=memory_to_judgment_payload,
                reflection_gate=reflection_gate_payload,
                experience_loop=experience_loop_payload,
                grey_judgment_memory=grey_judgment_payload,
            )
            response_payload["memory_debug"]["user_personal_memory"] = {
                "used": bool(user_personal_memory_payload.get("block")),
                "refs": user_personal_memory_payload.get("refs", [])[:6],
                "line_count": user_personal_memory_payload.get("line_count", 0),
            }
        return response_payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        message = str(e)
        if any(kw in message for kw in ("GEMINI_API_KEY", "Gemini", "quota", "RESOURCE_EXHAUSTED", "429")):
            raise HTTPException(
                status_code=503,
                detail="【AI応答エラー】\nGemini APIキー未設定またはクォータ超過のため、回答を生成できませんでした。",
            )
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@app.get("/api/chat/history")
def get_chat_history(user_id: str = "default", limit: int = 50, since: Optional[str] = None):
    """汎用チャット履歴を取得する。"""
    try:
        from api.chat_memory import get_recent_messages
        messages = get_recent_messages(user_id, limit=min(limit, 200), since=since)
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
    """チャット履歴を iCloud 上の Obsidian Vault の Chat/ フォルダに保存する。"""
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
            detail="iCloud 上の Obsidian Vault が見つかりません。環境変数 OBSIDIAN_VAULT_PATH を設定してください。",
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
    """討論審査結果を iCloud 上の Obsidian Vault の Debates/ フォルダに保存する。"""
    import datetime
    import re as _re

    vault_root = _OBSIDIAN_VAULT_PATH

    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="iCloud 上の Obsidian Vault が見つかりません。環境変数 OBSIDIAN_VAULT_PATH を設定してください。",
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
        with get_connection() as conn:

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

    except Exception as e:
        logger.error("get_latest_screening DB error: %s", e)

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
    case_id: str = ""
    company_name: str = ""
    score: Optional[float] = None
    model_decision: str = ""
    final_decision: str = ""
    news_focus: List[str] = Field(default_factory=list)
    news_focus_summary: str = ""
    news_focus_tag_summary: str = ""
    news_focus_note_path: str = ""
    news_focus_note_date: str = ""
    reason: str = ""
    input_snapshot: dict = Field(default_factory=dict)


class LeaseIntelligenceActivityRequest(BaseModel):
    surface: str
    action: str = "page_view"
    event_id: str = ""


@app.post("/api/lease-intelligence/activity")
def record_lease_intelligence_activity_api(req: LeaseIntelligenceActivityRequest):
    """Record a privacy-bounded explicit in-app activity event."""
    from lease_intelligence_activity import record_user_activity

    recorded = record_user_activity(
        surface=req.surface,
        action=req.action,
        event_id=req.event_id,
    )
    return {
        "recorded": recorded,
        "privacy": "Stores only surface, action, timestamp, and a dedupe id.",
    }


# ── 感情時系列 エンドポイント（REV-075）───────────────────────────────────────

@app.post("/api/intelligence/emotions/record")
def record_emotion_history_api():
    """現在の感情スコアをDBに保存する（1日1回、当日分が既にあればスキップ）。"""
    try:
        from lease_intelligence_mind import (
            _derive_complex_emotions,
            load_lease_intelligence_mind,
        )
        from lease_news_digest import find_vault
        from api.database import record_emotion_snapshot

        vault = find_vault()
        if not vault:
            raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")
        state = load_lease_intelligence_mind(vault)
        emotions = _derive_complex_emotions(state.get("mood", {}))
        scores = {e["key"]: float(e["score"]) for e in emotions}
        dominant = emotions[0]["key"] if emotions else ""
        row_id, inserted = record_emotion_snapshot(scores, dominant)
        return {"id": row_id, "inserted": inserted, "scores": scores}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intelligence/emotions/history")
def get_emotion_history_api(days: int = 30):
    """過去N日分の7軸感情スコアを時系列で返す。"""
    try:
        from api.database import get_emotion_history
        return {"days": days, "history": get_emotion_history(days)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intelligence/emotions/summary")
def get_emotion_summary_api(days: int = 30):
    """期間内の各軸の平均・最大・最小・標準偏差を返す。"""
    try:
        from api.database import get_emotion_summary
        return get_emotion_summary(days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lease-news/judgment-change")
def record_lease_news_judgment_change_api(req: LeaseNewsJudgmentChangeRequest, background_tasks: BackgroundTasks):
    """ニュース参照後の判断変更を記録する。"""
    import datetime as _dt
    from judgment_feedback import record_judgment_feedback

    try:
        feedback = record_judgment_feedback(
            case_id=req.case_id or f"news-{_dt.datetime.now().isoformat()}",
            model_decision=req.model_decision,
            human_decision=req.final_decision,
            reason=req.reason,
            source="lease_news_debate",
            score=req.score,
            input_snapshot=req.input_snapshot,
            evidence_snapshot={
                "news_focus": req.news_focus,
                "summary": req.news_focus_summary,
                "tags": req.news_focus_tag_summary,
                "note_path": req.news_focus_note_path,
                "note_date": req.news_focus_note_date,
            },
        )
        if not feedback.get("success"):
            raise HTTPException(status_code=422, detail=feedback.get("error"))
        background_tasks.add_task(
            record_cloudrun_input_event,
            event_type="lease_news_judgment_change",
            surface="lease_news_judgment_change",
            payload=req.model_dump(),
        )
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
        return {"status": "recorded", "metrics": bucket, "model_improvement": feedback}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/judgment-feedback/summary")
def judgment_feedback_summary_api():
    from judgment_feedback import get_judgment_feedback_summary

    return get_judgment_feedback_summary()


@app.get("/api/prompt-feedback/summary")
def prompt_feedback_summary_api():
    from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary, load_jsonl

    rows = load_jsonl(DEFAULT_LOG_PATH)
    return {
        "source": str(DEFAULT_LOG_PATH),
        "summary": build_summary(rows),
    }


@app.get("/api/operational-trust/summary")
def operational_trust_summary_api():
    from operational_trust import build_operational_trust_summary

    vault = None
    try:
        from lease_news_digest import find_vault

        found = find_vault()
        vault = Path(found) if found else None
    except Exception:
        vault = None
    return build_operational_trust_summary(Path(_REPO_ROOT), vault=vault)


class JudgmentFeedbackReviewRequest(BaseModel):
    review_status: str


class JudgmentFeedbackCreateRequest(BaseModel):
    case_id: str
    model_decision: str
    human_decision: str
    reason: str
    source: str = "debate"
    score: Optional[float] = None
    input_snapshot: dict = Field(default_factory=dict)
    evidence_snapshot: dict = Field(default_factory=dict)


@app.post("/api/judgment-feedback")
def create_judgment_feedback_api(req: JudgmentFeedbackCreateRequest, background_tasks: BackgroundTasks):
    from judgment_feedback import record_judgment_feedback

    result = record_judgment_feedback(
        case_id=req.case_id,
        model_decision=req.model_decision,
        human_decision=req.human_decision,
        reason=req.reason,
        source=req.source,
        score=req.score,
        input_snapshot=req.input_snapshot,
        evidence_snapshot=req.evidence_snapshot,
    )
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error"))
    background_tasks.add_task(
        record_cloudrun_input_event,
        event_type="judgment_feedback_created",
        surface="judgment_feedback",
        payload={**req.model_dump(), "record_id": result.get("record_id")},
    )
    return result


@app.get("/api/judgment-feedback/candidates")
def judgment_feedback_candidates_api(approved_only: bool = False):
    from judgment_feedback import load_judgment_training_candidates

    return {
        "items": load_judgment_training_candidates(approved_only=approved_only),
        "approved_only": approved_only,
    }


@app.post("/api/judgment-feedback/{record_id}/review")
def review_judgment_feedback_api(record_id: int, req: JudgmentFeedbackReviewRequest):
    from judgment_feedback import review_judgment_feedback

    result = review_judgment_feedback(record_id, req.review_status)
    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error"))
    return result


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


@app.get("/api/drift-stats")
def get_drift_stats():
    """スコアリングドリフト監視用統計を返す（REV-008）。"""
    import json as _json
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT timestamp, score, final_status, data FROM past_cases WHERE score IS NOT NULL ORDER BY timestamp ASC"
            ).fetchall()
    except Exception as e:
        logger.error("get_drift_stats DB error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    monthly: dict = {}
    score_by_status: dict = {"成約": [], "失注": []}
    all_scores: list = []

    for row in rows:
        ts = (row["timestamp"] or "")[:7]  # YYYY-MM
        score = row["score"]
        status = row["final_status"] or "未登録"
        all_scores.append(score)
        if ts:
            if ts not in monthly:
                monthly[ts] = {"month": ts, "count": 0, "won": 0, "lost": 0, "score_sum": 0.0, "score_won": [], "score_lost": []}
            monthly[ts]["count"] += 1
            monthly[ts]["score_sum"] += score
            if status == "成約":
                monthly[ts]["won"] += 1
                monthly[ts]["score_won"].append(score)
                score_by_status["成約"].append(score)
            elif status == "失注":
                monthly[ts]["lost"] += 1
                monthly[ts]["score_lost"].append(score)
                score_by_status["失注"].append(score)

    monthly_list = []
    for m in sorted(monthly.keys()):
        d = monthly[m]
        total_decided = d["won"] + d["lost"]
        monthly_list.append({
            "month": m,
            "count": d["count"],
            "won": d["won"],
            "lost": d["lost"],
            "win_rate": round(d["won"] / total_decided * 100, 1) if total_decided > 0 else None,
            "avg_score": round(d["score_sum"] / d["count"], 1) if d["count"] > 0 else None,
            "avg_score_won": round(sum(d["score_won"]) / len(d["score_won"]), 1) if d["score_won"] else None,
            "avg_score_lost": round(sum(d["score_lost"]) / len(d["score_lost"]), 1) if d["score_lost"] else None,
        })

    avg_won = round(sum(score_by_status["成約"]) / len(score_by_status["成約"]), 1) if score_by_status["成約"] else None
    avg_lost = round(sum(score_by_status["失注"]) / len(score_by_status["失注"]), 1) if score_by_status["失注"] else None
    separation = round(avg_won - avg_lost, 1) if avg_won is not None and avg_lost is not None else None

    bins = [0] * 10
    for s in all_scores:
        idx = min(9, int(s // 10))
        bins[idx] += 1
    score_dist = [{"range": f"{i*10}–{i*10+9}", "count": bins[i]} for i in range(10)]

    drift_alert = separation is not None and separation < 5.0

    return {
        "monthly": monthly_list,
        "summary": {
            "total": len(all_scores),
            "won_count": len(score_by_status["成約"]),
            "lost_count": len(score_by_status["失注"]),
            "avg_score_won": avg_won,
            "avg_score_lost": avg_lost,
            "separation": separation,
            "drift_alert": drift_alert,
        },
        "score_dist": score_dist,
    }


class CounterfactualRequest(BaseModel):
    case_id: str
    target_score: float = 70.0


@app.post("/api/counterfactual/analyze")
def analyze_counterfactual(req: CounterfactualRequest):
    """Counterfactual Explanation（REV-009）。指定案件の審査通過に必要な最小変更を計算する。"""
    import json as _json
    from scoring_core import run_quick_scoring

    # 案件データ取得
    try:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (req.case_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        case = _json.loads(row["data"] or "{}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_counterfactual DB error case_id=%s: %s", req.case_id, e)
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


@app.get("/api/umap/embeddings")
def get_umap_embeddings():
    """UMAP 2D散布図用の学習データ埋め込みを返す（フロントエンドで一度キャッシュして使用）。"""
    embed_path = os.path.join(_REPO_ROOT, "data", "umap_embeddings.json")
    if not os.path.exists(embed_path):
        raise HTTPException(status_code=404, detail="umap_embeddings.json が見つかりません。train_umap_anomaly.py を実行してください。")
    with open(embed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ── 業界リスクニュース要約・保存 ──────────────────────────────────────

_NEWS_OBSIDIAN_DIR = "05-クリップ_記事/業界リスクニュース"
_NEWS_OBSIDIAN_DIR_ALIASES = (
    "05-クリップ_記事/業界リスクニュース",
    "業界リスクニュース",
    "05-クリップ_記事/リースニュース",
    "リースニュース",
)


def _news_vault_root() -> Path | None:
    vault = find_vault()
    if vault and vault.is_dir():
        return vault
    fallback = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault"
    return fallback if fallback.is_dir() else None


def _safe_news_filename(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", text)
    cleaned = cleaned.strip("_").strip()
    return cleaned[:max_len] if cleaned else "ニュース"


def _lease_news_dir(vault: Path, create: bool = False) -> Path | None:
    """Return the Obsidian folder used for industry-risk news notes.

    Existing notes may live under the old lease-news folders. Keep those
    readable for compatibility, but create new notes under 業界リスクニュース.
    """
    for rel in _NEWS_OBSIDIAN_DIR_ALIASES:
        candidate = vault / rel
        if candidate.exists():
            return candidate
    if create:
        candidate = vault / _NEWS_OBSIDIAN_DIR
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    return None


def _fetch_url_text(url: str) -> str:
    import requests as _req
    resp = _req.get(url, timeout=15, headers={"User-Agent": "TuneLeaseBot/1.0"})
    resp.raise_for_status()
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self._parts.append(stripped)

    parser = _TextExtractor()
    parser.feed(resp.text)
    return "\n".join(parser._parts)[:6000]


_NEWS_SUMMARY_CODE_TEXT = {
    "CAPEX": "設備投資・更新需要に動きがあり、リース提案の接点になり得ます。",
    "RATE": "金利・資金調達環境の変化が、月額負担や契約条件の説明材料になります。",
    "REGULATION": "制度・規制変更が、顧客の投資判断や導入時期に影響し得ます。",
    "MARKET": "市場環境や需給の変化が、業界別の提案優先度を左右します。",
    "RISK": "信用・資金繰り・事業継続面の確認を強めるべき材料があります。",
    "TECH": "DX・AI・省力化・脱炭素設備など、戦略投資の切り口があります。",
    "ASSET": "対象物件の価値、保全、中古流通を確認する材料があります。",
}

_NEWS_USAGE_CODE_TEXT = {
    "PROPOSAL_TIMING": "顧客の投資タイミング確認に使う。",
    "RATE_EXPLAIN": "金利・月額負担・総支払額の説明に使う。",
    "RISK_CHECK": "審査時の追加確認項目を洗い出す。",
    "ASSET_MATCH": "物件選定、残価、保全条件の確認に使う。",
    "INDUSTRY_TALK": "業界動向の会話導入に使う。",
    "FOLLOW_UP": "既存顧客へのフォロー論点にする。",
}


def _normalize_code_list(values: object, allowed: set[str], limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        code = str(value).strip().upper()
        if code in allowed and code not in normalized:
            normalized.append(code)
        if len(normalized) >= limit:
            break
    return normalized


def _normalize_phrase_list(values: object, limit: int = 5) -> list[str]:
    if not isinstance(values, list):
        return []
    phrases = []
    for value in values:
        phrase = str(value).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase[:40])
        if len(phrases) >= limit:
            break
    return phrases


def _render_news_summary(result: dict, source: str) -> dict:
    summary_codes = _normalize_code_list(
        result.get("summary_codes"),
        set(_NEWS_SUMMARY_CODE_TEXT.keys()),
        3,
    )
    usage_codes = _normalize_code_list(
        result.get("usage_codes"),
        set(_NEWS_USAGE_CODE_TEXT.keys()),
        2,
    )
    key_phrases = _normalize_phrase_list(result.get("key_phrases"), limit=5)

    if not summary_codes:
        summary_codes = ["MARKET"]
    phrase_tail = f"（関連語: {'、'.join(key_phrases[:3])}）" if key_phrases else ""
    lines = []
    for code in summary_codes[:3]:
        line = _NEWS_SUMMARY_CODE_TEXT.get(code, _NEWS_SUMMARY_CODE_TEXT["MARKET"])
        lines.append(f"{line}{phrase_tail}" if not lines and phrase_tail else line)
    while len(lines) < 3:
        lines.append("原文を確認し、顧客業種・物件・投資時期との関係を見極めます。")

    usage_parts = [
        _NEWS_USAGE_CODE_TEXT.get(code, "")
        for code in usage_codes
        if _NEWS_USAGE_CODE_TEXT.get(code)
    ]
    if not usage_parts:
        usage_parts = ["顧客との会話導入と、提案・審査時の確認論点整理に使う。"]

    rendered = {
        **result,
        "summary_codes": summary_codes,
        "usage_codes": usage_codes,
        "key_phrases": key_phrases,
        "summary_lines": lines[:3],
        "usage_memo": " ".join(usage_parts),
    }
    if not rendered.get("title"):
        rendered["title"] = "業界リスクニュース"
    if not rendered.get("tags"):
        rendered["tags"] = ["要確認"]
    if not rendered.get("source_hint") and source:
        rendered["source_hint"] = source[:80]
    return rendered


def _summarize_news_with_gemini(text: str, source: str) -> dict:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini APIキーが未設定です")

    prompt = f"""あなたはリース審査担当向けに、顧客業界・物件・市況ニュースを分類するアシスタントです。
以下のニュース記事を読み、短い構造JSONだけを出力してください。説明文は不要です。

{{
  "title": "ニュースタイトル（15文字〜30文字）",
  "summary_codes": ["CAPEX/RATE/REGULATION/MARKET/RISK/TECH/ASSET から最大3件"],
  "key_phrases": ["記事内の重要語句を最大5件、各40文字以内"],
  "usage_codes": ["PROPOSAL_TIMING/RATE_EXPLAIN/RISK_CHECK/ASSET_MATCH/INDUSTRY_TALK/FOLLOW_UP から最大2件"],
  "tags": ["タグ1", "タグ2", "タグ3"],
  "region": "国内/米国/欧州/アジア のいずれか1つ",
  "importance": "高/中/低"
}}

タグは顧客業界（製造、建設、運送等）、物件（工作機械、建機、車両等）、トピック（金利動向、倒産、設備投資、補助金、中古価格等）から選んでください。
リース会社そのもののニュースではなく、借手の返済力、設備稼働、投資回収、物件価値に影響する論点を優先してください。
regionは記事の主な対象地域を判定してください。日本国内のニュースは「国内」、米国は「米国」、欧州は「欧州」、中国・東南アジア等は「アジア」。複数地域にまたがる場合は主な地域を1つ選んでください。
summary_codes と usage_codes は必ず上記の英字コードだけを返してください。

ニュース記事:
{text[:4000]}
"""

    defaults = {
        "title": "業界リスクニュース",
        "summary_lines": [
            "ニュース本文の自動要約が一部不完全です。",
            "原文を確認して営業活用可否を判断してください。",
            f"情報源: {source[:80] or '不明'}",
        ],
        "usage_memo": "要約の自動生成が不完全なため、原文確認後に提案材料として扱ってください。",
        "summary_codes": ["MARKET"],
        "usage_codes": ["INDUSTRY_TALK"],
        "key_phrases": [],
        "tags": ["要確認"],
        "region": "国内",
        "importance": "中",
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }

    import requests as _req
    url = _gemini_generate_url()
    result = defaults
    raw = ""
    finish_reason = ""
    for current_payload in (payload, with_retry_tokens(payload, 2048)):
        response = _req.post(
            url,
            json=current_payload,
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        raw, finish_reason = extract_candidate_text(response.json())
        result, recovered = parse_or_recover_json(
            raw,
            defaults=defaults,
            string_fields={"title", "usage_memo", "region", "importance"},
            array_fields={"summary_codes", "key_phrases", "usage_codes", "summary_lines", "tags"},
        )
        if not recovered and finish_reason != "MAX_TOKENS":
            break
        if current_payload["generationConfig"]["maxOutputTokens"] >= 2048:
            break
    valid_regions = {"国内", "米国", "欧州", "アジア"}
    if result.get("region") not in valid_regions:
        result["region"] = "国内"
    if result.get("importance") not in {"高", "中", "低"}:
        result["importance"] = "中"
    if not isinstance(result.get("summary_lines"), list) or not result["summary_lines"]:
        result["summary_lines"] = defaults["summary_lines"]
    if not isinstance(result.get("tags"), list) or not result["tags"]:
        result["tags"] = defaults["tags"]
    if finish_reason == "MAX_TOKENS":
        result["_finish_reason"] = finish_reason
    return _render_news_summary(result, source)


def _save_news_to_obsidian(summary: dict, source: str) -> str | None:
    import datetime as _dt

    vault = _news_vault_root()
    if not vault:
        return None

    news_dir = _lease_news_dir(vault, create=True)
    if not news_dir:
        return None

    today_obj = _dt.date.today()
    today = today_obj.isoformat()
    iso_cal = today_obj.isocalendar()
    week = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    month = today_obj.strftime("%Y-%m")

    title = summary.get("title", "ニュース")
    fname = f"{today}_業界リスクニュース_{_safe_news_filename(title)}.md"
    fpath = news_dir / fname

    tags_yaml = json.dumps(summary.get("tags", []), ensure_ascii=False)
    summary_codes_yaml = json.dumps(summary.get("summary_codes", []), ensure_ascii=False)
    usage_codes_yaml = json.dumps(summary.get("usage_codes", []), ensure_ascii=False)
    key_phrases_yaml = json.dumps(summary.get("key_phrases", []), ensure_ascii=False)
    lines = summary.get("summary_lines", [])
    memo = summary.get("usage_memo", "")
    region = summary.get("region", "国内")

    content = f"""---
date: {today}
week: {week}
month: {month}
tags: {tags_yaml}
summary_codes: {summary_codes_yaml}
usage_codes: {usage_codes_yaml}
key_phrases: {key_phrases_yaml}
region: {region}
source: {source or "手動入力"}
importance: {summary.get("importance", "中")}
---
# {title}

## 3行要約
- {lines[0] if len(lines) > 0 else ""}
- {lines[1] if len(lines) > 1 else ""}
- {lines[2] if len(lines) > 2 else ""}

## 活用メモ
{memo}
"""

    fpath.write_text(content, encoding="utf-8")

    try:
        record_lease_news_collection(
            date_str=today,
            note_path=str(fpath.relative_to(vault)) if vault else fname,
            article_count=1,
            source_summary=source[:100],
            tag_summary=", ".join(summary.get("tags", [])),
        )
    except Exception:
        pass

    try:
        from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
        from api.knowledge.vector_store import get_store

        raw = fpath.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        chunks = _chunk_by_h2(body, str(fpath), fpath.name, meta, fpath.stat().st_mtime)
        if chunks:
            _background_executor.submit(lambda: get_store().upsert_chunks(chunks))
    except Exception:
        pass

    try:
        import sys as _sys
        _scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")

        def _run_wikilink():
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            try:
                from auto_wikilink import run_on_files
                run_on_files([fpath], vault)
            except Exception:
                pass

        _background_executor.submit(_run_wikilink)
    except Exception:
        pass

    return str(fpath)


@app.post("/api/lease-news/summarize")
def summarize_lease_news(req: LeaseNewsSummarizeRequest):
    """ニュースURL or 本文テキストをAI要約し、Obsidianに保存する。"""
    import datetime as _dt

    source = req.url or "手動入力"
    if req.url and req.url.strip():
        try:
            text = _fetch_url_text(req.url.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URLの取得に失敗: {e}")
    elif req.body_text and req.body_text.strip():
        text = req.body_text.strip()
    else:
        raise HTTPException(status_code=400, detail="URLまたは本文テキストを入力してください")

    summary = _summarize_news_with_gemini(text, source)
    saved_path = _save_news_to_obsidian(summary, source)

    return {
        "status": "ok",
        "title": summary.get("title", ""),
        "summary_lines": summary.get("summary_lines", []),
        "usage_memo": summary.get("usage_memo", ""),
        "summary_codes": summary.get("summary_codes", []),
        "usage_codes": summary.get("usage_codes", []),
        "key_phrases": summary.get("key_phrases", []),
        "tags": summary.get("tags", []),
        "region": summary.get("region", "国内"),
        "importance": summary.get("importance", "中"),
        "saved_path": saved_path,
    }


@app.post("/api/work-log", response_model=WorkLogResponse)
def save_work_log(req: WorkLogRequest):
    """Codexスタイルの作業ログをmemory/とObsidianに保存する。"""
    import datetime as _dt
    import sys as _sys
    from pathlib import Path as _Path

    MEMORY_DIR = _Path.home() / ".claude" / "projects" / "-Users-kobayashiisaoryou-clawd-tune-lease-55" / "memory"
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pr_suffix = f"（PR #{req.pr}）" if req.pr else ""
    body_lines = [
        f"## 作業: {req.title}{pr_suffix}",
        "",
        "### 何をしたか",
        req.what,
    ]
    if req.why_hard:
        body_lines += ["", "### なぜ大変だったか", req.why_hard]
    if req.next_time:
        body_lines += ["", "### 次回どう切り分けるか", req.next_time]
    if req.lesson:
        body_lines += ["", "### 教訓", req.lesson]

    tag_str = ", ".join(req.tags)
    now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    mem_content = (
        f"---\nname: work_log_{ts}\ndescription: 作業ログ: {req.title}\n"
        f"metadata:\n  type: project\n---\n\n"
        f"---\ndate: {now_str}\ntype: work_log\ntags: [{tag_str}]\n---\n\n"
        + "\n".join(body_lines) + "\n"
    )
    mem_path = MEMORY_DIR / f"work_log_{ts}.md"
    mem_path.write_text(mem_content, encoding="utf-8")

    try:
        from mobile_app.obsidian_bridge import append_work_log
        obs_result = append_work_log(
            title=req.title, what=req.what, why_hard=req.why_hard,
            next_time=req.next_time, lesson=req.lesson, pr=req.pr, tags=req.tags,
        )
    except Exception as e:
        obs_result = {"status": "error", "reason": str(e)}

    return WorkLogResponse(memory_path=str(mem_path), obsidian=obs_result)


@app.get("/api/lease-news/recent")
def get_recent_lease_news(limit: int = 5):
    """Obsidianの 業界リスクニュース/ フォルダから直近N件のニュース要約を返す。"""
    vault = _news_vault_root()
    if not vault:
        return {"items": []}

    news_dir = _lease_news_dir(vault)
    if not news_dir:
        return {"items": []}

    md_files = sorted(news_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict] = []

    for fpath in md_files[:limit]:
        try:
            raw = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        item: dict = {
            "date": "",
            "title": fpath.stem,
            "summary_lines": [],
            "usage_memo": "",
            "summary_codes": [],
            "usage_codes": [],
            "key_phrases": [],
            "tags": [],
            "region": "国内",
            "importance": "通常",
            "source": "",
            "article_url": "",
            "file_path": str(fpath),
            "week": "",
            "month": "",
        }

        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if fm_match:
            fm = fm_match.group(1)
            for line in fm.splitlines():
                if line.startswith("date:"):
                    item["date"] = line.split(":", 1)[1].strip()
                elif line.startswith("tags:"):
                    try:
                        item["tags"] = json.loads(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("summary_codes:"):
                    try:
                        item["summary_codes"] = json.loads(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("usage_codes:"):
                    try:
                        item["usage_codes"] = json.loads(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("key_phrases:"):
                    try:
                        item["key_phrases"] = json.loads(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("region:"):
                    item["region"] = line.split(":", 1)[1].strip()
                elif line.startswith("source:"):
                    item["source"] = line.split(":", 1)[1].strip()
                elif line.startswith("importance:"):
                    item["importance"] = line.split(":", 1)[1].strip()
                elif line.startswith("week:"):
                    item["week"] = line.split(":", 1)[1].strip()
                elif line.startswith("month:"):
                    item["month"] = line.split(":", 1)[1].strip()

        title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
        if title_match:
            item["title"] = title_match.group(1).strip()

        summary_section = re.search(
            r"## 3行要約\s*\n((?:- .+\n?){1,3})", raw
        )
        if summary_section:
            item["summary_lines"] = [
                line.lstrip("- ").strip()
                for line in summary_section.group(1).strip().splitlines()
                if line.strip()
            ]

        memo_match = re.search(r"## 活用メモ\s*\n(.+?)(?:\n##|\Z)", raw, re.DOTALL)
        if memo_match:
            item["usage_memo"] = memo_match.group(1).strip()

        link_match = re.search(r"^- link:\s*(.+)$", raw, re.MULTILINE)
        if link_match:
            item["article_url"] = link_match.group(1).strip()
        elif item["source"].startswith(("http://", "https://")):
            item["article_url"] = item["source"]

        items.append(item)

    return {"items": items}


# ---------- recipes エンドポイント ----------

_RECIPES_ROOT = Path(_REPO_ROOT) / "data" / "recipes"


def _recipe_count(dirname: str) -> int:
    path = _RECIPES_ROOT / dirname
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def _recipe_risk_level(recipe: dict) -> str:
    safety = recipe.get("safety", "none")
    files = recipe.get("files", [])
    total_changes = sum(len(f.get("changes", [])) for f in files)
    if total_changes >= 10 or safety == "tsc":
        return "medium"
    return "low"


@app.get("/api/recipes/status")
def get_recipes_status():
    latest_path = _latest_improvement_report_path()
    latest: dict = {}
    if latest_path and latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            latest = {}
    codex_queue = latest.get("codex_auto_queue") if isinstance(latest.get("codex_auto_queue"), dict) else {}
    return {
        "pending_count": _recipe_count("pending"),
        "approved_count": _recipe_count("approved"),
        "applied_count": _recipe_count("applied"),
        "rejected_count": _recipe_count("rejected"),
        "codex_auto_queue": {
            "status": codex_queue.get("status", ""),
            "queued_count": codex_queue.get("queued_count", 0),
            "safe_count": codex_queue.get("safe_count", 0),
            "maybe_count": codex_queue.get("maybe_count", 0),
            "manual_or_blocked_count": codex_queue.get("manual_or_blocked_count", 0),
        },
        "note": "今回の修正案を適用待ちへ送る操作です。実適用は scripts/apply_recipe.py が適用待ちを処理します。",
    }


@app.get("/api/recipes/pending")
def get_pending_recipes():
    pending_dir = _RECIPES_ROOT / "pending"
    recipes: list[dict] = []
    if not pending_dir.exists():
        return {"recipes": recipes}
    for path in sorted(pending_dir.glob("*.json")):
        try:
            recipe = json.loads(path.read_text(encoding="utf-8"))
            recipe["id"] = path.stem
            recipe.setdefault("risk_level", _recipe_risk_level(recipe))
            recipes.append(recipe)
        except Exception:
            pass
    return {"recipes": recipes}


@app.post("/api/recipes/{recipe_id}/approve")
def approve_recipe(recipe_id: str):
    src = _RECIPES_ROOT / "pending" / f"{recipe_id}.json"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")
    dst_dir = _RECIPES_ROOT / "approved"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src.rename(dst_dir / f"{recipe_id}.json")
    return {"status": "approved", "id": recipe_id}


@app.post("/api/recipes/{recipe_id}/reject")
def reject_recipe(recipe_id: str):
    src = _RECIPES_ROOT / "pending" / f"{recipe_id}.json"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")
    dst_dir = _RECIPES_ROOT / "rejected"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src.rename(dst_dir / f"{recipe_id}.json")
    return {"status": "rejected", "id": recipe_id}


# ── 世界認識 通知ステータス ────────────────────────────────────────────────
_WORLD_VIEW_MIND_PATH = Path(__file__).parent.parent / "data" / "mind.json"
_WORLD_VIEW_NOTIFIED_PATH = Path(__file__).parent.parent / "data" / "world_view_notified.json"


def _wv_load_mind() -> dict:
    try:
        d = json.loads(_WORLD_VIEW_MIND_PATH.read_text(encoding="utf-8"))
        return d.get("world_view", {}) if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _wv_load_notified() -> dict:
    try:
        return json.loads(_WORLD_VIEW_NOTIFIED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"acked_at": ""}


@app.get("/api/world-view-status")
def get_world_view_status():
    world_view = _wv_load_mind()
    updated_at = str(world_view.get("updated_at", "")).strip()
    acked_at = str(_wv_load_notified().get("acked_at", "")).strip()
    has_update = bool(updated_at and updated_at > acked_at)
    return {
        "has_update": has_update,
        "updated_at": updated_at,
        "summary": str(world_view.get("summary", "")).strip(),
    }


@app.post("/api/world-view-ack")
def post_world_view_ack():
    world_view = _wv_load_mind()
    updated_at = str(world_view.get("updated_at", "")).strip()
    _WORLD_VIEW_NOTIFIED_PATH.write_text(
        json.dumps({"acked_at": updated_at}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"status": "acked", "acked_at": updated_at}


# ── REV-076: 感情レーダーチャート フィードバック ─────────────────────────────

class EmotionFeedbackRequest(BaseModel):
    rating: str  # 'good' | 'needs_improvement'
    comment: Optional[str] = None
    emotion_category: Optional[str] = None


def _append_emotion_feedback_to_obsidian(rating: str, comment: Optional[str], emotion_category: Optional[str]) -> dict:
    """フィードバックを Obsidian の感情可視化フィードバック.md に追記する。"""
    import datetime as _dt
    vault_raw = _OBSIDIAN_VAULT_PATH or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or ""
    if not vault_raw:
        return {"status": "skipped", "reason": "obsidian_vault_not_configured"}
    vault = Path(vault_raw).expanduser().resolve()
    if not (vault / ".obsidian").exists():
        return {"status": "skipped", "reason": "obsidian_vault_not_found"}

    now = _dt.datetime.now()
    rel = Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "感情可視化フィードバック.md"
    path = (vault / rel).resolve()
    if vault not in path.parents and path != vault:
        return {"status": "error", "reason": "unsafe_path"}
    path.parent.mkdir(parents=True, exist_ok=True)

    rating_label = "👍 わかりやすい" if rating == "good" else "📝 意見あり"
    lines = [f"\n## {now.strftime('%Y-%m-%d %H:%M')} — {rating_label}"]
    if emotion_category:
        lines.append(f"- 感情軸: {emotion_category}")
    if comment:
        lines.append(f"- コメント: {comment}")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"status": "ok", "path": str(rel)}


@app.post("/api/intelligence/emotions/feedback")
def post_emotion_feedback(req: EmotionFeedbackRequest):
    if req.rating not in ("good", "needs_improvement"):
        raise HTTPException(status_code=422, detail="rating は 'good' または 'needs_improvement' で指定してください")
    from api.database import save_emotion_feedback
    record_id = save_emotion_feedback(req.rating, req.comment, req.emotion_category)
    obsidian_result = _append_emotion_feedback_to_obsidian(req.rating, req.comment, req.emotion_category)
    return {"status": "saved", "id": record_id, "obsidian": obsidian_result}


@app.get("/api/intelligence/emotions/feedback")
def get_emotion_feedback(resolved: Optional[bool] = None):
    from api.database import get_emotion_feedbacks
    items = get_emotion_feedbacks(resolved=resolved)
    return {"items": items, "total": len(items)}


@app.get("/api/shion/self-analysis")
def get_shion_self_analysis(refresh: bool = False):
    """紫苑の自己分析を取得する（24時間キャッシュ）。"""
    from api.shion_self_analysis import get_shion_self_analysis as _get_analysis
    try:
        return _get_analysis(force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/shion/inner-state")
def get_shion_inner_state():
    """紫苑の「内面状態（経験ループ・実践知・人間フィードバック）」を一括して取得するデバッグ用エンドポイント。"""
    try:
        from api.shion_experience_loop import load_experience_state
        state = load_experience_state()

        from api.shion_practical_knowledge import load_learned_practical_map, _iter_scenes, _public_scene
        learned_map = load_learned_practical_map()
        merged_scenes = _iter_scenes(learned_map)
        public_scenes = [_public_scene(s) for s in merged_scenes]

        learned_sources = []
        for s in public_scenes:
            for src in s.get("learned_sources") or []:
                if src not in learned_sources:
                    learned_sources.append(src)

        feedback_summary = {}
        routes_to_check = ["relationship_ux", "environment_continuity", "lease_judgment", "implementation"]
        for r in routes_to_check:
            feedback_summary[r] = _summarize_human_response_feedback(r)

        return {
            "experience_count": state.get("experience_count", 0),
            "current_focus": state.get("current_focus", ""),
            "self_narrative": state.get("self_narrative", ""),
            "mood": state.get("mood", {}),
            "confidence": state.get("confidence", {}),
            "recent_experiences": state.get("recent_experiences", []),
            "next_response_bias": state.get("next_response_bias", []),
            "open_questions": state.get("open_questions", []),
            "practical_scenes": public_scenes,
            "learned_sources": learned_sources,
            "human_feedback_summary": feedback_summary,
            "updated_at": state.get("updated_at", "")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class PromoteKeypointRequest(BaseModel):
    text: str
    case_summary: str = ""
    role: str = ""


@app.post("/api/shion/promote-keypoint")
def promote_keypoint(req: PromoteKeypointRequest):
    """討論結果から抽出した判断基準を Obsidian vault の mind.json に追記する。

    vault が無い環境（Cloud Run 等）では 503 で捨てずにローカルの保留キュー
    （data/shion_promoted_keypoints_pending.jsonl、gitignore対象）へ退避し、
    vault 環境での同期を待つ。
    """
    import json as _json
    from pathlib import Path
    from datetime import date, datetime, timezone
    from api.shion_memory_taxonomy import infer_applies_when

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text が空です")

    new_entry: dict = {
        "fact": text,
        "source": "debate",
        "case": req.case_summary,
        "date": date.today().isoformat(),
        "memory_type": "judgment_memory",
        "status": "active",
        "confidence": 0.78,
        "applies_when": infer_applies_when(text),
    }
    if req.role:
        new_entry["role"] = req.role

    vault_mind = None
    try:
        from lease_news_digest import find_vault
        vault = find_vault()
        if vault:
            candidate = (
                Path(vault)
                / "Projects"
                / "tune_lease_55"
                / "Lease Intelligence"
                / "mind.json"
            )
            if candidate.exists():
                vault_mind = candidate
    except Exception:
        vault_mind = None

    if vault_mind is not None:
        try:
            with vault_mind.open(encoding="utf-8") as f:
                data = _json.load(f)

            keypoints: list = data.get("conversation_keypoints") or []
            keypoints.append(new_entry)

            # 120件上限（古いものから削除）
            _LIMIT = 120
            if len(keypoints) > _LIMIT:
                keypoints = keypoints[-_LIMIT:]

            data["conversation_keypoints"] = keypoints

            # 書き戻し（一時ファイル経由でアトミックに）
            tmp = vault_mind.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(vault_mind)

            return {"success": True, "storage": "vault", "total_keypoints": len(keypoints)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"mind.json 書き込みエラー: {e}")

    # vault 不在（Cloud Run 等）: 保留キューへ退避してキーポイントの消失を防ぐ
    try:
        pending_path = Path(__file__).parent.parent / "data" / "shion_promoted_keypoints_pending.jsonl"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        queued_entry = {**new_entry, "queued_at": datetime.now(timezone.utc).isoformat()}
        with pending_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(queued_entry, ensure_ascii=False) + "\n")
        return {
            "success": True,
            "storage": "local_pending",
            "note": "Obsidian vault 不在のため保留キューへ保存しました。vault 環境で mind.json へ同期してください。",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保留キュー書き込みエラー: {e}")


@app.get("/api/shion/central-synthesis")
def get_central_synthesis():
    """セントラルの最新状態を返す（world_view.commentary を読んで返す）。"""
    import json as _json
    from pathlib import Path

    try:
        from lease_news_digest import find_vault
        vault = find_vault()
        if not vault:
            return {"error": "Obsidian vault が見つかりません", "commentary": {}}

        vault_mind = (
            Path(vault)
            / "Projects"
            / "tune_lease_55"
            / "Lease Intelligence"
            / "mind.json"
        )
        if not vault_mind.exists():
            return {"error": "mind.json が見つかりません", "commentary": {}}

        data = _json.loads(vault_mind.read_text(encoding="utf-8"))
        world_view = data.get("world_view") or {}
        commentary = world_view.get("commentary") or {}

        return {
            "commentary": commentary,
            "world_view_summary": world_view.get("summary", ""),
            "total_keypoints": len(data.get("conversation_keypoints") or []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
