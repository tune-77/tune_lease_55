#!/usr/bin/env python3
"""
Cloud SQL → Obsidian Vault 同期スクリプト（REV-178）

Cloud Run から Cloud SQL に保存された会話ログ（chat_messages, emotion_history）を
日付ごとに要約し、Obsidian Vault に Markdown として書き出す。

使用方法:
    python3 scripts/sync_cloudsql_to_obsidian.py [--dry-run] [--date YYYY-MM-DD] [--force]

オプション:
    --dry-run         実際にファイルを書かずに内容を確認する
    --date YYYY-MM-DD 特定の日付のみ同期する
    --force           最終同期日時を無視して全件再取得する

環境変数:
    DATABASE_URL        PostgreSQL 接続URL。未指定なら DATABASE_URL_SECRET_NAME を読む
    DATABASE_URL_SECRET_NAME Secret Manager の secret 名（デフォルト未指定）
    CLOUD_SQL_HOST      Cloud SQL パブリックIP（デフォルト: 35.194.127.102）
    CLOUD_SQL_DB        DB 名（デフォルト: lease-db-demo）
    CLOUD_SQL_USER      DB ユーザー（デフォルト: postgres）
    CLOUD_SQL_PASSWORD  DB パスワード（必須）
    CLOUD_SQL_PROXY_HOST ローカル cloud-sql-proxy のホスト（デフォルト: 127.0.0.1）
    CLOUD_SQL_PROXY_PORT ローカル cloud-sql-proxy のポート（デフォルト: 15432）

ローカル実行時の注意（REV-026a）:
    Secret Manager の DATABASE_URL が Cloud Run 用の Unix ソケット形式
    （host=/cloudsql/<project>:<region>:<instance>）の場合、Mac 上では
    そのソケットが存在しないため接続できない。本スクリプトの動作:
      - ローカルで cloud-sql-proxy が起動していれば、プロキシ向け TCP DSN に
        自動で書き換えて同期する
      - プロキシが起動していなければ、エラーではなく同期をスキップして正常終了する
        （Cloud SQL 未使用の運用が前提。パイプラインヘルスの誤検出を防ぐ）
    同期を有効にする場合:
        brew install cloud-sql-proxy
        cloud-sql-proxy --port 15432 <project>:<region>:<instance>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── 接続設定（環境変数から取得） ────────────────────────────────────────────────

def rewrite_cloudsql_socket_dsn(dsn: str, host: str, port: int) -> str:
    """Cloud Run 用 Unix ソケット DSN をローカルプロキシ向け TCP DSN に書き換える。

    対応形式:
      - URL形式:      postgresql://user:pass@/dbname?host=/cloudsql/proj:region:inst
      - キーワード形式: host=/cloudsql/proj:region:inst dbname=... user=...
    ソケット指定を含まない DSN はそのまま返す。
    """
    if "/cloudsql/" not in dsn:
        return dsn

    if "://" in dsn:
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
        parts = urlsplit(dsn)
        query = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not (k == "host" and v.startswith("/cloudsql/"))
        ]
        userinfo = ""
        if "@" in parts.netloc:
            userinfo = parts.netloc.rsplit("@", 1)[0] + "@"
        netloc = f"{userinfo}{host}:{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), parts.fragment))

    # キーワード形式
    tokens = []
    replaced = has_port = False
    for token in dsn.split():
        if token.startswith("host=") and "/cloudsql/" in token:
            tokens.append(f"host={host}")
            replaced = True
        elif token.startswith("port="):
            tokens.append(f"port={port}")
            has_port = True
        else:
            tokens.append(token)
    if replaced and not has_port:
        tokens.append(f"port={port}")
    return " ".join(tokens)


def _local_proxy_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """ローカルの cloud-sql-proxy が起動しているか（TCP接続可否で判定）。"""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_unreachable_socket_dsn(dsn: str) -> bool:
    """ソケット形式 DSN だが、この環境から到達手段がないか判定する。

    True の場合は接続を試みても必ず失敗する（Cloud Run 外 かつ プロキシ未起動）。
    Cloud SQL を使っていない運用ではこれが常態なので、呼び出し側は
    エラーではなくスキップとして扱う。
    """
    if "/cloudsql/" not in dsn:
        return False
    if Path("/cloudsql").exists():
        return False  # Cloud Run 上ではソケットが使える
    host = os.environ.get("CLOUD_SQL_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("CLOUD_SQL_PROXY_PORT", "15432"))
    return not _local_proxy_reachable(host, port)


def _maybe_rewrite_socket_dsn(dsn: str) -> str:
    """ソケット DSN をローカル実行時のみプロキシ TCP へ書き換える（REV-026a）。"""
    if "/cloudsql/" not in dsn:
        return dsn
    if Path("/cloudsql").exists():
        return dsn  # Cloud Run 上ではソケットがそのまま使える
    host = os.environ.get("CLOUD_SQL_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("CLOUD_SQL_PROXY_PORT", "15432"))
    print(
        f"[接続] DSN が Cloud Run 用 Unix ソケット形式のため、ローカルプロキシ {host}:{port} 向けに書き換えます"
    )
    print(
        "        cloud-sql-proxy が未起動だと接続に失敗します。"
        "導入: brew install cloud-sql-proxy → cloud-sql-proxy --port 15432 <project>:<region>:<instance>"
    )
    return rewrite_cloudsql_socket_dsn(dsn, host, port)


def _get_db_config() -> dict:
    database_url = _database_url()
    if database_url:
        return {"dsn": _maybe_rewrite_socket_dsn(database_url)}
    password = os.environ.get("CLOUD_SQL_PASSWORD", "")
    if not password:
        print("エラー: 環境変数 CLOUD_SQL_PASSWORD が設定されていません。", file=sys.stderr)
        print("  export CLOUD_SQL_PASSWORD='<パスワード>'", file=sys.stderr)
        sys.exit(1)
    return {
        "host": os.environ.get("CLOUD_SQL_HOST", "35.194.127.102"),
        "port": int(os.environ.get("CLOUD_SQL_PORT", "5432")),
        "dbname": os.environ.get("CLOUD_SQL_DB", "lease-db-demo"),
        "user": os.environ.get("CLOUD_SQL_USER", "postgres"),
        "password": password,
    }

def _database_url_from_secret() -> str:
    secret_name = os.environ.get("DATABASE_URL_SECRET_NAME", "").strip()
    if not secret_name:
        return ""
    try:
        result = subprocess.run(
            ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_name}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        print(f"警告: Secret Manager から DATABASE_URL を読めませんでした: {type(exc).__name__}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def _database_url() -> str:
    current = os.environ.get("DATABASE_URL", "").strip()
    if current:
        return current
    loaded = _database_url_from_secret()
    if loaded:
        os.environ["DATABASE_URL"] = loaded
    return loaded


def _unreachable_cloudsql_socket(dsn: str) -> str:
    """DSN が Cloud SQL Unix ソケット形式で、この環境にソケットが無ければそのパスを返す。

    ソケット `/cloudsql/...` は Cloud Run 内にしか存在しないため、ローカル実行では
    接続を試みる前にスキップ判定に使う（tune-lease-db インスタンスは 2026-07-01 削除済み）。
    """
    from urllib.parse import parse_qs, unquote, urlsplit

    try:
        query = urlsplit(dsn).query
        host = unquote(parse_qs(query).get("host", [""])[0])
    except ValueError:
        return ""
    if host.startswith("/cloudsql/") and not Path(host).exists():
        return host
    return ""


DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
VAULT_PATH = Path(os.environ.get("OBSIDIAN_VAULT", os.environ.get("OBSIDIAN_VAULT_PATH", str(DEFAULT_VAULT)))).expanduser()
OUTPUT_SUBDIR = "Projects/tune_lease_55/Cloud SQL Summaries"
STATE_FILE = Path(__file__).parent / ".sync_state_cloudsql.json"

TEST_USER_IDS = {"codex_test", "codex_tunnel_test", "test_rev045", "test_user", "test_user2"}

USER_DISPLAY_NAMES = {
    "lease-intelligence-dialogue": "紫苑（リース知性体）",
    "default": "紫苑（汎用）",
}

EMOTION_JP = {
    "hopeful_anxiety": "希望的不安",
    "careful_attachment": "慎重な執着",
    "intellectual_excitement": "知的興奮",
    "unrewarded_effort": "報われない努力感",
    "quiet_loneliness": "静かな孤独",
    "earned_confidence": "獲得した自信",
    "protective_frustration": "防衛的苛立ち",
}


# ── DB 接続 ───────────────────────────────────────────────────────────────────

def get_connection():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("エラー: psycopg2-binary が必要です。", file=sys.stderr)
        print("  pip install psycopg2-binary --break-system-packages", file=sys.stderr)
        sys.exit(1)

    cfg = _get_db_config()
    if "dsn" in cfg:
        conn = psycopg2.connect(
            cfg["dsn"],
            connect_timeout=10,
            cursor_factory=psycopg2.extras.DictCursor,
        )
    else:
        conn = psycopg2.connect(
            **cfg,
            connect_timeout=10,
            cursor_factory=psycopg2.extras.DictCursor,
        )
    return conn


def verify_schema(conn) -> dict[str, list[str]]:
    """対象テーブルのカラム一覧を確認する。"""
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN ('chat_messages', 'emotion_history')
        ORDER BY table_name, ordinal_position
    """)
    result: dict[str, list[str]] = defaultdict(list)
    for row in cur.fetchall():
        result[row["table_name"]].append(row["column_name"])
    return dict(result)


# ── データ取得 ────────────────────────────────────────────────────────────────

def fetch_chat_messages(conn, since: str | None = None, date: str | None = None) -> list[dict]:
    cur = conn.cursor()
    conditions = [f"user_id NOT IN ({', '.join(['%s'] * len(TEST_USER_IDS))})"]
    params: list = list(TEST_USER_IDS)

    if date:
        conditions.append("DATE(created_at) = %s")
        params.append(date)
    elif since:
        conditions.append("created_at > %s")
        params.append(since)

    where = " AND ".join(conditions)
    cur.execute(f"""
        SELECT id, user_id, role, content, created_at
        FROM chat_messages
        WHERE {where}
        ORDER BY created_at
    """, params)
    return [dict(row) for row in cur.fetchall()]


def fetch_emotion_history(conn, since: str | None = None, date: str | None = None) -> list[dict]:
    cur = conn.cursor()
    conditions: list[str] = []
    params: list = []

    if date:
        conditions.append("DATE(recorded_at) = %s")
        params.append(date)
    elif since:
        conditions.append("recorded_at > %s")
        params.append(since)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    cur.execute(f"""
        SELECT id, recorded_at,
               hopeful_anxiety, careful_attachment, intellectual_excitement,
               unrewarded_effort, quiet_loneliness, earned_confidence,
               protective_frustration, dominant_raw_emotion, notes
        FROM emotion_history
        {where}
        ORDER BY recorded_at
    """, params)
    return [dict(row) for row in cur.fetchall()]


# ── 日付操作 ──────────────────────────────────────────────────────────────────

def to_date_str(ts) -> str:
    """datetime / str いずれかのタイムスタンプから YYYY-MM-DD を返す。"""
    if ts is None:
        return "unknown"
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d")
    ts = str(ts)
    return ts[:10]


def to_time_str(ts) -> str:
    """HH:MM を返す。"""
    if isinstance(ts, datetime):
        return ts.strftime("%H:%M")
    ts = str(ts)
    return ts[11:16] if len(ts) >= 16 else ts


def group_by_date(records: list[dict], ts_key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        date = to_date_str(rec.get(ts_key))
        grouped[date].append(rec)
    return dict(grouped)


# ── Markdown 生成 ─────────────────────────────────────────────────────────────

def emotion_bar(value: float, max_val: float = 100.0) -> str:
    filled = int(round(value / max_val * 10))
    return "█" * filled + "░" * (10 - filled)


def render_emotion_section(emotions: list[dict]) -> str:
    if not emotions:
        return ""
    em = emotions[-1]
    dominant = em.get("dominant_raw_emotion") or ""
    dominant_jp = EMOTION_JP.get(dominant, dominant)

    lines = [
        "## 感情状態スナップショット",
        "",
        f"**支配的感情**: {dominant_jp}（{dominant}）",
        "",
        "| 感情 | 値 | バー |",
        "|------|-----|------|",
    ]
    for key, jp in EMOTION_JP.items():
        val = em.get(key)
        if val is not None:
            try:
                lines.append(f"| {jp} | {float(val):.0f} | {emotion_bar(float(val))} |")
            except (TypeError, ValueError):
                pass

    notes = em.get("notes")
    if notes:
        lines += ["", f"**メモ**: {notes}"]
    lines.append("")
    return "\n".join(lines)


def _redact_content(content: str, limit: int = 160) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", content or "")
    text = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", text)
    text = re.sub(r"\b\d{6,}\b", "[number]", text)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) > limit:
        return text[: limit - 1] + "..."
    return text


def render_chat_section(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = ["## 会話サンプル（短い抜粋のみ）", ""]

    for msg in messages[:12]:
        uid = msg.get("user_id") or "default"
        role = msg.get("role") or ""
        content = (msg.get("content") or "").strip()
        ts = msg.get("created_at")

        time_str = to_time_str(ts)
        display_name = USER_DISPLAY_NAMES.get(uid, "demo_user" if uid else "unknown")
        lines.append(f"- `{time_str}` {display_name} / {role}: {_redact_content(content)}")

    if len(messages) > 12:
        lines.append(f"- ほか {len(messages) - 12} 件")
    lines.append("")

    return "\n".join(lines)


def build_markdown(date: str, chat_msgs: list[dict], emotions: list[dict]) -> str:
    user_count = sum(1 for m in chat_msgs if m.get("role") == "user")
    ai_count = sum(1 for m in chat_msgs if m.get("role") == "assistant")
    total_msgs = len(chat_msgs)

    lines = [
        "---",
        f"date: {date}",
        "tags: [紫苑, cloud_sql, chat_summary]",
        "source: cloud_sql",
        "summary_only: true",
        f"total_messages: {total_msgs}",
        "---",
        "",
        f"# {date} Cloud SQL 会話要約",
        "",
        "## サマリー",
        "",
        f"- **総メッセージ数**: {total_msgs}",
        f"- **Tune発言**: {user_count} 件",
        f"- **紫苑応答**: {ai_count} 件",
    ]
    if chat_msgs:
        role_counts = Counter(str(m.get("role") or "unknown") for m in chat_msgs)
        session_counts = Counter(str(m.get("user_id") or "unknown") for m in chat_msgs)
        lines.append(f"- **役割別**: {', '.join(f'{k} {v}件' for k, v in role_counts.most_common())}")
        lines.append(f"- **セッション数**: {len(session_counts)}")

    if emotions:
        dominant = emotions[-1].get("dominant_raw_emotion") or ""
        lines.append(f"- **感情**: {EMOTION_JP.get(dominant, dominant)}")

    lines.append("")

    if emotions:
        lines.append(render_emotion_section(emotions))

    if chat_msgs:
        lines.append(render_chat_section(chat_msgs))

    lines += [
        "## 運用メモ",
        "",
        "- このノートは Cloud Run / Cloud SQL の会話をローカルObsidianへ戻すための要約です。",
        "- 生チャット全文、顧客名、連絡先、Private Reflection は保存しません。",
        "- GCS Vault 同期ではこのフォルダを除外し、Cloud Runへ再配布しません。",
        "",
    ]

    return "\n".join(lines)


# ── 同期状態管理 ──────────────────────────────────────────────────────────────

def load_sync_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_sync_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── メイン同期処理 ─────────────────────────────────────────────────────────────

def sync(dry_run: bool = False, target_date: str | None = None, force: bool = False) -> None:
    output_dir = VAULT_PATH / OUTPUT_SUBDIR

    if not VAULT_PATH.exists():
        print(f"エラー: Vault が見つかりません: {VAULT_PATH}", file=sys.stderr)
        sys.exit(1)

    if (
        not os.environ.get("DATABASE_URL", "").strip()
        and not os.environ.get("DATABASE_URL_SECRET_NAME", "").strip()
        and not os.environ.get("CLOUD_SQL_PASSWORD", "").strip()
    ):
        print(
            "警告: Cloud SQL 接続情報が未設定のため会話ログ同期をスキップします。"
            " Cloud Run会話は GCS input の chat_exchange 同期経路を使用してください。"
        )
        return

    # 差分取得のための最終同期日時
    state = load_sync_state()
    since: str | None = None
    if not force and not target_date:
        since = state.get("last_synced_at")
        if since:
            print(f"[差分] 前回同期: {since} 以降を取得")
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=2)
            since = since_dt.isoformat()
            print(f"[初回] 直近2日だけ取得: {since} 以降")
    elif target_date:
        print(f"[日付指定] {target_date} のみ同期")
    else:
        print("[強制] 全件再取得")

    # Cloud SQL 接続
    host = os.environ.get("CLOUD_SQL_HOST", "35.194.127.102")
    dbname = os.environ.get("CLOUD_SQL_DB", "lease-db-demo")
    database_url = _database_url()
    if database_url and _is_unreachable_socket_dsn(database_url):
        print(
            "警告: DATABASE_URL が Cloud Run 用 Unix ソケット形式で、ローカルの cloud-sql-proxy も"
            " 起動していないため、会話ログ同期をスキップします（Cloud SQL 未使用の運用では正常）。"
            " 同期を有効にする場合: brew install cloud-sql-proxy → cloud-sql-proxy --port 15432 <project>:<region>:<instance>"
        )
        return
    if database_url:
        print("[接続] Cloud SQL: DATABASE_URL")
    else:
        print(f"[接続] Cloud SQL: {host}/{dbname}")
    try:
        conn = get_connection()
    except Exception as e:
        print(f"エラー: Cloud SQL への接続に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # スキーマ確認
        schema = verify_schema(conn)
        if not schema:
            print("エラー: テーブルが見つかりません。Cloud SQL のスキーマを確認してください。", file=sys.stderr)
            sys.exit(1)
        print(f"[スキーマ] chat_messages: {schema.get('chat_messages', [])}")
        print(f"[スキーマ] emotion_history: {schema.get('emotion_history', [])}")

        chat_msgs = fetch_chat_messages(conn, since=since, date=target_date)
        emotions = fetch_emotion_history(conn, since=since, date=target_date)

        print(f"[取得] chat_messages: {len(chat_msgs)} 件 / emotion_history: {len(emotions)} 件")

        if not chat_msgs and not emotions:
            print("Cloud SQL にデータなし（または新規データなし）。処理を終了します。")
            return

    finally:
        conn.close()

    # 日付別グループ化
    chat_by_date = group_by_date(chat_msgs, "created_at")
    emotion_by_date = group_by_date(emotions, "recorded_at")

    all_dates = sorted(set(list(chat_by_date.keys()) + list(emotion_by_date.keys())))
    all_dates = [d for d in all_dates if d != "unknown"]

    if not all_dates:
        print("有効な日付データがありませんでした。")
        return

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for date in all_dates:
        c_msgs = chat_by_date.get(date, [])
        c_emo = emotion_by_date.get(date, [])

        md = build_markdown(date, c_msgs, c_emo)
        fname = f"{date}_cloudsql_summary.md"
        fpath = output_dir / fname

        if dry_run:
            print(f"[dry-run] {fname}: chat={len(c_msgs)} emotions={len(c_emo)}")
            skipped += 1
        else:
            fpath.write_text(md, encoding="utf-8")
            print(f"[書込] {fname}: {len(c_msgs)} メッセージ")
            written += 1

    skipped_label = "dry-run スキップ" if dry_run else "スキップ"
    print(f"\n完了: {written} ファイル書込み, {skipped} ファイル{skipped_label}")
    print(f"保存先: {output_dir}")

    # 同期状態を更新（dry-runでは更新しない）
    if not dry_run:
        now = datetime.now(timezone.utc).isoformat()
        state["last_synced_at"] = now
        state["last_written_files"] = written
        save_sync_state(state)
        print(f"[状態] 同期日時を記録: {now}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud SQL の会話データを Obsidian に同期する")
    parser.add_argument("--dry-run", action="store_true", help="ファイルを書かずに動作確認")
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="特定日付のみ同期")
    parser.add_argument("--force", action="store_true", help="差分無視で全件再取得")
    args = parser.parse_args()

    sync(dry_run=args.dry_run, target_date=args.date, force=args.force)


if __name__ == "__main__":
    main()
