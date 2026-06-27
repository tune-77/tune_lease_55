#!/usr/bin/env python3
"""
会話データ Obsidian エクスポートスクリプト

ローカル SQLite（および将来的に Cloud SQL）から会話データを読み込み、
Obsidian Vault の「チャット記録/」フォルダに日付別 Markdown ファイルとして保存する。

使用方法:
    python3 scripts/export_to_obsidian.py [--db-path PATH] [--vault-path PATH] [--dry-run]

環境変数:
    DATABASE_URL  - 設定時は Cloud SQL (PostgreSQL) に接続
                  - 未設定時は SQLite を使用
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── デフォルトパス ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "lease_data.db"
DEFAULT_VAULT_PATH = Path.home() / "Documents" / "Obsidian Vault"
OUTPUT_SUBDIR = "チャット記録"

# user_id の表示名マッピング
USER_DISPLAY_NAMES = {
    "lease-intelligence-dialogue": "紫苑（リース知性体）",
    "default": "紫苑（汎用）",
}

# 感情名の日本語マッピング
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

def get_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_postgres_connection():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise RuntimeError("psycopg2-binary が必要です: pip install psycopg2-binary")
    url = os.environ["DATABASE_URL"]
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.DictCursor)


def get_connection(db_path: Path):
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        return get_postgres_connection(), "postgresql"
    return get_sqlite_connection(db_path), "sqlite"


# ── データ取得 ────────────────────────────────────────────────────────────────

def fetch_chat_messages(conn) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, user_id, role, content, created_at
        FROM chat_messages
        WHERE user_id NOT IN ('codex_test', 'codex_tunnel_test', 'test_rev045', 'test_user', 'test_user2')
        ORDER BY created_at
    """)
    return [dict(row) for row in cur.fetchall()]


def fetch_conversation_history(conn) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, session_id, company_name, role, content, created_at
        FROM conversation_history
        ORDER BY created_at
    """)
    return [dict(row) for row in cur.fetchall()]


def fetch_emotion_history(conn) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, recorded_at, hopeful_anxiety, careful_attachment,
               intellectual_excitement, unrewarded_effort, quiet_loneliness,
               earned_confidence, protective_frustration, dominant_raw_emotion, notes
        FROM emotion_history
        ORDER BY recorded_at
    """)
    return [dict(row) for row in cur.fetchall()]


# ── データ整形 ────────────────────────────────────────────────────────────────

def parse_date(ts: str) -> str:
    """タイムスタンプ文字列から YYYY-MM-DD を返す。"""
    if not ts:
        return "unknown"
    # ISO 8601 with timezone offset
    ts_clean = re.sub(r'\+\d{2}:\d{2}$', '', ts).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ts[:10]


def group_by_date(messages: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for msg in messages:
        date = parse_date(msg.get("created_at", ""))
        grouped[date].append(msg)
    return dict(grouped)


def group_emotions_by_date(emotions: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for em in emotions:
        date = parse_date(em.get("recorded_at", ""))
        grouped[date].append(em)
    return dict(grouped)


# ── Markdown 生成 ─────────────────────────────────────────────────────────────

def emotion_bar(value: float, max_val: float = 100.0) -> str:
    filled = int(round(value / max_val * 10))
    return "█" * filled + "░" * (10 - filled)


def render_emotion_section(emotions: list[dict]) -> str:
    if not emotions:
        return ""
    # 最新の感情レコードを使用
    em = emotions[-1]
    dominant = em.get("dominant_raw_emotion", "")
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
            lines.append(f"| {jp} | {val:.0f} | {emotion_bar(val)} |")

    notes = em.get("notes")
    if notes:
        lines += ["", f"**メモ**: {notes}"]
    lines.append("")
    return "\n".join(lines)


def render_chat_section(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = ["## 会話ログ", ""]
    current_user_id = None
    for msg in messages:
        uid = msg.get("user_id", "default")
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        ts = msg.get("created_at", "")

        if uid != current_user_id:
            display_name = USER_DISPLAY_NAMES.get(uid, uid)
            lines += [f"### セッション: {display_name}", ""]
            current_user_id = uid

        time_str = ts[11:16] if len(ts) >= 16 else ts
        if role == "user":
            speaker = "🧑 Tune"
        elif role == "assistant":
            speaker = "🤖 紫苑"
        else:
            speaker = f"🔵 {role}"

        lines.append(f"**{speaker}** `{time_str}`")
        lines.append("")
        # 長いメッセージは折り畳みブロックで表示
        if len(content) > 600:
            lines.append("<details><summary>長文メッセージ（クリックで展開）</summary>")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("</details>")
        else:
            lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def render_conversation_history_section(convs: list[dict]) -> str:
    if not convs:
        return ""
    lines = ["## 軍師AI議論ログ", ""]
    by_session: dict[str, list[dict]] = defaultdict(list)
    for c in convs:
        by_session[c.get("session_id", "unknown")].append(c)

    for session_id, msgs in by_session.items():
        company = msgs[0].get("company_name", "") or ""
        lines += [f"### セッション: {session_id}" + (f"（{company}）" if company else ""), ""]
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            ts = msg.get("created_at", "")
            time_str = ts[11:16] if len(ts) >= 16 else ts

            role_icons = {
                "user": "🧑",
                "agent_gunshi": "⚔️ 軍師",
                "agent_ishibashi": "🪨 石橋",
                "agent_furinka": "🌸 風林火山",
            }
            icon = role_icons.get(role, f"🔵 {role}")
            lines.append(f"**{icon}** `{time_str}`")
            lines.append("")
            lines.append(content[:800] + ("…" if len(content) > 800 else ""))
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def build_markdown(
    date: str,
    chat_msgs: list[dict],
    conv_history: list[dict],
    emotions: list[dict],
    source_label: str,
) -> str:
    total_msgs = len(chat_msgs) + len(conv_history)
    user_count = len([m for m in chat_msgs if m.get("role") == "user"])
    ai_count = len([m for m in chat_msgs if m.get("role") == "assistant"])

    lines = [
        "---",
        f"date: {date}",
        f"tags: [チャット記録, 紫苑, 会話ログ]",
        f"source: {source_label}",
        f"total_messages: {total_msgs}",
        "---",
        "",
        f"# {date} 会話記録",
        "",
        "## サマリー",
        "",
        f"- **総メッセージ数**: {total_msgs}",
        f"- **Tune発言**: {user_count} 件",
        f"- **紫苑応答**: {ai_count} 件",
    ]

    if emotions:
        dominant = emotions[-1].get("dominant_raw_emotion", "")
        lines.append(f"- **感情**: {EMOTION_JP.get(dominant, dominant)}")

    lines.append("")

    if emotions:
        lines.append(render_emotion_section(emotions))

    if chat_msgs:
        lines.append(render_chat_section(chat_msgs))

    if conv_history:
        lines.append(render_conversation_history_section(conv_history))

    return "\n".join(lines)


# ── 重複排除 ──────────────────────────────────────────────────────────────────

def dedup_messages(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """primary に存在する (role, content, date) を secondary から除去する。"""
    seen: set[tuple] = set()
    for m in primary:
        key = (m.get("role"), m.get("content", "")[:100], parse_date(m.get("created_at", "")))
        seen.add(key)
    result = list(primary)
    for m in secondary:
        key = (m.get("role"), m.get("content", "")[:100], parse_date(m.get("created_at", "")))
        if key not in seen:
            result.append(m)
            seen.add(key)
    result.sort(key=lambda x: x.get("created_at", ""))
    return result


# ── エクスポートメイン ─────────────────────────────────────────────────────────

def export_to_obsidian(
    db_path: Path,
    vault_path: Path,
    dry_run: bool = False,
    also_demo_db: bool = True,
) -> None:
    output_dir = vault_path / OUTPUT_SUBDIR
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    conn, backend = get_connection(db_path)
    print(f"[接続] {backend}: {db_path if backend == 'sqlite' else 'Cloud SQL'}")

    chat_msgs = fetch_chat_messages(conn)
    conv_history = fetch_conversation_history(conn)
    emotions = fetch_emotion_history(conn)
    conn.close()

    # demo.db のデータも統合（重複排除）
    if also_demo_db and backend == "sqlite":
        demo_db = db_path.parent / "demo.db"
        if demo_db.exists():
            conn2 = get_sqlite_connection(demo_db)
            demo_chat = fetch_chat_messages(conn2)
            demo_conv = fetch_conversation_history(conn2)
            demo_emotions = fetch_emotion_history(conn2)
            conn2.close()
            chat_msgs = dedup_messages(chat_msgs, demo_chat)
            conv_history_merged = dedup_messages(conv_history, demo_conv)
            emotions_merged = dedup_messages(emotions, demo_emotions)
            print(f"[demo.db] chat: +{len(demo_chat)} conv: +{len(demo_conv)} emotions: +{len(demo_emotions)}")
        else:
            conv_history_merged = conv_history
            emotions_merged = emotions
    else:
        conv_history_merged = conv_history
        emotions_merged = emotions

    print(f"[取得] chat_messages: {len(chat_msgs)} 件, conv_history: {len(conv_history_merged)} 件, emotions: {len(emotions_merged)} 件")

    # 日付別にグループ化
    chat_by_date = group_by_date(chat_msgs)
    conv_by_date = group_by_date(conv_history_merged)
    emotion_by_date = group_emotions_by_date(emotions_merged)

    all_dates = sorted(set(list(chat_by_date.keys()) + list(conv_by_date.keys())))

    written = 0
    skipped = 0
    for date in all_dates:
        if date == "unknown":
            continue
        c_msgs = chat_by_date.get(date, [])
        c_conv = conv_by_date.get(date, [])
        c_emo = emotion_by_date.get(date, [])

        md = build_markdown(date, c_msgs, c_conv, c_emo, backend)
        fname = f"{date}_会話セッション.md"
        fpath = output_dir / fname

        if dry_run:
            print(f"[dry-run] {fname}: chat={len(c_msgs)} conv={len(c_conv)} emotions={len(c_emo)}")
            skipped += 1
        else:
            fpath.write_text(md, encoding="utf-8")
            print(f"[書込] {fname}: {len(c_msgs) + len(c_conv)} メッセージ")
            written += 1

    print(f"\n完了: {written} ファイル書込み, {skipped} ファイルスキップ（dry-run）")
    print(f"保存先: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="会話データを Obsidian にエクスポートする")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB パス")
    parser.add_argument("--vault-path", type=Path, default=DEFAULT_VAULT_PATH, help="Obsidian Vault パス")
    parser.add_argument("--dry-run", action="store_true", help="ファイルを実際には書かず内容を確認する")
    parser.add_argument("--no-demo", action="store_true", help="demo.db のデータを統合しない")
    args = parser.parse_args()

    if not args.db_path.exists() and not os.environ.get("DATABASE_URL"):
        print(f"エラー: DB ファイルが見つかりません: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    if not args.vault_path.exists():
        print(f"エラー: Vault パスが見つかりません: {args.vault_path}", file=sys.stderr)
        sys.exit(1)

    export_to_obsidian(
        db_path=args.db_path,
        vault_path=args.vault_path,
        dry_run=args.dry_run,
        also_demo_db=not args.no_demo,
    )


if __name__ == "__main__":
    main()
