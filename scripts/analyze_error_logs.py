#!/usr/bin/env python3
"""FastAPI / アプリケーションエラーログを解析して改善台帳に追記するスクリプト。

対象ログ: logs/api.log, logs/app.log
同じエラーパターンが3回以上出現した場合に ledger_rules.json へ
type="manual", pending_review=true で追記する。
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"
LOGS_DIR = PROJECT_ROOT / "logs"

LOG_FILES = ["api.log", "app.log"]
LOOKBACK_DAYS = 7
MIN_COUNT = 3
_MAX_LINE_LEN = 2000  # これより長い行は悪意ある注入またはバイナリデータとみなしてスキップ

# ledger フィールドで許可する文字: 英数字・日本語・記号・スペース・改行なし制御文字除去済み
_SAFE_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str, max_len: int = 200) -> str:
    """制御文字・NULLバイトを除去し、max_len 文字に切り詰める。"""
    # unicodedata カテゴリが Cc（制御文字）のうち tab/LF/CR 以外を除去
    cleaned = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or not unicodedata.category(ch).startswith("C")
    )
    # ASCII 制御文字（0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f, 0x7f）を追加除去
    cleaned = _SAFE_TEXT_RE.sub("", cleaned)
    return cleaned[:max_len]


# 集計対象: ERROR / CRITICAL / Traceback / Exception
_ERROR_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"  # timestamp
    r".*?\[(ERROR|CRITICAL|Traceback|Exception)\]"
    r"\s*(.*)"
)

# モジュール名を抽出 (例: [module_name] → module_name)
_MODULE_RE = re.compile(r"\[([A-Za-z0-9_\.]+)\]")


def _parse_ts(ts_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _extract_error_key(line: str) -> str | None:
    """エラー行からグルーピングキーを抽出する。制御文字を除去してから処理する。"""
    line = _sanitize_text(line, max_len=_MAX_LINE_LEN)
    m = _ERROR_LINE_RE.search(line)
    if not m:
        # Traceback 行や Exception 行も拾う
        if "Traceback" in line or "Exception" in line or "Error:" in line:
            # 末尾の具体的な値（ファイルパス・ID等）を除いてキーにする
            key = re.sub(r"'[^']{20,}'", "'...'", line.strip())
            key = re.sub(r'"[^"]{20,}"', '"..."', key)
            key = re.sub(r"\b\d{4,}\b", "N", key)
            return key[:120]
        return None
    level = m.group(2)
    msg = m.group(3).strip()
    # 動的な値（数値・UUID・タイムスタンプ・長い文字列）をマスク
    msg = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "<uuid>", msg)
    msg = re.sub(r"\b\d{4,}\b", "<N>", msg)
    msg = re.sub(r"'[^']{20,}'", "'...'", msg)
    return f"[{level}] {msg[:100]}"


def load_error_counts(lookback_days: int = LOOKBACK_DAYS) -> dict[str, int]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    counts: dict[str, int] = defaultdict(int)

    for log_name in LOG_FILES:
        log_path = LOGS_DIR / log_name
        if not log_path.exists():
            continue
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        current_ts: datetime | None = None
        for line in lines:
            # 2000 文字超の行はバイナリデータまたは注入試行とみなしてスキップ
            if len(line) > _MAX_LINE_LEN:
                continue
            ts_m = re.match(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line)
            if ts_m:
                current_ts = _parse_ts(ts_m.group(1))
            if current_ts and current_ts < cutoff:
                continue
            key = _extract_error_key(line)
            if key:
                counts[key] += 1

    return dict(counts)


def load_ledger() -> list[dict]:
    if not LEDGER_FILE.exists():
        return []
    try:
        return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_ledger(rules: list[dict]) -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def already_exists(ledger: list[dict], error_key: str) -> bool:
    for entry in ledger:
        if error_key[:60] in str(entry.get("description") or ""):
            return True
    return False


def max_rev_number(ledger: list[dict]) -> int:
    max_num = 0
    for entry in ledger:
        m = re.match(r"REV-(\d+)", str(entry.get("rev_id") or ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def main() -> None:
    counts = load_error_counts()
    frequent = {k: v for k, v in counts.items() if v >= MIN_COUNT}

    if not frequent:
        print(
            f"[analyze_error_logs] 直近{LOOKBACK_DAYS}日で{MIN_COUNT}回以上のエラーパターンなし。",
            flush=True,
        )
        return

    ledger = load_ledger()
    base_rev = max_rev_number(ledger)
    now_iso = datetime.now(timezone.utc).isoformat()
    added = 0

    for error_key, count in sorted(frequent.items(), key=lambda x: -x[1]):
        if already_exists(ledger, error_key):
            print(f"[analyze_error_logs] スキップ（既存）: {error_key[:60]}", flush=True)
            continue
        base_rev += 1
        rev_id = f"REV-{base_rev:03d}e"
        safe_key = _sanitize_text(error_key, max_len=80)
        desc = _sanitize_text(
            f"[エラーログ自動検出] 直近{LOOKBACK_DAYS}日で{count}回出現: {safe_key}",
            max_len=200,
        )
        new_entry = {
            "rev_id": rev_id,
            "type": "manual",
            "pending_review": True,
            "category": "error_log_fix",
            "description": desc,
            "status": "pending_review",
            "source": "analyze_error_logs",
            "detected_at": now_iso,
            "error_count": count,
            "error_pattern": _sanitize_text(error_key, max_len=120),
            "affected_files": [],
            "risk": "medium",
            "auto_fix_allowed": False,
        }
        ledger.append(new_entry)
        added += 1
        print(f"[analyze_error_logs] 追記: {rev_id} ({count}回) — {error_key[:60]}", flush=True)

    if added > 0:
        save_ledger(ledger)
        print(f"[analyze_error_logs] {added} 件を台帳に追記しました。", flush=True)
    else:
        print("[analyze_error_logs] 新規候補なし。", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
