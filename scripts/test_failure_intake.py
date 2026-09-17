#!/usr/bin/env python3
"""pytest --junitxml 出力からテスト失敗を needs_review 候補として取り込む。

analyze_error_logs.py のログエラー検出と対になる、テスト失敗版の候補発見ソース。
recursive_self_improvement.py の main() から load_test_failure_intake() 経由で
report["needs_review"] にマージされる。target_module は常に空文字とし、
auto_fix_policy.evaluate_auto_fix_policy() が必ず手動確認（needs_review）に
倒れるようにする（このファイル自体は自動実行を一切行わない）。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TEST_FAILURE_INTAKE_PATH = _REPO_ROOT / "data" / "test_failure_intake.jsonl"

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str, max_len: int = 200) -> str:
    """制御文字・NULLバイトを除去し、max_len 文字に切り詰める。"""
    cleaned = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or not unicodedata.category(ch).startswith("C")
    )
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
    return cleaned[:max_len]


def _mask_dynamic_values(text: str) -> str:
    """テスト失敗メッセージ中の動的な値をマスクし、実行のたびに変わらない形にする。

    analyze_error_logs.py の _extract_error_key と同種の意図（UUID・数値・長い
    引用文字列の置換）だが、pytest のトレースバック/アサーション文言はログ行より
    構造が異なる（比較値の差分表示、ファイルパス+行番号が頻出する等）ため、
    ここでは独立実装とする（インポートせず、モジュール間の新規結合を避ける）。

    マスク後の文字列は candidate の description に使われ、canonical_key() の
    安定性（同じ失敗が日をまたいで同じキーになること）に直結する。

    設計判断（採用済み）: アサーション比較値まで含めて数値を一律マスクする方針
    （analyze_error_logs.py と同じ粒度）。比較値ごとの差分表示は失う代わりに、
    同じテストの失敗が実行のたびに別候補として積み上がることを防ぐ。
    """
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "<uuid>", text)
    text = re.sub(r"\b\d{4,}\b", "<N>", text)
    text = re.sub(r"'[^']{20,}'", "'...'", text)
    text = re.sub(r'"[^"]{20,}"', '"..."', text)
    return text


def _dedup_key(test_id: str, error_type: str) -> str:
    digest = hashlib.sha1(f"{test_id}:{error_type}".encode("utf-8")).hexdigest()[:12]
    return f"TESTFAIL-{digest}"


def _iter_failures(root: ET.Element) -> list[dict[str, str]]:
    """<testcase> 配下の <failure>/<error> を列挙する（<testsuites>/<testsuite> どちらの
    ルート構造でも root.iter() で拾えるため、ネスト差異を吸収する。"""
    failures: list[dict[str, str]] = []
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname") or ""
        name = testcase.get("name") or ""
        test_id = f"{classname}::{name}" if classname else name
        if not test_id:
            continue
        for tag in ("failure", "error"):
            elem = testcase.find(tag)
            if elem is None:
                continue
            error_type = elem.get("type") or tag
            message = elem.get("message") or (elem.text or "")
            failures.append({
                "test_id": test_id,
                "error_type": error_type,
                "message": message,
            })
    return failures


def _load_existing_dedup_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("id"):
            keys.add(str(record["id"]))
    return keys


def record_from_junit(junit_path: Path, source: str) -> int:
    """JUnit XML を読み、未記録の失敗テストを data/test_failure_intake.jsonl に追記する。

    戻り値は新規追記した件数。ファイル不在・パース失敗時は 0 を返す（呼び出し側の
    日次パイプラインを止めないため、例外を送出しない）。
    """
    if not junit_path.exists():
        return 0
    try:
        tree = ET.parse(junit_path)
    except ET.ParseError:
        return 0
    root = tree.getroot()

    failures = _iter_failures(root)
    if not failures:
        return 0

    existing_keys = _load_existing_dedup_keys(TEST_FAILURE_INTAKE_PATH)
    now_iso = datetime.now(timezone.utc).isoformat()

    new_lines: list[str] = []
    for failure in failures:
        dedup_id = _dedup_key(failure["test_id"], failure["error_type"])
        if dedup_id in existing_keys:
            continue
        existing_keys.add(dedup_id)
        masked_message = _mask_dynamic_values(_sanitize_text(failure["message"], max_len=500))
        title = _sanitize_text(f"テスト失敗: {failure['test_id']}", max_len=200)
        description = _sanitize_text(
            f"{failure['error_type']}: {masked_message}", max_len=400
        )
        entry = {
            "id": dedup_id,
            "test_id": failure["test_id"],
            "error_type": failure["error_type"],
            "title": title,
            "description": description,
            "source": source,
            "recorded_at": now_iso,
        }
        new_lines.append(json.dumps(entry, ensure_ascii=False))

    if not new_lines:
        return 0

    TEST_FAILURE_INTAKE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_FAILURE_INTAKE_PATH.open("a", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")
    return len(new_lines)


def load_test_failure_intake(path: Path | None = None) -> list[dict[str, Any]]:
    """test_failure_intake.jsonl を needs_review 候補の共通形に変換して読み込む。

    load_chat_quick_fix_intake() と同じ dict 形（id/title/description/target_module/
    source）を返す。target_module は常に空文字とし、auto_fix_policy が
    「対象ファイル未特定のため手動確認」で needs_review に倒すことを保証する。
    """
    intake_path = path or TEST_FAILURE_INTAKE_PATH
    if not intake_path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in intake_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id") or "")
        title = str(record.get("title") or "").strip()
        if not record_id or not title:
            continue
        items.append({
            "id": record_id,
            "title": title,
            "description": str(record.get("description") or "").strip(),
            "target_module": "",
            "source": str(record.get("source") or "test_failure_intake"),
        })
    return items
