"""Cloud Run 一時停止中（2026-10-01〜、PR #1188）は Cloud Run とのデータ受け渡しを止める。

切り替えは config/cloudrun_pause.json の `paused` 1か所（環境変数 CLOUDRUN_PAUSED=1/0 があればそちらを優先）。
停止中に止める処理と再開手順は config/cloudrun_pause.json の `_doc` を参照。

止めた処理は silent_failures に kind="paused_skip" で残す。失敗には数えず、朝報には
「Cloud Run 停止中：同期N件スキップ」の1行だけ出る（silent_failure_log.morning_report_lines）。

    from cloudrun_pause import skip_if_paused
    if skip_if_paused("backup.sync_ledger_to_gcs.upload"):
        return True

シェルからは `python cloudrun_pause.py <部品名>` が停止中なら 0（スキップする）、稼働中なら 1 を返す。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "cloudrun_pause.json"
PAUSED_KIND = "paused_skip"


def is_paused() -> bool:
    raw = os.environ.get("CLOUDRUN_PAUSED", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    try:
        return bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("paused"))
    except (OSError, ValueError, AttributeError):
        return False


def skip_if_paused(component: str) -> bool:
    """停止中なら「停止中のためスキップ」を記録して True。呼び出し側はそのまま正常終了する。"""
    if not is_paused():
        return False
    from silent_failure_log import record_silent_failure

    record_silent_failure(component, PAUSED_KIND, detail="Cloud Run 停止中のためスキップ")
    print(f"[SKIP] Cloud Run 停止中のため {component} をスキップ（config/cloudrun_pause.json）")
    return True


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(0 if skip_if_paused(sys.argv[1] if len(sys.argv) > 1 else "cloudrun.unknown") else 1)
