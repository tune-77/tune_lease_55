"""GenAI App Builder トライアルクレジットの間だけ Vertex AI Search を本番の根拠検索に広げるスイッチ。

on の時だけ有効になる拡張（off なら従来どおり: Search は従来どおり、Answer はヒント時のみ、
同期は手動、並べ替えなし）:
- 審査系の質問（lease_screening / lease_knowledge）で回答前に Answer API（グラウンディング付き）
- Obsidian→データストアの毎日自動同期（scripts/run_vertex_credit_daily.sh、launchd）
- 毎晩の ChromaDB vs Vertex 品質比較と Ranking API 並べ替えの shadow 評価、効果確認時のみ本番適用

off になる条件（どれか1つ）:
- 環境変数 VERTEX_CREDIT_MODE=off
- クレジット期限（2027-02-01）以降
- 推定累計利用額がクレジット額に達した（scripts/vertex_credit_monitor.py が状態ファイルに auto_off を書く）
- Cloud Run 上（K_SERVICE あり）で VERTEX_CREDIT_MODE=on が明示されていない。自動 off の状態ファイルは
  ローカルにしか無く Cloud Run には届かないため、Cloud Run では明示しない限り従来どおりにする。
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

CREDIT_NAME = "Trial credit for GenAI App Builder"
CREDIT_JPY = 152_846
CREDIT_EXPIRES = dt.date(2027, 2, 1)  # この日以降は自動 off
WARN_RATIO = 0.9
_REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.environ.get("VERTEX_CREDIT_STATE_PATH") or _REPO_ROOT / "data" / "vertex_credit_state.json")
SCREENING_CATEGORIES = {"lease_screening", "lease_knowledge"}


def load_state(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads((path or STATE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def credit_mode_status(*, today: dt.date | None = None, state_path: Path | None = None) -> dict[str, Any]:
    """{"active": bool, "reason": str}。理由は朝報と API 応答のデバッグ情報に出す。"""
    today = today or dt.date.today()
    raw = (os.environ.get("VERTEX_CREDIT_MODE") or "").strip().lower()
    if raw in {"off", "0", "false", "no"}:
        return {"active": False, "reason": "VERTEX_CREDIT_MODE=off"}
    if os.environ.get("K_SERVICE") and raw not in {"on", "1", "true", "yes"}:
        return {"active": False, "reason": "cloud_run_default_off"}
    if today >= CREDIT_EXPIRES:
        return {"active": False, "reason": f"credit_expired({CREDIT_EXPIRES.isoformat()})"}
    state = load_state(state_path)
    if state.get("auto_off"):
        return {"active": False, "reason": f"auto_off: {state.get('auto_off_reason') or 'credit_exhausted'}"}
    return {"active": True, "reason": "on"}


def is_active(**kwargs: Any) -> bool:
    return bool(credit_mode_status(**kwargs)["active"])


def rerank_promoted(state_path: Path | None = None) -> bool:
    """毎晩の shadow 評価で Ranking API の並べ替えが ChromaDB 単独より良いと確認できた時だけ True。"""
    return bool((load_state(state_path).get("rerank") or {}).get("promoted"))
