from __future__ import annotations

import datetime
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_HISTORY = PROJECT_ROOT / "data" / "model_history.jsonl"


def append_model_history(event_type: str, engine: str, from_version: str = "", to_version: str = "", reason: str = "", meta: dict | None = None) -> None:
    MODEL_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "engine": engine,
        "from_version": from_version,
        "to_version": to_version,
        "reason": reason,
        "meta": meta or {},
    }
    with open(MODEL_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
