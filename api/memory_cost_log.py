"""記憶構築コストの計測ログ。

Stanford の Agent Memory 論文が指摘する「construction コスト（LLM要約・
LLM再ランク・embedding生成など、記憶の書き込みにかかる処理）はクエリ側の
レイテンシしか見ていないと見落とされる」という点に対応するため、記憶を
書き込む処理の所要時間と件数を data/memory_construction_cost.jsonl へ
1行1レコードで追記する。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl

_LOG_PATH = DATA_DIR / "memory_construction_cost.jsonl"


def log_memory_cost(
    *,
    phase: str,
    func: str,
    elapsed_ms: float,
    item_count: int,
    tokens: int | None = None,
    path: Path = _LOG_PATH,
) -> None:
    """記憶構築/再ランクのコストを追記する。失敗しても呼び出し元を止めない。"""
    entry: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "func": func,
        "elapsed_ms": round(elapsed_ms, 1),
        "item_count": item_count,
    }
    if tokens is not None:
        entry["tokens"] = tokens
    try:
        append_jsonl(path, entry)
    except OSError:
        pass
