"""Mid-term memory prompt loading for chat surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from api.chat_identity_memory import extract_markdown_section, read_chat_memory_file
from api.chat_retrieval import chat_memory_roots


CHAT_MEMORY_REL_DIR = Path("Projects/tune_lease_55/Lease Intelligence/Public/Chat Memory")
CHAT_MEMORY_LAYER_FILE = "mid-term-continuity.md"
CHAT_MEMORY_LAYER_LABEL = "Mid-term Continuity Memory"
CHAT_MEMORY_FALLBACK_SECTIONS = (
    "中期継続メモリ",
    "中期の継続論点",
    "中期記憶",
)
CHAT_MEMORY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "payload": None}
CHAT_MEMORY_CACHE_TTL_SEC = 300


def load_chat_mid_term_memory_payload(obsidian_vault_path: str = "") -> dict[str, Any]:
    import time as _time

    now = _time.time()
    cached = CHAT_MEMORY_CACHE.get("payload")
    if cached is not None and now - float(CHAT_MEMORY_CACHE.get("loaded_at") or 0) < CHAT_MEMORY_CACHE_TTL_SEC:
        return cached

    block = ""
    refs: list[str] = []
    latest_pack_text = ""
    latest_pack_ref = ""

    for root in chat_memory_roots(obsidian_vault_path):
        memory_dir = root / CHAT_MEMORY_REL_DIR
        if not memory_dir.exists():
            continue
        path = memory_dir / CHAT_MEMORY_LAYER_FILE
        text = read_chat_memory_file(path, limit=5_000)
        if text:
            block = text
            refs.append(str(path))
            break
        if not latest_pack_text:
            latest_path = memory_dir / "latest_cloud_chat_memory_pack.md"
            latest_pack_text = read_chat_memory_file(latest_path, limit=5_000)
            latest_pack_ref = str(latest_path) if latest_pack_text else ""

    if not block and latest_pack_text:
        for section in CHAT_MEMORY_FALLBACK_SECTIONS:
            text = extract_markdown_section(latest_pack_text, section, limit=3_000)
            if text:
                block = text
                if latest_pack_ref:
                    refs.append(latest_pack_ref)
                break

    prompt_block = ""
    if block:
        prompt_block = "\n".join(
            [
                "【中期継続メモリ】",
                "以下は直近数日から1週間程度の継続論点です。短期の会話運びと長期の判断原則を混同せず、",
                "同じ不満・同じ論点・同じ振る舞いが続く時だけ、応答方針を少し調整してください。",
                "長期記憶のように断定せず、現在の流れとして自然に扱ってください。",
                "",
                block.strip(),
            ]
        ).strip()

    payload = {"block": prompt_block, "refs": refs[:4]}
    CHAT_MEMORY_CACHE.update(loaded_at=now, payload=payload)
    return payload


def build_chat_mid_term_memory_prompt_block(obsidian_vault_path: str = "") -> tuple[str, dict[str, Any]]:
    try:
        payload = load_chat_mid_term_memory_payload(obsidian_vault_path)
    except Exception as exc:
        print(f"[ChatMidTermMemory] 読み込みエラー: {exc}")
        payload = {"block": "", "refs": []}
    block = str(payload.get("block") or "").strip()
    return (f"\n\n{block}" if block else ""), payload


def invalidate_chat_mid_term_memory_cache() -> None:
    CHAT_MEMORY_CACHE.update(loaded_at=0.0, payload=None)
