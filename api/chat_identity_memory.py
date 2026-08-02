"""Identity-memory prompt loading for chat surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from api.chat_retrieval import chat_memory_roots


CHAT_MEMORY_REL_DIR = Path("Projects/tune_lease_55/Lease Intelligence/Public/Chat Memory")
CHAT_MEMORY_LAYER_FILES = {
    "identity": "identity.md",
    "judgment": "judgment-principles.md",
    "recent": "recent-continuity.md",
}
CHAT_MEMORY_FALLBACK_SECTIONS = {
    "identity": "長期記憶",
    "judgment": "長期記憶",
    "recent": "継続中の方針",
}
CHAT_MEMORY_LAYER_LABELS = {
    "identity": "Core Identity Memory",
    "judgment": "Judgment Memory",
    "recent": "Recent Continuity Memory",
}
CHAT_MEMORY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "payload": None}
CHAT_MEMORY_CACHE_TTL_SEC = 300


def read_chat_memory_file(path: Path, limit: int = 2_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def extract_markdown_section(markdown: str, heading: str, limit: int = 1_400) -> str:
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


def load_chat_identity_memory_payload(obsidian_vault_path: str = "") -> dict[str, Any]:
    import time as _time

    now = _time.time()
    cached = CHAT_MEMORY_CACHE.get("payload")
    if cached is not None and now - float(CHAT_MEMORY_CACHE.get("loaded_at") or 0) < CHAT_MEMORY_CACHE_TTL_SEC:
        return cached

    layers: dict[str, str] = {}
    refs: list[str] = []
    latest_pack_text = ""
    latest_pack_ref = ""

    for root in chat_memory_roots(obsidian_vault_path):
        memory_dir = root / CHAT_MEMORY_REL_DIR
        if not memory_dir.exists():
            continue
        if not latest_pack_text:
            latest_path = memory_dir / "latest_cloud_chat_memory_pack.md"
            latest_pack_text = read_chat_memory_file(latest_path, limit=5_000)
            latest_pack_ref = str(latest_path) if latest_pack_text else ""
        for layer, filename in CHAT_MEMORY_LAYER_FILES.items():
            if layer in layers:
                continue
            path = memory_dir / filename
            text = read_chat_memory_file(path)
            if text:
                layers[layer] = text
                refs.append(str(path))
        if len(layers) == len(CHAT_MEMORY_LAYER_FILES):
            break

    if latest_pack_text:
        for layer, section in CHAT_MEMORY_FALLBACK_SECTIONS.items():
            if layer in layers:
                continue
            text = extract_markdown_section(latest_pack_text, section)
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
            parts += ["", f"### {CHAT_MEMORY_LAYER_LABELS[layer]}", text]
        block = "\n".join(parts).strip()

    payload = {
        "block": block,
        "refs": refs[:8],
        "layers": {layer: bool(layers.get(layer, "").strip()) for layer in CHAT_MEMORY_LAYER_FILES},
    }
    CHAT_MEMORY_CACHE.update(loaded_at=now, payload=payload)
    return payload


def build_chat_identity_memory_prompt_block(obsidian_vault_path: str = "") -> tuple[str, dict[str, Any]]:
    try:
        payload = load_chat_identity_memory_payload(obsidian_vault_path)
    except Exception as exc:
        print(f"[ChatIdentityMemory] 読み込みエラー: {exc}")
        payload = {"block": "", "refs": [], "layers": {}}
    block = str(payload.get("block") or "").strip()
    return (f"\n\n{block}" if block else ""), payload


def invalidate_chat_identity_memory_cache() -> None:
    CHAT_MEMORY_CACHE.update(loaded_at=0.0, payload=None)
