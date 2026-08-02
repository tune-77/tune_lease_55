"""User-personal-memory prompt loading for chat surfaces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


USER_PERSONAL_MEMORY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "payload": None}
USER_PERSONAL_MEMORY_CACHE_TTL_SEC = 300

USER_PERSONAL_MEMORY_KEYWORDS = (
    "What to call",
    "Timezone",
    "Notes",
    "好み",
    "方針",
    "覚えて",
    "犬",
    "愛犬",
    "dog",
    "名前",
    "Mana",
    "Shion",
    "紫苑",
    "Relationship UX",
    "Core Motivation",
    "Personal Facts",
    "Priority Rule",
)


def read_personal_memory_lines(path: Path, *, limit: int = 24, all_lines: bool = False) -> tuple[list[str], str]:
    if not path.exists() or not path.is_file():
        return [], ""
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return [], ""
    selected: list[str] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line or len(line) > 900:
            continue
        if all_lines or any(keyword in line for keyword in USER_PERSONAL_MEMORY_KEYWORDS):
            selected.append(line)
        if len(selected) >= limit:
            break
    return selected, str(path)


def read_cloudrun_personal_memory_lines(
    *,
    recent_events_reader: Callable[..., list[dict[str, Any]]] | None = None,
    limit: int = 24,
) -> tuple[list[str], str]:
    if not (os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_PENDING_GCS_ENABLED") == "1"):
        return [], ""
    if recent_events_reader is None:
        return [], ""
    try:
        from api.user_personal_memory import derive_personal_memory_entries

        events = recent_events_reader(days=int(os.environ.get("CLOUDRUN_PERSONAL_MEMORY_GCS_DAYS", "45") or 45))
    except Exception:
        return [], ""

    lines: list[str] = []
    seen: set[str] = set()
    for event in reversed(events):
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidate_lines: list[str] = []
        if event_type == "personal_memory":
            candidate_lines = [str(line).strip() for line in payload.get("derived_lines") or [] if str(line).strip()]
            dog_name = str(payload.get("dog_name") or "").strip()
            if dog_name:
                candidate_lines.insert(0, f"- [confirmed] Dog name: {dog_name}")
        elif event_type == "chat_exchange":
            message = str(payload.get("user_message") or "").strip()
            derived = derive_personal_memory_entries(
                message,
                source=f"cloudrun:{event.get('surface') or 'chat'}",
                timestamp=str(event.get("ts") or ""),
            )
            candidate_lines = [str(line).strip() for line in derived.get("lines") or [] if str(line).strip()]
        for line in candidate_lines:
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) >= limit:
                return lines, "gcs://cloudrun-inputs/personal-memory"
    return lines, "gcs://cloudrun-inputs/personal-memory" if lines else ""


def _user_personal_memory_line_priority(line: str) -> int:
    if "[confirmed" in line or "[confirmed]" in line:
        return 0
    if "[sensitive" in line:
        return 1
    if "Priority Rule" in line or "Personal Facts" in line:
        return 2
    return 3


def load_user_personal_memory_payload(
    *,
    repo_root: str | Path,
    data_path_resolver: Callable[[str], str],
    recent_events_reader: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    import time as _time

    now = _time.time()
    cached = USER_PERSONAL_MEMORY_CACHE.get("payload")
    if (
        cached is not None
        and now - float(USER_PERSONAL_MEMORY_CACHE.get("loaded_at") or 0) < USER_PERSONAL_MEMORY_CACHE_TTL_SEC
    ):
        return cached

    repo_root_path = Path(repo_root)
    refs: list[str] = []
    lines: list[str] = []
    personal_path = Path(data_path_resolver("user_personal_memory.md"))
    root_personal_path = repo_root_path / "data" / "user_personal_memory.md"
    for path, all_lines, limit in (
        (personal_path, True, 80),
        (root_personal_path, True, 80),
        (repo_root_path / "PERSISTENT_MEMORY.md", False, 20),
        (repo_root_path / "USER.md", False, 24),
        (repo_root_path / "MEMORY.md", False, 32),
    ):
        found, ref = read_personal_memory_lines(path, limit=limit, all_lines=all_lines)
        if found:
            if ref not in refs:
                refs.append(ref)
            lines.extend(found)
        if len(lines) >= 80:
            break

    cloudrun_lines, cloudrun_ref = read_cloudrun_personal_memory_lines(
        limit=32,
        recent_events_reader=recent_events_reader,
    )
    if cloudrun_lines:
        if cloudrun_ref not in refs:
            refs.append(cloudrun_ref)
        lines.extend(cloudrun_lines)

    clean_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        clean_lines.append(line)
        if len(clean_lines) >= 80:
            break

    clean_lines.sort(key=_user_personal_memory_line_priority)

    block = ""
    if clean_lines:
        block = "\n".join([
            "【ユーザー個人記憶】",
            "以下はUser本人との会話でのみ使う最優先の個人文脈です。",
            "紫苑モードでは、[confirmed] の個人記憶を審査知識・一般RAG・会話ノリより先に尊重してください。",
            "[candidate] は断定せず、必要なら確認してください。[sensitive] はむやみに持ち出さず、関係する時だけ慎重に扱ってください。",
            "ユーザーが覚えているはずの個人事実を尋ねた時、ここに無い場合は推測せず、未記録だと認めて次に保存する姿勢を示してください。",
            "犬の名前などの個人記憶は、Userを個別の相手として扱うための関係性UXアンカーです。リース審査の直接の判断資産として大げさに扱わず、自然に短く参照してください。",
            "個人記憶を語る時は、AIが人間を深く理解したかのように演出しすぎないでください。覚えている事実、使う理由、使わない境界を短く分けてください。",
            "外部向け説明や一般論には広げず、関係する時だけ自然に反映してください。",
            *[f"- {line}" for line in clean_lines],
        ])

    payload = {"block": block, "refs": refs[:6], "line_count": len(clean_lines)}
    USER_PERSONAL_MEMORY_CACHE.update(loaded_at=now, payload=payload)
    return payload


def build_user_personal_memory_prompt_block(
    *,
    repo_root: str | Path,
    data_path_resolver: Callable[[str], str],
    recent_events_reader: Callable[..., list[dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    try:
        payload = load_user_personal_memory_payload(
            repo_root=repo_root,
            data_path_resolver=data_path_resolver,
            recent_events_reader=recent_events_reader,
        )
    except Exception as exc:
        print(f"[UserPersonalMemory] 読み込みエラー: {exc}")
        payload = {"block": "", "refs": [], "line_count": 0}
    block = str(payload.get("block") or "").strip()
    return (f"\n\n{block}" if block else ""), payload


def invalidate_user_personal_memory_cache() -> None:
    USER_PERSONAL_MEMORY_CACHE.update(loaded_at=0.0, payload=None)
