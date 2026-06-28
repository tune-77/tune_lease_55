"""
Question-time recall for Shion memory index.

Reads data/shion_memory_index.json and selects a small set of memory records
based on the taxonomy recall routes. No network calls and no writes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from api.shion_practical_knowledge import infer_practical_scene
from api.shion_memory_taxonomy import MemoryType, RECALL_ROUTES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"

_CASE_TERMS = ("審査", "案件", "承認", "否決", "条件", "スコア", "与信", "リスク", "リース")
_IDENTITY_TERMS = ("紫苑", "Mana", "良心", "人格", "価値", "迷", "中核", "記憶", "内省")
_IMPLEMENTATION_TERMS = ("実装", "コード", "api", "frontend", "テスト", "エラー", "Cloud Run", "RAG", "ChromaDB")
_USER_PREF_TERMS = ("好み", "方針", "覚えて", "嫌", "どう思う", "やって", "優先")

_INDUSTRY_TERMS = (
    "製造業", "建設業", "医療", "介護", "運送", "運輸", "物流", "小売", "卸売",
    "飲食", "宿泊", "サービス", "不動産", "農業", "食品", "工作機械",
)
_ASSET_TERMS = (
    "建機", "ショベル", "フォークリフト", "トラック", "車両", "工作機械", "機械",
    "医療機器", "CT", "MRI", "設備", "太陽光", "コンテナ", "厨房", "印刷機",
)
_DECISION_TERMS = ("承認", "否決", "条件付き", "条件付", "警戒", "保留", "稟議")

def load_memory_index(path: Path = _INDEX_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def infer_recall_route(question: str) -> str:
    text = question or ""
    if _contains_any(text, _IMPLEMENTATION_TERMS):
        return "implementation"
    if _contains_any(text, _IDENTITY_TERMS):
        return "shion_identity"
    if _contains_any(text, _CASE_TERMS):
        return "case_screening"
    if _contains_any(text, _USER_PREF_TERMS):
        return "user_preference"
    return "policy_review"


def recall_memories(question: str, *, limit: int = 5, index_path: Path = _INDEX_PATH) -> dict[str, Any]:
    index = load_memory_index(index_path)
    records = index.get("records") or []
    if not isinstance(records, list):
        records = []
    route = infer_recall_route(question)
    preferred_types = RECALL_ROUTES.get(route, RECALL_ROUTES["policy_review"])
    query_terms = _query_terms(question)
    case_profile = _extract_case_profile(question) if route == "case_screening" else {}

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "active")
        if status in {"private", "deprecated"}:
            continue
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        memory_type = str(record.get("memory_type") or "")
        score = _score_record(content, memory_type, preferred_types, query_terms, case_profile, record)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = _select_records(scored, route=route, limit=max(0, limit))
    practical_scene = infer_practical_scene(question)
    return {
        "route": route,
        "preferred_types": list(preferred_types),
        "case_profile": case_profile,
        "practical_scene": practical_scene,
        "memories": selected,
        "refs": [str(r.get("id") or "") for r in selected if r.get("id")],
    }


def build_recall_prompt_block(question: str, *, limit: int = 5) -> tuple[str, dict[str, Any]]:
    recalled = recall_memories(question, limit=limit)
    memories = recalled.get("memories") or []
    practical_scene = recalled.get("practical_scene") or {}
    if not memories and not practical_scene:
        return "", recalled
    lines = [
        "【紫苑の想起メモ】",
        f"想起ルート: {recalled.get('route')}",
        "以下は今回の質問に関連しそうな記憶です。回答では必要なものだけ自然に使い、無関係なら無理に触れないでください。",
    ]
    if practical_scene:
        lines.extend(_format_practical_scene_block(practical_scene))
    for idx, record in enumerate(memories, start=1):
        mtype = str(record.get("memory_type") or "memory")
        status = str(record.get("status") or "active")
        content = str(record.get("content") or "").strip()
        lines.append(f"{idx}. [{mtype}/{status}] {content[:260]}")
    return "\n".join(lines), recalled


def _format_practical_scene_block(scene: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "【実践知マップ】",
        f"場面: {scene.get('label')}",
        "この場面では、手順だけでなく「なぜそうするか」と「例外時にどう判断するか」まで使ってください。",
    ]
    learned_count = int(scene.get("learned_entry_count") or 0)
    if learned_count:
        lines.append(f"学習候補: Obsidian/過去判断由来の三層候補 {learned_count}件を含む。")
    layer_labels = (
        ("第一層 手順層", "procedure_layer"),
        ("第二層 意味層", "meaning_layer"),
        ("第三層 判断層", "judgment_layer"),
    )
    for label, key in layer_labels:
        values = [str(item).strip() for item in scene.get(key) or [] if str(item).strip()]
        if values:
            lines.append(f"{label}: " + " / ".join(values[:2]))
    return lines


def _select_records(scored: list[tuple[float, dict[str, Any]]], *, route: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    caps = {"value_memory": 1, "technical_memory": 0} if route == "case_screening" else {}
    for _, record in scored:
        mtype = str(record.get("memory_type") or "")
        cap = caps.get(mtype)
        if cap is not None and type_counts.get(mtype, 0) >= cap:
            continue
        selected.append(record)
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _score_record(
    content: str,
    memory_type: str,
    preferred_types: list[MemoryType],
    query_terms: set[str],
    case_profile: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> float:
    score = 0.0
    if memory_type in preferred_types:
        score += 3.0 - (preferred_types.index(memory_type) * 0.4)
    content_lower = content.lower()
    overlap = sum(1 for term in query_terms if term.lower() in content_lower)
    score += min(overlap, 5) * 0.7
    if query_terms and overlap == 0 and memory_type not in preferred_types:
        return 0.0
    if query_terms and overlap == 0 and memory_type in {"factual_memory", "technical_memory"}:
        score *= 0.25
    if case_profile:
        score += _case_profile_bonus(content, case_profile, record or {})
    if "Mana" in content or "良心" in content:
        score += 0.4
    return score


def _extract_case_profile(question: str) -> dict[str, Any]:
    text = question or ""
    industries = [term for term in _INDUSTRY_TERMS if term in text]
    assets = [term for term in _ASSET_TERMS if term.lower() in text.lower()]
    decisions = [term for term in _DECISION_TERMS if term in text]
    score_band = _extract_score_band(text)
    return {
        "industries": industries,
        "assets": assets,
        "decisions": decisions,
        "score_band": score_band,
        "is_boundary": score_band == "boundary" or "境界" in text,
    }


def _extract_score_band(text: str) -> str:
    scores = []
    for match in re.finditer(r"(?<!\d)(\d{1,3})(?:\.\d+)?\s*(?:点|スコア|%)?", text):
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        if 0 <= val <= 100:
            scores.append(val)
    if not scores:
        return ""
    score = scores[0]
    if 40 < score < 70:
        return "boundary"
    if score <= 40:
        return "low"
    if score >= 70:
        return "high"
    return "middle"


def _case_profile_bonus(content: str, profile: dict[str, Any], record: dict[str, Any]) -> float:
    bonus = 0.0
    hay = content.lower()
    source_path = str(record.get("source_path") or "").lower()
    applies = " ".join(map(str, record.get("applies_when") or [])).lower()
    combined = f"{hay} {applies} {source_path}"
    matched_asset_or_industry = False

    for term in profile.get("industries") or []:
        if term.lower() in combined:
            bonus += 2.0
            matched_asset_or_industry = True
    for term in profile.get("assets") or []:
        if term.lower() in combined:
            bonus += 2.8
            matched_asset_or_industry = True
    for term in profile.get("decisions") or []:
        if term.lower() in combined:
            bonus += 1.0

    band = profile.get("score_band")
    if band == "boundary" and ("境界" in combined or "条件" in combined or "40" in combined or "60" in combined):
        bonus += 1.2
    elif band == "low" and ("否決" in combined or "警戒" in combined or "低スコア" in combined):
        bonus += 1.2
    elif band == "high" and ("承認" in combined or "高スコア" in combined):
        bonus += 0.8

    if record.get("memory_type") == "judgment_memory" and bonus > 0:
        bonus += 0.8
    if record.get("memory_type") == "factual_memory" and matched_asset_or_industry and "knowledge_base" in source_path:
        bonus += 4.0
    if (profile.get("assets") or profile.get("industries")) and not matched_asset_or_industry:
        bonus -= 1.4
    return bonus


def _query_terms(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9_./-]{2,}|[一-龥ぁ-んァ-ヶー]{2,}", text or "")
    stop = {"これ", "それ", "どう", "して", "です", "ます", "ある", "いる", "やって", "かな"}
    return {t for t in raw if t not in stop}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    hay = (text or "").lower()
    return any(term.lower() in hay for term in terms)
