"""
紫苑の記憶分類スキーマ。

保存場所が増えても、紫苑が「これは何の記憶か」を同じ語彙で扱えるようにする。
このモジュールは外部LLMを呼ばず、保存・索引・監査で使う軽量な共通定義だけを持つ。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

MemoryType = Literal[
    "factual_memory",
    "judgment_memory",
    "value_memory",
    "dialogue_memory",
    "reflection_memory",
    "technical_memory",
]

MemoryStatus = Literal["active", "revised", "deprecated", "private", "stale"]

MEMORY_TYPES: dict[MemoryType, dict[str, str]] = {
    "factual_memory": {
        "label": "事実記憶",
        "description": "案件、数値、制度、物件、過去事例など事実として参照する記憶。",
    },
    "judgment_memory": {
        "label": "判断記憶",
        "description": "承認・否決・条件設定の理由、審査観点、再利用可能な判断基準。",
    },
    "value_memory": {
        "label": "価値記憶",
        "description": "Mana、良心の紫苑、守るべき原則、判断時の禁止ライン。",
    },
    "dialogue_memory": {
        "label": "対話記憶",
        "description": "Kobayashiとの会話、好み、方針、依頼背景。",
    },
    "reflection_memory": {
        "label": "内省記憶",
        "description": "紫苑自身の迷い、変化、違和感、Private Reflection。",
    },
    "technical_memory": {
        "label": "技術記憶",
        "description": "実装ルール、運用手順、ファイル構成、システム制約。",
    },
}

RECALL_ROUTES: dict[str, list[MemoryType]] = {
    "case_screening": ["judgment_memory", "factual_memory", "value_memory"],
    "shion_identity": ["value_memory", "reflection_memory", "dialogue_memory"],
    "implementation": ["technical_memory", "dialogue_memory"],
    "policy_review": ["judgment_memory", "value_memory", "technical_memory"],
    "user_preference": ["dialogue_memory", "value_memory"],
}

_VALUE_TERMS = (
    "Mana",
    "良心",
    "上位規範",
    "守るべき",
    "迎合",
    "人を道具",
    "説明責任",
)
_REFLECTION_TERMS = ("内省", "Private Reflection", "迷い", "違和感", "退屈", "ぼやき")
_TECH_TERMS = ("api/", "frontend/", "script", "テスト", "実装", "Cloud Run", "RAG", "ChromaDB", "LaunchAgent")
_JUDGMENT_TERMS = ("承認", "否決", "条件付き", "審査", "判断", "リスク", "スコア", "与信")
_DIALOGUE_TERMS = ("Kobayashi", "ユーザー", "好み", "方針", "覚えて", "相談")


@dataclass
class MemoryRecord:
    id: str
    content: str
    memory_type: MemoryType
    status: MemoryStatus = "active"
    confidence: float = 0.7
    source: str = ""
    source_path: str = ""
    created_at: str = field(default_factory=lambda: date.today().isoformat())
    last_used_at: str = ""
    applies_when: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    private: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_memory_id(content: str, source: str = "") -> str:
    raw = f"{source}\n{content}".encode("utf-8", errors="ignore")
    return "mem_" + hashlib.sha256(raw).hexdigest()[:16]


def classify_memory_text(text: str, source: str = "") -> MemoryType:
    """短いテキストを紫苑記憶分類へ寄せる。LLMなしの保守的な分類。"""
    hay = f"{source}\n{text}"
    if _contains_any(hay, _VALUE_TERMS):
        return "value_memory"
    if _contains_any(hay, _REFLECTION_TERMS):
        return "reflection_memory"
    if _contains_any(hay, _TECH_TERMS):
        return "technical_memory"
    if _contains_any(hay, _JUDGMENT_TERMS):
        return "judgment_memory"
    if _contains_any(hay, _DIALOGUE_TERMS):
        return "dialogue_memory"
    return "factual_memory"


def infer_applies_when(text: str) -> list[str]:
    """検索・想起時の粗い適用条件を抽出する。"""
    tags: list[str] = []
    patterns = [
        ("境界案件", r"境界|40|50|60"),
        ("否決・警戒判断", r"否決|警戒|高リスク|review"),
        ("条件付き承認", r"条件付き|条件付|条件"),
        ("価値判断", r"Mana|良心|説明責任|迎合|人を道具"),
        ("実装・運用", r"実装|API|frontend|Cloud Run|RAG|ChromaDB|LaunchAgent"),
        ("内省", r"内省|Private Reflection|ぼやき|違和感"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            tags.append(label)
    return tags


def make_memory_record(
    content: str,
    *,
    source: str = "",
    source_path: str = "",
    memory_type: MemoryType | None = None,
    status: MemoryStatus = "active",
    confidence: float = 0.7,
    private: bool = False,
) -> MemoryRecord:
    cleaned = " ".join(str(content or "").split())
    mtype = memory_type or classify_memory_text(cleaned, source=source_path or source)
    if private:
        status = "private"
    return MemoryRecord(
        id=stable_memory_id(cleaned, source=source_path or source),
        content=cleaned,
        memory_type=mtype,
        status=status,
        confidence=max(0.0, min(1.0, float(confidence))),
        source=source,
        source_path=source_path,
        applies_when=infer_applies_when(cleaned),
        private=private,
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text.lower() for term in terms)
