"""Practical knowledge map for Shion.

This module keeps the hand-authored seed scenes and merges locally learned
three-layer entries. It does not call external services and does not write.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_PATH = REPO_ROOT / "data" / "shion_practical_knowledge_map.json"

PRACTICAL_SCENES: tuple[dict[str, Any], ...] = (
    {
        "id": "initial_screening",
        "label": "初回審査入力前",
        "triggers": ("初回", "審査", "入力", "案件", "始め", "決算", "財務"),
        "procedure_layer": (
            "企業属性、財務、物件、契約条件、導入目的を最低限そろえる。",
            "不足している資料と仮置き入力を分ける。",
        ),
        "meaning_layer": (
            "最初に見るべきなのは点数ではなく、返済原資・物件価値・導入目的が同じ方向を向いているか。",
            "入力値の不確実性が高いままAI判断へ進むと、後続の稟議コメントが一般論になる。",
        ),
        "judgment_layer": (
            "導入目的や設置場所が曖昧なら、スコア実行より先に確認質問を出す。",
            "財務が薄くても物件保全や銀行支援が強い場合は、即否決ではなく条件設計へ回す。",
        ),
    },
    {
        "id": "ocr_review",
        "label": "決算書OCR後",
        "triggers": ("OCR", "決算書", "読み取り", "財務", "転記", "数字"),
        "procedure_layer": (
            "OCR値を売上、利益、純資産、総資産、借入、リース債務へ分けて確認する。",
            "桁、単位、マイナス、前期比較の異常を先に見る。",
        ),
        "meaning_layer": (
            "OCR後の審査品質は、モデル精度より入力値の信頼性に左右される。",
            "単年度の数字だけでなく、利益と資金繰りが同じ方向に悪化しているかを見る。",
        ),
        "judgment_layer": (
            "重要項目に読み取り違いがある場合は、AIコメントを出す前に手入力で補正する。",
            "赤字でも一過性か構造的かを分け、構造的なら承認条件より回収可能性を優先する。",
        ),
    },
    {
        "id": "borderline_decision",
        "label": "承認・否決の境界",
        "triggers": ("境界", "微妙", "条件付き", "条件付", "保留", "60点", "50点", "承認", "否決"),
        "procedure_layer": (
            "スコア、財務、物件、代表者・銀行支援、契約条件を横並びで見る。",
            "承認条件、追加確認、否決理由を同時に作る。",
        ),
        "meaning_layer": (
            "境界案件は、点数の大小よりも何を条件にすればリスクが可視化・低減されるかが重要。",
            "審査部に突かれる点を先に言語化できると、稟議の説得力が上がる。",
        ),
        "judgment_layer": (
            "返済原資が弱く、物件保全も弱いなら条件付き承認に逃げない。",
            "弱点が限定的で、確認資料や保証・頭金・短期化で潰せるなら条件付き承認を検討する。",
        ),
    },
    {
        "id": "external_research",
        "label": "外部調査後",
        "triggers": ("外部調査", "Research", "research", "ニュース", "市況", "制度", "業界", "調査"),
        "procedure_layer": (
            "調査結果を事実、推論、未確認事項へ分ける。",
            "保存したResearchノートから、審査で使う確認質問と承認条件だけ抜き出す。",
        ),
        "meaning_layer": (
            "外部情報は回答を濃くするためではなく、案件判断を一般論から外すために使う。",
            "業界ニュースは、当該顧客の資金繰り・稼働・物件価値に接続して初めて判断材料になる。",
        ),
        "judgment_layer": (
            "参照元が補助情報だけなら、スコアや承認可否を直接変えない。",
            "一次情報や業界団体情報で制度・市況変化が確認できた時だけ、確認質問や条件へ反映する。",
        ),
    },
    {
        "id": "ringi_comment",
        "label": "稟議コメント作成時",
        "triggers": ("稟議", "コメント", "審査コメント", "説明", "理由", "承認理由", "否決理由"),
        "procedure_layer": (
            "結論、根拠、弱点、補完条件、確認事項の順に書く。",
            "モデル値、財務、物件、外部調査を混ぜずに根拠として分ける。",
        ),
        "meaning_layer": (
            "稟議コメントはAI回答ではなく、審査部が後から検証できる判断記録である。",
            "弱点を隠すより、弱点と条件の対応関係を明示した方が通りやすい。",
        ),
        "judgment_layer": (
            "強い断定は、根拠が数値・資料・外部情報のどれにあるかを明示できる時だけ使う。",
            "否決や条件付き承認では、相手を雑に切らず、再検討条件を残す。",
        ),
    },
    {
        "id": "competitor_pricing",
        "label": "競合・料率条件がある時",
        "triggers": ("競合", "料率", "金利", "提示", "他社", "利回り", "見積", "価格"),
        "procedure_layer": (
            "競合条件、採算、信用リスク、物件保全、取引継続価値を分けて見る。",
            "料率を下げる場合は、代わりに何でリスクを抑えるかを決める。",
        ),
        "meaning_layer": (
            "価格競争だけで判断すると、信用リスクと採算が見えなくなる。",
            "競合対応では、料率より契約条件・保全・継続取引の意味づけが重要になる。",
        ),
        "judgment_layer": (
            "信用リスクが強い案件では、競合料率に合わせる前に頭金・保証・期間短縮を検討する。",
            "優良先かつ継続取引価値が高い場合は、採算下限を明示して戦略的に寄せる。",
        ),
    },
    {
        "id": "subsidy_policy",
        "label": "補助金・制度変更が絡む時",
        "triggers": ("補助金", "制度", "税制", "会計", "助成", "優遇", "変更", "2026"),
        "procedure_layer": (
            "対象要件、採択時期、入金時期、自己資金、つなぎ資金を確認する。",
            "制度情報は有効日と対象範囲を必ず見る。",
        ),
        "meaning_layer": (
            "補助金は採算を良く見せるが、入金ズレが資金繰りリスクを作ることがある。",
            "制度変更は全案件に効くのではなく、対象設備・対象企業・契約時期に該当して初めて効く。",
        ),
        "judgment_layer": (
            "採択前提で返済余力を組み立てない。",
            "補助金入金までの資金繰りが弱い場合は、承認条件に入金確認やつなぎ資金確認を入れる。",
        ),
    },
)

_LAYER_KEYS = ("procedure_layer", "meaning_layer", "judgment_layer")


def infer_practical_scene(question: str, *, map_path: Path = DEFAULT_MAP_PATH) -> dict[str, Any]:
    text = question or ""
    learned = load_learned_practical_map(map_path)
    best: tuple[float, int, dict[str, Any]] | None = None
    for scene in _iter_scenes(learned):
        triggers = tuple(scene.get("triggers") or ())
        trigger_score = sum(1 for term in triggers if term and term.lower() in text.lower())
        if trigger_score <= 0:
            continue
        score = float(trigger_score) + min(0.3, int(scene.get("learned_entry_count") or 0) * 0.03)
        if best is None or (score, trigger_score) > (best[0], best[1]):
            best = (score, trigger_score, scene)
    if best is None:
        return {}
    return _public_scene(best[2])


def load_learned_practical_map(path: Path = DEFAULT_MAP_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def extract_practical_entries_from_memory_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = {scene["id"]: _empty_learned_scene(scene) for scene in PRACTICAL_SCENES}
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "active")
        if status in {"private", "deprecated"}:
            continue
        memory_type = str(record.get("memory_type") or "")
        source = str(record.get("source") or "")
        content = _clean_text(record.get("content") or "")
        if not _is_practical_candidate(content, memory_type=memory_type, source=source):
            continue
        scene = _match_scene_for_text(content)
        layer = classify_practical_layer(content)
        if not scene or not layer:
            continue
        entry = {
            "text": content[:260],
            "source": str(record.get("source") or "memory_index"),
            "source_path": str(record.get("source_path") or ""),
            "memory_id": str(record.get("id") or ""),
            "confidence": float(record.get("confidence") or 0.65),
        }
        _append_unique_entry(scenes[scene["id"]][layer], entry)
    return _compact_learned_scenes(scenes)


def classify_practical_layer(text: str) -> str:
    hay = str(text or "")
    if not hay.strip():
        return ""
    if hay.startswith("根拠:"):
        return "meaning_layer"
    if hay.startswith("要点:"):
        return "meaning_layer"
    if hay.startswith("判断ルール:"):
        return "judgment_layer"
    judgment_score = _count_matches(
        hay,
        (
            r"なら", r"場合", r"時だけ", r"べき", r"しない", r"避け", r"優先",
            r"承認", r"否決", r"条件付き", r"条件付", r"警戒", r"検討", r"判断",
        ),
    )
    procedure_score = _count_matches(
        hay,
        (
            r"確認", r"見る", r"分け", r"入力", r"保存", r"抽出", r"作成", r"実行",
            r"補正", r"記録", r"並べ", r"横並び", r"順",
        ),
    )
    meaning_score = _count_matches(
        hay,
        (
            r"なぜ", r"意味", r"重要", r"左右", r"説得力", r"一般論", r"品質",
            r"効く", r"理由", r"ため", r"見える", r"接続", r"資産",
        ),
    )
    scores = {
        "procedure_layer": procedure_score,
        "meaning_layer": meaning_score,
        "judgment_layer": judgment_score,
    }
    layer, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return ""
    if judgment_score >= 2 and ("場合" in hay or "なら" in hay or "承認" in hay or "否決" in hay):
        return "judgment_layer"
    return layer


def _iter_scenes(learned: dict[str, Any]) -> list[dict[str, Any]]:
    learned_by_id = {
        str(scene.get("id") or ""): scene
        for scene in learned.get("scenes", [])
        if isinstance(scene, dict)
    }
    merged: list[dict[str, Any]] = []
    for base in PRACTICAL_SCENES:
        scene = dict(base)
        learned_scene = learned_by_id.get(str(base["id"])) or {}
        learned_count = 0
        for key in _LAYER_KEYS:
            values = list(scene.get(key) or [])
            learned_values = [
                str(item.get("text") or "").strip()
                for item in learned_scene.get(key, [])
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            learned_count += len(learned_values)
            scene[key] = values + learned_values[:4]
        scene["learned_entry_count"] = learned_count
        if learned_scene:
            scene["learned_sources"] = _scene_sources(learned_scene)
        merged.append(scene)
    return merged


def _public_scene(scene: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": scene["id"],
        "label": scene["label"],
        "procedure_layer": list(scene.get("procedure_layer") or []),
        "meaning_layer": list(scene.get("meaning_layer") or []),
        "judgment_layer": list(scene.get("judgment_layer") or []),
        "learned_entry_count": int(scene.get("learned_entry_count") or 0),
        "learned_sources": list(scene.get("learned_sources") or [])[:8],
    }
    return result


def _match_scene_for_text(text: str) -> dict[str, Any] | None:
    best: tuple[int, dict[str, Any]] | None = None
    for scene in PRACTICAL_SCENES:
        triggers = tuple(scene.get("triggers") or ())
        score = sum(1 for term in triggers if term and term.lower() in text.lower())
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, scene)
    return best[1] if best else None


def _is_practical_candidate(text: str, *, memory_type: str, source: str) -> bool:
    if len(text) < 18:
        return False
    if text.startswith(("title:", "tags:", "User reported", "[2026", "`", "**")):
        return False
    if "`" in text:
        return False
    if any(term in text for term in ("api/", "frontend/", "Cloud Run", "FastAPI", "LaunchAgent", "pytest", "git ")):
        return False
    if "感情" in text:
        return False
    identity_terms = ("紫苑", "知性体", "感情", "Mana", "良心", "関係性UX", "意識")
    concrete_terms = (
        "審査", "承認", "否決", "条件付き", "条件付", "物件", "財務", "決算",
        "稟議", "補助金", "料率", "競合", "外部調査", "スコア", "与信",
    )
    if any(term in text for term in identity_terms) and not any(term in text for term in concrete_terms):
        return False
    if memory_type in {"technical_memory", "dialogue_memory", "reflection_memory", "value_memory"}:
        return source == "judgment_feedback"
    domain_terms = (
        "リース", "審査", "承認", "否決", "条件", "物件", "財務", "決算", "OCR",
        "稟議", "補助金", "制度", "料率", "競合", "外部調査", "業界", "市況",
        "スコア", "与信", "返済", "保証", "前受金", "銀行", "資金繰り",
    )
    return any(term.lower() in text.lower() for term in domain_terms)


def _empty_learned_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": scene["id"],
        "label": scene["label"],
        "triggers": list(scene.get("triggers") or []),
        "procedure_layer": [],
        "meaning_layer": [],
        "judgment_layer": [],
    }


def _append_unique_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    text = entry["text"]
    if any(existing.get("text") == text for existing in entries):
        return
    entries.append(entry)


def _compact_learned_scenes(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compact = []
    for scene in scenes.values():
        total = sum(len(scene[key]) for key in _LAYER_KEYS)
        if total <= 0:
            continue
        for key in _LAYER_KEYS:
            scene[key] = sorted(
                scene[key],
                key=lambda item: float(item.get("confidence") or 0),
                reverse=True,
            )[:16]
        scene["learned_entry_count"] = total
        compact.append(scene)
    return {"scenes": compact}


def _scene_sources(scene: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in _LAYER_KEYS:
        for item in scene.get(key, []):
            if not isinstance(item, dict):
                continue
            ref = str(item.get("source_path") or item.get("source") or "").strip()
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def _count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())
