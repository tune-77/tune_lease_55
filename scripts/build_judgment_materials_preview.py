#!/usr/bin/env python3
"""Preview extractor for reusable lease judgment materials.

This script is intentionally read-only with respect to Obsidian and sidecar-only
with respect to the app. It reads recent conversation notes and writes preview
artifacts only:

- data/judgment_materials_preview.jsonl
- reports/judgment_materials_preview_YYYYMMDD.md

It does not connect to RAG, scoring, chat prompts, or Obsidian sync.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "judgment_materials_preview.jsonl"
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
SOURCE_DIRS = (
    Path("Projects") / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log",
    Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Dialogue",
)

MATERIAL_TYPES = ("judgment_rule", "risk_signal", "user_preference")
MAX_ASSISTANT_PER_DAY_TYPE = 8

RISK_TERMS = (
    "リスク",
    "通りづらい",
    "弱い",
    "難しい",
    "廃業",
    "不確実",
    "懸念",
    "違和感",
    "換金性",
    "返済",
    "撤退",
    "未採択",
    "依存",
)
RULE_TERMS = (
    "見る",
    "確認",
    "必要",
    "重要",
    "多い",
    "期間",
    "条件",
    "判断",
    "評価",
    "審査",
    "補助金",
    "銀行支援",
    "リース",
    "物件",
)
STRONG_RULE_TERMS = (
    "確認",
    "求める",
    "示されているか",
    "具体性",
    "判断",
    "評価",
    "見る",
    "見るべき",
    "審査",
    "通りづらい",
    "リース期間",
    "判断材料",
    "本体",
    "覚えて",
)
PREFERENCE_TERMS = (
    "覚えて",
    "重視",
    "大事",
    "正しい答え",
    "本体",
    "交換可能",
    "スピード",
    "好き",
    "嫌",
)
DOMAIN_TERMS = (
    "リース",
    "審査",
    "稟議",
    "厨房",
    "ラーメン",
    "飲食",
    "銀行支援",
    "補助金",
    "事業計画",
    "Q_risk",
    "AURION",
    "AI",
    "物件",
    "換金性",
    "ハッカソン",
    "判断材料",
    "購入選択権",
    "残価",
    "信頼",
)
PROMPT_NOISE_TERMS = (
    "紫苑レビュー依頼",
    "前提:",
    "前提：",
    "出力は",
    "以下の観点",
    "テンプレート",
    "JSON",
    "JSON形式",
    "【保存済み経験ケース】",
    "直感スコア",
    "① 調査結果",
    "② 推論",
    "デモケース",
    "デモ精密",
    "私の役割",
    "以下の専門領域",
    "注意:",
    "注意：",
    "点数の再説明",
)
ASSISTANT_EXAMPLE_TERMS = (
    "例えば",
    "仮定",
    "程度は必要",
    "必要になるでしょう",
    "必要になります",
    "とすると",
    "など）",
)
CONVERSATIONAL_NOISE_TERMS = (
    "何か次に確認したい",
    "確認したいことはありますか",
    "他に確認したい点",
    "開発チームにご確認",
    "後ほどお伝え",
    "お勧めします",
    "ご相談ください",
    "お手伝いできます",
    "User様",
    "私にとって",
    "感じています",
    "記憶しました",
    "先ほど確認",
    "となりますね",
    "になりますね",
)


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or str(DEFAULT_VAULT)
    return Path(raw).expanduser()


def _date_range(end_date: dt.date, days: int) -> list[dt.date]:
    return [end_date - dt.timedelta(days=offset) for offset in range(days)]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)


def _clean_text(value: str, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", text)
    text = re.sub(r"\b\d{6,}\b", "[number]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text)
    raw_parts = re.split(r"(?<=[。！？!?])\s+|[。！？!?]\s*", compact)
    parts: list[str] = []
    for part in raw_parts:
        cleaned = _clean_text(part, limit=260)
        if 8 <= len(cleaned) <= 260:
            parts.append(cleaned)
    return parts


def _looks_like_noise(sentence: str, role: str) -> bool:
    if any(term in sentence for term in PROMPT_NOISE_TERMS):
        return True
    if any(term in sentence for term in CONVERSATIONAL_NOISE_TERMS):
        return True
    if sentence.count("・") >= 4 or sentence.count(" - ") >= 3:
        return True
    if role == "assistant" and any(term in sentence for term in ASSISTANT_EXAMPLE_TERMS):
        return True
    if re.search(r"\d+人[〜～-]\d+人|\d+ルート|\d+店舗", sentence):
        return True
    if role == "user" and sentence.endswith("したい") and not (_has_strong_rule_signal(sentence) or _has_preference_signal(sentence)):
        return True
    if role == "user" and len(sentence) < 22 and not (_has_strong_rule_signal(sentence) or _has_preference_signal(sentence)):
        return True
    if sentence.startswith(("もちろん", "承知しました", "了解しました", "はい、")):
        return True
    return False


def _rank_material(item: dict[str, Any]) -> tuple[int, int, float, int, str]:
    role_rank = 1 if item.get("source_role") == "user" else 0
    axis_rank = len(item.get("risk_axis") or [])
    confidence = float(item.get("confidence") or 0)
    return (role_rank, axis_rank, confidence, -len(item.get("claim") or ""), item.get("claim") or "")


def _limit_preview_materials(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    assistant_counts: dict[tuple[str, str], int] = {}
    ranked = sorted(materials, key=_rank_material, reverse=True)
    for item in ranked:
        if item.get("source_role") != "assistant":
            kept.append(item)
            continue
        key = (item["date"], item["material_type"])
        if assistant_counts.get(key, 0) >= MAX_ASSISTANT_PER_DAY_TYPE:
            continue
        assistant_counts[key] = assistant_counts.get(key, 0) + 1
        kept.append(item)
    return sorted(kept, key=lambda item: (item["date"], item["material_type"], item["claim"]))


def _has_strong_rule_signal(sentence: str) -> bool:
    return any(term in sentence for term in STRONG_RULE_TERMS)


def _has_preference_signal(sentence: str) -> bool:
    return any(term in sentence for term in PREFERENCE_TERMS)


def _extract_sections(text: str) -> list[dict[str, str]]:
    """Extract user/assistant blocks from Cloud Run and Dialogue markdown."""
    blocks: list[dict[str, str]] = []
    patterns = [
        ("user", r"###\s*User\s*\n(.*?)(?=\n###\s*Assistant|\n<!--|\Z)"),
        ("assistant", r"###\s*Assistant\s*\n(.*?)(?=\n###\s*User|\n<!--|\Z)"),
        ("user", r"\*\*ユーザー\*\*\s*\n+(.*?)(?=\n\*\*リース知性体\*\*|\nsource_ts:|\n<!--|\Z)"),
        ("assistant", r"\*\*リース知性体\*\*\s*\n+(.*?)(?=\n\*\*ユーザー\*\*|\nsource_ts:|\n<!--|\Z)"),
    ]
    for role, pattern in patterns:
        for match in re.findall(pattern, text, flags=re.DOTALL):
            content = _clean_text(match, limit=1800)
            if content:
                blocks.append({"role": role, "text": content})
    return blocks


def _source_files(vault: Path, end_date: dt.date, days: int) -> list[tuple[dt.date, Path, str]]:
    files: list[tuple[dt.date, Path, str]] = []
    for day in _date_range(end_date, days):
        for rel_dir in SOURCE_DIRS:
            path = vault / rel_dir / f"{day.isoformat()}.md"
            if path.exists():
                files.append((day, path, str(rel_dir)))
    return files


def _looks_domain_relevant(sentence: str) -> bool:
    return any(term in sentence for term in DOMAIN_TERMS)


def _classify(sentence: str, role: str) -> tuple[str, float] | None:
    if _looks_like_noise(sentence, role):
        return None
    risk_score = sum(1 for term in RISK_TERMS if term in sentence)
    rule_score = sum(1 for term in RULE_TERMS if term in sentence)
    preference_score = sum(1 for term in PREFERENCE_TERMS if term in sentence)
    domain_relevant = _looks_domain_relevant(sentence)
    if not domain_relevant and max(risk_score, rule_score, preference_score) == 0:
        return None

    if risk_score >= 1 and risk_score >= rule_score:
        if role == "assistant" and not (_risk_axes(sentence) or _has_strong_rule_signal(sentence)):
            return None
        base = 0.72 + min(0.16, risk_score * 0.04)
        return "risk_signal", min(0.9, base + (0.04 if role == "user" else 0.0))
    if preference_score >= 1 and role == "user":
        if not (domain_relevant or any(term in sentence for term in ("AI", "信頼", "判断", "本体", "交換可能", "スピード"))):
            return None
        base = 0.68 + min(0.18, preference_score * 0.05)
        return "user_preference", min(0.9, base)
    if rule_score >= 1:
        if role == "assistant" and not _has_strong_rule_signal(sentence):
            return None
        if role == "user" and not _has_strong_rule_signal(sentence):
            return None
        base = 0.66 + min(0.2, rule_score * 0.04)
        return "judgment_rule", min(0.88, base + (0.04 if role == "user" else 0.0))
    return None


def _risk_axes(sentence: str) -> list[str]:
    axes: list[str] = []
    mapping = {
        "asset_life": ("期間", "厨房", "物件", "耐用", "換金性", "設備"),
        "industry_risk": ("飲食", "ラーメン", "業種", "廃業"),
        "cash_flow": ("返済", "資金繰り", "キャッシュ", "入金"),
        "support_specificity": ("銀行支援", "支援", "補助金"),
        "ai_ops": ("内省", "ハッカソン", "判断材料", "交換可能"),
    }
    for axis, terms in mapping.items():
        if any(term in sentence for term in terms):
            axes.append(axis)
    return axes[:4]


def _use_when(material_type: str, sentence: str) -> str:
    if "ラーメン" in sentence or "飲食" in sentence or "厨房" in sentence:
        return "飲食業・厨房機器・店舗設備のリース判断をするとき"
    if "銀行支援" in sentence or "補助金" in sentence:
        return "外部支援を返済原資や保全材料として扱うとき"
    if "ハッカソン" in sentence or "判断材料" in sentence:
        return "AI Agent Opsや判断資産化の説明・改善をするとき"
    if material_type == "risk_signal":
        return "案件の見落としリスクや追加確認事項を洗い出すとき"
    if material_type == "user_preference":
        return "ユーザーの判断基準・重視点に沿って回答するとき"
    return "類似案件の判断理由や稟議コメントを作るとき"


def _domain(sentence: str) -> str:
    if any(term in sentence for term in ("ハッカソン", "内省", "判断材料", "交換可能")):
        return "ai_agent_ops"
    return "lease_screening"


def _material_id(date: str, source: str, sentence: str, material_type: str) -> str:
    raw = f"{date}|{source}|{material_type}|{sentence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_materials(*, vault: Path, end_date: dt.date, days: int) -> list[dict[str, Any]]:
    materials: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    for day, path, source_label in _source_files(vault, end_date, days):
        text = _strip_frontmatter(_read_text(path))
        for block in _extract_sections(text):
            role = block["role"]
            for sentence in _split_sentences(block["text"]):
                classified = _classify(sentence, role)
                if not classified:
                    continue
                material_type, confidence = classified
                claim = _clean_text(sentence, limit=220)
                dedupe_key = (material_type, claim)
                if dedupe_key in seen_claims:
                    continue
                seen_claims.add(dedupe_key)
                rel_path = path.relative_to(vault) if str(path).startswith(str(vault)) else path
                materials.append(
                    {
                        "id": _material_id(day.isoformat(), source_label, claim, material_type),
                        "date": day.isoformat(),
                        "source": source_label,
                        "source_role": role,
                        "material_type": material_type,
                        "domain": _domain(claim),
                        "claim": claim,
                        "use_when": _use_when(material_type, claim),
                        "risk_axis": _risk_axes(claim),
                        "confidence": round(confidence, 2),
                        "evidence_path": str(rel_path),
                        "private": False,
                        "preview": True,
                    }
                )
    return _limit_preview_materials(materials)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _markdown(materials: list[dict[str, Any]], *, end_date: dt.date, days: int) -> str:
    counts: dict[str, int] = {material_type: 0 for material_type in MATERIAL_TYPES}
    for item in materials:
        counts[item["material_type"]] = counts.get(item["material_type"], 0) + 1
    lines = [
        f"# Judgment Materials Preview ({(end_date - dt.timedelta(days=days - 1)).isoformat()} to {end_date.isoformat()})",
        "",
        "## Summary",
        "",
        f"- Materials: {len(materials)}",
        f"- judgment_rule: {counts.get('judgment_rule', 0)}",
        f"- risk_signal: {counts.get('risk_signal', 0)}",
        f"- user_preference: {counts.get('user_preference', 0)}",
        "",
        "## Safety",
        "",
        "- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.",
        "- Sources are recent Cloud Run Conversation Log and Lease Intelligence Dialogue notes.",
        "- Private Reflection is intentionally excluded from this extractor.",
        "",
        "## Materials",
        "",
    ]
    for item in materials:
        axes = ", ".join(item.get("risk_axis") or [])
        lines += [
            f"### {item['date']} / {item['material_type']} / confidence={item['confidence']}",
            "",
            f"- Claim: {item['claim']}",
            f"- Use when: {item['use_when']}",
            f"- Axis: {axes or 'n/a'}",
            f"- Evidence: `{item['evidence_path']}`",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(materials: list[dict[str, Any]], *, end_date: dt.date, days: int) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_key = end_date.isoformat().replace("-", "")
    md_path = REPORTS_DIR / f"judgment_materials_preview_{date_key}.md"
    latest_md = REPORTS_DIR / "judgment_materials_preview_latest.md"
    summary_json = REPORTS_DIR / f"judgment_materials_preview_{date_key}.json"
    latest_json = REPORTS_DIR / "judgment_materials_preview_latest.json"
    md = _markdown(materials, end_date=end_date, days=days)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "end_date": end_date.isoformat(),
        "days": days,
        "materials": len(materials),
        "counts": {material_type: sum(1 for item in materials if item["material_type"] == material_type) for material_type in MATERIAL_TYPES},
        "output_jsonl": str(DEFAULT_OUTPUT_JSONL),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "markdown": str(md_path),
        "latest_markdown": str(latest_md),
        "summary_json": str(summary_json),
        "latest_summary_json": str(latest_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build judgment materials preview from recent Obsidian conversation notes")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--vault", default="", help="Obsidian Vault path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL), help="JSONL output path")
    args = parser.parse_args()

    end_date = dt.date.fromisoformat(args.date)
    vault = Path(args.vault).expanduser() if args.vault else _vault_path()
    days = max(1, args.days)
    materials = extract_materials(vault=vault, end_date=end_date, days=days)
    output_path = Path(args.output)
    write_jsonl(output_path, materials)
    paths = write_report(materials, end_date=end_date, days=days)
    print(
        json.dumps(
            {
                "materials": len(materials),
                "output_jsonl": str(output_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
