#!/usr/bin/env python3
"""Build canonical judgment-rule candidates from preview materials.

This is a second-stage, read-only sidecar. It consumes
data/judgment_materials_preview.jsonl and writes compressed canonical preview
artifacts only. It does not connect to RAG, chat prompts, scoring, or Obsidian
sync.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_INPUT_JSONL = DATA_DIR / "judgment_materials_preview.jsonl"
DEFAULT_OUTPUT_JSON = DATA_DIR / "canonical_judgment_rules_preview.json"


CONCEPT_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "asset_life_and_residual",
        ("リース期間", "残価", "耐用", "経済的寿命", "再販", "再リース", "換金性", "使用状況"),
        "リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。",
    ),
    (
        "support_specificity",
        ("銀行支援", "補助金", "直接支援", "支援", "具体性"),
        "銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。",
    ),
    (
        "business_plan_specificity",
        ("事業計画", "具体的な事業計画", "受注", "収益", "資金繰り", "返済原資", "稼働計画"),
        "事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。",
    ),
    (
        "industry_operating_risk",
        ("飲食", "ラーメン", "運送", "燃料費", "人件費", "廃業", "倒産", "業態"),
        "業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。",
    ),
    (
        "intuition_gap",
        ("違和感", "数字は悪くない", "定性的", "見落とし", "警戒"),
        "数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。",
    ),
    (
        "conditional_approval_checks",
        ("条件付き承認", "承認条件", "確認すべき", "条件設計", "資料不足"),
        "条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。",
    ),
    (
        "judgment_asset_ops",
        ("判断資産", "再利用", "判断基準", "審査判断", "今回なら何を確認"),
        "会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。",
    ),
    (
        "demo_readiness",
        ("ハッカソン", "審査員", "説明", "公開", "デモ"),
        "公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。",
    ),
    (
        "user_decision_preference",
        ("覚えて", "スピード", "正しい答え", "本体", "交換可能", "信頼"),
        "ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _clean_claim(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:220]


def _concept_for(item: dict[str, Any]) -> tuple[str, str] | None:
    claim = _clean_claim(item.get("claim", ""))
    axes = " ".join(item.get("risk_axis") or [])
    haystack = f"{claim} {axes}"
    for concept, terms, statement in CONCEPT_RULES:
        if any(term in haystack for term in terms):
            return concept, statement
    return None


def _canonical_id(material_type: str, domain: str, concept: str) -> str:
    raw = f"{material_type}|{domain}|{concept}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _rank_evidence(item: dict[str, Any]) -> tuple[int, float, int]:
    user_rank = 1 if item.get("source_role") == "user" else 0
    confidence = float(item.get("confidence") or 0)
    axis_count = len(item.get("risk_axis") or [])
    return (user_rank, confidence, axis_count)


def _status(evidence_count: int, user_evidence_count: int) -> str:
    if user_evidence_count >= 1 and evidence_count >= 2:
        return "accepted_preview"
    if evidence_count >= 3:
        return "accepted_preview"
    return "candidate"


def build_canonical_rules(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in materials:
        if item.get("private") is True:
            continue
        material_type = item.get("material_type") or "judgment_rule"
        domain = item.get("domain") or "lease_screening"
        concept_pair = _concept_for(item)
        if not concept_pair:
            continue
        concept, statement = concept_pair
        key = (material_type, domain, concept)
        group = groups.setdefault(
            key,
            {
                "id": _canonical_id(material_type, domain, concept),
                "material_type": material_type,
                "domain": domain,
                "concept": concept,
                "canonical_statement": statement,
                "claims": [],
                "evidence_paths": [],
                "risk_axis": [],
                "source_roles": [],
                "confidences": [],
                "preview": True,
                "private": False,
            },
        )
        group["claims"].append(_clean_claim(item.get("claim", "")))
        evidence_path = item.get("evidence_path")
        if evidence_path and evidence_path not in group["evidence_paths"]:
            group["evidence_paths"].append(evidence_path)
        for axis in item.get("risk_axis") or []:
            if axis not in group["risk_axis"]:
                group["risk_axis"].append(axis)
        group["source_roles"].append(item.get("source_role") or "unknown")
        group["confidences"].append(float(item.get("confidence") or 0))

    canonical: list[dict[str, Any]] = []
    for group in groups.values():
        ranked_claims = sorted(set(group["claims"]), key=lambda claim: (-len(claim), claim))
        user_evidence_count = sum(1 for role in group["source_roles"] if role == "user")
        evidence_count = len(group["claims"])
        avg_confidence = sum(group["confidences"]) / max(1, len(group["confidences"]))
        confidence = min(0.98, avg_confidence + min(0.12, evidence_count * 0.015) + (0.04 if user_evidence_count else 0))
        canonical.append(
            {
                "id": group["id"],
                "material_type": group["material_type"],
                "domain": group["domain"],
                "concept": group["concept"],
                "status": _status(evidence_count, user_evidence_count),
                "canonical_statement": group["canonical_statement"],
                "evidence_count": evidence_count,
                "user_evidence_count": user_evidence_count,
                "confidence": round(confidence, 2),
                "risk_axis": group["risk_axis"][:5],
                "sample_claims": ranked_claims[:5],
                "evidence_paths": group["evidence_paths"][:8],
                "preview": True,
                "private": False,
            }
        )
    canonical.sort(
        key=lambda item: (
            item["status"] != "accepted_preview",
            -item["evidence_count"],
            -item["user_evidence_count"],
            item["material_type"],
            item["concept"],
        )
    )
    return canonical


def write_json(path: Path, rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "preview": True,
        "private": False,
        "canonical_rules": rules,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown(rules: list[dict[str, Any]]) -> str:
    accepted = sum(1 for item in rules if item["status"] == "accepted_preview")
    lines = [
        "# Canonical Judgment Rules Preview",
        "",
        "## Summary",
        "",
        f"- Canonical rules: {len(rules)}",
        f"- accepted_preview: {accepted}",
        f"- candidate: {len(rules) - accepted}",
        "",
        "## Safety",
        "",
        "- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.",
        "- Built from `data/judgment_materials_preview.jsonl`.",
        "- Similar materials are compressed into representative rules; evidence paths remain linked for review.",
        "",
        "## Rules",
        "",
    ]
    for item in rules:
        axes = ", ".join(item.get("risk_axis") or [])
        lines += [
            f"### {item['concept']} / {item['status']} / evidence={item['evidence_count']}",
            "",
            f"- Rule: {item['canonical_statement']}",
            f"- Type: {item['material_type']}",
            f"- Confidence: {item['confidence']}",
            f"- User evidence: {item['user_evidence_count']}",
            f"- Axis: {axes or 'n/a'}",
            "- Sample claims:",
        ]
        for claim in item.get("sample_claims", [])[:3]:
            lines.append(f"  - {claim}")
        lines += ["- Evidence paths:"]
        for evidence in item.get("evidence_paths", [])[:3]:
            lines.append(f"  - `{evidence}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(rules: list[dict[str, Any]], *, date: dt.date) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_key = date.isoformat().replace("-", "")
    md_path = REPORTS_DIR / f"canonical_judgment_rules_preview_{date_key}.md"
    latest_md = REPORTS_DIR / "canonical_judgment_rules_preview_latest.md"
    summary_path = REPORTS_DIR / f"canonical_judgment_rules_preview_{date_key}.json"
    latest_summary = REPORTS_DIR / "canonical_judgment_rules_preview_latest.json"
    md = _markdown(rules)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": date.isoformat(),
        "canonical_rules": len(rules),
        "accepted_preview": sum(1 for item in rules if item["status"] == "accepted_preview"),
        "candidate": sum(1 for item in rules if item["status"] == "candidate"),
        "output_json": str(DEFAULT_OUTPUT_JSON),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "markdown": str(md_path),
        "latest_markdown": str(latest_md),
        "summary_json": str(summary_path),
        "latest_summary_json": str(latest_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical judgment-rule preview from material preview JSONL")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    rules = build_canonical_rules(read_jsonl(Path(args.input)))
    output_path = Path(args.output)
    write_json(output_path, rules)
    paths = write_report(rules, date=dt.date.fromisoformat(args.date))
    print(
        json.dumps(
            {
                "canonical_rules": len(rules),
                "accepted_preview": sum(1 for item in rules if item["status"] == "accepted_preview"),
                "output_json": str(output_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
