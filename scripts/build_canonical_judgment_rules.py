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


# --- L0/L1/L2 三階層ローディング ---------------------------------------------
# memory_layers/README.md の Retrieval Boundary に対応する。想起側は候補を層付きで
# 返すだけで、どの層をプロンプトへ載せるかは context 側が投入予算から決める。
# ここは層の中身とトークン概算を用意することに徹し、選択はしない。
#
#   L0 (abstract): 関連性判定用の短文。検索ヒット時にまず読む。
#   L1 (overview): 適用条件 / 失敗条件 / 質問観点。この案件で使えるかを判断する。
#   L2 (details):  代表クレーム全文と一次エビデンス。稟議で根拠を示す時だけ読む。
#
# L0/L1 は concept と canonical_statement だけから決まる。この2つは昇格時の
# マージキー（promote_canonical_judgment_rules._semantic_key）と一致するため、
# マージを経ても内容がずれない。L2 は sample_claims / evidence_paths の描画なので、
# マージ後は昇格側で組み直す。
_L0_MAX_CHARS = 80


# concept ごとの L1 要素。LLM 生成にすると日次パイプラインがネットワークと
# 非決定性に依存するため、判断軸は人が書いた静的定義を正とする。
# 未定義の concept は canonical_statement とリスク軸だけで L1 を組む。
CONCEPT_LAYER_HINTS: dict[str, dict[str, str]] = {
    "asset_life_and_residual": {
        "applies_when": "リース期間・残価・満了後の出口を決める時",
        "fails_when": "法定耐用年数だけで期間を決め、実使用状況と再販市場を見ていない時",
        "ask": "実際の稼働状況は？ 経済的寿命は法定耐用年数より短くないか？ 満了後の再販・再リース先はあるか？",
    },
    "support_specificity": {
        "applies_when": "銀行支援・補助金を返済原資として評価する時",
        "fails_when": "支援の存在だけを根拠にし、本件リースへの直接性と入金時期を確認していない時",
        "ask": "その支援は本件リースへの直接支援か？ 入金時期はいつか？ 返済原資にどう効くか？",
    },
    "business_plan_specificity": {
        "applies_when": "事業計画を返済可能性の根拠として使う時",
        "fails_when": "売上見込みの数字だけを見て、受注根拠と資金繰りを確認していない時",
        "ask": "受注の裏付けは何か？ 稼働計画は現実的か？ 資金繰り上いつ返済原資が立つか？",
    },
    "industry_operating_risk": {
        "applies_when": "業種特有の定性リスクを審査判断へ反映する時",
        "fails_when": "財務指標だけで判断し、業種の倒産率・費用変動・人員確保を見ていない時",
        "ask": "この業種の廃業・倒産率は？ 主要費用の変動耐性は？ 稼働継続の前提は何か？",
    },
    "intuition_gap": {
        "applies_when": "数字は基準を満たすのに違和感が残る時",
        "fails_when": "違和感を言語化せず見送り、稟議に判断根拠が残らない時",
        "ask": "違和感の正体はどの項目か？ 追加で何を確認すれば消えるか？ 消えない場合の条件は何か？",
    },
    "conditional_approval_checks": {
        "applies_when": "条件付き承認として通す時",
        "fails_when": "不確実性を残したまま通し、追加資料・確認条件・撤退条件が明文化されていない時",
        "ask": "承認条件は何か？ どの資料が揃えば条件解除か？ 撤退条件はどこに置くか？",
    },
    "judgment_asset_ops": {
        "applies_when": "会話や案件対応で得た判断基準を次回向けに残す時",
        "fails_when": "結論だけを残し、代表ルールと出典が分離されていない時",
        "ask": "この判断は他案件でも使えるか？ 出典はどのノートか？ 次回どの場面で想起させたいか？",
    },
    "demo_readiness": {
        "applies_when": "外部向けに紫苑の判断プロセスを説明する時",
        "fails_when": "機能一覧の説明に終始し、判断がどう更新されたかを示せていない時",
        "ask": "どの判断が更新されたか？ 更新の根拠は何か？ 次回どう使われるか？",
    },
    "user_decision_preference": {
        "applies_when": "ユーザーが明示した判断基準・信頼条件を回答へ反映する時",
        "fails_when": "一般論を優先し、ユーザーの明示方針を上書きしている時",
        "ask": "ユーザーは何を明示したか？ 一般論と衝突していないか？ 最新の指示はどれか？",
    },
}


def token_estimate(text: str) -> int:
    """日本語混在テキストの概算トークン数。

    scripts/build_obsidian_retrieval_graph.py の _token_estimate と同じ概算則
    （3文字≒1トークン）を使い、レポート間で数値の意味を揃える。
    """
    return max(1, len(str(text or "")) // 3)


def _l0_abstract(statement: str) -> str:
    """検索・関連性判定用の一文サマリー。"""
    text = re.sub(r"\s+", " ", str(statement or "")).strip()
    if not text:
        return ""
    head = text.split("。", 1)[0].strip()
    if not head:
        return ""
    if len(head) > _L0_MAX_CHARS:
        return head[: _L0_MAX_CHARS - 1].rstrip("、 ") + "…"
    return head + "。"


def _l1_overview(concept: str, statement: str, risk_axis: list[str]) -> str:
    """適用条件 / 失敗条件 / 質問観点。この案件で使えるかを判断するための層。"""
    hints = CONCEPT_LAYER_HINTS.get(str(concept or ""), {})
    lines = [f"判断: {str(statement or '').strip()}"]
    for label, key in (("適用条件", "applies_when"), ("失敗条件", "fails_when"), ("質問観点", "ask")):
        value = str(hints.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    lines.append(f"リスク軸: {'、'.join(risk_axis or []) or 'n/a'}")
    return "\n".join(lines)


def render_l2_details(sample_claims: list[str], evidence_paths: list[str]) -> str:
    """代表クレーム全文と一次エビデンス。稟議で根拠を示す時だけ読む層。

    昇格時に sample_claims / evidence_paths がマージされるため、
    promote_canonical_judgment_rules 側からも同じ描画に使う。
    """
    lines = ["代表クレーム:"]
    lines += [f"- {claim}" for claim in sample_claims or []] or ["- (なし)"]
    lines.append("一次エビデンス:")
    lines += [f"- {path}" for path in evidence_paths or []] or ["- (なし)"]
    return "\n".join(lines)


def build_layers(rule: dict[str, Any]) -> dict[str, Any]:
    """canonical rule から L0/L1/L2 とトークン概算を組む。

    層を選ぶのはここではない。context 側が投入予算を決められるよう、
    層ごとのトークン概算を添えて返すところまでを担う。
    """
    l0 = _l0_abstract(str(rule.get("canonical_statement") or ""))
    l1 = _l1_overview(
        str(rule.get("concept") or ""),
        str(rule.get("canonical_statement") or ""),
        list(rule.get("risk_axis") or []),
    )
    l2 = render_l2_details(
        list(rule.get("sample_claims") or []),
        list(rule.get("evidence_paths") or []),
    )
    return {
        "l0_abstract": l0,
        "l1_overview": l1,
        "l2_details": l2,
        "tokens_estimate": {
            "l0": token_estimate(l0),
            "l1": token_estimate(l1),
            "l2": token_estimate(l2),
        },
    }


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
                "sources": [],
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
        group["sources"].append(item.get("source") or "")
        group["confidences"].append(float(item.get("confidence") or 0))

    canonical: list[dict[str, Any]] = []
    for group in groups.values():
        ranked_claims = sorted(set(group["claims"]), key=lambda claim: (-len(claim), claim))
        user_evidence_count = sum(1 for role in group["source_roles"] if role == "user")
        evidence_count = len(group["claims"])
        avg_confidence = sum(group["confidences"]) / max(1, len(group["confidences"]))
        confidence = min(0.98, avg_confidence + min(0.12, evidence_count * 0.015) + (0.04 if user_evidence_count else 0))
        vertex_only = bool(group["sources"]) and all(source == "vertex_distilled_review" for source in group["sources"])
        status = "candidate" if vertex_only else _status(evidence_count, user_evidence_count)
        rule = {
            "id": group["id"],
            "material_type": group["material_type"],
            "domain": group["domain"],
            "concept": group["concept"],
            "status": status,
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
        # 層は組み上がったルールから導く。構造化フィールドが正で、層はその描画。
        rule["layers"] = build_layers(rule)
        canonical.append(rule)
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
    layer_tokens = {"l0": 0, "l1": 0, "l2": 0}
    for item in rules:
        estimate = (item.get("layers") or {}).get("tokens_estimate") or {}
        for layer in layer_tokens:
            layer_tokens[layer] += int(estimate.get(layer) or 0)
    lines = [
        "# Canonical Judgment Rules Preview",
        "",
        "## Summary",
        "",
        f"- Canonical rules: {len(rules)}",
        f"- accepted_preview: {accepted}",
        f"- candidate: {len(rules) - accepted}",
        f"- Layer tokens (L0/L1/L2 total): {layer_tokens['l0']}/{layer_tokens['l1']}/{layer_tokens['l2']}",
        "",
        "## Safety",
        "",
        "- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.",
        "- Built from `data/judgment_materials_preview.jsonl`.",
        "- Similar materials are compressed into representative rules; evidence paths remain linked for review.",
        "- L0/L1/L2 layers are prepared for budgeted loading; layer selection stays on the context side.",
        "",
        "## Rules",
        "",
    ]
    for item in rules:
        axes = ", ".join(item.get("risk_axis") or [])
        layers = item.get("layers") or {}
        estimate = layers.get("tokens_estimate") or {}
        lines += [
            f"### {item['concept']} / {item['status']} / evidence={item['evidence_count']}",
            "",
            f"- L0: {layers.get('l0_abstract') or 'n/a'}",
            f"- Layer tokens: L0={estimate.get('l0', 0)} / L1={estimate.get('l1', 0)} / L2={estimate.get('l2', 0)}",
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
    parser.add_argument(
        "--extra-input",
        action="append",
        default=[],
        help="Additional JSONL file(s) to merge in (e.g. Vertex Distilled review bridge output). May be repeated.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    materials = read_jsonl(Path(args.input))
    for extra_path in args.extra_input:
        materials.extend(read_jsonl(Path(extra_path)))
    rules = build_canonical_rules(materials)
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
