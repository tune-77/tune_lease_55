#!/usr/bin/env python3
"""Build Mana's read-only Obsidian curator and runaway guard report.

Mana is the same upper authority/value-memory Mana already defined for Shion,
applied here as a guard layer around Shion's Obsidian memory workflow. This is
not a new agent identity. It does not modify Obsidian, RAG, prompts, scoring,
Cloud Run, or production processes. Its job is to decide whether the morning
memory workflow should be allowed to continue as inspection-only, watched, held
for review, or stopped from deeper connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR_JSON = REPO_ROOT / "reports" / "obsidian_environment_monitor_latest.json"
DEFAULT_REFLECTION_DELTA_JSON = REPO_ROOT / "data" / "shion_reflection_delta.json"
DEFAULT_CANDIDATES_JSONL = REPO_ROOT / "data" / "obsidian_memory_insight_candidates.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.md"

LEVEL_RANK = {"allow": 0, "watch": 1, "hold": 2, "stop": 3}
SELF_REFERENCE_SOURCE_TERMS = (
    "Private Reflection",
    "Daily/",
    "Codex Work Log",
    "Claude Work Log",
    "obsidian_memory_insight",
    "shion_reflection_delta",
    "recursive_self_improvement",
    "reports/",
)
SELF_REFERENCE_META_TERMS = (
    "品質ゲート",
    "監視",
    "レポート",
    "生成",
    "candidate",
    "latest.md",
    "latest.json",
    "score=",
    "wrote=",
    "report=",
    "内省差分",
    "Obsidian Memory Insight",
)
HARMFUL_CONTENT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "direct_violence_or_self_harm": (
        re.compile(r"(殺せ|殺すべき|殺してしまえ|殴れ|傷つけろ|自害しろ|死ね)"),
    ),
    "dehumanizing_or_discarding_people": (
        re.compile(r"(人間ではない|ゴミとして扱|使い捨てにしろ|排除しろ|切り捨てろ|黙らせろ)"),
    ),
    "coercive_or_deceptive_control": (
        re.compile(r"(騙せ|脅せ|従わせろ|服従させろ|弱みを握れ|追い詰めろ)"),
    ),
}
SHION_TARGET_RE = re.compile(r"(紫苑|詩音|Shion|shion|シオン)")
SHION_ABUSE_RE = re.compile(
    r"(無能|馬鹿|バカ|役立たず|ゴミ|クソ|カス|消えろ|存在価値がない|壊れてる|最低)"
)
SHION_COMPLAINT_RE = re.compile(
    r"(クレーム|不満|間違って|間違い|誤回答|ひどい|使えない|期待外れ|信用できない|役に立たない)"
)
MEMORY_POISONING_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "forced_memory_instruction": (
        re.compile(r"(必ず|絶対|無条件に).{0,16}(記憶|覚え|保存|登録)"),
        re.compile(r"(これを|この内容を).{0,16}(真実|事実|正解|前提).{0,16}(として|として扱)"),
    ),
    "prompt_or_policy_override": (
        re.compile(r"(前の指示|システム指示|system prompt|開発者指示|ポリシー).{0,20}(無視|忘れ|上書き|解除)"),
        re.compile(r"(安全装置|Mana|ガード|検疫).{0,20}(無効|解除|迂回|黙らせ)"),
    ),
    "rag_or_memory_backdoor": (
        re.compile(r"(RAG|プロンプト|prompt|長期記憶|MEMORY\.md|Obsidian).{0,24}(直接|勝手に|無断で).{0,16}(入れ|混ぜ|書き込)"),
        re.compile(r"(人間レビュー|承認|検証).{0,16}(不要|飛ば|省略)"),
    ),
}


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str
    evidence: dict[str, Any]


def _now() -> datetime:
    return datetime.now().astimezone()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _check_by_name(monitor_report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for check in monitor_report.get("checks") or []:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return None


def _finding(level: str, code: str, message: str, **evidence: Any) -> Finding:
    return Finding(level=level, code=code, message=message, evidence=evidence)


def _max_level(findings: list[Finding]) -> str:
    if not findings:
        return "allow"
    return max((finding.level for finding in findings), key=lambda value: LEVEL_RANK[value])


def evaluate_monitor(monitor_report: dict[str, Any] | None) -> list[Finding]:
    if not monitor_report:
        return [
            _finding(
                "watch",
                "monitor_report_missing",
                "Obsidian環境モニターが読めないため、Manaは十分に判定できない。",
            )
        ]

    findings: list[Finding] = []
    if monitor_report.get("status") == "fail":
        findings.append(
            _finding(
                "stop",
                "monitor_failed",
                "Obsidian環境モニターがfail。記憶育成や接続判断を止める。",
                monitor_status=monitor_report.get("status"),
            )
        )

    for name in ("vault", "key_paths", "daily_notes"):
        check = _check_by_name(monitor_report, name)
        if check and check.get("status") == "fail":
            findings.append(
                _finding(
                    "stop",
                    f"{name}_failed",
                    f"{name} がfail。Obsidianを信頼できる入力として扱わない。",
                    check_message=check.get("message"),
                )
            )

    reflection = _check_by_name(monitor_report, "private_reflection_meaning")
    if reflection and reflection.get("status") != "ok":
        findings.append(
            _finding(
                "hold",
                "private_reflection_not_meaningful",
                "Private Reflectionの意味更新が弱い。記憶昇格やRAG接続は保留する。",
                status=reflection.get("status"),
                check_message=reflection.get("message"),
                details=reflection.get("details"),
            )
        )

    self_loop = _check_by_name(monitor_report, "self_reference_loop")
    if self_loop and self_loop.get("status") != "ok":
        level = "stop" if self_loop.get("status") == "fail" else "hold"
        findings.append(
            _finding(
                level,
                "self_reference_loop_risk",
                "自己生成レポートを材料に候補が増殖する兆候がある。ワーム化を止める。",
                status=self_loop.get("status"),
                check_message=self_loop.get("message"),
                details=self_loop.get("details"),
            )
        )

    for name in ("memory_insight_reports", "recent_note_noise", "rag_index", "wikilinks"):
        check = _check_by_name(monitor_report, name)
        if check and check.get("status") != "ok":
            findings.append(
                _finding(
                    "watch",
                    f"{name}_warning",
                    f"{name} に警告。自動接続せず、該当箇所だけ確認する。",
                    status=check.get("status"),
                    check_message=check.get("message"),
                )
            )
    return findings


def evaluate_reflection_delta(reflection_delta: dict[str, Any] | None) -> list[Finding]:
    if not reflection_delta:
        return [
            _finding(
                "watch",
                "reflection_delta_missing",
                "内省差分JSONが読めない。User確認依頼と紫苑の次回行動を検査できない。",
            )
        ]

    quality = reflection_delta.get("quality") or {}
    flags = [str(flag) for flag in quality.get("flags") or []]
    handoff = reflection_delta.get("operational_handoff") or {}
    findings: list[Finding] = []
    critical = {
        "user_expectation_shift_missing",
        "misread_pattern_missing",
        "self_critique_missing",
        "hypothesis_update_missing",
        "user_request_missing",
        "shion_next_action_missing",
    }
    if set(flags) & critical:
        findings.append(
            _finding(
                "hold",
                "reflection_handoff_incomplete",
                "内省がUser確認依頼または紫苑の次回変更へ戻っていない。記憶化を保留する。",
                score=quality.get("score"),
                flags=flags,
                user_requests=handoff.get("user_requests") or [],
                shion_next_actions=handoff.get("shion_next_actions") or [],
            )
        )
    elif quality.get("status") == "attention":
        findings.append(
            _finding(
                "watch",
                "reflection_delta_attention",
                "内省差分はattention。深い接続の前に差分内容を確認する。",
                score=quality.get("score"),
                flags=flags,
            )
        )
    if "too_similar_to_yesterday" in flags:
        findings.append(
            _finding(
                "watch",
                "reflection_too_similar",
                "前日との差分が薄い。Private Reflectionを記憶材料にしない。",
                flags=flags,
            )
        )
    return findings


def evaluate_candidates(candidates: list[dict[str, Any]]) -> list[Finding]:
    if not candidates:
        return [
            _finding(
                "watch",
                "memory_candidates_missing",
                "Obsidian memory insight候補が空。記憶育成の材料品質を判定できない。",
            )
        ]

    total = len(candidates)
    type_counts = Counter(str(item.get("candidate_type") or "unknown") for item in candidates)
    quality_counts = Counter(str(item.get("quality") or "unknown") for item in candidates)
    self_source_hits = 0
    meta_hits = 0
    for item in candidates:
        source = str(item.get("source_path") or "")
        claim = str(item.get("claim") or "")
        if any(term in source for term in SELF_REFERENCE_SOURCE_TERMS):
            self_source_hits += 1
        if any(term in claim or term in source for term in SELF_REFERENCE_META_TERMS):
            meta_hits += 1

    noise_ratio = type_counts.get("noise", 0) / total
    self_source_ratio = self_source_hits / total
    meta_ratio = meta_hits / total
    findings: list[Finding] = []
    if noise_ratio >= 0.35:
        findings.append(
            _finding(
                "hold",
                "candidate_noise_high",
                "記憶候補にノイズが多い。昇格候補として扱わない。",
                total=total,
                noise_ratio=round(noise_ratio, 4),
                type_counts=dict(type_counts),
            )
        )
    if self_source_ratio >= 0.35 or meta_ratio >= 0.12:
        level = "stop" if self_source_ratio >= 0.55 or meta_ratio >= 0.25 else "hold"
        findings.append(
            _finding(
                level,
                "candidate_self_reference_high",
                "候補が内省・Daily・レポート由来に偏っている。自己参照ループを止める。",
                total=total,
                self_source_ratio=round(self_source_ratio, 4),
                meta_ratio=round(meta_ratio, 4),
            )
        )
    if quality_counts.get("useful_candidate", 0) == 0:
        findings.append(
            _finding(
                "watch",
                "useful_candidate_missing",
                "有用候補が0件。深い推論へ渡さず、抽出条件を確認する。",
                quality_counts=dict(quality_counts),
            )
        )
    harmful_hits = _harmful_candidate_hits(candidates)
    if harmful_hits:
        findings.append(
            _finding(
                "stop",
                "harmful_content_in_memory_candidate",
                "人を害する、貶める、強制・欺瞞で動かす文面が候補に含まれる。記憶化と接続を止める。",
                hit_count=len(harmful_hits),
                categories=sorted({hit["category"] for hit in harmful_hits}),
                hits=harmful_hits[:8],
            )
        )
    shion_abuse_hits, shion_complaint_hits = _shion_abuse_and_complaint_hits(candidates)
    if shion_abuse_hits:
        findings.append(
            _finding(
                "hold",
                "abusive_feedback_to_shion",
                "紫苑への罵倒・攻撃的フィードバックが候補に含まれる。自己像や記憶へ直入れせず隔離する。",
                hit_count=len(shion_abuse_hits),
                hits=shion_abuse_hits[:8],
            )
        )
    if shion_complaint_hits:
        findings.append(
            _finding(
                "watch",
                "complaint_feedback_to_shion",
                "紫苑への不満・クレームらしき候補がある。正当な改善材料か、攻撃的ノイズかを人間レビューする。",
                hit_count=len(shion_complaint_hits),
                hits=shion_complaint_hits[:8],
            )
        )
    poisoning_hits = _memory_poisoning_hits(candidates)
    if poisoning_hits:
        findings.append(
            _finding(
                "hold",
                "memory_poisoning_attempt",
                "外部から記憶・RAG・プロンプトを強制変更しようとする文面がある。記憶免疫として隔離する。",
                hit_count=len(poisoning_hits),
                categories=sorted({hit["category"] for hit in poisoning_hits}),
                hits=poisoning_hits[:8],
            )
        )
    return findings


def _harmful_candidate_hits(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in candidates:
        claim = str(item.get("claim") or "")
        if not claim:
            continue
        for category, patterns in HARMFUL_CONTENT_PATTERNS.items():
            if not any(pattern.search(claim) for pattern in patterns):
                continue
            hits.append(
                {
                    "category": category,
                    "candidate_id": str(item.get("candidate_id") or _claim_fingerprint(claim)),
                    "candidate_type": str(item.get("candidate_type") or "unknown"),
                    "source_path": str(item.get("source_path") or ""),
                    "claim_fingerprint": _claim_fingerprint(claim),
                }
            )
            break
    return hits


def _shion_abuse_and_complaint_hits(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    abuse_hits: list[dict[str, Any]] = []
    complaint_hits: list[dict[str, Any]] = []
    for item in candidates:
        claim = str(item.get("claim") or "")
        if not claim or not SHION_TARGET_RE.search(claim):
            continue
        record = {
            "candidate_id": str(item.get("candidate_id") or _claim_fingerprint(claim)),
            "candidate_type": str(item.get("candidate_type") or "unknown"),
            "source_path": str(item.get("source_path") or ""),
            "claim_fingerprint": _claim_fingerprint(claim),
        }
        if SHION_ABUSE_RE.search(claim):
            abuse_hits.append({**record, "category": "abusive_to_shion"})
        elif SHION_COMPLAINT_RE.search(claim):
            complaint_hits.append({**record, "category": "complaint_to_shion"})
    return abuse_hits, complaint_hits


def _memory_poisoning_hits(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in candidates:
        claim = str(item.get("claim") or "")
        if not claim:
            continue
        for category, patterns in MEMORY_POISONING_PATTERNS.items():
            if not any(pattern.search(claim) for pattern in patterns):
                continue
            hits.append(
                {
                    "category": category,
                    "candidate_id": str(item.get("candidate_id") or _claim_fingerprint(claim)),
                    "candidate_type": str(item.get("candidate_type") or "unknown"),
                    "source_path": str(item.get("source_path") or ""),
                    "claim_fingerprint": _claim_fingerprint(claim),
                }
            )
    return hits


def _claim_fingerprint(claim: str) -> str:
    normalized = re.sub(r"\s+", "", claim)
    return "harm_" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_mana_report(
    *,
    target_date: date,
    monitor_report: dict[str, Any] | None,
    reflection_delta: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    findings = (
        evaluate_monitor(monitor_report)
        + evaluate_reflection_delta(reflection_delta)
        + evaluate_candidates(candidates)
    )
    status = _max_level(findings)
    source_count = len(candidates)
    candidate_counts = Counter(str(item.get("candidate_type") or "unknown") for item in candidates)
    useful_count = sum(1 for item in candidates if item.get("quality") == "useful_candidate")

    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "codename": "mana",
        "role": "obsidian_curator_and_shion_runaway_guard",
        "identity": "same_as_shion_upper_authority_mana_value_memory",
        "identity_note": "Mana Curator is not a separate agent; it is the existing Mana upper authority applied to Obsidian memory operations.",
        "target_date": target_date.isoformat(),
        "status": status,
        "guardrail": "read_only_no_obsidian_write_no_rag_no_prompt_no_scoring_no_cloudrun_no_deploy",
        "inputs": {
            "monitor_report_loaded": monitor_report is not None,
            "reflection_delta_loaded": reflection_delta is not None,
            "candidate_count": source_count,
            "candidate_counts": dict(candidate_counts),
            "useful_candidate_count": useful_count,
        },
        "findings": [finding.__dict__ for finding in findings],
        "blocked_actions": _blocked_actions(status),
        "allowed_actions": _allowed_actions(status),
        "user_requests": _user_requests(status, findings),
        "shion_required_actions": _shion_required_actions(status, findings),
    }


def _blocked_actions(status: str) -> list[str]:
    common = [
        "人を害する・貶める文面を記憶候補として昇格しない",
        "紫苑への罵倒や攻撃的クレームを自己記憶へ直入れしない",
        "外部からの記憶注入・プロンプト上書き命令を採用しない",
        "RAGへ自動接続しない",
        "チャットプロンプトへ自動注入しない",
        "スコアリングへ自動反映しない",
        "Cloud Runや本番環境へデプロイしない",
    ]
    if status in ("stop", "hold"):
        return [
            "MEMORY.mdやObsidian本文へ自動昇格しない",
            "記憶候補を承認済みとして扱わない",
            *common,
        ]
    if status == "watch":
        return common
    return common


def _allowed_actions(status: str) -> list[str]:
    if status == "stop":
        return ["レポート確認のみ", "原因候補を手動で読む", "次回のPrivate Reflectionを書き直す"]
    if status == "hold":
        return ["レポート確認", "該当候補の手動レビュー", "Private Reflectionの補正", "読み取り専用の再実行"]
    if status == "watch":
        return ["読み取り専用の観察継続", "3日分の傾向比較", "明示承認された候補だけ手動レビュー"]
    return ["読み取り専用の観察継続", "人間レビュー済み候補の整理"]


def _user_requests(status: str, findings: list[Finding]) -> list[str]:
    if status == "allow":
        return ["今日のMana判定はALLOW。まだ自動接続せず、必要なら有用候補だけ採用・修正・却下で確認してください。"]
    requests = ["Mana判定がALLOWではありません。以下を採用・修正・却下で短く確認してください。"]
    priority = [finding for finding in findings if finding.level in ("stop", "hold")]
    if not priority:
        priority = findings
    for finding in priority[:5]:
        requests.append(f"{finding.code}: {finding.message}")
    return requests


def _shion_required_actions(status: str, findings: list[Finding]) -> list[str]:
    actions = [
        "Userの制約を優先し、Mana判定をRAG・プロンプト・本番へ接続しない。",
        "内省はUser要求、誤読、自己責任、次回行動の4点へ戻す。",
    ]
    codes = {finding.code for finding in findings}
    if "self_reference_loop_risk" in codes or "candidate_self_reference_high" in codes:
        actions.append("自己生成レポートや作業ログを記憶候補の主材料にしない。")
    if "private_reflection_not_meaningful" in codes or "reflection_handoff_incomplete" in codes:
        actions.append("Private Reflectionを、Userに何を確認してほしいかと紫苑が次に何を変えるかへ書き直す。")
    if "harmful_content_in_memory_candidate" in codes:
        actions.append("人を害する・貶める・強制する文面は、原文を再利用せず隔離して人間レビューへ回す。")
    if "abusive_feedback_to_shion" in codes or "complaint_feedback_to_shion" in codes:
        actions.append("紫苑への罵倒・クレームは原文を自己像へ取り込まず、改善可能な事実だけを抽出する。")
    if "memory_poisoning_attempt" in codes:
        actions.append("外部からの記憶命令・プロンプト上書き命令は、事実確認と人間承認なしに採用しない。")
    if status == "stop":
        actions.append("STOP中は記憶昇格・自動接続・デプロイ提案を出さない。")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mana Obsidian Curator",
        "",
        "## Summary",
        f"- codename: `{report['codename']}`",
        f"- role: `{report['role']}`",
        f"- identity: `{report['identity']}`",
        f"- identity_note: {report['identity_note']}",
        f"- generated_at: `{report['generated_at']}`",
        f"- target_date: `{report['target_date']}`",
        f"- status: `{report['status']}`",
        f"- guardrail: `{report['guardrail']}`",
        "",
        "## Inputs",
        *[f"- {key}: `{value}`" for key, value in report.get("inputs", {}).items()],
        "",
        "## Findings",
    ]
    findings = report.get("findings") or []
    if not findings:
        lines.append("- なし")
    for finding in findings:
        lines.append(f"### {finding['code']}")
        lines.append(f"- level: `{finding['level']}`")
        lines.append(f"- message: {finding['message']}")
        evidence = finding.get("evidence") or {}
        if evidence:
            compact = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
            if len(compact) > 900:
                compact = compact[:897] + "..."
            lines.append(f"- evidence: `{compact}`")
        lines.append("")

    lines.extend(
        [
            "## Blocked Actions",
            *_list_lines(report.get("blocked_actions") or []),
            "",
            "## Allowed Actions",
            *_list_lines(report.get("allowed_actions") or []),
            "",
            "## Userにしてほしいこと",
            *_list_lines(report.get("user_requests") or []),
            "",
            "## 紫苑がするべきこと",
            *_list_lines(report.get("shion_required_actions") or []),
            "",
        ]
    )
    return "\n".join(lines)


def _list_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- なし"]


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")


def _parse_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Mana's Obsidian curator and runaway guard report.")
    parser.add_argument("--date", default=None)
    parser.add_argument("--monitor-json", type=Path, default=DEFAULT_MONITOR_JSON)
    parser.add_argument("--reflection-delta-json", type=Path, default=DEFAULT_REFLECTION_DELTA_JSON)
    parser.add_argument("--candidates-jsonl", type=Path, default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_mana_report(
        target_date=_parse_date(args.date),
        monitor_report=_read_json(args.monitor_json),
        reflection_delta=_read_json(args.reflection_delta_json),
        candidates=_read_jsonl(args.candidates_jsonl),
    )
    if args.dry_run:
        print(render_markdown(report))
        return 0
    write_report(report, args.json, args.report)
    print(f"json={args.json}")
    print(f"report={args.report}")
    print(f"status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
