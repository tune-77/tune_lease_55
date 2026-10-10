#!/usr/bin/env python3
"""Export selected Obsidian lease knowledge notes for Vertex AI Search.

This creates a sanitized local text corpus from the normal iCloud Obsidian
Vault. It intentionally excludes chat logs, private reflection, raw memory, and
daily-style conversational material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from obsidian_query import list_vault_md_files
from api.knowledge.obsidian_loader import note_excludes_rag  # noqa: E402

from runtime_paths import resolve_obsidian_vault  # noqa: E402

DEFAULT_VAULT = resolve_obsidian_vault()
DEFAULT_PROJECT_REL = Path("Projects") / "tune_lease_55"
DEFAULT_OUTPUT = Path("data") / "agent_search" / "lease_knowledge_export"
DEFAULT_GCS_PREFIX = "gs://tune-lease-55-data/agent-search/lease-knowledge"

KEYWORDS = (
    "リース",
    "審査",
    "稟議",
    "判断資産",
    "再リース",
    "耐用年数",
    "残価",
    "物件",
    "与信",
    "倒産",
    "業界",
    "補助金",
    "所有権",
    "Q_risk",
    "Mahalanobis",
    "スコア",
    "リスク",
    "leasing",
    "lease",
    "credit",
    "risk",
)

EXCLUDED_PARTS = {
    ".obsidian",
    "AI Chat",
    "Alerts",
    "Cloud Run Conversation Log",
    "Cloud Run Return",
    "Daily",
    "Dialogue",
    "Improvement Log",
    "Lease Intelligence",
    "Lease Intelligence/Memory",
    "Memory",
    "OS",
    "Private Reflection",
    "Slack",
}

EXCLUDED_FILENAME_KEYWORDS = (
    "AIChat",
    "Obsidian検索修正",
    "ユーモア",
    "実装",
    "改善",
    "ニュース判断変更記録",
    "単位統一",
    "After Hackathon",
    "Claude",
    "Mana",
    "Webデザイン",
    "外部調査器官",
    "Inbox",
)

PREFERRED_PARTS = {
    "Research",
    "Judgment Assets",
}

# Lease Intelligence 配下は丸ごと除外だが、審査知識の Knowledge だけは根拠検索の対象にする
# （Memory・Private Reflection・Dialogue 等の除外は維持。秘密・private マーカー・品質ゲートも通常どおり）
INCLUDED_SUBTREES = (Path("Lease Intelligence") / "Knowledge",)
CANONICAL_RULES_PATH = _REPO_ROOT / "data" / "canonical_judgment_rules.json"
CANONICAL_SOURCE_PREFIX = "judgment_assets/canonical"

ROOT_NOTE_KEYWORDS = (
    "リース審査AI_知識分解",
    "リースvs銀行借入",
    "審査ナレッジ",
    "LightGBMスコアリング",
    "Q_risk",
    "Mahalanobis",
)

SECRET_PATTERNS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
    "DATABASE_URL",
    "GEMINI_API_KEY",
)


@dataclass(frozen=True)
class ExportedNote:
    source_path: str
    output_path: str
    title: str
    canonical_topic: str
    source_bucket: str
    quality_score: int
    digest: str
    size_chars: int


@dataclass(frozen=True)
class CandidateNote:
    path: Path
    source_path: str
    title: str
    canonical_topic: str
    source_bucket: str
    quality_score: int
    cleaned: str
    digest: str
    size_chars: int


@dataclass(frozen=True)
class RejectedNote:
    source_path: str
    reason: str
    title: str = ""


def _in_included_subtree(rel: Path) -> bool:
    return any(rel.parts[: len(sub.parts)] == sub.parts for sub in INCLUDED_SUBTREES)


def should_exclude(path: Path, project_root: Path) -> bool:
    rel = path.relative_to(project_root)
    if _in_included_subtree(rel):
        # 許可したサブツリーでも、ファイル名の除外キーワードと隠しファイルは従来どおり
        return any(keyword in path.name for keyword in EXCLUDED_FILENAME_KEYWORDS) or path.name.startswith(".")
    rel_text = str(rel)
    parts = set(rel.parts)
    if any(part in parts for part in EXCLUDED_PARTS):
        return True
    if any(excluded in rel_text for excluded in EXCLUDED_PARTS):
        return True
    if any(keyword in path.name for keyword in EXCLUDED_FILENAME_KEYWORDS):
        return True
    if path.name.startswith("."):
        return True
    return False


def is_preferred_location(path: Path, project_root: Path) -> bool:
    rel = path.relative_to(project_root)
    parts = set(rel.parts)
    if parts & PREFERRED_PARTS or _in_included_subtree(rel):
        return True
    return any(keyword in path.name for keyword in ROOT_NOTE_KEYWORDS)


def has_secret_like_text(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in SECRET_PATTERNS)


def has_private_marker(text: str) -> bool:
    if note_excludes_rag(text):
        return True
    head = "\n".join(text.splitlines()[:40]).lower()
    private_patterns = (
        "vertex_exclude: true",
        "agent_search_exclude: true",
        "public: false",
        "visibility: private",
        "confidential: true",
        "private: true",
        "非公開",
        "外部投入禁止",
        "vertex投入禁止",
    )
    return any(pattern in head for pattern in private_patterns)


def matches_lease_knowledge(path: Path, text: str) -> bool:
    haystack = f"{path.name}\n{text[:8000]}"
    return any(keyword in haystack for keyword in KEYWORDS)


def strip_obsidian_noise(text: str) -> str:
    text = re.sub(r"!\[\[[^\]]+\]\]", "", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def source_bucket(path: Path, project_root: Path) -> str:
    rel = path.relative_to(project_root)
    parts = set(rel.parts)
    if "Judgment Assets" in parts:
        return "judgment_assets"
    if "Auto Research" in parts:
        return "auto_research"
    if "Research" in parts:
        return "research"
    return "root_note"


def canonical_topic_from_note(path: Path, text: str) -> str:
    frontmatter_match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if frontmatter_match:
        for line in frontmatter_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "canonical_topic":
                topic = value.strip().strip("\"'")
                if topic:
                    return normalize_canonical_topic(topic)

    title = title_from_text(path, strip_obsidian_noise(text))
    return normalize_canonical_topic(title)


def normalize_canonical_topic(value: str) -> str:
    topic = re.sub(r"\s+", " ", value).strip()
    topic = re.sub(r"^\d{4}[-_/年]\d{1,2}[-_/月]\d{1,2}日?[_\s-]*", "", topic)
    topic = re.sub(r"^\d{8}[_\s-]*", "", topic)
    topic = re.sub(r"\s*[-_]*\s*リース判断Auto Research\s*$", "", topic)
    topic = re.sub(r"\s*[-_]*\s*Auto Research\s*$", "", topic, flags=re.I)
    topic = re.sub(r"\s*[_-]*report\s*$", "", topic, flags=re.I)
    topic = re.sub(r"\s*レポート\s*$", "", topic)
    topic = re.sub(r"\s*総合\s*$", "", topic)
    return topic[:120] or "untitled"


def canonical_key(topic: str) -> str:
    lowered = topic.lower()
    lowered = re.sub(r"[ 　\t\r\n・、。/／_-]+", "", lowered)
    return lowered


def quality_score_for_note(path: Path, text: str) -> int:
    haystack = f"{path.name}\n{text[:12000]}"
    domain_hits = sum(1 for keyword in KEYWORDS if keyword in haystack)
    usable_markers = (
        "確認",
        "論点",
        "条件",
        "リスク",
        "判断",
        "根拠",
        "出典",
        "適用",
        "反証",
        "更新",
        "注意",
        "質問",
        "兆候",
    )
    marker_hits = sum(1 for marker in usable_markers if marker in haystack)
    length_score = 0
    if len(text) >= 500:
        length_score += 1
    if len(text) >= 1200:
        length_score += 1
    if len(text) >= 2500:
        length_score += 1
    heading_score = min(2, len(re.findall(r"^#{2,3}\s+", text, flags=re.M)))
    return min(12, min(domain_hits, 4) + min(marker_hits, 4) + length_score + heading_score)


def passes_quality_gate(path: Path, project_root: Path, text: str, score: int) -> bool:
    if source_bucket(path, project_root) == "root_note":
        return score >= 4
    return score >= 5


def candidate_rank(candidate: CandidateNote) -> tuple[int, int, int, str]:
    bucket_priority = {
        "judgment_assets": 4,
        "research": 3,
        "root_note": 2,
        "auto_research": 1,
    }.get(candidate.source_bucket, 0)
    return (candidate.quality_score, bucket_priority, candidate.size_chars, candidate.source_path)


def output_name(rel: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", rel.with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}__{digest}.txt"


def canonical_rule_candidates(rejected: list[RejectedNote], path: Path | None = None) -> list[CandidateNote]:
    """判断資産の正本のうち active（ユーザーが教えたものを含む）を1件1文書にする。

    一般化された canonical_statement と概念・リスク軸だけを出す。個別案件の文が混ざりうる
    sample_claims は出さず、本文は mask_for_vertex で伏せ、秘密らしい文字列があれば外す。
    """
    from api.vertex_query_mask import mask_for_vertex

    try:
        data = json.loads((path or CANONICAL_RULES_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = data.get("rules") if isinstance(data, dict) else data
    candidates: list[CandidateNote] = []
    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("status") != "active":
            continue
        statement = mask_for_vertex(str(rule.get("canonical_statement") or "").strip())
        rule_id = str(rule.get("id") or "")
        concept = str(rule.get("concept") or "judgment_asset")
        source_rel = f"{CANONICAL_SOURCE_PREFIX}/{rule_id}.md"
        if len(statement) < 20 or not rule_id:
            rejected.append(RejectedNote(source_rel, "too_short"))
            continue
        if has_secret_like_text(statement):
            rejected.append(RejectedNote(source_rel, "secret_like_text", concept))
            continue
        axes = "、".join(str(a) for a in rule.get("risk_axis") or [])
        cleaned = "\n".join(
            [
                f"# 判断資産: {concept}",
                "",
                statement,
                "",
                f"- 種別: {rule.get('material_type') or ''}",
                f"- 領域: {rule.get('domain') or ''}",
                f"- リスク軸: {axes}",
                f"- 根拠件数: {rule.get('evidence_count') or 0}（うちユーザー由来 {rule.get('user_evidence_count') or 0}）",
                f"- 確信度: {rule.get('confidence') or ''}",
            ]
        )
        candidates.append(
            CandidateNote(
                path=Path(source_rel),
                source_path=source_rel,
                title=f"判断資産: {concept}",
                canonical_topic=f"judgment_asset:{rule_id}",
                source_bucket="judgment_asset_canonical",
                quality_score=int(round(float(rule.get("confidence") or 0) * 10)),
                cleaned=cleaned,
                digest=hashlib.sha1(cleaned.encode("utf-8")).hexdigest(),
                size_chars=len(cleaned),
            )
        )
    return candidates


def export_notes(
    vault: Path, output: Path, max_docs: int, gcs_prefix: str, *, canonical_rules: Path | None = None
) -> list[ExportedNote]:
    """canonical_rules を渡すと判断資産の正本（active）も同じコーパスに入れる（日次同期は渡す）。"""
    project_root = vault / DEFAULT_PROJECT_REL
    if not project_root.exists():
        raise SystemExit(f"Project notes not found: {project_root}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    rejected: list[RejectedNote] = []
    best_by_topic: dict[str, CandidateNote] = {}
    # FULL reconciliation can delete remote documents, so this export must not
    # treat an interrupted/partial Vault walk as a valid empty corpus.
    candidates = sorted(
        list_vault_md_files(project_root, suppress_errors=False),
        key=lambda p: str(p.relative_to(project_root)),
    )

    for path in candidates:
        source_rel = path.relative_to(vault).as_posix()
        if should_exclude(path, project_root):
            rejected.append(RejectedNote(source_rel, "excluded_path"))
            continue
        if not is_preferred_location(path, project_root):
            rejected.append(RejectedNote(source_rel, "not_preferred_location"))
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            rejected.append(RejectedNote(source_rel, "unicode_decode_error"))
            continue
        if has_secret_like_text(raw):
            rejected.append(RejectedNote(source_rel, "secret_like_text", path.stem))
            continue
        if has_private_marker(raw):
            rejected.append(RejectedNote(source_rel, "private_marker", path.stem))
            continue
        if len(raw.strip()) < 120:
            rejected.append(RejectedNote(source_rel, "too_short"))
            continue
        if not matches_lease_knowledge(path, raw):
            rejected.append(RejectedNote(source_rel, "not_lease_knowledge", path.stem))
            continue

        cleaned = strip_obsidian_noise(raw)
        if len(cleaned) < 180:
            rejected.append(RejectedNote(source_rel, "quality_gate", title_from_text(path, cleaned)))
            continue
        score = quality_score_for_note(path, cleaned)
        if not passes_quality_gate(path, project_root, cleaned, score):
            rejected.append(RejectedNote(source_rel, "quality_gate", title_from_text(path, cleaned)))
            continue

        rel = path.relative_to(vault)
        title = title_from_text(path, cleaned)
        canonical_topic = canonical_topic_from_note(path, raw)
        digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
        candidate = CandidateNote(
            path=path,
            source_path=rel.as_posix(),
            title=title,
            canonical_topic=canonical_topic,
            source_bucket=source_bucket(path, project_root),
            quality_score=score,
            cleaned=cleaned,
            digest=digest,
            size_chars=len(cleaned),
        )
        key = canonical_key(canonical_topic)
        previous = best_by_topic.get(key)
        if previous is None or candidate_rank(candidate) > candidate_rank(previous):
            if previous is not None:
                rejected.append(RejectedNote(previous.source_path, "duplicate_canonical_topic", previous.title))
            best_by_topic[key] = candidate
        else:
            rejected.append(RejectedNote(source_rel, "duplicate_canonical_topic", title))

    exported: list[ExportedNote] = []
    selected = sorted(best_by_topic.values(), key=lambda item: item.source_path)[:max_docs]
    if canonical_rules is not None:
        selected += canonical_rule_candidates(rejected, canonical_rules)

    for candidate in selected:
        rel = Path(candidate.source_path)
        out_path = output / output_name(rel)
        body = "\n".join(
            [
                f"Title: {candidate.title}",
                f"Source: {candidate.source_path}",
                f"Canonical-Topic: {candidate.canonical_topic}",
                f"Source-Bucket: {candidate.source_bucket}",
                f"Quality-Score: {candidate.quality_score}",
                "Corpus: tune_lease_55_obsidian_lease_knowledge",
                "",
                candidate.cleaned,
                "",
            ]
        )
        out_path.write_text(body, encoding="utf-8")
        exported.append(
            ExportedNote(
                source_path=candidate.source_path,
                output_path=out_path.relative_to(output).as_posix(),
                title=candidate.title,
                canonical_topic=candidate.canonical_topic,
                source_bucket=candidate.source_bucket,
                quality_score=candidate.quality_score,
                digest=candidate.digest,
                size_chars=candidate.size_chars,
            )
        )

    rejected_summary: dict[str, int] = {}
    for item in rejected:
        rejected_summary[item.reason] = rejected_summary.get(item.reason, 0) + 1

    manifest = {
        "corpus": "tune_lease_55_obsidian_lease_knowledge",
        "vault": str(vault),
        "project": DEFAULT_PROJECT_REL.as_posix(),
        "excluded": sorted(EXCLUDED_PARTS),
        "quality_gate": {
            "dedupe": "canonical_topic",
            "min_score": {"root_note": 4, "default": 5},
            "private_markers": [
                "vertex_exclude",
                "agent_search_exclude",
                "public: false",
                "visibility: private",
                "confidential",
            ],
        },
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "rejected_summary": rejected_summary,
        "rejected_samples": [item.__dict__ for item in rejected[:80]],
        "documents": [note.__dict__ for note in exported],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "documents.jsonl").open("w", encoding="utf-8") as f:
        for note in exported:
            source_digest = hashlib.sha1(note.source_path.encode("utf-8")).hexdigest()
            doc_id = f"doc-{source_digest[:24]}"
            record = {
                "id": doc_id,
                "structData": {
                    "title": note.title,
                    "source_path": note.source_path,
                    "canonical_topic": note.canonical_topic,
                    "source_bucket": note.source_bucket,
                    "quality_score": note.quality_score,
                    "corpus": "tune_lease_55_obsidian_lease_knowledge",
                },
                "content": {
                    "mimeType": "text/plain",
                    "uri": f"{gcs_prefix.rstrip('/')}/{note.output_path}",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "README.txt").write_text(
        "Vertex AI Search pilot corpus for tune_lease_55.\n"
        "Generated from selected Obsidian project notes. Chat logs, daily memory, "
        "private reflection, and secret-like files are excluded.\n",
        encoding="utf-8",
    )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-docs", type=int, default=180)
    parser.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    args = parser.parse_args()

    exported = export_notes(args.vault, args.output, args.max_docs, args.gcs_prefix, canonical_rules=CANONICAL_RULES_PATH)
    print(f"exported={len(exported)}")
    print(f"output={args.output}")
    for note in exported[:20]:
        print(f"- {note.output_path} <- {note.source_path}")
    if len(exported) > 20:
        print(f"... {len(exported) - 20} more")


if __name__ == "__main__":
    main()
