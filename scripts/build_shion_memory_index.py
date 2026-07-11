"""Build a lightweight Shion memory index from existing local memory sources.

This is intentionally read-mostly and local: it does not call LLMs, does not
write to Obsidian, and does not alter the daily improvement pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.shion_memory_taxonomy import MEMORY_TYPES, RECALL_ROUTES, make_memory_record

DEFAULT_OUTPUT = REPO_ROOT / "data" / "shion_memory_index.json"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _memory_bullets_from_markdown(path: Path, source: str) -> list[dict[str, Any]]:
    text = _read_text(path)
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        if _skip_markdown_bullet(content):
            continue
        private = "Private Reflection" in content or "私室" in content
        records.append(
            make_memory_record(
                content,
                source=source,
                source_path=str(path.relative_to(REPO_ROOT)),
                private=private,
            ).to_dict()
        )
    return records


def _skip_markdown_bullet(content: str) -> bool:
    if len(content) < 12:
        return True
    if "自動生成プレースホルダー" in content:
        return True
    # Section labels such as "**Key Features**:" are headings, not memories.
    if content.startswith("**") and content.endswith(":") and len(content) < 80:
        return True
    if content in {"", "-"}:
        return True
    return False


def _mind_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    records: list[dict[str, Any]] = []

    upper = data.get("upper_authority")
    if isinstance(upper, dict):
        values = upper.get("values") or []
        content = (
            f"{upper.get('name', 'Mana')}: {upper.get('role', '')} / "
            f"{upper.get('boundary', '')} / values={'; '.join(map(str, values))}"
        )
        records.append(
            make_memory_record(
                content,
                source="mind.upper_authority",
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_type="value_memory",
                confidence=0.95,
            ).to_dict()
        )

    world_view = data.get("world_view")
    if isinstance(world_view, dict):
        summary = str(world_view.get("summary") or "").strip()
        if summary:
            records.append(
                make_memory_record(
                    summary,
                    source="mind.world_view",
                    source_path=str(path.relative_to(REPO_ROOT)),
                    memory_type="factual_memory",
                    confidence=0.75,
                ).to_dict()
            )
        for signal in world_view.get("key_signals") or []:
            records.append(
                make_memory_record(
                    str(signal),
                    source="mind.world_view.key_signal",
                    source_path=str(path.relative_to(REPO_ROOT)),
                    memory_type="factual_memory",
                    confidence=0.7,
                ).to_dict()
            )

    for kp in data.get("conversation_keypoints") or []:
        if not isinstance(kp, dict):
            continue
        content = str(kp.get("fact") or kp.get("content") or "").strip()
        if not content:
            continue
        records.append(
            make_memory_record(
                content,
                source=str(kp.get("source") or "mind.conversation_keypoint"),
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_type=kp.get("memory_type") or None,
                confidence=float(kp.get("confidence") or 0.75),
            ).to_dict()
        )

    return records


def _knowledge_markdown_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_dirs = (REPO_ROOT / "knowledge_base" / "okf_lease_concepts",)
    for root in source_dirs:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            rel = str(path.relative_to(REPO_ROOT))
            text = _read_text(path)
            snippets = _markdown_snippets(text)
            mtype = "judgment_memory" if "/rules/" in rel else "factual_memory"
            # ノートタイトル（主題語）。content には含めない（id が変わり
            # created_at / last_used_at の引き継ぎが壊れるため）。ベクトル索引側が
            # 埋め込みテキストの前置に使う。
            title = _note_title(text)
            for snippet in snippets:
                record = make_memory_record(
                    snippet,
                    source="knowledge_base",
                    source_path=rel,
                    memory_type=mtype,  # type: ignore[arg-type]
                    confidence=0.82,
                ).to_dict()
                if title:
                    record["topic"] = title
                records.append(record)
    return records


def _canonical_judgment_rule_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = payload.get("rules") if isinstance(payload, dict) else []
    records: list[dict[str, Any]] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("private") is True or str(rule.get("status") or "") != "active":
            continue
        statement = str(rule.get("canonical_statement") or "").strip()
        if len(statement) < 12:
            continue
        record = make_memory_record(
            statement,
            source="canonical_judgment_rules",
            source_path=str(path.relative_to(REPO_ROOT)),
            memory_type="judgment_memory",
            confidence=float(rule.get("confidence") or 0.82),
        ).to_dict()
        record["topic"] = str(rule.get("concept") or "")
        record["evidence_count"] = int(rule.get("evidence_count") or 0)
        record["user_evidence_count"] = int(rule.get("user_evidence_count") or 0)
        record["evidence_paths"] = list(rule.get("evidence_paths") or [])[:6]
        records.append(record)
    return records


def _note_title(text: str) -> str:
    """frontmatter の title:、なければ最初の H1 見出しを返す。"""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.startswith("title:"):
                return stripped[len("title:"):].strip().strip("\"'")
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _markdown_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    current_heading = ""
    lines = text.splitlines()
    start = 0
    # YAML frontmatter はノートのメタデータであり記憶ではないため索引に入れない
    # （"tags: [...]" や "confidence: medium" が記憶レコード化して想起枠を奪っていた）
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line or line == "---" or line.startswith("<!--"):
            continue
        if line.startswith("#"):
            current_heading = line.lstrip("#").strip()
            continue
        if line.startswith("- "):
            content = line[2:].strip()
        else:
            content = line
        if len(content) < 18 or content.startswith("```"):
            continue
        if current_heading:
            content = f"{current_heading}: {content}"
        snippets.append(content)
    return snippets[:24]


def _load_previous_fields(path: Path) -> dict[str, dict[str, str]]:
    """前回索引から、引き継ぐべきフィールド（初出日・最終使用日）をIDごとに読む。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    previous: dict[str, dict[str, str]] = {}
    for record in data.get("records") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        if rid:
            previous[rid] = {
                "created_at": str(record.get("created_at") or ""),
                "last_used_at": str(record.get("last_used_at") or ""),
            }
    return previous


def build_index(
    previous_index_path: Path | None = None, *, demo_safe: bool = False
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    memory_path = REPO_ROOT / "MEMORY.md"
    if memory_path.exists():
        records.extend(_memory_bullets_from_markdown(memory_path, "long_term_memory"))

    memory_dir = REPO_ROOT / "memory"
    if memory_dir.exists():
        for path in sorted(memory_dir.glob("20*.md"))[-14:]:
            records.extend(_memory_bullets_from_markdown(path, "daily_memory"))

    mind_path = REPO_ROOT / "data" / "mind.json"
    if mind_path.exists():
        records.extend(_mind_records(mind_path))

    # 会話から承認を経て昇格した長期記憶（apply_shion_memory_promotions.py が追記）
    promoted_path = REPO_ROOT / "knowledge_base" / "shion_promoted_memories.md"
    if promoted_path.exists():
        records.extend(_memory_bullets_from_markdown(promoted_path, "promoted_memory"))

    records.extend(_knowledge_markdown_records())

    canonical_rules_path = REPO_ROOT / "data" / "canonical_judgment_rules.json"
    if canonical_rules_path.exists():
        records.extend(_canonical_judgment_rule_records(canonical_rules_path))

    # Deduplicate by stable id, keeping the first occurrence.
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = str(record.get("id") or "")
        if rid and rid not in deduped:
            deduped[rid] = record

    final_records = list(deduped.values())

    # 再生成のたびに created_at（初出日）が今日へリセットされると、鮮度更新の
    # 「作成から45日超かつ未使用 → stale」が永久に発火しないため、前回索引から
    # 初出日と最終使用日を引き継ぐ。
    previous = _load_previous_fields(
        previous_index_path or (REPO_ROOT / "data" / "shion_memory_index.json")
    )
    for record in final_records:
        prev = previous.get(str(record.get("id") or ""))
        if not prev:
            continue
        if prev.get("created_at"):
            record["created_at"] = prev["created_at"]
        if prev.get("last_used_at") and not record.get("last_used_at"):
            record["last_used_at"] = prev["last_used_at"]

    # 改訂宣言（data/shion_memory_revisions.jsonl）を再適用する。
    # 宣言ファイルが真実の源なので、索引を再生成しても revised / supersedes が消えない。
    from scripts.revise_shion_memory import apply_revisions, load_revisions

    revisions = load_revisions(REPO_ROOT / "data" / "shion_memory_revisions.jsonl")
    if revisions:
        holder: dict[str, Any] = {"records": final_records}
        apply_revisions(holder, revisions)
        final_records = holder["records"]

    if demo_safe:
        # 公開デモ環境には対話・内省・private の記憶を載せない
        final_records = [r for r in final_records if not _is_demo_unsafe(r)]

    counts = Counter(str(r.get("memory_type") or "unknown") for r in final_records)
    status_counts = Counter(str(r.get("status") or "active") for r in final_records)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "taxonomy": MEMORY_TYPES,
        "recall_routes": RECALL_ROUTES,
        "summary": {
            "total_records": len(final_records),
            "by_type": dict(sorted(counts.items())),
            "by_status": dict(sorted(status_counts.items())),
        },
        "records": final_records,
    }


def _is_demo_unsafe(record: dict[str, Any]) -> bool:
    """公開デモバンドルへ載せてはいけない記憶か（対話・内省・private）。"""
    if record.get("private"):
        return True
    if str(record.get("status") or "") == "private":
        return True
    return str(record.get("memory_type") or "") in {"dialogue_memory", "reflection_memory"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Shion memory taxonomy index.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--demo-safe",
        action="store_true",
        help="公開デモ向けに dialogue_memory / reflection_memory / private を除外する",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 引き継ぎ元は出力先の既存索引。初回出力先（デモ用の別パス等）はローカル既定索引から引き継ぐ
    previous = args.output if args.output.exists() else None
    index = build_index(previous_index_path=previous, demo_safe=args.demo_safe)
    text = json.dumps(index, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(text)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.output)
    print(f"wrote={args.output}")
    print(f"total_records={index['summary']['total_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
