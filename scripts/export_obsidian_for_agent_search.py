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
)

PREFERRED_PARTS = {
    "Research",
    "Judgment Assets",
}

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
    digest: str
    size_chars: int


def should_exclude(path: Path, project_root: Path) -> bool:
    rel = path.relative_to(project_root)
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
    if parts & PREFERRED_PARTS:
        return True
    return any(keyword in path.name for keyword in ROOT_NOTE_KEYWORDS)


def has_secret_like_text(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in SECRET_PATTERNS)


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


def output_name(rel: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", rel.with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}__{digest}.txt"


def export_notes(vault: Path, output: Path, max_docs: int, gcs_prefix: str) -> list[ExportedNote]:
    project_root = vault / DEFAULT_PROJECT_REL
    if not project_root.exists():
        raise SystemExit(f"Project notes not found: {project_root}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedNote] = []
    candidates = sorted(project_root.rglob("*.md"), key=lambda p: str(p.relative_to(project_root)))

    for path in candidates:
        if len(exported) >= max_docs:
            break
        if should_exclude(path, project_root):
            continue
        if not is_preferred_location(path, project_root):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(raw.strip()) < 120:
            continue
        if has_secret_like_text(raw):
            continue
        if not matches_lease_knowledge(path, raw):
            continue

        cleaned = strip_obsidian_noise(raw)
        rel = path.relative_to(vault)
        title = title_from_text(path, cleaned)
        digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
        out_path = output / output_name(rel)
        body = "\n".join(
            [
                f"Title: {title}",
                f"Source: {rel.as_posix()}",
                "Corpus: tune_lease_55_obsidian_lease_knowledge",
                "",
                cleaned,
                "",
            ]
        )
        out_path.write_text(body, encoding="utf-8")
        exported.append(
            ExportedNote(
                source_path=rel.as_posix(),
                output_path=out_path.relative_to(output).as_posix(),
                title=title,
                digest=digest,
                size_chars=len(cleaned),
            )
        )

    manifest = {
        "corpus": "tune_lease_55_obsidian_lease_knowledge",
        "vault": str(vault),
        "project": DEFAULT_PROJECT_REL.as_posix(),
        "excluded": sorted(EXCLUDED_PARTS),
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

    exported = export_notes(args.vault, args.output, args.max_docs, args.gcs_prefix)
    print(f"exported={len(exported)}")
    print(f"output={args.output}")
    for note in exported[:20]:
        print(f"- {note.output_path} <- {note.source_path}")
    if len(exported) > 20:
        print(f"... {len(exported) - 20} more")


if __name__ == "__main__":
    main()
