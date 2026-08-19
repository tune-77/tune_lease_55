#!/usr/bin/env python3
"""Tag and file newly added Obsidian vault notes, and surface related notes.

For each vault markdown file not seen in a previous run:
  1. Propose 1-3 tags and a destination folder via Vertex AI text generation,
     given the vault's existing tag/folder vocabulary as context.
  2. Find up to 3 related existing notes via local keyword overlap (no RAG
     dependency — this is a batch curator script, not a chat path; see
     .claude/skills/obsidian-search-rule).
  3. With --apply: write the frontmatter tags and move the file into the
     suggested folder, after backing up the original file.

Read-only by default (--dry-run behavior); pass --apply to actually write.
Every Vertex call failure degrades to "tags/folder not proposed" for that
file rather than raising, matching the rest of the Obsidian tooling.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._obsidian_common import (  # noqa: E402
    FRONTMATTER_RE,
    iter_vault_markdown_files,
    note_tags,
    parse_note_frontmatter,
    strip_frontmatter,
)
from scripts._vertex_text_gen import generate_text  # noqa: E402
from scripts.obsidian_taxonomy_audit import collect_tags_and_folders  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_STATE_PATH = DATA_DIR / "obsidian_note_curator_state.json"
DEFAULT_BACKUP_DIR = DATA_DIR / "obsidian_note_curator_backup"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "obsidian_note_curator_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "obsidian_note_curator_latest.md"
DEFAULT_MAX_FILES_PER_RUN = 20

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[一-龥ぁ-んァ-ン]{2,}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_run": "", "seen_files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_run": "", "seen_files": {}}
    if not isinstance(data, dict):
        return {"last_run": "", "seen_files": {}}
    data.setdefault("seen_files", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_new_or_changed_files(vault: Path, state: dict[str, Any]) -> list[Path]:
    seen: dict[str, str] = state.get("seen_files", {})
    changed: list[Path] = []
    for path in iter_vault_markdown_files(vault):
        rel = str(path.relative_to(vault))
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        if seen.get(rel) != mtime:
            changed.append(path)
    return changed


def _tokens(text: str) -> set[str]:
    stopwords = {"こと", "ため", "よう", "する", "した", "ある", "ない", "これ", "それ"}
    return {token for token in TOKEN_RE.findall(text) if token not in stopwords}


def find_related_notes(target_path: Path, target_text: str, vault: Path, *, limit: int = 3) -> list[dict[str, Any]]:
    target_tokens = _tokens(strip_frontmatter(target_text))
    if not target_tokens:
        return []
    scored: list[tuple[float, str]] = []
    for path in iter_vault_markdown_files(vault):
        if path == target_path:
            continue
        text = _read_text(path)
        if not text:
            continue
        tokens = _tokens(strip_frontmatter(text))
        if not tokens:
            continue
        overlap = target_tokens & tokens
        if not overlap:
            continue
        score = len(overlap) / len(target_tokens | tokens)
        scored.append((score, str(path.relative_to(vault))))
    scored.sort(key=lambda item: -item[0])
    return [{"path": rel, "score": round(score, 4)} for score, rel in scored[:limit] if score > 0]


def _sanitize_folder(raw: str) -> str:
    parts = [part.strip() for part in re.split(r"[\\/]+", raw.strip()) if part.strip()]
    parts = [part for part in parts if part not in (".", "..")]
    return "/".join(parts[:3])


def _parse_proposal(text: str) -> dict[str, Any]:
    tags_match = re.search(r"TAGS?\s*[:：]\s*(.+)", text, re.IGNORECASE)
    folder_match = re.search(r"FOLDER\s*[:：]\s*(.+)", text, re.IGNORECASE)
    tags: list[str] = []
    if tags_match:
        raw_tags = re.split(r"[,、]", tags_match.group(1).splitlines()[0])
        tags = [tag.strip() for tag in raw_tags if tag.strip()][:3]
    folder = ""
    if folder_match:
        folder = _sanitize_folder(folder_match.group(1).splitlines()[0])
    return {"tags": tags, "folder": folder, "parsed": bool(tags or folder)}


def propose_tags_and_folder(
    note_text: str,
    *,
    existing_tags: list[str],
    existing_folders: list[str],
    generate_fn=generate_text,
) -> dict[str, Any]:
    prompt = (
        "あなたはObsidian Vaultの整理を手伝うアシスタントです。\n"
        "以下のノート本文に対して、1〜3個のタグと移動先フォルダ名を1つ提案してください。\n"
        "できる限り既存のタグ・フォルダを再利用し、新規タグ・フォルダは本当に必要な場合だけ提案してください。\n\n"
        f"既存タグ一覧: {', '.join(existing_tags[:60]) or '(なし)'}\n"
        f"既存フォルダ一覧: {', '.join(existing_folders[:60]) or '(なし)'}\n\n"
        f"ノート本文（先頭2000文字）:\n{note_text[:2000]}\n\n"
        "以下の形式**のみ**で回答してください（説明文は不要）:\n"
        "TAGS: タグ1, タグ2\n"
        "FOLDER: フォルダ名"
    )
    result = generate_fn(prompt)
    if not result.get("used"):
        return {"tags": [], "folder": "", "parsed": False, "status": result.get("status", "unavailable")}
    proposal = _parse_proposal(result.get("text", ""))
    proposal["status"] = "ok" if proposal["parsed"] else "unparseable"
    return proposal


def _split_frontmatter_and_body(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        frontmatter = None
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    body = text[match.end() :]
    return frontmatter, body


def _render_note(frontmatter: dict[str, Any], body: str) -> str:
    if not frontmatter:
        return body
    dumped = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{dumped}\n---\n{body}"


def apply_tags_and_move(
    path: Path,
    vault: Path,
    proposal: dict[str, Any],
    *,
    backup_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    original_text = _read_text(path)
    rel = path.relative_to(vault)
    backup_path = backup_dir / run_id / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(original_text, encoding="utf-8")

    frontmatter, body = _split_frontmatter_and_body(original_text)
    merged_tags = list(dict.fromkeys([*note_tags(frontmatter), *proposal.get("tags", [])]))
    if merged_tags:
        frontmatter["tags"] = merged_tags
    new_text = _render_note(frontmatter, body)

    target_path = path
    moved = False
    folder = proposal.get("folder") or ""
    if folder:
        destination_dir = vault / folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        candidate = destination_dir / path.name
        if candidate.resolve() != path.resolve():
            target_path = candidate
            moved = True

    if new_text != original_text or moved:
        target_path.write_text(new_text, encoding="utf-8")
        if moved and target_path != path:
            path.unlink()

    return {
        "applied": True,
        "backup_path": str(backup_path),
        "new_path": str(target_path.relative_to(vault)),
        "moved": moved,
        "tags_written": merged_tags,
    }


def build_report(
    vault: Path,
    *,
    state_path: Path,
    backup_dir: Path,
    apply: bool,
    max_files: int,
    generate_fn=generate_text,
) -> dict[str, Any]:
    state = load_state(state_path)
    candidates = detect_new_or_changed_files(vault, state)[:max_files]
    tag_counts, folder_counts = collect_tags_and_folders(vault)
    existing_tags = sorted(tag_counts)
    existing_folders = sorted(folder_counts)
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    entries: list[dict[str, Any]] = []
    for path in candidates:
        text = _read_text(path)
        proposal = propose_tags_and_folder(text, existing_tags=existing_tags, existing_folders=existing_folders, generate_fn=generate_fn)
        related = find_related_notes(path, text, vault)
        entry: dict[str, Any] = {
            "path": str(path.relative_to(vault)),
            "proposed_tags": proposal.get("tags", []),
            "proposed_folder": proposal.get("folder", ""),
            "proposal_status": proposal.get("status"),
            "related_notes": related,
            "applied": False,
        }
        if apply and proposal.get("parsed"):
            applied = apply_tags_and_move(path, vault, proposal, backup_dir=backup_dir, run_id=run_id)
            entry.update(applied)
            state["seen_files"][applied["new_path"]] = dt.datetime.fromtimestamp(
                (vault / applied["new_path"]).stat().st_mtime
            ).isoformat()
        else:
            rel = str(path.relative_to(vault))
            state["seen_files"][rel] = dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        entries.append(entry)

    if apply:
        state["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
        save_state(state_path, state)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "apply": apply,
        "run_id": run_id,
        "candidate_count": len(candidates),
        "processed_count": len(entries),
        "entries": entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Note Curator",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Apply mode: `{report['apply']}`",
        f"- Candidates found: {report['candidate_count']}",
        "",
        "## Entries",
    ]
    if not report["entries"]:
        lines.append("- なし")
    for entry in report["entries"]:
        lines.append(f"### {entry['path']}")
        lines.append(f"- proposed_tags: {entry.get('proposed_tags')}")
        lines.append(f"- proposed_folder: {entry.get('proposed_folder')}")
        lines.append(f"- status: {entry.get('proposal_status')}")
        lines.append(f"- applied: {entry.get('applied')}")
        if entry.get("related_notes"):
            lines.append("- related_notes:")
            for related in entry["related_notes"]:
                lines.append(f"  - {related['path']} (score={related['score']})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--apply", action="store_true", help="Actually write tags/move files (default: report only)")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES_PER_RUN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true", help="Print report to stdout, write nothing")
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else resolve_obsidian_vault()
    if not vault.exists():
        print(f"[ERROR] Vault not found: {vault}")
        return 1

    report = build_report(
        vault,
        state_path=args.state.expanduser(),
        backup_dir=args.backup_dir.expanduser(),
        apply=args.apply and not args.dry_run,
        max_files=max(1, args.max_files),
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
