#!/usr/bin/env python3
"""Small helper for writing Codex notes into an Obsidian vault."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path


def _home_candidates() -> list[Path]:
    home = Path.home()
    roots = [
        home / "Documents",
        home / "Obsidian",
        home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
        home / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
    ]
    return [p for p in roots if p.exists()]


def find_vaults() -> list[Path]:
    vaults: list[Path] = []
    env = os.getenv("OBSIDIAN_VAULT")
    if env and (Path(env) / ".obsidian").exists():
        vaults.append(Path(env).expanduser().resolve())
    for root in _home_candidates():
        for marker in root.rglob(".obsidian"):
            if marker.is_dir():
                vaults.append(marker.parent.resolve())
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in vaults:
        key = str(path)
        if key not in seen:
            uniq.append(path)
            seen.add(key)
    return uniq


def require_vault(raw: str | None) -> Path:
    if raw:
        vault = Path(raw).expanduser().resolve()
        if not (vault / ".obsidian").exists():
            raise SystemExit(f"Not an Obsidian vault: {vault}")
        return vault
    vaults = find_vaults()
    if len(vaults) == 1:
        return vaults[0]
    if not vaults:
        raise SystemExit("No Obsidian vault found. Pass --vault or set OBSIDIAN_VAULT.")
    raise SystemExit("Multiple vaults found. Pass --vault:\n" + "\n".join(str(v) for v in vaults))


def safe_note_path(vault: Path, rel: str) -> Path:
    target = (vault / rel).expanduser().resolve()
    if vault not in target.parents and target != vault:
        raise SystemExit("Refusing to write outside the vault")
    if target.suffix.lower() != ".md":
        target = target.with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def append_text(path: Path, text: str) -> None:
    body = text.strip()
    if not body:
        raise SystemExit("Nothing to write")
    prefix = "\n\n" if path.exists() and path.read_text(encoding="utf-8").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + body + "\n")


def cmd_find_vaults(_: argparse.Namespace) -> None:
    for vault in find_vaults():
        print(vault)


def cmd_append_daily(args: argparse.Namespace) -> None:
    vault = require_vault(args.vault)
    day = args.date or dt.date.today().isoformat()
    folder = args.folder.strip("/") if args.folder else ""
    rel = f"{folder}/{day}.md" if folder else f"{day}.md"
    path = safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    section = f"## {now} Codex\n\n{args.text.strip()}"
    append_text(path, section)
    print(path)


def cmd_write_note(args: argparse.Namespace) -> None:
    vault = require_vault(args.vault)
    path = safe_note_path(vault, args.path)
    title = args.title.strip() if args.title else path.stem
    if path.exists() and not args.append:
        raise SystemExit(f"Note already exists. Use --append: {path}")
    if args.text_file:
        body = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        body = args.text.strip()
    if args.append:
        append_text(path, body)
    else:
        created = dt.date.today().isoformat()
        path.write_text(f"---\ncreated: {created}\nsource: codex\n---\n\n# {title}\n\n{body}\n", encoding="utf-8")
    print(path)


def cmd_search(args: argparse.Namespace) -> None:
    vault = require_vault(args.vault)
    query = args.query.lower()
    limit = max(1, args.limit)
    count = 0
    for path in vault.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if query in text.lower() or query in path.name.lower():
            print(path.relative_to(vault))
            count += 1
            if count >= limit:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("find-vaults")
    p.set_defaults(func=cmd_find_vaults)

    p = sub.add_parser("append-daily")
    p.add_argument("--vault")
    p.add_argument("--folder", default="Daily")
    p.add_argument("--date")
    p.add_argument("--text", required=True)
    p.set_defaults(func=cmd_append_daily)

    p = sub.add_parser("write-note")
    p.add_argument("--vault")
    p.add_argument("--path", required=True)
    p.add_argument("--title")
    p.add_argument("--text", default="")
    p.add_argument("--text-file")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_write_note)

    p = sub.add_parser("search")
    p.add_argument("--vault")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
