#!/usr/bin/env python3
"""Obsidian 改善ログのアーカイバ.

✅ マーク付きエントリを「改善ログ/アーカイブ/YYYY-MM.md」へ移動する。

Usage:
    python scripts/obsidian_archive.py --dry-run   # 対象を表示するのみ（変更なし）
    python scripts/obsidian_archive.py --archive   # 実際に移動する
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Vault 検索 ──────────────────────────────────────────────────────────────


def _resolve_vault() -> Path:
    """obsidian_bridge.find_vault() でパスを解決する。見つからなければ終了。"""
    _mobile_app = Path(__file__).resolve().parent.parent / "mobile_app"
    if _mobile_app.exists() and str(_mobile_app) not in sys.path:
        sys.path.insert(0, str(_mobile_app))
    try:
        from obsidian_bridge import find_vault  # type: ignore[import]

        vault = find_vault()
        if vault:
            return vault
    except ImportError:
        pass
    env = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    logger.error(
        "iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH 環境変数を設定してください。"
    )
    sys.exit(1)


# ─── 改善ノート探索 ────────────────────────────────────────────────────────────

_INDEX_PATTERNS = [
    "tuneLease55/改善策インデックス_2026.md",
    "tuneLease55/改善*.md",
    "改善.md",
    "**/改善策インデックス*.md",
    "**/改善.md",
]


def find_improvement_files(vault: Path) -> list[Path]:
    """改善ログ関連の Markdown ファイルを返す（アーカイブ済みディレクトリは除外）。"""
    archive_dir = vault / "改善ログ" / "アーカイブ"
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in _INDEX_PATTERNS:
        if "*" in pattern or "**" in pattern:
            candidates = sorted(vault.glob(pattern))
        else:
            candidates = [vault / pattern]
        for p in candidates:
            if not p.exists():
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            if archive_dir in resolved.parents or resolved == archive_dir:
                continue
            found.append(resolved)
            seen.add(resolved)
    return found


# ─── エントリ抽出 ──────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"✅.*?(\d{4}-\d{2})-\d{2}")


class ArchiveEntry(NamedTuple):
    lines: list[str]  # 移動するテキスト行（末尾改行付き）
    yearmonth: str    # "YYYY-MM"
    source: Path      # 元ファイル


def extract_entries(path: Path) -> list[ArchiveEntry]:
    """✅ マーク付き行（＋直後のインデント継続行）を ArchiveEntry として返す。"""
    raw = path.read_text(encoding="utf-8").splitlines(keepends=True)
    entries: list[ArchiveEntry] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if "✅" not in line:
            i += 1
            continue
        # ✅ 行本体 + 直後のインデント行をまとめてブロックとする
        block: list[str] = [line]
        j = i + 1
        while j < len(raw):
            nxt = raw[j]
            if nxt.startswith((" ", "\t")) or nxt.strip() == "":
                block.append(nxt)
                j += 1
            else:
                break
        # 末尾の空行はブロックから除外
        while block and block[-1].strip() == "":
            block.pop()
        # YYYY-MM を ✅ 行から取得、なければ当月
        m = _DATE_RE.search(line)
        ym = m.group(1) if m else datetime.now().strftime("%Y-%m")
        entries.append(ArchiveEntry(lines=block, yearmonth=ym, source=path))
        i = j
    return entries


# ─── アーカイブ書き込み ────────────────────────────────────────────────────────


def write_to_archive(
    vault: Path, entries: list[ArchiveEntry], dry_run: bool
) -> None:
    """エントリを YYYY-MM.md へ追記する。"""
    grouped: dict[str, list[ArchiveEntry]] = {}
    for e in entries:
        grouped.setdefault(e.yearmonth, []).append(e)

    archive_root = vault / "改善ログ" / "アーカイブ"

    for ym, batch in sorted(grouped.items()):
        archive_path = archive_root / f"{ym}.md"
        content = "".join(line for e in batch for line in e.lines)
        if not content.endswith("\n"):
            content += "\n"

        print(f"\n[アーカイブ先] {archive_path.relative_to(vault)}")
        for e in batch:
            preview = "".join(e.lines[:1]).strip()[:80]
            print(f"  ← {e.source.relative_to(vault)}: {preview}")

        if dry_run:
            continue

        archive_root.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            header = f"# 改善ログ アーカイブ {ym}\n\n"
            archive_path.write_text(header + content, encoding="utf-8")
        else:
            with archive_path.open("a", encoding="utf-8") as f:
                f.write(content)
        logger.info("アーカイブ書き込み完了: %s (%d件)", archive_path.name, len(batch))


# ─── ソースから削除 ────────────────────────────────────────────────────────────


def remove_entries_from_sources(
    entries: list[ArchiveEntry], dry_run: bool
) -> None:
    """元ファイルから ✅ エントリ行を削除する。"""
    by_source: dict[Path, list[ArchiveEntry]] = {}
    for e in entries:
        by_source.setdefault(e.source, []).append(e)

    for src, batch in by_source.items():
        original = src.read_text(encoding="utf-8")
        result = original
        for e in batch:
            block_text = "".join(e.lines)
            result = result.replace(block_text, "", 1)
        # 連続する空行を2行以内に縮める
        result = re.sub(r"\n{3,}", "\n\n", result)

        removed = len(original) - len(result)
        if dry_run:
            print(f"  [dry-run] {src.name}: {removed} 文字を削除予定")
        else:
            src.write_text(result, encoding="utf-8")
            logger.info("ソース更新完了: %s (-%d文字)", src.name, removed)


# ─── メイン ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obsidian ✅ マーク付きエントリをアーカイブに移動する"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run", action="store_true", help="移動対象を表示するのみ（変更なし）"
    )
    group.add_argument("--archive", action="store_true", help="実際に移動する")
    args = parser.parse_args()
    dry_run: bool = args.dry_run

    vault = _resolve_vault()
    logger.info("Vault: %s", vault)

    files = find_improvement_files(vault)
    if not files:
        logger.warning("改善ノートが見つかりませんでした。終了します。")
        return

    all_entries: list[ArchiveEntry] = []
    for f in files:
        entries = extract_entries(f)
        if entries:
            logger.info("%s: %d件の✅エントリを検出", f.name, len(entries))
        all_entries.extend(entries)

    if not all_entries:
        print("✅ マーク付きエントリは見つかりませんでした。")
        return

    mode = "[DRY-RUN] " if dry_run else ""
    print(f"\n{mode}合計 {len(all_entries)} 件のエントリを処理します。")

    write_to_archive(vault, all_entries, dry_run)
    remove_entries_from_sources(all_entries, dry_run)

    if dry_run:
        print("\n[dry-run] 変更は適用されていません。--archive で実行してください。")
    else:
        print(f"\n✅ アーカイブ完了: {len(all_entries)} 件を移動しました。")


if __name__ == "__main__":
    main()
