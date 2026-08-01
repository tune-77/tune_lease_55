#!/usr/bin/env python3
"""iCloud 上の Obsidian Vault の孤立ファイルに wikiリンクを自動付与する。

対象フォルダ: Projects/tune_lease_55/, リースニュース/, リース知識/
除外フォルダ: Daily/, Clippings/

使い方:
    python scripts/auto_wikilink.py                  # Vault を自動検出
    python scripts/auto_wikilink.py --vault /path    # Vault を直接指定
    python scripts/auto_wikilink.py --dry-run        # 変更なしで確認
    python scripts/auto_wikilink.py --file foo.md    # 特定ファイルのみ処理

graph effect レポート連携（scripts/build_obsidian_graph_judgment_effect.py）:
    reports/obsidian_graph_judgment_effect_latest.json が存在する場合、
    isolated_but_used / bridge_candidate に分類されたノートへ、
    判断に効いている effective_hub への [[wikilink]] を自動で1本追加する
    （レポートの next_action 提案をそのまま実行するだけで、新規の判定ロジックは持たない）。
    --no-graph-effect-links で無効化できる。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from obsidian_query import iter_vault_md_files  # noqa: E402

# ---------------------------------------------------------------------------
# Vault detection
# ---------------------------------------------------------------------------

_DEFAULT_VAULT_CANDIDATES = [
    Path(os.environ.get("OBSIDIAN_VAULT", "")).expanduser()
    if os.environ.get("OBSIDIAN_VAULT") else None,
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault",
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
]

TARGET_FOLDERS = (
    "Projects/tune_lease_55",
    "リースニュース",
    "リース知識",
)

EXCLUDED_PARTS = ("Daily", "Clippings", ".obsidian", ".claude")

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
# 日付ログファイル: Cases/ AI Chat/ Asset Finance/ 直下の YYYY-MM-DD.md
_DATE_LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

DEFAULT_GRAPH_EFFECT_JSON = _REPO_ROOT / "reports" / "obsidian_graph_judgment_effect_latest.json"
_GRAPH_EFFECT_LINK_BUCKETS = ("isolated_but_used", "bridge_candidates")
_GRAPH_EFFECT_SECTION_HEADER = "## 関連（判断効果分析による自動リンク）"
_GRAPH_EFFECT_MAX_HUBS_PER_NOTE = 2


def find_vault(override: str | None = None) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Vault not found: {p}")
    for p in _DEFAULT_VAULT_CANDIDATES:
        if p and p.is_dir():
            return p.resolve()
    raise FileNotFoundError(
        "iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT を設定するか --vault を指定してください。"
    )


# ---------------------------------------------------------------------------
# Title index
# ---------------------------------------------------------------------------

def build_title_index(vault: Path) -> dict[str, str]:
    """Vault 内の対象フォルダの全 .md ファイルを走査し {stem: rel_path} を返す。

    ファイル名が衝突する場合は最初に見つかったものを使う。
    日本語ファイル名も含めてインデックス化する。
    """
    index: dict[str, str] = {}
    for path in iter_vault_md_files(vault, TARGET_FOLDERS, EXCLUDED_PARTS):
        stem = path.stem
        rel = str(path.relative_to(vault))
        if stem not in index:
            index[stem] = rel
    return index


# ---------------------------------------------------------------------------
# Graph judgment effect report integration
# (scripts/build_obsidian_graph_judgment_effect.py の next_action 提案を反映する)
# ---------------------------------------------------------------------------

def load_graph_effect_link_targets(report_path: Path) -> dict[str, list[str]]:
    """graph judgment effect レポートから『どのノートにハブへのリンクを足すか』を返す。

    isolated_but_used / bridge_candidate に分類されたノート（vault相対パス）を key に、
    上位 effective_hub の wikilink stem（拡張子なし相対パス）のリストを value として返す。
    レポートが読めない・hub が無い場合は空 dict（no-op）。
    """
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(report, dict):
        return {}

    hub_stems: list[str] = []
    for item in report.get("top_effective_hubs") or []:
        path = str(item.get("path") or "").strip()
        if path.lower().endswith(".md"):
            path = path[:-3]
        if path:
            hub_stems.append(path)
    if not hub_stems:
        return {}

    targets: dict[str, list[str]] = {}
    for bucket in _GRAPH_EFFECT_LINK_BUCKETS:
        for item in report.get(bucket) or []:
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            targets[path] = hub_stems[:_GRAPH_EFFECT_MAX_HUBS_PER_NOTE]
    return targets


def add_graph_effect_links(body: str, rel_path: str, targets: dict[str, list[str]]) -> tuple[str, int]:
    """孤立/橋渡しノートへ、判断に効いているハブへの [[wikilink]] を追加する。

    既にセクションを追加済み、または全ハブへ既にリンク済みなら何もしない。
    Returns:
        (new_body, added_count)
    """
    hub_stems = targets.get(rel_path)
    if not hub_stems or _GRAPH_EFFECT_SECTION_HEADER in body:
        return body, 0

    linked_stems = {m.group(1).split("|", 1)[0].split("#", 1)[0].strip() for m in _WIKILINK_RE.finditer(body)}
    new_hubs = [stem for stem in hub_stems if stem not in linked_stems]
    if not new_hubs:
        return body, 0

    section = "\n\n" + _GRAPH_EFFECT_SECTION_HEADER + "\n" + "\n".join(f"- [[{stem}]]" for stem in new_hubs) + "\n"
    return body.rstrip("\n") + section, len(new_hubs)


# ---------------------------------------------------------------------------
# Frontmatter handling
# ---------------------------------------------------------------------------

def split_frontmatter(text: str) -> tuple[str, str]:
    """(frontmatter_block, body) を返す。frontmatter がなければ ("", text)。"""
    m = _FRONTMATTER_RE.match(text)
    if m:
        return m.group(0), text[m.end():]
    return "", text


# ---------------------------------------------------------------------------
# Linkification
# ---------------------------------------------------------------------------

def _existing_wikilink_spans(text: str) -> list[tuple[int, int]]:
    """既存の [[...]] の (start, end) スパンリストを返す。"""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find("[[", pos)
        if start < 0:
            break
        end = text.find("]]", start + 2)
        if end < 0:
            break
        spans.append((start, end + 2))
        pos = end + 2
    return spans


def _in_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def linkify_body(body: str, title_index: dict[str, str], own_stem: str) -> tuple[str, int]:
    """本文中の他ノートへのテキスト言及を [[ノート名]] に変換する。

    Returns:
        (new_body, change_count)
    """
    if not body.strip():
        return body, 0

    # 対象ステムを文字数の多い順（長い名前を優先）でソートして検索
    candidates = [
        stem for stem in title_index
        if stem != own_stem and len(stem) >= 4
    ]
    candidates.sort(key=len, reverse=True)

    changes = 0
    for stem in candidates:
        if stem not in body:
            continue
        # 既存 wikilink スパンを再計算（変換後にオフセットがずれるため毎回）
        spans = _existing_wikilink_spans(body)

        # stem の全出現位置を探して置換
        new_body_parts: list[str] = []
        prev = 0
        for m in re.finditer(re.escape(stem), body):
            start, end = m.start(), m.end()
            # 既存 wikilink 内ならスキップ
            if _in_span(start, spans):
                new_body_parts.append(body[prev:end])
                prev = end
                continue
            # 直前文字が [ ならスキップ（[[... の途中）
            if start > 0 and body[start - 1] == "[":
                new_body_parts.append(body[prev:end])
                prev = end
                continue
            # コードブロック内はスキップ（簡易判定: バッククォート行）
            line_start = body.rfind("\n", 0, start) + 1
            line_text = body[line_start:body.find("\n", start)]
            if line_text.lstrip().startswith(("```", "    ", "\t")):
                new_body_parts.append(body[prev:end])
                prev = end
                continue
            # 置換
            new_body_parts.append(body[prev:start])
            new_body_parts.append(f"[[{stem}]]")
            prev = end
            changes += 1

        new_body_parts.append(body[prev:])
        body = "".join(new_body_parts)

    return body, changes


# ---------------------------------------------------------------------------
# Date navigation for daily log files
# ---------------------------------------------------------------------------

def _find_date_neighbors(path: Path) -> tuple[Path | None, Path | None]:
    """同ディレクトリ内の前日・翌日 YYYY-MM-DD.md を返す。"""
    m = _DATE_LOG_RE.match(path.name)
    if not m:
        return None, None
    try:
        date = dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None, None

    parent = path.parent
    prev_date = date - dt.timedelta(days=1)
    next_date = date + dt.timedelta(days=1)
    prev_path = parent / f"{prev_date.isoformat()}.md"
    next_path = parent / f"{next_date.isoformat()}.md"
    return (
        prev_path if prev_path.exists() else None,
        next_path if next_path.exists() else None,
    )


_NAV_HEADER_RE = re.compile(r"^---\s*ナビ.*?---\s*\n", re.DOTALL | re.MULTILINE)


def add_date_nav(path: Path, vault: Path, body: str) -> tuple[str, bool]:
    """日付ログファイルに前日/翌日ナビゲーションを追加する。

    既存のナビブロックがあれば更新、なければ先頭に挿入。
    Returns:
        (new_body, changed)
    """
    prev_path, next_path = _find_date_neighbors(path)
    if prev_path is None and next_path is None:
        return body, False

    def rel_stem(p: Path) -> str:
        return str(p.relative_to(vault))[:-3]  # .md を除去

    nav_parts: list[str] = []
    if prev_path:
        nav_parts.append(f"← [[{rel_stem(prev_path)}|前日]]")
    if next_path:
        nav_parts.append(f"→ [[{rel_stem(next_path)}|翌日]]")
    nav_line = "  ".join(nav_parts)
    nav_block = f"---ナビ\n{nav_line}\n---\n\n"

    existing = _NAV_HEADER_RE.match(body)
    if existing:
        if body[existing.start():existing.end()] == nav_block:
            return body, False
        new_body = nav_block + body[existing.end():]
    else:
        new_body = nav_block + body
    return new_body, True


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    vault: Path,
    title_index: dict[str, str],
    *,
    dry_run: bool = False,
    graph_effect_targets: dict[str, list[str]] | None = None,
) -> dict:
    """1ファイルを処理して変更内容を返す。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"path": str(path), "error": str(e)}

    frontmatter, body = split_frontmatter(raw)
    own_stem = path.stem
    rel_path = path.relative_to(vault).as_posix()

    link_changes = 0
    nav_changed = False
    graph_effect_links_added = 0

    # wikiリンク付与
    body, link_changes = linkify_body(body, title_index, own_stem)

    # 日付ログの時系列ナビ（Cases/, AI Chat/, Asset Finance/ 直下のみ）
    if _DATE_LOG_RE.match(path.name):
        body, nav_changed = add_date_nav(path, vault, body)

    # graph judgment effect レポート連携（isolated_but_used / bridge_candidate へのハブリンク）
    if graph_effect_targets:
        body, graph_effect_links_added = add_graph_effect_links(body, rel_path, graph_effect_targets)

    total_changes = link_changes + (1 if nav_changed else 0) + graph_effect_links_added
    if total_changes == 0:
        return {"path": rel_path, "changes": 0}

    new_content = frontmatter + body
    if not dry_run:
        path.write_text(new_content, encoding="utf-8")

    return {
        "path": rel_path,
        "link_changes": link_changes,
        "nav_changed": nav_changed,
        "graph_effect_links_added": graph_effect_links_added,
        "changes": total_changes,
    }


def process_vault(
    vault: Path,
    *,
    dry_run: bool = False,
    target_paths: list[Path] | None = None,
    graph_effect_targets: dict[str, list[str]] | None = None,
) -> list[dict]:
    """対象フォルダを走査してwikiリンクを付与する。"""
    title_index = build_title_index(vault)
    print(f"[auto_wikilink] title index: {len(title_index)} notes", file=sys.stderr)

    if target_paths is not None:
        paths = target_paths
    else:
        paths = list(iter_vault_md_files(vault, TARGET_FOLDERS, EXCLUDED_PARTS))

    results = []
    for p in sorted(paths):
        r = process_file(p, vault, title_index, dry_run=dry_run, graph_effect_targets=graph_effect_targets)
        if r.get("changes", 0) > 0 or r.get("error"):
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Public helper for use from other scripts
# ---------------------------------------------------------------------------

def run_on_files(
    file_paths: list[Path],
    vault: Path | None = None,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """外部スクリプトから特定ファイルに対して呼び出すエントリポイント。"""
    if vault is None:
        try:
            vault = find_vault()
        except FileNotFoundError:
            return []
    title_index = build_title_index(vault)
    results = []
    for p in file_paths:
        if not p.is_file():
            continue
        r = process_file(p, vault, title_index, dry_run=dry_run)
        if r.get("changes", 0) > 0 or r.get("error"):
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Obsidian wikiリンク自動付与スクリプト")
    p.add_argument("--vault", default=None, help="iCloud 上の Obsidian Vault パス（省略時は自動検出）")
    p.add_argument("--dry-run", action="store_true", help="ファイルを変更せずに確認のみ")
    p.add_argument("--file", dest="files", metavar="PATH", nargs="+",
                   help="処理対象ファイルを直接指定（省略時は対象フォルダ全体）")
    p.add_argument("--verbose", "-v", action="store_true", help="変更なしのファイルも表示")
    p.add_argument("--graph-effect-report", type=Path, default=DEFAULT_GRAPH_EFFECT_JSON,
                   help="scripts/build_obsidian_graph_judgment_effect.py の出力JSON")
    p.add_argument("--no-graph-effect-links", action="store_true",
                   help="graph effect レポートに基づくハブリンク自動追加を無効化")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        vault = find_vault(args.vault)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"[auto_wikilink] vault={vault}", file=sys.stderr)

    target_paths: list[Path] | None = None
    if args.files:
        target_paths = [Path(f).expanduser().resolve() for f in args.files]

    graph_effect_targets: dict[str, list[str]] = {}
    if not args.no_graph_effect_links:
        graph_effect_targets = load_graph_effect_link_targets(args.graph_effect_report)
        if graph_effect_targets:
            print(
                f"[auto_wikilink] graph effect report: {len(graph_effect_targets)} note(s) targeted "
                f"({args.graph_effect_report})",
                file=sys.stderr,
            )

    results = process_vault(
        vault,
        dry_run=args.dry_run,
        target_paths=target_paths,
        graph_effect_targets=graph_effect_targets,
    )

    changed = [r for r in results if r.get("changes", 0) > 0]
    errors = [r for r in results if r.get("error")]
    graph_effect_total = sum(r.get("graph_effect_links_added", 0) for r in results)

    for r in results:
        if r.get("error"):
            print(f"  ERROR {r['path']}: {r['error']}")
        elif r.get("changes", 0) > 0 or args.verbose:
            link_n = r.get("link_changes", 0)
            nav = " +nav" if r.get("nav_changed") else ""
            hub_n = r.get("graph_effect_links_added", 0)
            hub = f" +{hub_n} hub-link" if hub_n else ""
            mode = "[dry-run]" if args.dry_run else "[updated]"
            print(f"  {mode} {r['path']} (+{link_n} links{nav}{hub})")

    print(f"\ntotal changed: {len(changed)}, errors: {len(errors)}, graph-effect hub-links: {graph_effect_total}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
