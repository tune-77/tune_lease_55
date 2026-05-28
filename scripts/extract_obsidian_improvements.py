"""Obsidian [[改善]] インデックスから改善案を抽出し、パイプライン用テキストに変換するスクリプト."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Vault パス候補（環境変数 > iCloud パス > Documents パス）
_DEFAULT_VAULT_PATHS = [
    Path("/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"),
    Path("/Users/kobayashiisaoryou/Documents/Obsidian Vault"),
]

OUTPUT_FILE = Path("/tmp/obsidian_improvements_export.txt")

# wikiリンク展開の最大深さ（1〜3を推奨; 深くするほど取得件数が増える）
MAX_WIKI_DEPTH: int = 2
# フォールバック抽出（タグ無し全行拾い）の有効フラグ
ENABLE_FALLBACK: bool = False
# BFS展開するノートの上限（循環・爆発防止）
_MAX_WIKI_NODES: int = 50

# 改善インデックスのファイル名パターン（優先順）
_INDEX_PATTERNS = [
    "tuneLease55/改善策インデックス_2026.md",
    "tuneLease55/改善*.md",
    "改善.md",
    "**/改善策インデックス*.md",
    "**/改善.md",
]


def _get_vault_path() -> Path:
    """有効な Vault パスを返す（環境変数 > デフォルト候補）."""
    env_vault = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env_vault:
        p = Path(env_vault)
        if p.exists():
            return p
        print(f"警告: OBSIDIAN_VAULT_PATH '{env_vault}' が存在しません。デフォルトを使用します。", file=sys.stderr)

    for candidate in _DEFAULT_VAULT_PATHS:
        if candidate.exists():
            return candidate

    print("エラー: Obsidian Vault が見つかりません。", file=sys.stderr)
    sys.exit(1)


def find_index_file(vault: Path) -> Path | None:
    """改善インデックスファイルを検索する."""
    for pattern in _INDEX_PATTERNS:
        if "*" in pattern or "**" in pattern:
            matches = sorted(vault.glob(pattern))
            if matches:
                return matches[0]
        else:
            candidate = vault / pattern
            if candidate.exists():
                return candidate
    return None


def find_note(vault: Path, link_name: str) -> Path | None:
    """[[link_name]] に対応するノートファイルを Vault 内で検索する."""
    # [[path|表示名]] 形式のパイプ以降は表示名なので除去
    link_name = link_name.split("|")[0].strip()

    # スラッシュを含む場合は直接パスとして試みる
    if "/" in link_name:
        candidate = vault / f"{link_name}.md"
        if candidate.exists():
            return candidate

    # Vault 全体をグロブ検索（ファイル名のみで照合）
    safe_name = re.escape(link_name)
    matches = list(vault.glob(f"**/{link_name}.md"))
    if matches:
        return matches[0]

    return None


def _extract_wiki_links(text: str) -> list[str]:
    """テキスト中の [[wiki リンク]] を全て抽出する."""
    return re.findall(r'\[\[([^\]]+)\]\]', text)


def _convert_to_pipeline_text(content: str, source_name: str) -> str:
    """
    Obsidian マークダウンをパイプライン用 [改善]/[TODO] タグ付きテキストに変換する.

    変換ルール:
    - 「未解決課題」セクションのリスト項目 → [改善]
    - 🔴 高優先度セクション内の箇条書き → [改善]
    - Phase1 の 🔲 未チェック項目 → [TODO]
    - 優先度マトリクスの行 → スキップ（冗長になるため）
    """
    lines = content.split("\n")
    output_lines: list[str] = []

    current_section = ""
    in_improvement_section = False
    is_high_priority = False
    in_priority_matrix = False  # 優先度マトリクステーブルはスキップ

    for line in lines:
        stripped = line.strip()

        # 優先度マトリクステーブルの検出（スキップ）
        if "優先度マトリクス" in stripped:
            in_priority_matrix = True
        if in_priority_matrix:
            if stripped.startswith("---"):
                in_priority_matrix = False
            continue

        # セクションヘッダー検出
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
            in_improvement_section = any(kw in current_section for kw in [
                "改善", "課題", "未解決", "ロードマップ", "Phase",
            ])
            is_high_priority = any(kw in current_section for kw in [
                "高", "Phase1", "Phase 1", "未解決",
            ])
            continue

        # 未解決課題セクションのリスト項目
        if "未解決" in current_section and stripped.startswith("-"):
            item = stripped.lstrip("- ").strip()
            if item and len(item) > 5:
                output_lines.append(f"[改善] {item}")
                output_lines.append(f"理由：{source_name} の未解決課題として記録済み")
                output_lines.append("")
            continue

        if in_improvement_section:
            # 優先度マーカー更新
            if "🔴" in stripped:
                is_high_priority = True
            elif "🟡" in stripped or "🟢" in stripped:
                is_high_priority = False

            # 高優先度セクション内の箇条書き → [改善]
            if is_high_priority and stripped.startswith("- ") and not stripped.startswith("- - "):
                item = stripped[2:].strip()
                # マークダウン装飾除去
                item = re.sub(r'\*\*(.+?)\*\*', r'\1', item)
                item = re.sub(r'`(.+?)`', r'\1', item)
                item = re.sub(r'\[\[.+?\]\]', '', item).strip()
                if item and len(item) > 10:
                    output_lines.append(f"[改善] {item}")
                    output_lines.append(f"理由：{current_section}（高優先度）の改善案 — {source_name}")
                    output_lines.append("")

            # Phase1 の未チェック項目 → [TODO]
            elif "Phase" in current_section and "🔲" in stripped:
                item = stripped.replace("🔲", "").strip()
                if item:
                    output_lines.append(f"[TODO] {item}")
                    output_lines.append(f"理由：Phase1 ロードマップの未実装タスク — {source_name}")
                    output_lines.append("")

    return "\n".join(output_lines)


def extract_improvements_from_index(index_file: Path, vault: Path) -> str:
    """
    インデックスファイルとリンク先ノート（BFS 深さ MAX_WIKI_DEPTH）から改善案を抽出する.

    Returns:
        パイプライン用テキスト（[改善]/[TODO] タグ付き）
    """
    from collections import deque

    output_parts: list[str] = []

    index_content = index_file.read_text(encoding="utf-8")
    output_parts.append(f"# 改善案抽出元: {index_file.name}\n")

    # インデックス本体から抽出
    index_improvements = _convert_to_pipeline_text(index_content, index_file.stem)
    if index_improvements.strip():
        output_parts.append(index_improvements)

    # BFS で [[wiki リンク]] を深さ MAX_WIKI_DEPTH まで展開
    visited: set[str] = {index_file.stem, index_file.name.replace(".md", "")}
    # キュー要素: (link_name, depth, content_to_scan)
    queue: deque[tuple[str, int, str]] = deque()
    for link in _extract_wiki_links(index_content):
        queue.append((link, 1, index_content))

    while queue and len(visited) < _MAX_WIKI_NODES:
        link, depth, _ = queue.popleft()
        link_name = link.split("|")[0].strip()

        if link_name in visited or depth > MAX_WIKI_DEPTH:
            continue
        visited.add(link_name)

        linked_file = find_note(vault, link_name)
        if not linked_file:
            continue

        try:
            linked_content = linked_file.read_text(encoding="utf-8")
            linked_improvements = _convert_to_pipeline_text(linked_content, linked_file.stem)
            if linked_improvements.strip():
                output_parts.append(f"\n# リンク先ノート (深さ{depth}): {linked_file.name}\n")
                output_parts.append(linked_improvements)

            # さらに深い階層のリンクをキューへ追加
            if depth < MAX_WIKI_DEPTH:
                for child_link in _extract_wiki_links(linked_content):
                    child_name = child_link.split("|")[0].strip()
                    if child_name not in visited:
                        queue.append((child_link, depth + 1, linked_content))
        except Exception as e:
            print(f"警告: {linked_file} 読み込み失敗: {e}", file=sys.stderr)

    return "\n".join(output_parts)


def main() -> None:
    vault = _get_vault_path()
    print(f"Obsidian Vault: {vault}")

    index_file = find_index_file(vault)
    if not index_file:
        print(f"エラー: 改善インデックスファイルが見つかりません（Vault: {vault}）", file=sys.stderr)
        sys.exit(1)

    print(f"改善インデックス: {index_file}")

    pipeline_text = extract_improvements_from_index(index_file, vault)

    OUTPUT_FILE.write_text(pipeline_text, encoding="utf-8")

    tag_count = pipeline_text.count("[改善]") + pipeline_text.count("[TODO]")
    print(f"抽出完了: {OUTPUT_FILE} — {tag_count}件の改善案タグ")


if __name__ == "__main__":
    main()
