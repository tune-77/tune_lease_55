"""Obsidian [[改善]] インデックスから改善案を抽出し、パイプライン用テキストに変換するスクリプト."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Vault パス候補（環境変数 OBSIDIAN_VAULT_PATH > iCloud パス）
_DEFAULT_VAULT_PATHS = [
    Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
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

        # ✅ 実装済タグがある行はスキップ
        _DONE_MARKERS = ("✅ 実装済", "✅実装済", "✅ 完了", "✅完了", "✅ done", "✅done")
        if any(m in stripped for m in _DONE_MARKERS):
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


_AI_CHAT_LOG_SUBPATH = "Projects/tune_lease_55/AI Chat/Improvement Log"

# AIチャット改善ログのフォーマット（[high]/[medium]/[low] + (accept)/(reject) 形式）
_AI_CHAT_LOG_ITEM_RE = re.compile(
    r"^- \*\*(.+?)\*\*\s*\[(?:high|medium|low)\].*$", re.MULTILINE
)


def extract_improvements_from_ai_chat_logs(vault: Path) -> str:
    """
    AI Chat/Improvement Log/ 配下の全 .md を直接スキャンして改善案を抽出する.

    BFS 展開とは独立して実行し、[改善] タグ付きテキストを返す。
    """
    log_dir = vault / _AI_CHAT_LOG_SUBPATH
    if not log_dir.exists():
        print(f"警告: AI Chat ログディレクトリが存在しません: {log_dir}", file=sys.stderr)
        return ""

    md_files = sorted(log_dir.glob("*.md"))
    if not md_files:
        return ""

    output_parts: list[str] = [f"# AIチャット改善ログ ({_AI_CHAT_LOG_SUBPATH})\n"]

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"警告: {md_file} 読み込み失敗: {e}", file=sys.stderr)
            continue

        file_parts: list[str] = []
        for m in _AI_CHAT_LOG_ITEM_RE.finditer(content):
            item_title = m.group(1).strip()
            if len(item_title) > 5:
                file_parts.append(f"[改善] {item_title}")
                file_parts.append(f"理由：AI Chat 改善ログ ({md_file.stem}) に記録された改善案")
                file_parts.append("")

        if file_parts:
            output_parts.append(f"\n## {md_file.stem}\n")
            output_parts.extend(file_parts)

    return "\n".join(output_parts)


def _normalize_title(title: str) -> str:
    """タイトルを比較用に正規化する（前後空白削除・連続空白を1つに）."""
    return re.sub(r'\s+', ' ', title.strip())


def _parse_improvements(text: str) -> list[dict]:
    """パイプラインテキストから改善案リストを生成する（[改善]/[TODO]タグ行を抽出）."""
    items: list[dict] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("[改善]") or line.startswith("[TODO]"):
            tag = "改善" if line.startswith("[改善]") else "TODO"
            title = _normalize_title(line[len(f"[{tag}]"):])
            reason = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("理由："):
                reason = lines[i + 1][3:].strip()
                i += 1
            if title:
                items.append({"tag": tag, "title": title, "reason": reason})
        i += 1
    return items


def _jaccard_similarity(a: str, b: str) -> float:
    """2文字列のJaccard類似度（2-gram）を返す。0.0〜1.0。"""
    def bigrams(s: str) -> set[str]:
        s = re.sub(r'\s+', '', s)
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()
    sa, sb = bigrams(a), bigrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# テーマグループ：同グループ内は最初の1件のみ残す
_THEME_GROUPS: list[tuple[str, ...]] = [
    ("曖昧な質問", "曖昧な発言", "曖昧な比較", "曖昧な要求", "曖昧な予算", "漠然とした"),
    ("Q_risk", "Q_riskの説明", "Q_risk説明", "量子干渉リスク"),
    ("補助金情報", "補助金案件", "補助金情報提供"),
    ("業種別成約率", "業種別成約率データ", "業種別成約率の傾向"),
    ("知識宇宙", "知識宇宙マップ", "知識宇宙のファイル名"),
    ("リース対象外", "リース対象外資産", "リース対象外物件"),
    ("リース対象物件", "リース対象物件の判断", "リース対象物件の明確化"),
    ("リース見積もり", "リース初期情報", "リース可否判断"),
    ("情報不足", "案件開始時の情報不足", "リース可否判断に必要な情報"),
    ("条件付き承認", "条件付承認", "条件付き承認の推奨"),
    ("八奈見杏奈", "八奈見杏奈の応答", "八奈見杏奈の発言"),
    ("機能範囲外", "機能範囲外の質問", "範囲外コマンド"),
]

_JACCARD_THRESHOLD = 0.55  # この値以上なら重複とみなす


def deduplicate_improvements(improvements: list[dict]) -> list[dict]:
    """改善案リストから重複を排除し duplicate_count を付与する.

    重複判定基準（優先順）:
    1. タイトル完全一致
    2. タイトル先頭40文字一致
    3. 一方が他方に含まれる（8文字以上）
    4. テーマグループ一致（同グループは先頭1件のみ残す）
    5. Jaccard 2-gram 類似度 >= 0.55
    """
    _SUBSET_MIN_LEN = 8

    def _in_same_theme(a: str, b: str) -> bool:
        for group in _THEME_GROUPS:
            a_match = any(kw in a for kw in group)
            b_match = any(kw in b for kw in group)
            if a_match and b_match:
                return True
        return False

    result: list[dict] = []

    for imp in improvements:
        title = imp["title"]
        title_prefix = title[:40]

        matched_idx = -1
        for i, kept in enumerate(result):
            kept_title = kept["title"]
            kept_prefix = kept_title[:40]

            # 1. 完全一致 / 先頭40文字一致
            if title == kept_title or title_prefix == kept_prefix:
                matched_idx = i
                break

            # 2. サブセット判定
            if len(title) >= _SUBSET_MIN_LEN and len(kept_title) >= _SUBSET_MIN_LEN:
                if title in kept_title or kept_title in title:
                    matched_idx = i
                    break

            # 3. テーマグループ一致
            if _in_same_theme(title, kept_title):
                matched_idx = i
                break

            # 4. Jaccard 類似度
            if _jaccard_similarity(title, kept_title) >= _JACCARD_THRESHOLD:
                matched_idx = i
                break

        if matched_idx >= 0:
            kept = result[matched_idx]
            kept["duplicate_count"] += 1
            # 短い方のタイトルを保持（サブセット排除）
            if len(title) < len(kept["title"]) and title in kept["title"]:
                kept["title"] = title
                kept["reason"] = imp["reason"]
        else:
            result.append({**imp, "duplicate_count": 1})

    return result


def _format_deduplicated(improvements: list[dict]) -> str:
    """deduplicate_improvements の結果をパイプライン用テキストに変換する."""
    lines: list[str] = []
    for imp in improvements:
        tag = imp["tag"]
        count = imp.get("duplicate_count", 1)
        count_note = f"（重複元: {count}件）" if count > 1 else ""
        lines.append(f"[{tag}] {imp['title']}")
        lines.append(f"理由：{imp['reason']}{count_note}")
        lines.append("")
    return "\n".join(lines)


def _load_consolidator() -> object | None:
    """improvement_consolidator モジュールを動的ロードして consolidate_with_ai を返す."""
    import importlib.util
    mod_path = Path(__file__).parent / "improvement_consolidator.py"
    if not mod_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("improvement_consolidator", mod_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "consolidate_with_ai", None)
    except Exception as e:
        print(f"警告: improvement_consolidator のロード失敗: {e}", file=sys.stderr)
        return None


def main() -> None:
    vault = _get_vault_path()
    print(f"Obsidian Vault: {vault}")

    index_file = find_index_file(vault)
    if not index_file:
        print(f"エラー: 改善インデックスファイルが見つかりません（Vault: {vault}）", file=sys.stderr)
        sys.exit(1)

    print(f"改善インデックス: {index_file}")

    pipeline_text = extract_improvements_from_index(index_file, vault)

    # AI Chat Improvement Log を独立スキャンして結合
    ai_chat_text = extract_improvements_from_ai_chat_logs(vault)
    if ai_chat_text.strip():
        pipeline_text = pipeline_text + "\n\n" + ai_chat_text

    # 重複排除
    raw_improvements = _parse_improvements(pipeline_text)
    before_count = len(raw_improvements)
    deduped = deduplicate_improvements(raw_improvements)
    after_count = len(deduped)

    # 実装済み除外リスト（インデックスファイルの「実装済み改善一覧」セクションから読み込む）
    implemented_titles: set[str] = set()
    try:
        idx_content = index_file.read_text(encoding="utf-8")
        in_impl_section = False
        for line in idx_content.splitlines():
            stripped = line.strip()
            if "実装済み改善一覧" in stripped and stripped.startswith("#"):
                in_impl_section = True
                continue
            if in_impl_section and stripped.startswith("#"):
                break  # 次セクションで終了
            if in_impl_section and stripped.startswith("✅ 実装済"):
                title = _normalize_title(stripped.replace("✅ 実装済", "").lstrip())
                if title:
                    implemented_titles.add(title)
    except Exception:
        pass

    if implemented_titles:
        before_impl = len(deduped)
        deduped = [
            imp for imp in deduped
            if not any(
                t in _normalize_title(imp["title"]) or _normalize_title(imp["title"]) in t
                for t in implemented_titles
            )
        ]
        skipped = before_impl - len(deduped)
        if skipped:
            print(f"実装済み除外: {skipped}件スキップ（残り{len(deduped)}件）")
        after_count = len(deduped)

    # AI統合（Gemini APIが使えない場合は deduped をそのまま使用）
    consolidate_with_ai = _load_consolidator()
    if consolidate_with_ai is not None:
        final = consolidate_with_ai(deduped)
    else:
        final = deduped
    final_count = len(final)

    final_text = _format_deduplicated(final)
    OUTPUT_FILE.write_text(final_text, encoding="utf-8")

    ai_note = f" → AI統合後: {final_count}件" if final_count != after_count else ""
    print(
        f"抽出完了: {OUTPUT_FILE} — {final_count}件の改善案"
        f"（重複排除前: {before_count}件、排除後: {after_count}件{ai_note}）"
    )


if __name__ == "__main__":
    main()
