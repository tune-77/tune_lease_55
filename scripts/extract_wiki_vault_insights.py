"""
lease-wiki-vault の @AI_Insight_Evolved_*.md から「## 3. 結論」セクションを読み込み、
改善候補を [改善] タグ付きテキストとして stdout に出力する。

前回処理済みの日付を /tmp/wiki_vault_last_processed.txt で管理し、
新規ファイル分のみ処理することで毎日同じ内容を繰り返さない。
run_daily_improvement_pipeline.sh から >> EXPORT_FILE でキャプチャされる。
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime
from pathlib import Path

_WIKI_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "lease-wiki-vault"
)
_LAST_PROCESSED_FILE = Path("/tmp/wiki_vault_last_processed.txt")

# 「行動を促す」パターン → タイトルへの変換ルール
# (正規表現, グループ番号または置換文字列)
_ACTION_PATTERNS: list[tuple[str, str]] = [
    (r"^(.+?)(?:へ進める|に進める)。?$", r"\1"),
    (r"^(.+?)を優先する。?$", r"\1"),
    (r"^(.+?)する必要がある。?$", r"\1"),
    (r"^(.+?)へ移す。?$", r"\1"),
    (r"^(.+?)を優先して.*$", r"\1"),
    (r"^(.+?)を検討(?:する|すべき)。?$", r"\1の検討"),
    (r"^(.+?)を実装(?:する|すべき)。?$", r"\1の実装"),
    (r"^(.+?)を追加(?:する|すべき)。?$", r"\1の追加"),
]


def _get_last_processed() -> date | None:
    if not _LAST_PROCESSED_FILE.exists():
        return None
    try:
        return datetime.strptime(_LAST_PROCESSED_FILE.read_text().strip(), "%Y-%m-%d").date()
    except (ValueError, OSError):
        return None


def _set_last_processed(d: date) -> None:
    try:
        _LAST_PROCESSED_FILE.write_text(d.isoformat())
    except OSError as e:
        print(f"警告: last_processed の更新失敗: {e}", file=sys.stderr)


def _extract_date_from_filename(name: str) -> date | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _convert_to_improvement(bullet: str, source: str) -> str | None:
    """箇条書き行を [改善] タグ付き文字列に変換する。該当しない場合は None。"""
    text = re.sub(r"^[-・\s]+", "", bullet).strip()
    if len(text) < 10:
        return None

    for pattern, repl in _ACTION_PATTERNS:
        if re.match(pattern, text):
            title = re.sub(pattern, repl, text).strip()
            # Markdown 装飾除去
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
            title = re.sub(r"`(.+?)`", r"\1", title)
            title = title[:80]
            if len(title) >= 8:
                return (
                    f"[改善] {title}\n"
                    f"理由：lease-wiki-vault/{source} の横断推論結論から自動抽出\n"
                )
    return None


def process_insight_file(md_file: Path) -> list[str]:
    """1ファイルの「## 3. 結論」から改善案リストを抽出する。"""
    try:
        content = md_file.read_text(encoding="utf-8")
    except OSError as e:
        print(f"警告: {md_file.name} 読み込み失敗: {e}", file=sys.stderr)
        return []

    if "## 3. 結論" not in content:
        return []

    start = content.find("## 3. 結論")
    end = content.find("\n## ", start + 1)
    section = content[start : end if end > 0 else start + 3000]

    results: list[str] = []
    for line in section.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("-") and not stripped.startswith("・"):
            continue
        improvement = _convert_to_improvement(stripped, md_file.name)
        if improvement:
            results.append(improvement)

    return results


def main() -> None:
    if not _WIKI_VAULT.exists():
        print(f"警告: lease-wiki-vault が見つかりません: {_WIKI_VAULT}", file=sys.stderr)
        return

    last_processed = _get_last_processed()
    today = date.today()

    insight_files = sorted(_WIKI_VAULT.glob("@AI_Insight_Evolved_*.md"))
    if not insight_files:
        print("警告: @AI_Insight_Evolved_*.md が見つかりません", file=sys.stderr)
        return

    output_parts: list[str] = []
    max_date: date | None = last_processed
    processed_count = 0

    for f in insight_files:
        file_date = _extract_date_from_filename(f.name)
        if file_date is None:
            continue
        # 前回処理済み以前のファイルはスキップ
        if last_processed is not None and file_date <= last_processed:
            continue

        improvements = process_insight_file(f)
        if improvements:
            output_parts.extend(improvements)
            processed_count += 1

        if max_date is None or file_date > max_date:
            max_date = file_date

    if not output_parts:
        print(
            f"extract_wiki_vault_insights: 新規改善案なし"
            f"（前回処理: {last_processed or '初回'}、Insight ファイル {len(insight_files)}件）",
            file=sys.stderr,
        )
    else:
        header = f"# lease-wiki-vault @AI_Insight_Evolved 自動抽出（{today}）\n\n"
        print(header + "\n".join(output_parts))
        print(
            f"extract_wiki_vault_insights: {len(output_parts)}件の改善案を出力"
            f"（{processed_count}ファイル処理）",
            file=sys.stderr,
        )

    if max_date and max_date != last_processed:
        _set_last_processed(max_date)


if __name__ == "__main__":
    main()
