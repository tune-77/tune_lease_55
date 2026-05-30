"""git コミット履歴から実装済みREVを検出し Obsidian 改善インデックスに自動追記する."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VAULT_PATHS = [
    Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
]
_INDEX_REL = "tuneLease55/改善策インデックス_2026.md"
_IMPL_SECTION = "## 実装済み改善一覧（パイプライン除外リスト）"
_EXPORT_FILE = Path("/tmp/obsidian_improvements_export.txt")

# コミットメッセージのキーワード → 改善タイトルの部分一致マッピング
_COMMIT_TO_TITLE_HINTS: dict[str, list[str]] = {
    "フォームUX":          ["入力欄の説明文", "入力導線"],
    "REV-018":             ["入力欄の説明文の明確化"],
    "REV-023":             ["入力導線の明確化"],
    "REV-019":             ["条件付き承認時の推奨アクション"],
    "REV-022":             ["補助金情報提供"],
    "REV-025":             ["業種平均との比較表示"],
    "REV-026":             ["条件付承認時の主要リスク"],
    "REV-035":             ["業種別成約率の傾向分析"],
    "REV-040":             ["新規事業計画の審査ガイダンス"],
    "REV-041":             ["PD表示箇所の明確化"],
    "REV-048":             ["金利決定ロジックの説明強化"],
    "REV-050":             ["リース期間バリデーション警告"],
    "REV-064":             ["物件ごとの法定耐用年数"],
    "REV-068":             ["推奨業種バナー"],
    "REV-085":             ["契約種類に割賦"],
    "REV-089":             ["Q_risk", "量子干渉リスク"],
    "REV-109":             ["ビジュアルインサイト AI コメント"],
    "rate-engine":         ["動的金利提案エンジン MVP"],
    "RAGナレッジ":          ["RAGナレッジQ&A基盤"],
    "ドリフト":             ["データドリフト監視ダッシュボード"],
    "counterfactual":      ["Counterfactual Explanation"],
    "マルチエージェント":   ["軍師AIマルチエージェント化"],
    "AUC.*リーケージ":     ["AUC=1.00 データリーケージ疑義"],
    "デフォルト率.*警告":  ["デフォルト率モデル", "PD警告"],
}


def _get_vault() -> Path | None:
    for p in _VAULT_PATHS:
        if p.exists():
            return p
    return None


def _get_recent_commits(days: int = 3) -> list[str]:
    """直近 N 日のコミットメッセージ一覧を返す。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "log",
             f"--since={days} days ago", "--oneline", "--no-merges"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip().splitlines()
    except Exception:
        return []


def _load_export_titles() -> list[str]:
    """抽出済み改善タイトル一覧を読み込む。"""
    if not _EXPORT_FILE.exists():
        return []
    titles = []
    for line in _EXPORT_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("[改善]") or line.startswith("[TODO]"):
            tag = "[改善]" if line.startswith("[改善]") else "[TODO]"
            title = line[len(tag):].strip()
            if title:
                titles.append(title)
    return titles


def _load_existing_impl(index_file: Path) -> set[str]:
    """既存の「実装済み改善一覧」セクションからタイトルセットを返す。"""
    implemented: set[str] = set()
    content = index_file.read_text(encoding="utf-8")
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if _IMPL_SECTION in stripped:
            in_section = True
            continue
        if in_section and stripped.startswith("#"):
            break
        if in_section and stripped.startswith("✅ 実装済"):
            title = stripped.replace("✅ 実装済", "").lstrip()
            if title:
                implemented.add(title.strip())
    return implemented


def _append_to_index(index_file: Path, new_entries: list[str]) -> None:
    """「実装済み改善一覧」セクションの末尾に新エントリを追記する。"""
    today = date.today().strftime("%Y-%m-%d")
    content = index_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    insert_at = None
    in_section = False
    for i, line in enumerate(lines):
        if _IMPL_SECTION in line:
            in_section = True
            continue
        if in_section and line.strip().startswith("#"):
            insert_at = i  # 次セクション直前
            break

    addition = [f"✅ 実装済 {e}  <!-- auto {today} -->" for e in new_entries]

    if insert_at is not None:
        lines[insert_at:insert_at] = addition + [""]
    else:
        lines += addition

    index_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    vault = _get_vault()
    if vault is None:
        print("警告: Obsidian Vault が見つかりません。スキップします。", file=sys.stderr)
        return

    index_file = vault / _INDEX_REL
    if not index_file.exists():
        print(f"警告: インデックスファイルが見つかりません: {index_file}", file=sys.stderr)
        return

    commits = _get_recent_commits(days=3)
    if not commits:
        print("情報: 直近コミットなし。スキップします。")
        return

    export_titles = _load_export_titles()
    existing_impl = _load_existing_impl(index_file)

    new_entries: list[str] = []
    for commit_line in commits:
        commit_msg = commit_line[8:] if len(commit_line) > 8 else commit_line  # ハッシュを除去

        for keyword, hints in _COMMIT_TO_TITLE_HINTS.items():
            if not re.search(keyword, commit_msg, re.IGNORECASE):
                continue
            for hint in hints:
                # export_titles から部分一致するタイトルを探す
                matched = next(
                    (t for t in export_titles if hint in t or t in hint),
                    hint  # export にない場合はhintをそのまま使う
                )
                if matched not in existing_impl and matched not in new_entries:
                    new_entries.append(matched)

    if new_entries:
        _append_to_index(index_file, new_entries)
        print(f"✅ 実装済み {len(new_entries)} 件を Obsidian に追記しました:")
        for e in new_entries:
            print(f"   - {e}")
    else:
        print("情報: 新規実装済み項目なし。")


if __name__ == "__main__":
    main()
