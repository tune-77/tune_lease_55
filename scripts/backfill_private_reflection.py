#!/usr/bin/env python3
"""Jun 15-21 の Private Reflection バックフィルスクリプト。

対話ログが 04:00 AM のパイプライン実行時に未生成だったため
スキップされた日分（および 06:00 の上書きで失われた日分）を対象に、
現在存在する Dialogue ファイルを読んで内省を生成・追記する。

実行:
    cd /Users/kobayashiisaoryou/clawd/tune_lease_55
    .venv/bin/python scripts/backfill_private_reflection.py [--dry-run] [--dates 2026-06-15,2026-06-16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# バックフィル対象日（Jun 19 は 16:24 に手動生成済みのためスキップ）
DEFAULT_DATES = [
    "2026-06-15",
    "2026-06-16",
    "2026-06-17",
    "2026-06-18",
    # "2026-06-19" はコンテンツあり（5856 bytes）スキップ
    "2026-06-20",
    "2026-06-21",
]


def _reflection_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection"


def _dialogue_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Dialogue"


def _needs_backfill(vault: Path, date_str: str) -> tuple[bool, str]:
    """バックフィルが必要かチェック。(需要フラグ, 理由) を返す。"""
    reflection_path = _reflection_dir(vault) / f"{date_str}.md"
    dialogue_path = _dialogue_dir(vault) / f"{date_str}.md"

    if not dialogue_path.exists():
        return False, f"対話ログなし ({dialogue_path.name})"

    dialogue_size = dialogue_path.stat().st_size
    if dialogue_size < 100:
        return False, f"対話ログが空に近い ({dialogue_size} bytes)"

    if reflection_path.exists():
        content = reflection_path.read_text(encoding="utf-8")
        if "## 今日の対話について" in content:
            return False, "既に内省あり (スキップ)"

    return True, f"バックフィル対象 (対話ログ {dialogue_size:,} bytes)"


def backfill_date(vault: Path, date_str: str, dry_run: bool = False) -> str:
    from lease_intelligence_reflection import generate_and_append_reflection

    needs, reason = _needs_backfill(vault, date_str)
    if not needs:
        return f"[SKIP] {date_str}: {reason}"

    if dry_run:
        return f"[DRY-RUN] {date_str}: {reason} → generate_and_append_reflection を実行予定"

    try:
        result = generate_and_append_reflection(vault, date_str)
        return f"[OK] {result}"
    except Exception as exc:
        return f"[ERROR] {date_str}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Reflection バックフィル")
    parser.add_argument("--dry-run", action="store_true", help="実行せずに対象確認のみ")
    parser.add_argument(
        "--dates",
        type=lambda s: [d.strip() for d in s.split(",")],
        default=DEFAULT_DATES,
        help="バックフィル対象日（カンマ区切り）。デフォルト: Jun 15-21",
    )
    args = parser.parse_args()

    try:
        from lease_intelligence_reflection import _find_vault
        vault = _find_vault()
    except Exception:
        vault = None

    if not vault:
        print("[ERROR] Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH 環境変数を確認してください。")
        sys.exit(1)

    print(f"Vault: {vault}")
    print(f"対象日: {', '.join(args.dates)}")
    if args.dry_run:
        print("** DRY-RUN モード（実際の書き込みは行いません）**")
    print()

    for date_str in args.dates:
        result = backfill_date(vault, date_str, dry_run=args.dry_run)
        print(result)


if __name__ == "__main__":
    main()
