#!/usr/bin/env python3
"""Gist へ公開する前に latest.json の機微情報を検査する。

改善候補には Cloud Run チャット由来の自由文が混ざるため、顧客名・連絡先・
案件情報がそのまま公開 Gist へ出るリスクがある。疑わしいパターンを検出したら
終了コード 1 を返し、呼び出し側（run_daily_improvement_core.sh）は
Gist アップロードをスキップする（パイプライン自体は失敗させない）。

検出は保守的（偽陽性を許容）: 公開が1日止まるコストより漏えいのコストが大きい。

使い方:
  python scripts/check_gist_payload_safety.py --file reports/latest.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SUSPECT_PATTERNS: list[tuple[str, str]] = [
    ("メールアドレス", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("電話番号", r"0\d{1,4}-\d{1,4}-\d{3,4}"),
    ("会社名（株式会社）", r"株式会社[^\s、。」)]{1,20}"),
    ("会社名（有限会社）", r"有限会社[^\s、。」)]{1,20}"),
    ("会社名（合同会社）", r"合同会社[^\s、。」)]{1,20}"),
    ("代表者情報", r"代表者[:：]\s*\S+"),
    ("住所らしき記載", r"(東京都|大阪府|京都府|北海道|[一-龠]{2,3}県)[一-龠]{1,6}[市区町村]"),
]

# デモ用に犬種化した会社名（dogify_demo_company_names.py）は許可する
ALLOWLIST_PATTERNS = [
    r"株式会社(柴犬|プードル|ブルドッグ|チワワ|ハスキー|コーギー|ダックス|レトリバー)",
]


def scan(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for label, pattern in SUSPECT_PATTERNS:
        for match in re.finditer(pattern, text):
            value = match.group(0)
            if any(re.fullmatch(allow, value) for allow in ALLOWLIST_PATTERNS):
                continue
            # 値はマスクして出力する（このログ自体から漏らさない）
            masked = value[:4] + "…" if len(value) > 4 else value
            findings.append((label, masked))
            break  # 種別ごとに1件で十分
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()

    if not args.file.exists():
        print(f"[gist_safety] 対象ファイルなし: {args.file}（チェックをスキップ）")
        return 0
    try:
        text = args.file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[gist_safety] 読み取り失敗のため安全側でブロック: {exc}")
        return 1

    findings = scan(text)
    if findings:
        print(f"[gist_safety] ⚠️ 機微情報の疑い {len(findings)} 種別を検出。Gist公開をスキップすべきです:")
        for label, masked in findings:
            print(f"  - {label}: {masked}")
        return 1
    print("[gist_safety] 機微情報の疑いなし（公開可）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
