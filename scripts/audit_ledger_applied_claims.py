#!/usr/bin/env python3
"""改善パイプライン台帳(ledger.jsonl)の "applied" 記録を監査するバックフィルスクリプト。

根本対応③: step3_auto_apply.py の以前の実装では、commit/push/PR作成が確認
される前に台帳へ "applied" を記録していた（現在は修正済みで、"applied" は
commit確認後にのみ記録される）。しかし過去に書き込まれた記録の中には、実際
にはコードが変更されていない「見せかけの applied」が残っている可能性がある。
台帳の "applied" は永続的に再評価をスキップする状態のため、これが残ったままだと
同じ不具合が二度と自動修正の対象にならない。

このスクリプトは:
1. 台帳の各キーについて最新ステータスを求める
2. status == "applied" かつ pr_url が空のものを抽出する
3. git log 全履歴からタイトル文字列に一致するコミットが見つかるか確認する
   （見つかれば実際に対応するコードがあるとみなし対象外にする）
4. 見つからなければ「未検証」と判定し、--apply 指定時のみ台帳へ訂正記録
   （status="apply_failed"、再評価可能）を追記する

デフォルトは dry-run。実際に台帳へ書き込むには --apply を指定する。
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_latest_by_key(ledger_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not ledger_path.exists():
        return latest
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = entry.get("key")
        if key:
            latest[key] = entry
    return latest


def _clean_title_for_grep(title: str) -> str:
    cleaned = re.sub(r"<!--.*?-->", "", title).strip()
    return cleaned


def has_matching_commit(title: str) -> bool:
    cleaned = _clean_title_for_grep(title)
    if len(cleaned) < 4:
        return False
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "-F", "--grep", cleaned],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return True  # 確認できない場合は安全側（訂正しない）に倒す
    return bool(result.stdout.strip())


def audit(ledger_path: Path, apply: bool) -> dict:
    latest = load_latest_by_key(ledger_path)
    applied = {k: e for k, e in latest.items() if e.get("status") == "applied"}
    no_pr = {k: e for k, e in applied.items() if not e.get("pr_url")}

    unverified: list[dict] = []
    corroborated: list[dict] = []
    for key, entry in no_pr.items():
        title = str(entry.get("title") or "")
        if has_matching_commit(title):
            corroborated.append({"key": key, "title": title})
        else:
            unverified.append(
                {"key": key, "title": title, "recorded_at": entry.get("recorded_at", "")}
            )

    if apply:
        with ledger_path.open("a", encoding="utf-8") as f:
            for item in unverified:
                correction = {
                    "key": item["key"],
                    "status": "apply_failed",
                    "title": item["title"],
                    "canonical_key": "",
                    "pr_url": "",
                    "reason": (
                        "バックフィル監査(audit_ledger_applied_claims.py): "
                        "PR URL不在かつgit履歴に対応するコミットが見つからないため、"
                        "実際のコード反映が未確認と判定。再評価可能な状態へ訂正。"
                    ),
                    "recorded_at": datetime.datetime.now().isoformat(),
                }
                f.write(json.dumps(correction, ensure_ascii=False) + "\n")

    return {
        "total_keys": len(latest),
        "applied_keys": len(applied),
        "applied_without_pr_url": len(no_pr),
        "corroborated_by_git_log": len(corroborated),
        "unverified": unverified,
        "corrected": apply,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に台帳へ訂正記録を追記する（指定なしは dry-run）",
    )
    parser.add_argument("--json", action="store_true", help="結果をJSONで出力する")
    args = parser.parse_args()

    result = audit(args.ledger, args.apply)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"台帳キー総数: {result['total_keys']}")
    print(f"最新ステータスが applied: {result['applied_keys']}")
    print(f"うち pr_url が空: {result['applied_without_pr_url']}")
    print(f"  - git履歴に対応コミットあり（対象外）: {result['corroborated_by_git_log']}")
    print(f"  - 未検証（訂正対象）: {len(result['unverified'])}")
    if not args.apply and result["unverified"]:
        print("\n※ dry-run のため台帳は変更していません。--apply で実際に訂正記録を書き込みます。")
    print()
    for item in result["unverified"][:50]:
        print(f"  [{item['key']}] {item['title'][:60]} (recorded_at={item['recorded_at']})")
    if len(result["unverified"]) > 50:
        print(f"  ... 他 {len(result['unverified']) - 50} 件")


if __name__ == "__main__":
    main()
