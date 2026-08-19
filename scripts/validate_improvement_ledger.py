#!/usr/bin/env python3
"""改善台帳（scripts/improvement_ledger.jsonl）の健全性検査。

なぜ必要か:
`.github/workflows/ledger-sync.yml` は PAT を廃止し GITHUB_TOKEN で台帳PRを作る。
GitHub の再帰防止により GITHUB_TOKEN が作った PR では `pull_request` イベントが
発火せず `pr-checks.yml` が起動しない。台帳PRだけが無検査で流れることになるため、
PR を作る前にこのスクリプトでその場で検査する。

検査内容:
  1. 全行が JSON として読めること（追記が壊れていないこと）
  2. 各行が dict で、status が既知の値であること
  3. 追記のみであること（既存行が削除・改変されていないこと）
     — 台帳は「追記形式・最後のエントリが有効」が前提（CLAUDE.md / scripts/README_ledger.md）

使い方:
    python3 scripts/validate_improvement_ledger.py
    python3 scripts/validate_improvement_ledger.py --path <ledger> --base-ref HEAD
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "scripts" / "improvement_ledger.jsonl"

# 既存台帳に実在する値（applied/done/needs_review/pending/doing/rejected）に、
# cleanup_improvement_reviews.py と pipeline_ledger.py が書きうる値を加えた集合。
# 未知の値を弾くのが目的なので、実データを調べずに推測で狭めないこと
# （初版で done / doing を落として既存295行が全滅しかけた）。
KNOWN_STATUSES = {
    "applied",
    "rejected",
    "deferred",
    "needs_review",
    "parked",
    "deleted",
    "pending",
    "dismissed",
    "done",
    "doing",
}


def parse_lines(text: str) -> tuple[list[dict], list[str]]:
    """JSONL をパースし、(エントリ一覧, 問題一覧) を返す。"""
    entries: list[dict] = []
    problems: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"L{lineno}: JSONとして読めない ({exc.msg})")
            continue
        if not isinstance(value, dict):
            problems.append(f"L{lineno}: dict ではない ({type(value).__name__})")
            continue
        status = str(value.get("status") or "")
        if status and status not in KNOWN_STATUSES:
            problems.append(f"L{lineno}: 未知の status={status!r}")
        entries.append(value)
    return entries, problems


def check_append_only(current: str, baseline: str) -> list[str]:
    """既存行が削除・改変されていないこと（追記のみ）を検査する。"""
    base_lines = baseline.splitlines()
    cur_lines = current.splitlines()
    if len(cur_lines) < len(base_lines):
        return [
            f"行数が減っている（{len(base_lines)} → {len(cur_lines)}）。"
            "台帳は追記形式のため既存行を消してはいけない"
        ]
    problems: list[str] = []
    for index, (base, cur) in enumerate(zip(base_lines, cur_lines), start=1):
        if base != cur:
            problems.append(f"L{index}: 既存行が書き換えられている（追記のみのはず）")
            if len(problems) >= 5:
                problems.append("... 以降は省略")
                break
    return problems


def _baseline_from_git(path: Path, base_ref: str) -> str | None:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{rel.as_posix()}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout


def validate(path: Path, base_ref: str | None = "HEAD") -> list[str]:
    if not path.exists():
        return [f"台帳が見つからない: {path}"]
    current = path.read_text(encoding="utf-8")
    entries, problems = parse_lines(current)
    if not entries and not problems:
        problems.append("エントリが1件も無い（生成に失敗している可能性）")
    if base_ref:
        baseline = _baseline_from_git(path, base_ref)
        if baseline is not None:
            problems.extend(check_append_only(current, baseline))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="追記のみ検査の比較対象。空文字を渡すとこの検査を省略する",
    )
    args = parser.parse_args()

    problems = validate(args.path, args.base_ref or None)
    if problems:
        print(f"NG: 改善台帳の検査で {len(problems)} 件の問題を検出しました", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    entries, _ = parse_lines(args.path.read_text(encoding="utf-8"))
    print(f"OK: 改善台帳は健全です（{len(entries)} エントリ）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
