#!/usr/bin/env python3
"""
api/rule_engine/batch_apply.py

台帳ルール（ledger_rules.json）を一括適用するバッチスクリプト。

使い方:
  python api/rule_engine/batch_apply.py --dry-run           # 実行せずに結果を表示
  python api/rule_engine/batch_apply.py --apply             # 実際に適用
  python api/rule_engine/batch_apply.py --apply --rev REV-007-a  # 特定REVのみ適用
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# プロジェクトルートを sys.path に追加（スクリプト直接実行時のため）
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LEDGER_PATH = Path(__file__).parent / "ledger_rules.json"

# batch_apply 側で弾く保護対象ターゲット（rule_engine 側でも弾いているが二重防衛）
_PROTECTED_PATTERNS = [
    "data/*.db",
    "data/*.sqlite",
    "data/*.jsonl",
    "data/*.json",
    "data/*.pkl",
    ".streamlit/secrets.toml",
    "lease_intelligence_*.py",
    "mind.json",
]


def _is_protected(target: str) -> bool:
    """ターゲットパスが保護対象に該当するか判定する。"""
    p = PurePosixPath(target)
    for pattern in _PROTECTED_PATTERNS:
        if fnmatch.fnmatch(str(p), pattern):
            return True
        # ファイル名単体でもチェック（例: mind.json）
        if fnmatch.fnmatch(p.name, pattern):
            return True
    return False


def _candidate_paths(rule: dict) -> list[Path]:
    """ルール適用で変更されうるファイルの一覧（検証・ロールバック対象）。"""
    paths: list[Path] = []
    target = rule.get("target", "")
    if target:
        paths.append(_PROJECT_ROOT / target)
    rule_type = rule.get("type", "")
    if rule_type == "scoring_weight":
        paths.append(_PROJECT_ROOT / "api" / "scoring_weights.json")
    elif rule_type == "ui_text":
        paths.append(_PROJECT_ROOT / "frontend" / "src" / "lib" / "ui_labels.json")
    return paths


def _snapshot_paths(paths: list[Path]) -> dict[Path, bytes | None]:
    """適用前のファイル内容を保存する（None = ファイルが存在しなかった）。"""
    snapshot: dict[Path, bytes | None] = {}
    for path in paths:
        try:
            snapshot[path] = path.read_bytes() if path.exists() else None
        except OSError:
            snapshot[path] = None
    return snapshot


def _restore_snapshot(snapshot: dict[Path, bytes | None]) -> None:
    """スナップショットの状態へ戻す。適用前に無かったファイルは削除する。"""
    for path, content in snapshot.items():
        try:
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(content)
        except OSError as exc:
            print(f"  ⚠️  ロールバック失敗: {path} ({exc})", file=sys.stderr)


def _verify_file(path: Path) -> tuple[bool, str]:
    """適用後のファイルが壊れていないか検証する（JSONパース / Python構文）。"""
    if not path.exists():
        return True, "対象なし"
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True, "JSON OK"
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"JSON検証エラー ({path.name}): {exc}"
    if path.suffix == ".py":
        import py_compile
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
                py_compile.compile(str(path), cfile=tmp.name, doraise=True)
            return True, "py_compile OK"
        except py_compile.PyCompileError as exc:
            return False, f"Python構文エラー ({path.name}): {exc}"
        except OSError as exc:
            return False, f"検証I/Oエラー ({path.name}): {exc}"
    return True, "検証対象外の拡張子"


def _verify_after_apply(rule: dict, changed_file: str | None) -> tuple[bool, str]:
    """変更されたファイルと候補ファイルすべてを検証する。"""
    paths = list(_candidate_paths(rule))
    if changed_file:
        changed_path = _PROJECT_ROOT / changed_file
        if changed_path not in paths:
            paths.append(changed_path)
    for path in paths:
        ok, msg = _verify_file(path)
        if not ok:
            return False, msg
    return True, "OK"


def load_rules() -> list[dict]:
    if not _LEDGER_PATH.exists():
        print(f"❌ 台帳ファイルが見つかりません: {_LEDGER_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(_LEDGER_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_applied_at(rules: list[dict], applied_rev_ids: set[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = False
    for rule in rules:
        if rule.get("rev_id") in applied_rev_ids and "applied_at" not in rule:
            rule["applied_at"] = now
            changed = True
    if changed:
        with open(_LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
            f.write("\n")


class _Results:
    def __init__(self) -> None:
        self.success: list[tuple[str, str]] = []
        self.idempotent: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped_manual: list[tuple[str, str]] = []
        self.skipped_llm: list[tuple[str, str]] = []
        self.skipped_pending_review: list[tuple[str, str]] = []
        self.skipped_applied: list[tuple[str, str]] = []
        self.skipped_protected: list[tuple[str, str]] = []
        self.dry_run_pending: list[tuple[str, str, str]] = []  # (rev_id, type, desc)


def run_batch(rules: list[dict], dry_run: bool, rev_filter: str | None) -> None:
    from api.rule_engine.applier import apply_rule  # noqa: PLC0415

    res = _Results()
    applied_rev_ids: set[str] = set()

    for rule in rules:
        rev_id = rule.get("rev_id", "?")
        rule_type = rule.get("type", "")
        description = rule.get("description", "")

        # REV フィルタ
        if rev_filter and rev_id != rev_filter:
            continue

        # 既に適用済みはスキップ
        if rule.get("applied_at"):
            res.skipped_applied.append((rev_id, f"適用済み ({rule['applied_at']})"))
            print(f"  ⏭️  {rev_id} [{rule_type}] 適用済みのためスキップ")
            continue

        # manual はスキップ
        if rule_type == "manual":
            reason = rule.get("manual_reason", description)
            res.skipped_manual.append((rev_id, reason))
            print(f"  ⚠️  {rev_id} [manual] {description[:70]}")
            continue

        # llm_diff で pending_llm フラグが立っている場合はスキップ
        if rule_type == "llm_diff" and rule.get("pending_llm", False):
            res.skipped_llm.append((rev_id, description))
            print(f"  ⚠️  {rev_id} [llm_diff/pending] {description[:70]}")
            continue

        # pending_review フラグが立っている場合はスキップ（承認待ち）
        if rule.get("pending_review", False):
            res.skipped_pending_review.append((rev_id, description))
            print(f"  ⚠️  {rev_id} [pending_review] {description[:70]}")
            continue

        # 保護ファイルチェック
        target = rule.get("target", "")
        if target and _is_protected(target):
            res.skipped_protected.append((rev_id, f"保護対象ファイル: {target}"))
            print(f"  🔒 {rev_id} [{rule_type}] 保護対象のためスキップ: {target}")
            continue

        # dry-run では実行せずに記録のみ
        if dry_run:
            res.dry_run_pending.append((rev_id, rule_type, description))
            print(f"  🔍 {rev_id} [{rule_type}] 適用予定: {description[:70]}")
            continue

        # 実際に適用（適用前スナップショット → 適用 → 検証 → 失敗ならロールバック）
        snapshot = _snapshot_paths(_candidate_paths(rule))
        try:
            result = apply_rule(rule)
        except Exception as exc:
            _restore_snapshot(snapshot)
            res.failed.append((rev_id, str(exc)))
            print(f"  ❌ {rev_id} 例外（ロールバック済み）: {exc}")
            continue

        if result.success:
            if "冪等スキップ" in result.message:
                res.idempotent.append((rev_id, result.message))
                print(f"  ⏭️  {rev_id} 冪等スキップ: {result.message}")
            else:
                verify_ok, verify_msg = _verify_after_apply(rule, result.changed_file)
                if not verify_ok:
                    _restore_snapshot(snapshot)
                    res.failed.append((rev_id, f"適用後検証に失敗したためロールバック: {verify_msg}"))
                    print(f"  ↩️  {rev_id} 検証失敗→ロールバック: {verify_msg}")
                    continue
                res.success.append((rev_id, result.message))
                applied_rev_ids.add(rev_id)
                print(f"  ✅ {rev_id} 成功（検証OK）: {result.message}")
                if result.diff_summary:
                    for line in result.diff_summary.splitlines():
                        print(f"       {line}")
        else:
            res.failed.append((rev_id, result.message))
            print(f"  ❌ {rev_id} 失敗: {result.message}")

    # applied_at を台帳ファイルに書き戻す
    if not dry_run and applied_rev_ids:
        _save_applied_at(rules, applied_rev_ids)

    # ─── サマリー ───────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"📊 実行サマリー {'[DRY-RUN]' if dry_run else '[APPLY]'}")
    print("=" * 62)

    if dry_run:
        print(f"  🔍 適用予定 (--apply で実行) : {len(res.dry_run_pending):>3} 件")
    else:
        print(f"  ✅ 適用成功 (変更あり)       : {len(res.success):>3} 件")
        print(f"  ⏭️  冪等スキップ              : {len(res.idempotent):>3} 件")
        print(f"  ❌ 適用失敗                  : {len(res.failed):>3} 件")

    print(f"  ⚠️  手動対応必要 [manual]        : {len(res.skipped_manual):>3} 件")
    print(f"  ⚠️  LLM確認待ち [pending_llm]   : {len(res.skipped_llm):>3} 件")
    print(f"  ⚠️  承認待ち [pending_review]    : {len(res.skipped_pending_review):>3} 件")
    print(f"  ⏭️  適用済みスキップ             : {len(res.skipped_applied):>3} 件")
    if res.skipped_protected:
        print(f"  🔒 保護ファイルスキップ       : {len(res.skipped_protected):>3} 件")
    print()

    if res.skipped_manual:
        print("⚠️  手動対応が必要なルール:")
        for rev_id, reason in res.skipped_manual:
            print(f"   {rev_id}: {reason[:90]}")
        print()

    if res.skipped_llm:
        print("⚠️  LLM 確認待ちルール（llm_diff / pending_llm）:")
        print("   これらは --apply では自動実行されません。")
        print("   内容を確認し、pending_llm を false にしてから個別に実行してください。")
        for rev_id, desc in res.skipped_llm:
            print(f"   {rev_id}: {desc[:90]}")
        print()

    if res.skipped_pending_review:
        print("⚠️  承認待ちルール（pending_review）:")
        print("   改善ログ画面の「台帳ルール」タブから承認するか、pending_review を false にしてください。")
        for rev_id, desc in res.skipped_pending_review:
            print(f"   {rev_id}: {desc[:90]}")
        print()

    if not dry_run and res.failed:
        print("❌ 失敗したルール:")
        for rev_id, reason in res.failed:
            print(f"   {rev_id}: {reason}")
        print()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ledger_rules.json に定義された改善ルールを一括適用する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例:
  python api/rule_engine/batch_apply.py --dry-run
  python api/rule_engine/batch_apply.py --apply
  python api/rule_engine/batch_apply.py --apply --rev REV-007-a
""",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="実行せずに結果を表示")
    mode.add_argument("--apply", action="store_true", help="実際に適用")
    parser.add_argument(
        "--rev", metavar="REV_ID",
        help="特定の rev_id のルールのみ対象にする（例: --rev REV-007-a）",
    )
    args = parser.parse_args()

    rules = load_rules()

    mode_label = "DRY-RUN" if args.dry_run else "APPLY"
    rev_info = f" (REV フィルタ: {args.rev})" if args.rev else ""
    print(f"🚀 batch_apply.py — {mode_label}{rev_info}")
    print(f"   台帳ファイル : {_LEDGER_PATH}")
    print(f"   総ルール数   : {len(rules)} 件")
    print()

    run_batch(rules, dry_run=args.dry_run, rev_filter=args.rev)


if __name__ == "__main__":
    main()
