#!/usr/bin/env python3
"""昇格候補キューの承認分を長期記憶ソースへ追記する。

build_shion_memory_promotion_queue.py が出したキューから、ユーザーが承認した
候補だけを knowledge_base/shion_promoted_memories.md へ bullet として追記する。
このファイルは build_shion_memory_index.py のソースなので、次回の
インデックス再構築（夜間 or デプロイ時）で記憶として想起可能になる。

適用済みは data/shion_memory_promotions.jsonl に記録し、二重昇格を防ぐ。

使い方:
    python3 scripts/apply_shion_memory_promotions.py --ids promo_xxx,promo_yyy
    python3 scripts/apply_shion_memory_promotions.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "reports" / "shion_memory_promotion_queue_latest.json"
DEFAULT_TARGET = REPO_ROOT / "knowledge_base" / "shion_promoted_memories.md"
DEFAULT_APPLIED_LOG = REPO_ROOT / "data" / "shion_memory_promotions.jsonl"

_HEADER = """# 紫苑 昇格済み長期記憶

会話から承認を経て昇格した長期記憶。build_shion_memory_index.py が
このファイルの構造化 bullet を記憶レコードとして取り込む。
編集する場合は `- content:` ブロック単位で。削除ではなく revise
（scripts/revise_shion_memory.py）を優先すること。
"""

_AUTO_SAFE_DOMAIN_TERMS = (
    "リース",
    "審査",
    "判断資産",
    "購入選択権",
    "買取",
    "サプライヤー",
    "契約",
    "厨房機器",
)
_AUTO_SKIP_TERMS = (
    "正しい答え",
    "正解",
    "偉い",
    "明日",
    "審査員",
    "犬",
    "タム",
    "Git",
)
_AUTO_SKIP_PATTERNS = (
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日だ"),
    re.compile(r"企業名[:：]"),
    re.compile(r"デモ[一-龥A-Za-z0-9_ -]+"),
)


def _auto_safe_reason(candidate: dict) -> tuple[bool, str]:
    """Return whether a candidate is safe enough for user-requested auto promotion.

    This deliberately stays narrower than "all candidates": it accepts explicit
    durable lease knowledge or Shion operating principles, and rejects praise,
    temporary event instructions, personal trivia, broad tool bans, and case
    specific snippets.
    """
    content = str(candidate.get("proposed_content") or "").strip()
    kind = str(candidate.get("kind") or "")
    if not content:
        return False, "empty_content"
    if kind == "recurring_topic":
        return False, "recurring_topic_needs_context_review"
    if any(term in content for term in _AUTO_SKIP_TERMS):
        return False, "ephemeral_or_non_domain_or_praise"
    if any(pattern.search(content) for pattern in _AUTO_SKIP_PATTERNS):
        return False, "date_or_case_specific"
    if len(content) < 18:
        return False, "too_short"
    if not any(term in content for term in _AUTO_SAFE_DOMAIN_TERMS):
        return False, "not_lease_or_shion_principle"
    if "覚えて" not in content and "記憶" not in content:
        return False, "not_explicit_teaching"
    return True, "auto_safe_explicit_durable_memory"


def _clean_promoted_content(content: str) -> str:
    content = str(content or "").strip()
    content = re.sub(r"^覚えておいて\s*", "", content)
    content = re.sub(r"\s*覚えて(?:おいて|いて)?$", "", content)
    return " ".join(content.split())


def _metadata_for_candidate(candidate: dict) -> dict[str, object]:
    content = _clean_promoted_content(str(candidate.get("proposed_content") or ""))
    if "判断資産" in content:
        memory_type = "value_memory"
        domain = "judgment_asset_ops"
        use_when = "判断資産化するか迷う情報を扱う時"
        judgment_asset_candidate = False
    elif "スピード" in content:
        memory_type = "value_memory"
        domain = "lease_sales"
        use_when = "初回回答、営業向け助言、審査コメントの優先順位を決める時"
        judgment_asset_candidate = True
    elif "サプライヤー" in content:
        memory_type = "judgment_memory"
        domain = "sales_timing"
        use_when = "申込増加時期、サプライヤー起点案件、期末前後の案件背景を見る時"
        judgment_asset_candidate = True
    elif "購入選択権" in content or "買取" in content:
        memory_type = "factual_memory"
        domain = "lease_contract"
        use_when = "満了後買取、購入選択権、残価設定の説明をする時"
        judgment_asset_candidate = True
    elif "厨房機器" in content:
        memory_type = "factual_memory"
        domain = "asset_life"
        use_when = "飲食店・厨房機器のリース期間を確認する時"
        judgment_asset_candidate = True
    else:
        memory_type = "dialogue_memory"
        domain = "user_teaching"
        use_when = "Userから明示教示された前提を確認する時"
        judgment_asset_candidate = False
    return {
        "content": content,
        "type": memory_type,
        "domain": domain,
        "confidence": "user_taught",
        "use_when": use_when,
        "judgment_asset_candidate": judgment_asset_candidate,
        "source": str(candidate.get("candidate_id") or ""),
        "kind": str(candidate.get("kind") or ""),
    }


def _format_structured_memory(candidate: dict, today: str) -> str:
    meta = _metadata_for_candidate(candidate)
    return "\n".join(
        [
            f"- content: {meta['content']}",
            f"  type: {meta['type']}",
            f"  domain: {meta['domain']}",
            f"  confidence: {meta['confidence']}",
            f"  use_when: {meta['use_when']}",
            f"  judgment_asset_candidate: {str(bool(meta['judgment_asset_candidate'])).lower()}",
            f"  source: {meta['source']}",
            f"  kind: {meta['kind']}",
            f"  promoted_at: {today}",
        ]
    )


def _load_applied_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("candidate_id"):
            ids.add(str(row["candidate_id"]))
    return ids


def apply_promotions(
    queue_path: Path,
    target_path: Path,
    applied_log_path: Path,
    *,
    ids: set[str] | None,
    apply_all: bool,
    auto_safe: bool = False,
    dry_run: bool,
) -> int:
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"エラー: キューを読めません: {exc}", file=sys.stderr)
        return 1
    candidates = queue.get("candidates") if isinstance(queue, dict) else None
    if not isinstance(candidates, list) or not candidates:
        print("キューに候補がありません")
        return 0

    applied_ids = _load_applied_ids(applied_log_path)
    selected = []
    skipped_by_auto: list[tuple[str, str]] = []
    for c in candidates:
        cid = str(c.get("candidate_id") or "")
        if not cid or cid in applied_ids:
            continue
        if auto_safe:
            allowed, reason = _auto_safe_reason(c)
            if not allowed:
                skipped_by_auto.append((cid, reason))
                continue
            selected.append(c)
        elif apply_all or (ids and cid in ids):
            selected.append(c)

    if not selected:
        print("承認対象がありません（既に適用済みか、ID不一致）")
        if auto_safe and skipped_by_auto:
            print("[auto-safe] 自動除外:")
            for cid, reason in skipped_by_auto:
                print(f"  {cid}: {reason}")
        return 0

    today = datetime.now().date().isoformat()
    bullets = [_format_structured_memory(c, today) for c in selected]

    if dry_run:
        print(f"[dry-run] {target_path} へ {len(selected)} 件追記予定:")
        for b in bullets:
            print(" ", b[:120])
        if auto_safe and skipped_by_auto:
            print("[auto-safe] 自動除外:")
            for cid, reason in skipped_by_auto:
                print(f"  {cid}: {reason}")
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    current = target_path.read_text(encoding="utf-8") if target_path.exists() else _HEADER
    target_path.write_text(current.rstrip() + "\n\n" + "\n".join(bullets) + "\n", encoding="utf-8")

    applied_log_path.parent.mkdir(parents=True, exist_ok=True)
    with applied_log_path.open("a", encoding="utf-8") as fh:
        for c in selected:
            fh.write(
                json.dumps(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "kind": c.get("kind"),
                        "content": c.get("proposed_content"),
                        "applied_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(selected)} 件を {target_path} へ追記しました（次回インデックス再構築で記憶化）")
    if auto_safe and skipped_by_auto:
        print("[auto-safe] 自動除外:")
        for cid, reason in skipped_by_auto:
            print(f"  {cid}: {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="記憶昇格候補の承認・適用")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--applied-log", type=Path, default=DEFAULT_APPLIED_LOG)
    parser.add_argument("--ids", default="", help="承認する candidate_id（カンマ区切り）")
    parser.add_argument("--all", action="store_true", help="キューの全候補を承認する")
    parser.add_argument("--auto-safe", action="store_true", help="安全な明示教示候補だけを自動承認する")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = {token.strip() for token in args.ids.split(",") if token.strip()} or None
    if not ids and not args.all and not args.auto_safe:
        print("エラー: --ids、--all、--auto-safe のいずれかを指定してください", file=sys.stderr)
        return 1
    return apply_promotions(
        args.queue,
        args.target,
        args.applied_log,
        ids=ids,
        apply_all=args.all,
        auto_safe=args.auto_safe,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
