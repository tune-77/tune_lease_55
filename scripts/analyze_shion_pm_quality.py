#!/usr/bin/env python3
"""紫苑トリアージの事後検証（Phase 3: planning/shion_improvement_loop_plan.md）。

P3-1: 台帳（ledger）で applied/rejected/deleted へ解決した候補の結果を
      トリアージ記録の outcome へ書き戻す（追記形式・最後のエントリ有効）。
      ※ cleanup_improvement_reviews.py は CI 上でも走り data/ を持たないため、
        書き戻しは台帳を読む本スクリプト（Mac夜間実行）側で行う。
P3-3: トリアージ的中率・Overrule率・リードタイムを集計し、
      reports/shion_pm_quality_latest.{json,md} へ出力する。

使い方:
  python scripts/analyze_shion_pm_quality.py --date 2026-07-18
  python scripts/analyze_shion_pm_quality.py --dry-run   # 書き込みなしで表示
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from shion_triage import TRIAGE_FILE_RELPATH, load_triage_latest, triage_record_for_item  # noqa: E402

RESOLVED_STATUSES = {"applied", "rejected", "deleted"}
OUTCOME_LABELS = {"applied": "効いた(マージ済み)", "rejected": "外した(却下)", "deleted": "外した(削除)"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return


def load_ledger_statuses(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """台帳から (canonical_key→status, rev_id→status) を返す（最後のエントリ有効）。

    リポジトリ台帳（scripts/improvement_ledger.jsonl）とランタイム台帳
    （~/Library/Logs/tunelease/ledger.jsonl）の両方を読む。後者が新しい。
    """
    by_key: dict[str, str] = {}
    by_rev: dict[str, str] = {}
    paths = [
        root / "scripts" / "improvement_ledger.jsonl",
        Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl",
    ]
    for path in paths:
        for row in _iter_jsonl(path) or []:
            status = str(row.get("status") or "").lower()
            if not status:
                continue
            key = str(row.get("canonical_key") or row.get("key") or "")
            if key:
                by_key[key] = status
            rev_id = str(row.get("rev_id") or "")
            if rev_id:
                by_rev[rev_id] = status
    return by_key, by_rev


def resolve_outcome(record: dict, by_key: dict[str, str], by_rev: dict[str, str]) -> str:
    """トリアージ記録に対応する台帳の解決結果を返す（未解決なら空文字）。"""
    key = str(record.get("canonical_key") or "")
    status = by_key.get(key, "")
    if status not in RESOLVED_STATUSES:
        item_id = str(record.get("item_id") or "").strip()
        status = by_rev.get(item_id, "") if item_id else ""
    return status if status in RESOLVED_STATUSES else ""


def sync_outcomes(
    root: Path,
    triage_latest: dict[str, dict],
    by_key: dict[str, str],
    by_rev: dict[str, str],
    apply: bool,
) -> list[dict]:
    """P3-1: 解決済み候補の outcome をトリアージ記録へ追記する（冪等）。"""
    updates: list[dict] = []
    for record in triage_latest.values():
        outcome = resolve_outcome(record, by_key, by_rev)
        if not outcome:
            continue
        if str(record.get("outcome") or "") == outcome:
            continue  # 記録済み（冪等性）
        updated = dict(record)
        updated["outcome"] = outcome
        updated["outcome_recorded_at"] = dt.datetime.now().isoformat(timespec="seconds")
        updates.append(updated)
    if apply and updates:
        path = root / TRIAGE_FILE_RELPATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for updated in updates:
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
    return updates


def _avg_days(pairs: list[tuple[str, str]]) -> float | None:
    deltas: list[float] = []
    for start, end in pairs:
        try:
            t0 = dt.datetime.fromisoformat(start)
            t1 = dt.datetime.fromisoformat(end)
        except ValueError:
            continue
        deltas.append((t1 - t0).total_seconds() / 86400)
    if not deltas:
        return None
    return round(sum(deltas) / len(deltas), 1)


def load_report_candidates(root: Path) -> list[dict]:
    """網羅率の分母: 最新改善レポートのレビュー対象候補（needs_review）を返す。"""
    path = root / "reports" / "latest.json"
    if not path.exists():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in report.get("needs_review") or [] if isinstance(item, dict)]


def compute_coverage(triage_latest: dict[str, dict], candidates: list[dict]) -> dict[str, Any]:
    """トリアージ網羅率: 改善候補のうち判断が確定した割合（「中心」の実態確認）。"""
    if not candidates:
        return {"candidates": 0, "triaged": 0, "rate": None}
    triaged = sum(1 for item in candidates if triage_record_for_item(triage_latest, item))
    return {
        "candidates": len(candidates),
        "triaged": triaged,
        "rate": round(triaged / len(candidates), 3),
    }


def compute_kpis(triage_latest: dict[str, dict]) -> dict[str, Any]:
    """P3-3: 的中率（classified_by 別）・Overrule率・リードタイムを集計する。"""
    records = list(triage_latest.values())
    decision_counts: dict[str, int] = {}
    for record in records:
        decision = str(record.get("decision") or "")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    # 的中率: 「今日やる」判定のうち outcome が applied になった割合
    hit_stats: dict[str, dict[str, int]] = {}
    lead_pairs: list[tuple[str, str]] = []
    for record in records:
        if str(record.get("decision") or "") != "today":
            continue
        outcome = str(record.get("outcome") or "")
        if not outcome:
            continue
        classifier = str(record.get("classified_by") or "user")
        bucket = hit_stats.setdefault(classifier, {"resolved": 0, "applied": 0})
        bucket["resolved"] += 1
        if outcome == "applied":
            bucket["applied"] += 1
            if record.get("decided_at") and record.get("outcome_recorded_at"):
                lead_pairs.append((str(record["decided_at"]), str(record["outcome_recorded_at"])))
    hit_rates = {
        classifier: {
            **bucket,
            "hit_rate": round(bucket["applied"] / bucket["resolved"], 3) if bucket["resolved"] else None,
        }
        for classifier, bucket in hit_stats.items()
    }

    # Overrule率: ルール分類の初期値を User が覆した割合（迎合・逆張り検知の入力）
    with_rule = [r for r in records if str(r.get("rule_decision") or "")]
    overruled = [
        r for r in with_rule
        if str(r.get("classified_by") or "") == "user"
        and str(r.get("decision") or "") != str(r.get("rule_decision") or "")
    ]

    return {
        "triage_total": len(records),
        "decision_counts": decision_counts,
        "hit_rates_by_classifier": hit_rates,
        "overrule": {
            "with_rule_decision": len(with_rule),
            "overruled": len(overruled),
            "rate": round(len(overruled) / len(with_rule), 3) if with_rule else None,
        },
        "lead_time_days_avg": _avg_days(lead_pairs),
    }


def latest_improvement_quality(root: Path) -> dict[str, Any]:
    """analyze_improvement_quality.py の最新記録（P3-2 の材料）。"""
    last: dict[str, Any] = {}
    for row in _iter_jsonl(root / "data" / "improvement_quality_log.jsonl") or []:
        if isinstance(row, dict):
            last = row
    return last


def render_markdown(payload: dict[str, Any]) -> str:
    kpis = payload.get("kpis") or {}
    counts = kpis.get("decision_counts") or {}
    lines = [
        "# 紫苑トリアージ事後検証レポート",
        "",
        f"- 生成: {payload.get('generated_at')}",
        f"- トリアージ累計: {kpis.get('triage_total', 0)} 件"
        f"（今日やる {counts.get('today', 0)} / 後回し {counts.get('later', 0)} / 捨てる {counts.get('discard', 0)}）",
        f"- outcome 書き戻し: 今回 {payload.get('outcomes_synced', 0)} 件",
        "",
        "## トリアージ網羅率（最新レポートの候補のうち判断確定済み）",
    ]
    coverage = kpis.get("coverage") or {}
    coverage_rate = coverage.get("rate")
    lines.append(
        f"- {coverage.get('triaged', 0)}/{coverage.get('candidates', 0)}"
        + (f" = {coverage_rate * 100:.0f}%" if isinstance(coverage_rate, (int, float)) else "（候補なし・計測前）")
    )
    lines += [
        "",
        "## 的中率（「今日やる」→ applied）",
    ]
    hit_rates = kpis.get("hit_rates_by_classifier") or {}
    if hit_rates:
        for classifier, bucket in sorted(hit_rates.items()):
            rate = bucket.get("hit_rate")
            rate_text = f"{rate * 100:.0f}%" if isinstance(rate, (int, float)) else "計測前"
            lines.append(f"- {classifier}: {bucket.get('applied', 0)}/{bucket.get('resolved', 0)} = {rate_text}")
    else:
        lines.append("- 解決済みの「今日やる」判定がまだありません（計測前）")
    overrule = kpis.get("overrule") or {}
    rate = overrule.get("rate")
    lines += [
        "",
        "## Overrule率（ルール初期値をUserが覆した割合）",
        f"- {overrule.get('overruled', 0)}/{overrule.get('with_rule_decision', 0)}"
        + (f" = {rate * 100:.0f}%" if isinstance(rate, (int, float)) else "（計測前）"),
        "",
        "## リードタイム",
        f"- 判断→マージ 平均: {kpis.get('lead_time_days_avg') if kpis.get('lead_time_days_avg') is not None else '計測前'} 日",
        "",
        "## 監視先行率",
        "- 未計測（紫苑の失敗報告時刻の記録機構が未実装のため。記録設計後に追加）",
    ]
    quality = payload.get("improvement_quality") or {}
    if quality:
        lines += [
            "",
            "## 改善レポート品質（analyze_improvement_quality 最新）",
            f"- {json.dumps({k: v for k, v in quality.items() if k in ('date', 'success_rate', 'queued_count', 'success_count')}, ensure_ascii=False)}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="書き込みなしで結果を表示")
    args = parser.parse_args()

    root = repo_root()
    triage_latest = load_triage_latest(root)
    if not triage_latest:
        print("[shion_pm_quality] トリアージ記録がまだありません（スキップ）")
        return 0

    by_key, by_rev = load_ledger_statuses(root)
    updates = sync_outcomes(root, triage_latest, by_key, by_rev, apply=not args.dry_run)
    for updated in updates:
        label = OUTCOME_LABELS.get(str(updated.get("outcome")), updated.get("outcome"))
        print(f"[shion_pm_quality] outcome書き戻し: {updated.get('canonical_key')} → {label}")

    # 書き戻し後の状態で集計する
    triage_after = load_triage_latest(root) if not args.dry_run else {
        **triage_latest,
        **{str(u.get("canonical_key")): u for u in updates},
    }
    kpis = compute_kpis(triage_after)
    kpis["coverage"] = compute_coverage(triage_after, load_report_candidates(root))
    # 監視先行率は「紫苑がSlack通知より先に失敗を報告した」時刻の記録機構が
    # 未実装のため計測不能。記録の設計ができるまで未計測と明示する
    kpis["monitoring_lead"] = {"status": "not_instrumented"}
    payload = {
        "date": args.date,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "outcomes_synced": len(updates),
        "kpis": kpis,
        "improvement_quality": latest_improvement_quality(root),
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_compact = args.date.replace("-", "")
    for path in (reports_dir / f"shion_pm_quality_{date_compact}.json", reports_dir / "shion_pm_quality_latest.json"):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "shion_pm_quality_latest.md").write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"[shion_pm_quality] 集計完了: outcome同期 {len(updates)} 件 / "
        f"トリアージ {payload['kpis']['triage_total']} 件 → {reports_dir / 'shion_pm_quality_latest.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
