#!/usr/bin/env python3
"""
パイプラインステップの成否ログを集計し、
失敗率の高いステップを改善台帳(ledger_rules.json)に pending_review: true で追記する。

入力: data/pipeline_step_log.jsonl
  各行: {"ts": "2026-06-20T07:00:00", "run_date": "20260620", "step": "extract_obsidian_improvements", "exit_code": 0, "duration_s": 1.2}

出力: api/rule_engine/ledger_rules.json への追記（重複はrev_idで排除）
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from rev_ledger_utils import load_all_rev_sources, max_rev_number  # noqa: E402

LOG_FILE = PROJECT_ROOT / "data" / "pipeline_step_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"

FAILURE_RATE_THRESHOLD = 0.5
MIN_TOTAL_RUNS = 3
LOOKBACK_DAYS = 7

# 所要秒サマリー用。duration_s は pipeline_log_step.sh が自動計測した
# 「前ステップからの経過秒」で、ステップ単体の実行時間の上限値。
# 0 は「0秒」と「未計測（第3引数なしの旧ログ）」を区別できないため集計から除外する。
SLOW_STEP_TOP_N = 5
SLOW_STEP_MIN_SECONDS = 30

# 2026-09-10に外部Gist配布を廃止。過去7日ログがウィンドウから抜けるまで、
# 廃止済みステップを新しいパイプライン障害として再起票しない。
NON_BLOCKING_STEPS = {"gist_update"}

# auto_fix_allowed=true にするための条件
AUTO_FIX_MIN_FAILURE_DAYS = 5   # 7日中5日以上失敗していること
AUTO_FIX_RULE_TYPES = {"patch_json", "config_value"}  # 対応ルール型

# 週次ゲート（run_daily_improvement_post.sh の DETAILED_SIDECAR_REPORT_FREQUENCY=weekly、
# 既定で月曜のみ実行）で走るステップ。7日ウィンドウでは実測が最大1件しか溜まらないため
# MIN_TOTAL_RUNS=3 を永久に満たせず、毎週必ず失敗しても失敗検知に乗らなかった。
# これらのステップだけ評価窓を4週間に広げ、週次の実行頻度に見合う閾値で判定する。
#
# build_judgment_asset_graph は週次ゲートの中にあるが、else 側が同じステップ名で
# exit 0 を記録する（run_daily_improvement_post.sh:375,378）ため毎日ログが出る。
# 通常ステップとして扱うのが正しいので、ここには含めない。
WEEKLY_STEPS = frozenset(
    {
        "check_orphaned_scripts",
        "build_instruction_debt_report",
        "build_judgment_asset_ab_report",
        "build_shion_growth_brief",
        "evaluate_shion_growth",
        "build_shion_architecture_layer_audit",
    }
)
WEEKLY_LOOKBACK_DAYS = 28
WEEKLY_MIN_TOTAL_RUNS = 2          # 4週のうち2回以上の実測があれば判定する
WEEKLY_AUTO_FIX_MIN_FAILURE_DAYS = 2   # 2週連続で失敗していること

# ログ読み込みは最長窓で行い、判定はステップごとの窓で切る（二段フィルタ）。
# 読み込み窓だけ広げると通常ステップが3週前の失敗を数え始めてしまう。
MAX_LOOKBACK_DAYS = max(LOOKBACK_DAYS, WEEKLY_LOOKBACK_DAYS)


def step_lookback_days(step: str) -> int:
    """当該ステップの評価窓（日数）。週次ステップだけ長い窓を使う。"""
    return WEEKLY_LOOKBACK_DAYS if step in WEEKLY_STEPS else LOOKBACK_DAYS


def step_min_total_runs(step: str) -> int:
    """当該ステップを判定対象にするための最小実測件数。"""
    return WEEKLY_MIN_TOTAL_RUNS if step in WEEKLY_STEPS else MIN_TOTAL_RUNS


def step_auto_fix_min_days(step: str) -> int:
    """auto_fix_allowed=true にするための最小失敗日数。"""
    return WEEKLY_AUTO_FIX_MIN_FAILURE_DAYS if step in WEEKLY_STEPS else AUTO_FIX_MIN_FAILURE_DAYS


def load_recent_logs():
    if not LOG_FILE.exists():
        print("pipeline_step_log.jsonl が存在しません。スキップします。", flush=True)
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_LOOKBACK_DAYS)).strftime("%Y%m%d")
    entries = []
    with LOG_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("run_date", "") >= cutoff:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def filter_to_step_windows(entries):
    """最長窓で読み込んだログを、ステップごとの評価窓まで絞り込む。

    load_recent_logs() が MAX_LOOKBACK_DAYS で読むため、この関数を通さないと
    通常ステップが3週間前の失敗まで数えてしまう。"""
    now = datetime.now(timezone.utc)
    cutoff_by_days = {
        days: (now - timedelta(days=days)).strftime("%Y%m%d")
        for days in {LOOKBACK_DAYS, WEEKLY_LOOKBACK_DAYS}
    }
    kept = []
    for e in entries:
        step = e.get("step", "unknown")
        if str(e.get("run_date", "")) >= cutoff_by_days[step_lookback_days(step)]:
            kept.append(e)
    return kept


def resolution_cutoff(ledger: list, step: str) -> str:
    """当該ステップについてledger上で直近に解決済みとされたresolved_atを返す（無ければ空文字）。

    REV-304a→REV-392aのように、コード修正で直った後も7日ウィンドウに残る
    修正前の失敗ログだけで同一ステップが再度penalty_stepsに乗り、重複REVが
    起票される再発を防ぐためのアンカー。"""
    latest = ""
    for entry in ledger:
        if entry.get("status") not in {"resolved", "stale_resolved"}:
            continue
        if step not in str(entry.get("description") or ""):
            continue
        resolved_at = str(entry.get("resolved_at") or "")
        if resolved_at > latest:
            latest = resolved_at
    return latest


def aggregate(entries, cutoffs: dict | None = None):
    cutoffs = cutoffs or {}
    counts = defaultdict(
        lambda: {
            "good": 0,
            "bad": 0,
            "bad_days": set(),
            "latest_exit_code": None,
            "latest_ts": "",
            "durations": [],
        }
    )
    for e in entries:
        step = e.get("step", "unknown")
        ts = str(e.get("ts") or "")
        cutoff = cutoffs.get(step, "")
        if cutoff and ts and ts <= cutoff:
            # 直近の解決（コード修正の反映含む）より前のログは再検出の判定に含めない
            continue
        if e.get("exit_code", 1) == 0:
            counts[step]["good"] += 1
        else:
            counts[step]["bad"] += 1
            counts[step]["bad_days"].add(e.get("run_date", ""))
        duration = e.get("duration_s")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            counts[step]["durations"].append(float(duration))
        if ts >= counts[step]["latest_ts"]:
            counts[step]["latest_ts"] = ts
            counts[step]["latest_exit_code"] = e.get("exit_code", 1)
    return counts


def has_auto_fix_rule(ledger: list, step: str) -> bool:
    """ledger_rules.json に対象ステップに対応する patch_json / config_value 型ルールが存在するか。"""
    for entry in ledger:
        if entry.get("type") not in AUTO_FIX_RULE_TYPES:
            continue
        if step in str(entry.get("description") or ""):
            return True
    return False


def slow_steps(counts, top_n=SLOW_STEP_TOP_N, min_seconds=SLOW_STEP_MIN_SECONDS):
    """所要秒が記録されているステップから「遅い」ものを選び、遅い順のリストで返す。

    引数:
        counts: aggregate() の戻り値。counts[step]["durations"] に 0 より大きい
                実測秒のリストが入る（未計測=0 は除外済み。空リストのステップもある）
        top_n: 返す最大件数
        min_seconds: これ未満のステップは報告しない下限秒

    戻り値: [(step, 代表秒, 実測サンプル数), ...] を代表秒の降順で最大 top_n 件

    代表秒は期間内の**最大値**。平均や最新値ではなくピークを採るのは、日次1回のため
    サンプルが最大7件しかなく平均では単発の悪化が埋もれるから。反面、NTP補正や
    一時的な外部API待ちによる1回だけのスパイクも拾うので、サンプル数を併記して
    読み手が単発か慢性かを判断できるようにしている。
    """
    rows = []
    for step, c in counts.items():
        durations = c.get("durations") or []
        if not durations:
            # 未計測のステップ（旧ログや第3引数なしの0のみ）は報告対象外
            continue
        peak = max(durations)
        if peak < min_seconds:
            continue
        rows.append((step, peak, len(durations)))
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]


def print_duration_summary(counts):
    """所要秒サマリーを標準出力に出す（朝レポートのログに残る）。"""
    measured = sum(len(c["durations"]) for c in counts.values())
    if measured == 0:
        print(
            f"\n所要秒の実測ログがまだありません（通常{LOOKBACK_DAYS}日 / 週次{WEEKLY_LOOKBACK_DAYS}日）。"
            "pipeline_log_step.sh の自動計測が入った翌朝から溜まります。",
            flush=True,
        )
        return

    rows = slow_steps(counts)
    print(
        f"\n=== 所要秒の重いステップ（通常{LOOKBACK_DAYS}日 / 週次{WEEKLY_LOOKBACK_DAYS}日 / {SLOW_STEP_MIN_SECONDS}秒以上） ===",
        flush=True,
    )
    if not rows:
        print(f"  {SLOW_STEP_MIN_SECONDS}秒以上かかっているステップはありません。", flush=True)
        return
    for step, seconds, samples in rows:
        print(f"  {step}: {seconds:.0f}秒 (実測{samples}回)", flush=True)


def load_ledger():
    if not LEDGER_FILE.exists():
        return []
    with LEDGER_FILE.open() as f:
        return json.load(f)


def already_exists(ledger, step):
    for entry in ledger:
        if step in entry.get("description", "") and entry.get("status") not in {"resolved", "stale_resolved"}:
            return True
    return False


def resolve_recovered_entries(ledger: list, counts: dict, now_iso: str) -> int:
    """最新実行が成功している過去のパイプライン障害検出を解決済みにする。"""
    resolved = 0
    for entry in ledger:
        if entry.get("source") != "analyze_pipeline_health":
            continue
        if entry.get("status") in {"resolved", "stale_resolved"}:
            continue
        description = str(entry.get("description") or "")
        for step, c in counts.items():
            if step not in description:
                continue
            if c.get("latest_exit_code") == 0 or step in NON_BLOCKING_STEPS:
                entry["status"] = "stale_resolved"
                entry["pending_review"] = False
                entry["resolved_at"] = now_iso
                entry["resolution_reason"] = (
                    "任意配布ステップであり、ローカルの改善適用・朝レポート生成を停止しないため解決済みに更新"
                    if step in NON_BLOCKING_STEPS
                    else "直近の同ステップ実行が成功しているため、過去検出を解決済みに更新"
                )
                resolved += 1
            break
    return resolved


def main():
    entries = load_recent_logs()
    if not entries:
        return

    entries = filter_to_step_windows(entries)
    if not entries:
        print("評価窓内のステップログがありません。スキップします。", flush=True)
        return

    ledger = load_ledger()
    steps_seen = {e.get("step", "unknown") for e in entries}
    cutoffs = {step: cutoff for step in steps_seen if (cutoff := resolution_cutoff(ledger, step))}
    counts = aggregate(entries, cutoffs)
    print_duration_summary(counts)

    penalty_steps = []
    for step, c in counts.items():
        if step in NON_BLOCKING_STEPS:
            continue
        total = c["good"] + c["bad"]
        if total < step_min_total_runs(step):
            continue
        rate = c["bad"] / total
        if rate >= FAILURE_RATE_THRESHOLD:
            penalty_steps.append((step, c["bad"], total, rate, len(c["bad_days"])))

    base_rev = max_rev_number(load_all_rev_sources())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved = resolve_recovered_entries(ledger, counts, now_iso)

    if not penalty_steps:
        if resolved > 0:
            with LEDGER_FILE.open("w") as f:
                json.dump(ledger, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"復旧済みパイプライン障害を自動整理: {resolved} 件: {LEDGER_FILE}", flush=True)
        print(
            f"評価窓（通常{LOOKBACK_DAYS}日 / 週次ステップ{WEEKLY_LOOKBACK_DAYS}日）で"
            f"失敗率閾値({FAILURE_RATE_THRESHOLD*100:.0f}%)超のステップはありません。",
            flush=True,
        )
        return

    added = 0
    for step, bad, total, rate, bad_days in sorted(penalty_steps, key=lambda x: -x[3]):
        if counts[step].get("latest_exit_code") == 0:
            print(f"スキップ（直近成功）: {step}", flush=True)
            continue
        if already_exists(ledger, step):
            print(f"スキップ（既存）: {step}", flush=True)
            continue

        base_rev += 1
        rev_id = f"REV-{base_rev:03d}a"
        pct = int(rate * 100)
        window_days = step_lookback_days(step)
        description = f"[パイプライン自動検出] {step} が過去{window_days}日で失敗率{pct}%（{bad}/{total}件, {bad_days}日失敗）"

        # 評価窓内で規定日数以上失敗 かつ patch_json/config_value 型ルールが存在する場合のみ自動修正許可
        can_auto_fix = (
            bad_days >= step_auto_fix_min_days(step)
            and has_auto_fix_rule(ledger, step)
        )

        new_entry = {
            "rev_id": rev_id,
            "type": "patch_json" if can_auto_fix else "manual",
            "pending_review": True,
            "category": "pipeline_fix",
            "description": description,
            "status": "pending_review",
            "source": "analyze_pipeline_health",
            "detected_at": now_iso,
            "affected_files": [],
            "risk": "low" if can_auto_fix else "medium",
            "auto_fix_allowed": can_auto_fix,
        }
        ledger.append(new_entry)
        added += 1
        print(f"追記: {rev_id} — {description}", flush=True)

    if added > 0 or resolved > 0:
        with LEDGER_FILE.open("w") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n台帳更新: 追記 {added} 件 / 解決 {resolved} 件: {LEDGER_FILE}", flush=True)
    else:
        print("追記なし（すべて既存エントリと重複）。", flush=True)

    # サマリー出力
    print(
        f"\n=== パイプラインヘルス サマリー（通常{LOOKBACK_DAYS}日 / 週次ステップ{WEEKLY_LOOKBACK_DAYS}日） ===",
        flush=True,
    )
    for step, c in sorted(counts.items()):
        total = c["good"] + c["bad"]
        rate = c["bad"] / total if total else 0
        bad_days = len(c["bad_days"])
        active_failure = c.get("latest_exit_code") != 0
        flag = (
            " ⚠️"
            if active_failure and rate >= FAILURE_RATE_THRESHOLD and total >= step_min_total_runs(step)
            else ""
        )
        weekly = f" [週次{step_lookback_days(step)}日窓]" if step in WEEKLY_STEPS else ""
        print(
            f"  {step}: 成功{c['good']}/失敗{c['bad']}({bad_days}日) (失敗率{rate*100:.0f}%){weekly}{flag}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
