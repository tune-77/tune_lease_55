#!/usr/bin/env python3
"""Apply allowlisted pipeline recovery recipes learned from resolved incidents.

Resolved pipeline REV entries are used as evidence and reporting context, never as
commands.  Only deterministic handlers registered in ``RECOVERY_RECIPES`` may
change state.  Each step is attempted at most once per run date and must verify
success before a healthy step result is appended to the structured pipeline log.

日次パイプラインの最後（core と post の全手順の後）に1回走る。その日に失敗した手順を
3つに振り分ける:
- 個別レシピ（RECOVERY_RECIPES）: 解決済み障害の証拠がある手順だけ、決め打ちの手当てをする
- 汎用レシピ（GENERIC_RETRY_STEPS）: 冪等と確認した手順だけ。ログから一時的な原因
  （タイムアウト・ネットワーク/API・iCloud/ファイル欠落・ロック競合）が読めた時に限り
  1回だけ再実行し、exit 0 と手順ごとの成功条件（出力の更新）で検証する
- 品質チェック（評価・テスト・ヘルスチェック・検知器）: 再実行で通すと本当の劣化を隠すので
  再実行せず「劣化の疑い」として朝報に出す
直せなかった失敗・品質チェック失敗・2日以上連続の失敗は morning_report_lines() が朝報上部に出す。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_shion_memory_health import (  # noqa: E402
    check as check_memory_health,
    load_index_summary,
    load_state as load_memory_health_state,
    save_state as save_memory_health_state,
)

DEFAULT_LOG = ROOT / "data" / "pipeline_step_log.jsonl"
DEFAULT_LEDGER = ROOT / "api" / "rule_engine" / "ledger_rules.json"
DEFAULT_STATE = ROOT / "data" / "pipeline_auto_recovery_state.json"
DEFAULT_REPORT = ROOT / "reports" / "pipeline_auto_recovery_latest.json"
PIPELINE_LOG_DIR = Path.home() / "Library" / "Logs" / "tunelease"
GENERIC_RECIPE_ID = "generic_transient_retry"


RECOVERY_RECIPES: dict[str, dict[str, str]] = {
    "check_shion_memory_health": {
        "recipe_id": "migrate_memory_health_layer_baseline",
        "description": "旧形式の件数基準を層別基準へ一度だけ移行し、直後に再検証する",
    },
    "build_agent_worklog_digest": {
        "recipe_id": "rerun_agent_worklog_digest_after_source_refresh",
        "description": "既知の2保存形式を再走査してダイジェストを再生成する",
    },
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latest_step_results(log_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return latest
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        step = str(entry.get("step") or "")
        if not step:
            continue
        if str(entry.get("ts") or "") >= str(latest.get(step, {}).get("ts") or ""):
            latest[step] = entry
    return latest


def build_recovery_catalog(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize resolved pipeline incidents without executing ledger text."""
    learned: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in ledger:
        if entry.get("category") != "pipeline_fix":
            continue
        if entry.get("status") not in {"resolved", "stale_resolved"}:
            continue
        if entry.get("auto_fix_allowed") is not True:
            continue
        description = str(entry.get("description") or "")
        for step in RECOVERY_RECIPES:
            if step not in description:
                continue
            learned[step].append(
                {
                    "rev_id": str(entry.get("rev_id") or ""),
                    "resolved_at": str(entry.get("resolved_at") or ""),
                    "resolution_reason": str(entry.get("resolution_reason") or "")[:300],
                }
            )
            break

    recipes = []
    for step, definition in RECOVERY_RECIPES.items():
        history = learned.get(step, [])
        recipes.append(
            {
                "step": step,
                **definition,
                "resolved_incident_count": len(history),
                "learned_from": history[-5:],
            }
        )
    return {"recipes": recipes, "resolved_incident_count": sum(len(v) for v in learned.values())}


def plan_recoveries(
    latest: dict[str, dict[str, Any]],
    state: dict[str, Any],
    run_date: str,
    *,
    eligible_steps: set[str] | None = None,
) -> list[dict[str, Any]]:
    attempts = state.get("attempts") if isinstance(state.get("attempts"), dict) else {}
    plans: list[dict[str, Any]] = []
    for step, definition in RECOVERY_RECIPES.items():
        if eligible_steps is not None and step not in eligible_steps:
            continue
        observed = latest.get(step)
        if not observed or int(observed.get("exit_code", 1)) == 0:
            continue
        attempt_key = f"{run_date}:{step}"
        if attempt_key in attempts:
            continue
        plans.append(
            {
                "step": step,
                "attempt_key": attempt_key,
                "last_failure": observed,
                **definition,
            }
        )
    plans.sort(
        key=lambda plan: sum(
            1 for key in attempts if str(key).endswith(f":{plan['step']}")
        )
    )
    return plans


def _recover_memory_health(root: Path) -> tuple[bool, str]:
    index_path = root / "data" / "shion_memory_index.json"
    state_path = root / "data" / "shion_memory_health_state.json"
    previous_summary_path = root / "data" / "shion_memory_index_previous_summary.json"
    summary = load_index_summary(index_path)
    previous = load_memory_health_state(state_path)
    if summary is None or int(summary.get("total") or 0) <= 0:
        return False, "記憶インデックスが読めないか0件のため自動移行しない"
    if isinstance(previous.get("by_layer"), dict) and previous.get("by_layer"):
        return False, "層別基準は移行済み。実データ減少の可能性があるため自動修正しない"
    previous_total = int(previous.get("total") or 0)
    if previous_total <= 0:
        return False, "旧基準が存在しないため通常ヘルスチェックに委ねる"

    previous_summary = _load_json(previous_summary_path, {})
    if not isinstance(previous_summary, dict) or not previous_summary.get("by_layer"):
        return False, "置換前インデックスの層別証拠がないため自動移行しない"
    if int(previous_summary.get("total") or 0) != previous_total:
        return False, "置換前インデックスと旧基準の総件数が一致しないため自動移行しない"

    stable_healthy, stable_message = check_memory_health(summary, previous_summary)
    if not stable_healthy:
        return False, f"置換前インデックスとの永続層比較が異常のため自動移行しない: {stable_message}"

    layers = summary.get("by_layer") if isinstance(summary.get("by_layer"), dict) else {}
    stable_total = sum(int(layers.get(name) or 0) for name in ("long_term", "persistent", "retrieval"))
    if stable_total <= 0:
        return False, "永続層が0件のため基準移行を拒否"

    save_memory_health_state(state_path, summary)
    healthy, message = check_memory_health(summary, summary)
    return healthy, f"置換前インデックスで永続層を検証し、層別基準へ移行: {message}"


def _recover_worklog_digest(root: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_agent_worklog_digest.py"),
            "--days",
            "14",
            "--limit",
            "12",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    detail = (proc.stdout or proc.stderr or "no output").strip()[-800:]
    if proc.returncode != 0:
        return False, detail
    digest = _load_json(root / "reports" / "agent_worklog_digest_latest.json", {})
    source_count = int(digest.get("source_count") or 0) if isinstance(digest, dict) else 0
    if source_count <= 0:
        return False, f"再生成は終了したが作業録が0件のため復旧未確認: {detail}"
    return True, f"作業録{source_count}件を再検出: {detail}"


HANDLERS: dict[str, Callable[[Path], tuple[bool, str]]] = {
    "migrate_memory_health_layer_baseline": _recover_memory_health,
    "rerun_agent_worklog_digest_after_source_refresh": _recover_worklog_digest,
}


def _append_verified_step_result(
    log_path: Path,
    *,
    run_date: str,
    step: str,
    recipe_id: str,
    duration_s: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": run_date,
        "step": step,
        "exit_code": 0,
        "duration_s": duration_s,
        "recovery": True,
        "recipe_id": recipe_id,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def execute_plan(
    plan: dict[str, Any],
    *,
    root: Path,
    log_path: Path,
    run_date: str,
) -> dict[str, Any]:
    recipe_id = str(plan["recipe_id"])
    handler = HANDLERS[recipe_id]
    started = time.monotonic()
    try:
        success, detail = handler(root)
    except Exception as exc:  # noqa: BLE001 - failure must be recorded, not stop the pipeline
        success, detail = False, f"{type(exc).__name__}: {exc}"
    duration_s = max(0, int(time.monotonic() - started))
    if success:
        _append_verified_step_result(
            log_path,
            run_date=run_date,
            step=str(plan["step"]),
            recipe_id=recipe_id,
            duration_s=duration_s,
        )
    return {
        "step": plan["step"],
        "recipe_id": recipe_id,
        "success": success,
        "detail": detail,
        "duration_s": duration_s,
    }


# ── 汎用レシピ：一時的な失敗だけを1回再実行 ─────────────────────────────

# 冪等と確認した手順だけ（出典: run_daily_improvement_core.sh / post.sh の呼び出し行）。
# outputs は「再実行開始後に更新された・空でない・JSONなら読める」ことを成功条件にするファイル。
# Slack 送信・GCS/Vertex へのアップロード・記憶の追記昇格など外部送信や追記型は入れない。
_GCS_SYNC = {"cmd": ["scripts/sync_cloudrun_inputs_from_gcs.py"], "outputs": []}  # event_id で重複排除・DB は UNIQUE
_OBSIDIAN_SUMMARY = {"cmd": ["scripts/sync_cloudrun_inputs_to_obsidian.py"], "outputs": []}  # 日次要約の上書き
_MEMORY_INDEX = {"cmd": ["scripts/build_shion_memory_index.py"], "outputs": ["data/shion_memory_index.json"]}
_FRESHNESS = {"cmd": ["scripts/update_shion_memory_freshness.py"], "outputs": ["data/shion_memory_index.json"]}
GENERIC_RETRY_STEPS: dict[str, dict[str, list[str]]] = {
    "sync_cloudrun_inputs_from_gcs": _GCS_SYNC,
    "sync_cloudrun_inputs_from_gcs_post": _GCS_SYNC,
    "sync_cloudrun_inputs_to_obsidian": _OBSIDIAN_SUMMARY,
    "sync_cloudrun_inputs_to_obsidian_post": _OBSIDIAN_SUMMARY,
    "fetch_estat_industry": {"cmd": ["scripts/fetch_estat_industry.py"], "outputs": []},
    "sync_codex_pr_status": {"cmd": ["scripts/sync_codex_pr_status.py"], "outputs": []},
    "build_shion_memory_index": _MEMORY_INDEX,
    "build_shion_memory_index_post_promotion": _MEMORY_INDEX,
    "build_shion_memory_index_after_auto_promotions": _MEMORY_INDEX,
    "update_shion_memory_freshness": _FRESHNESS,
    "update_shion_memory_freshness_post_promotion": _FRESHNESS,
    "build_shion_practical_knowledge_map": {
        "cmd": ["scripts/build_shion_practical_knowledge_map.py"],
        "outputs": ["data/shion_practical_knowledge_map.json"],
    },
}

# 失敗＝「品質が落ちた」という信号の手順。再実行で通すと劣化を隠す（2026-10-03 の想起劣化が実例）
QUALITY_CHECK_STEPS = {"loop_metrics", "sync_memory_from_daily", "memory_chat_regression_tests"}
_QUALITY_PREFIXES = ("eval_", "check_", "audit_", "test_", "validate_")
_QUALITY_SUFFIXES = ("_tests", "_health", "_eval")

TRANSIENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("timeout", re.compile(r"TimeoutExpired|timed out|ReadTimeout|ConnectTimeout|DeadlineExceeded|タイムアウト", re.I)),
    (
        "network",
        re.compile(
            r"ConnectionError|Connection (reset|refused|aborted)|Temporary failure in name resolution|"
            r"nodename nor servname|NewConnectionError|RemoteDisconnected|SSLError|ServiceUnavailable|"
            r"Too Many Requests|\b(429|500|502|503|504)\b.*(error|status|HTTP)|HTTP(Error)?\s*(429|5\d\d)",
            re.I,
        ),
    ),
    ("icloud_file", re.compile(r"Resource deadlock avoided|\.icloud\b|FileNotFoundError|No such file or directory", re.I)),
    ("lock", re.compile(r"database is locked|could not acquire lock|Resource temporarily unavailable|BlockingIOError", re.I)),
]
_STEP_MARKER_RE = re.compile(r"^\[step\] (\S+) exit=(-?\d+)\s*$")
_ERROR_LINE_RE = re.compile(r"error|exception|失敗|警告|traceback|fatal", re.I)


def is_quality_check(step: str) -> bool:
    return step in QUALITY_CHECK_STEPS or step.startswith(_QUALITY_PREFIXES) or step.endswith(_QUALITY_SUFFIXES)


def step_log_excerpt(log_text: str, step: str, max_lines: int = 80) -> str | None:
    """pipeline_log_step.sh の `[step] name exit=N` 印を頼りに、その手順自身の出力だけを切り出す。"""
    lines = log_text.splitlines()
    end = None
    for i in range(len(lines) - 1, -1, -1):
        m = _STEP_MARKER_RE.match(lines[i])
        if m and m.group(1) == step:
            end = i
            break
    if end is None:
        return None
    start = 0
    for i in range(end - 1, -1, -1):
        if _STEP_MARKER_RE.match(lines[i]):
            start = i + 1
            break
    return "\n".join(lines[start:end][-max_lines:])


def classify_transient(excerpt: str | None) -> str | None:
    if not excerpt:
        return None
    for cause, pattern in TRANSIENT_PATTERNS:
        if pattern.search(excerpt):
            return cause
    return None


def summarize_error(excerpt: str | None) -> str:
    if not excerpt:
        return "ログに手順の出力なし"
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    hits = [line for line in lines if _ERROR_LINE_RE.search(line)]
    return ((hits or lines or ["出力なし"])[-1])[:200]


def pipeline_log_path(run_date: str, log_dir: Path = PIPELINE_LOG_DIR) -> Path:
    return log_dir / f"improvement_{run_date}.log"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def failed_steps_for_run(log_path: Path, run_date: str) -> dict[str, dict[str, Any]]:
    """その run_date の各手順の最終結果のうち、失敗しているものを返す。"""
    final: dict[str, dict[str, Any]] = {}
    for raw in _read_text(log_path).splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or str(entry.get("run_date") or "") != run_date or not entry.get("step"):
            continue
        step = str(entry["step"])
        if str(entry.get("ts") or "") >= str(final.get(step, {}).get("ts") or ""):
            final[step] = entry
    return {step: e for step, e in final.items() if int(e.get("exit_code", 0) or 0) != 0}


def _outputs_verified(root: Path, outputs: list[str], started_wall: float) -> tuple[bool, str]:
    for rel in outputs:
        path = root / rel
        if not path.exists() or path.stat().st_size <= 0:
            return False, f"成功条件未達: {rel} が無いか空"
        if path.stat().st_mtime < started_wall - 1:
            return False, f"成功条件未達: {rel} が更新されていない"
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False, f"成功条件未達: {rel} が JSON として読めない"
    return True, ""


def rerun_generic_step(step: str, *, root: Path, timeout_s: int = 900) -> tuple[bool, int, str]:
    definition = GENERIC_RETRY_STEPS[step]
    started_wall = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, *[str(root / part) if part.startswith("scripts/") else part for part in definition["cmd"]]],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, 124, f"再実行も{timeout_s}秒でタイムアウト"
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, proc.returncode, f"再実行も exit {proc.returncode}: {summarize_error(output)}"
    ok, why = _outputs_verified(root, definition["outputs"], started_wall)
    if not ok:
        return False, 1, why
    return True, 0, f"再実行で exit 0・成功条件確認: {summarize_error(output)}"


def _append_step_entry(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **entry}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def triage_and_retry(
    *,
    root: Path,
    log_path: Path,
    pipeline_log: Path,
    state: dict[str, Any],
    run_date: str,
    apply: bool,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """その日の失敗を、汎用再実行・品質チェック・直せないものに振り分ける。個別レシピの手順は除く。"""
    log_text = _read_text(pipeline_log)
    attempts = state.setdefault("attempts", {})
    outcome: dict[str, list[dict[str, Any]]] = {"recovered": [], "unrecovered": [], "quality_failures": []}
    retried = 0
    for step, failure in sorted(failed_steps_for_run(log_path, run_date).items()):
        if step == "pipeline_auto_recovery":
            continue
        excerpt = step_log_excerpt(log_text, step)
        summary = summarize_error(excerpt)
        if is_quality_check(step):
            outcome["quality_failures"].append({"step": step, "summary": summary, "exit_code": failure.get("exit_code")})
            continue
        if step in RECOVERY_RECIPES:
            outcome["unrecovered"].append({"step": step, "cause": "individual_recipe_not_applied", "summary": summary})
            continue
        if step not in GENERIC_RETRY_STEPS:
            outcome["unrecovered"].append({"step": step, "cause": "not_allowlisted", "summary": summary})
            continue
        cause = classify_transient(excerpt)
        if cause is None:
            reason = "no_log_evidence" if excerpt is None else "not_transient"
            outcome["unrecovered"].append({"step": step, "cause": reason, "summary": summary})
            continue
        attempt_key = f"{run_date}:{step}"
        if attempt_key in attempts or not apply or retried >= limit:
            reason = "already_attempted" if attempt_key in attempts else "dry_run" if not apply else "daily_limit"
            outcome["unrecovered"].append({"step": step, "cause": f"{cause}/{reason}", "summary": summary})
            continue
        retried += 1
        started = time.monotonic()
        success, exit_code, detail = rerun_generic_step(step, root=root)
        duration_s = max(0, int(time.monotonic() - started))
        attempts[attempt_key] = {
            "attempted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "recipe_id": GENERIC_RECIPE_ID,
            "cause": cause,
            "success": success,
            "detail": detail[:500],
        }
        _append_step_entry(
            log_path,
            {
                "run_date": run_date,
                "step": step,
                "exit_code": exit_code,
                "duration_s": duration_s,
                "recovery": True,
                "recipe_id": GENERIC_RECIPE_ID,
                "cause": cause,
            },
        )
        bucket = "recovered" if success else "unrecovered"
        outcome[bucket].append({"step": step, "cause": cause, "summary": detail if success else f"{summary} → {detail}"})
    return outcome


def consecutive_failure_days(log_path: Path) -> tuple[str, dict[str, int]]:
    """最新 run_date に走った手順のうち、最終結果が何回連続で失敗しているか（run_date 単位）。"""
    final: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _read_text(log_path).splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or not entry.get("step") or not entry.get("run_date"):
            continue
        key = (str(entry["step"]), str(entry["run_date"]))
        if str(entry.get("ts") or "") >= str(final.get(key, {}).get("ts") or ""):
            final[key] = entry
    if not final:
        return "", {}
    latest_run = max(run for _, run in final)
    by_step: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (step, run), entry in final.items():
        by_step[step].append((run, int(entry.get("exit_code", 0) or 0)))
    streaks: dict[str, int] = {}
    for step, runs in by_step.items():
        runs.sort(reverse=True)
        if runs[0][0] != latest_run:
            continue
        count = 0
        for _, code in runs:
            if code == 0:
                break
            count += 1
        if count:
            streaks[step] = count
    return latest_run, streaks


def morning_report_lines(
    log_path: Path | None = None,
    state_path: Path | None = None,
    log_dir: Path = PIPELINE_LOG_DIR,
) -> list[str]:
    """朝報の上部に置く警告ブロック（警告が無ければ修復件数の1行だけ）。"""
    log_path = log_path or DEFAULT_LOG
    state = _load_json(state_path or DEFAULT_STATE, {})
    run_date, streaks = consecutive_failure_days(log_path)
    last = state.get("last_run") if isinstance(state, dict) and isinstance(state.get("last_run"), dict) else {}
    if last.get("run_date") != run_date:
        last = {}
    recovered = last.get("recovered") or []
    quality = {item["step"]: item for item in last.get("quality_failures") or []}
    unrecovered = {item["step"]: item for item in last.get("unrecovered") or []}
    # 自動修復が走らなかった日でも、品質チェック失敗と連続失敗は手順ログから拾う
    for step in streaks:
        if step not in quality and step not in unrecovered and is_quality_check(step):
            quality[step] = {"step": step}
    long_streaks = {step for step, days in streaks.items() if days >= 2}
    log_file = pipeline_log_path(run_date, log_dir) if run_date else None
    log_text = _read_text(log_file) if log_file else ""

    def detail(step: str, item: dict[str, Any] | None = None) -> str:
        summary = (item or {}).get("summary") or summarize_error(step_log_excerpt(log_text, step))
        cause = (item or {}).get("cause")
        parts = [f"`{step}`", f"連続{streaks.get(step, 1)}日"]
        if cause:
            parts.append(f"原因: {cause}")
        parts.append(f"最後のエラー: {summary}")
        return " / ".join(parts)

    rec_line = f"- 🔧 パイプライン自動修復: {len(recovered)}件" + (
        "（" + ", ".join(f"{r['step']}←{r.get('cause', '')}" for r in recovered) + "）" if recovered else ""
    )
    warn: list[str] = []
    for step in sorted(quality):
        warn.append(f"> - 🧪 品質チェック失敗（劣化の疑い・再実行せず）: {detail(step, quality[step])}")
    for step in sorted(unrecovered):
        if step not in quality:
            warn.append(f"> - ❌ 自動修復できず: {detail(step, unrecovered[step])}")
    for step in sorted(long_streaks - set(quality) - set(unrecovered)):
        warn.append(f"> - 🔁 2日以上連続失敗: {detail(step)}")
    if not warn:
        return [rec_line]
    return [
        f"> [!warning] 日次パイプライン要確認（{run_date}）",
        *warn,
        f"> - ログ: `{log_file}`",
        "",
        rec_line,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="allowlisted recipesを実行する")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--run-date", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--generic-limit", type=int, default=5, help="汎用再実行の1日あたり上限")
    parser.add_argument("--pipeline-log", type=Path, default=None, help="その日の improvement_YYYYMMDD.log")
    args = parser.parse_args()

    root = args.root.resolve()
    log_path = args.log or root / "data" / "pipeline_step_log.jsonl"
    ledger_path = args.ledger or root / "api" / "rule_engine" / "ledger_rules.json"
    state_path = args.state or root / "data" / "pipeline_auto_recovery_state.json"
    report_path = args.report or root / "reports" / "pipeline_auto_recovery_latest.json"

    ledger = _load_json(ledger_path, [])
    ledger = ledger if isinstance(ledger, list) else []
    state = _load_json(state_path, {"attempts": {}})
    state = state if isinstance(state, dict) else {"attempts": {}}
    if not isinstance(state.get("attempts"), dict):
        state["attempts"] = {}

    catalog = build_recovery_catalog(ledger)
    eligible_steps = {
        str(recipe.get("step") or "")
        for recipe in catalog.get("recipes", [])
        if int(recipe.get("resolved_incident_count") or 0) > 0
    }
    plans = plan_recoveries(
        latest_step_results(log_path),
        state,
        args.run_date,
        eligible_steps=eligible_steps,
    )
    selected = plans[: max(0, args.limit)]
    results: list[dict[str, Any]] = []
    if args.apply:
        for plan in selected:
            result = execute_plan(plan, root=root, log_path=log_path, run_date=args.run_date)
            results.append(result)
            state["attempts"][plan["attempt_key"]] = {
                "attempted_at": dt.datetime.now().isoformat(timespec="seconds"),
                "recipe_id": plan["recipe_id"],
                "success": result["success"],
                "detail": result["detail"][:500],
            }

    triage = triage_and_retry(
        root=root,
        log_path=log_path,
        pipeline_log=args.pipeline_log or pipeline_log_path(args.run_date),
        state=state,
        run_date=args.run_date,
        apply=args.apply,
        limit=max(0, args.generic_limit),
    )
    triage["recovered"] = [
        {"step": r["step"], "cause": r["recipe_id"], "summary": r["detail"][:200]} for r in results if r.get("success")
    ] + triage["recovered"]
    attempted_steps = {r["step"] for r in results}
    triage["unrecovered"] = [u for u in triage["unrecovered"] if u["step"] not in attempted_steps] + [
        {"step": r["step"], "cause": r["recipe_id"], "summary": r["detail"][:200]} for r in results if not r.get("success")
    ]
    if args.apply:
        state["last_run"] = {
            "run_date": args.run_date,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            **triage,
        }
        _dump_json(state_path, state)

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "catalog": catalog,
        "planned_count": len(plans),
        "selected_count": len(selected),
        "plans": selected,
        "results": results,
        "triage": triage,
        "guardrails": {
            "allowlisted_recipes_only": True,
            "arbitrary_commands_from_history": False,
            "max_attempts_per_step_per_run_date": 1,
            "daily_limit": max(0, args.limit),
            "resolved_evidence_required": True,
            "generic_retry_allowlist": sorted(GENERIC_RETRY_STEPS),
            "generic_retry_requires_transient_log_evidence": True,
            "quality_checks_never_retried": True,
        },
    }
    _dump_json(report_path, report)
    print(
        f"pipeline_auto_recovery: planned={len(plans)} selected={len(selected)} "
        f"succeeded={sum(1 for item in results if item.get('success'))} "
        f"generic_recovered={len(triage['recovered'])} unrecovered={len(triage['unrecovered'])} "
        f"quality_failures={len(triage['quality_failures'])}"
    )
    if args.apply and any(not item.get("success") for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
