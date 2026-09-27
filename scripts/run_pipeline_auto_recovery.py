#!/usr/bin/env python3
"""Apply allowlisted pipeline recovery recipes learned from resolved incidents.

Resolved pipeline REV entries are used as evidence and reporting context, never as
commands.  Only deterministic handlers registered in ``RECOVERY_RECIPES`` may
change state.  Each step is attempted at most once per run date and must verify
success before a healthy step result is appended to the structured pipeline log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    summary = load_index_summary(index_path)
    previous = load_memory_health_state(state_path)
    if summary is None or int(summary.get("total") or 0) <= 0:
        return False, "記憶インデックスが読めないか0件のため自動移行しない"
    if isinstance(previous.get("by_layer"), dict) and previous.get("by_layer"):
        return False, "層別基準は移行済み。実データ減少の可能性があるため自動修正しない"
    previous_total = int(previous.get("total") or 0)
    if previous_total <= 0:
        return False, "旧基準が存在しないため通常ヘルスチェックに委ねる"

    current_total = int(summary.get("total") or 0)
    drop_records = max(0, previous_total - current_total)
    drop_ratio = drop_records / previous_total
    if drop_records > 100 or drop_ratio > 0.3:
        return False, (
            f"旧基準から{drop_records}件（{drop_ratio:.1%}）減少しており"
            "通常ヘルスチェックの安全閾値を超えるため、"
            "自動移行しない"
        )

    layers = summary.get("by_layer") if isinstance(summary.get("by_layer"), dict) else {}
    stable_total = sum(int(layers.get(name) or 0) for name in ("long_term", "persistent", "retrieval"))
    if stable_total <= 0:
        return False, "永続層が0件のため基準移行を拒否"

    save_memory_health_state(state_path, summary)
    healthy, message = check_memory_health(summary, summary)
    return healthy, f"層別基準へ移行して再検証: {message}"


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
        _dump_json(state_path, state)

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "catalog": catalog,
        "planned_count": len(plans),
        "selected_count": len(selected),
        "plans": selected,
        "results": results,
        "guardrails": {
            "allowlisted_recipes_only": True,
            "arbitrary_commands_from_history": False,
            "max_attempts_per_step_per_run_date": 1,
            "daily_limit": max(0, args.limit),
            "resolved_evidence_required": True,
        },
    }
    _dump_json(report_path, report)
    print(
        f"pipeline_auto_recovery: planned={len(plans)} selected={len(selected)} "
        f"succeeded={sum(1 for item in results if item.get('success'))}"
    )
    if args.apply and any(not item.get("success") for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
