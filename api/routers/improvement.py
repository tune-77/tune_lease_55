"""改善ループ・トリアージルーター (REV-234 Phase2)"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cloudrun_writeback import record_cloudrun_input_event
from api.schemas import ReviewImprovementRequest, PromptRuleRegisterRequest

router = APIRouter(tags=["improvement"])

# ── リポジトリルート (main.py と同じ解決方法) ─────────────────────────────────
_SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)
_REPO_ROOT = str(Path(_SCRIPT_DIR).parent)


# ── _latest_improvement_report_path 依存ヘルパー ────────────────────────────────
def _candidate_report_dirs() -> list[Path]:
    bundle_root = Path(os.environ.get("CLOUDRUN_BUNDLE_DIR") or (Path(_REPO_ROOT) / ".cloudrun_bundle"))
    dirs = [Path(_REPO_ROOT) / "reports", bundle_root / "reports"]
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def _latest_improvement_report_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        latest = reports_dir / "latest.json"
        if latest.exists():
            return latest
    candidates: list[Path] = []
    for reports_dir in _candidate_report_dirs():
        candidates.extend(reports_dir.glob("improvement_report_*.json"))
    candidates = sorted(candidates)
    return candidates[-1] if candidates else None


class DismissImprovementRequest(BaseModel):
    key: str
    title: str


def _save_improvement_log_to_obsidian(entry: dict) -> dict:
    """Mirror an improvement ledger entry to the user's Obsidian vault."""
    status = str(entry.get("status") or "")
    title = str(entry.get("title") or "改善ログ")
    lines = [
        f"- status: {status or 'unknown'}",
        f"- title: {title}",
        f"- key: {entry.get('key') or ''}",
        f"- canonical_key: {entry.get('canonical_key') or entry.get('key') or ''}",
        f"- reason: {entry.get('reason') or ''}",
        f"- recorded_at: {entry.get('recorded_at') or ''}",
        "- source: improvement-log API",
    ]
    body = "\n".join(lines)
    try:
        from mobile_app.obsidian_bridge import append_improvement_note

        return append_improvement_note(f"改善ログ: {status or 'recorded'}", body)
    except Exception as exc:
        return {"status": "skipped", "reason": f"Obsidian save failed: {exc}"}


def _save_prompt_rule_to_obsidian(entry: dict) -> dict:
    """Mirror a prompt-rule registration to the user's Obsidian vault."""
    title = str(entry.get("title") or "PDCA修正登録")
    lines = [
        f"- source: {entry.get('source') or 'manual'}",
        f"- surface: {entry.get('surface') or ''}",
        f"- title: {title}",
        f"- rule: {entry.get('rule') or ''}",
        f"- reason: {entry.get('reason') or ''}",
        f"- recorded_at: {entry.get('recorded_at') or ''}",
        "- source: prompt-feedback API",
    ]
    body = "\n".join(lines)
    try:
        from mobile_app.obsidian_bridge import append_improvement_note

        return append_improvement_note(f"PDCA修正登録: {title}", body)
    except Exception as exc:
        return {"status": "skipped", "reason": f"Obsidian save failed: {exc}"}


@router.post("/api/improvement-log/dismiss")
def dismiss_improvement(req: DismissImprovementRequest):
    """改善案を「実装済み」としてledgerとObsidianに追記する。"""
    import json as _json
    from datetime import datetime as _dt
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    entry = {
        "key": req.key,
        "status": "applied",
        "title": req.title,
        "pr_url": "",
        "reason": "UI経由で手動実装済みマーク",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_improvement_log_to_obsidian(entry)
    writeback = record_cloudrun_input_event(
        event_type="improvement_dismiss",
        surface="improvement_log",
        payload={**entry, "obsidian": obsidian_result},
    )
    return {"ok": True, "key": req.key, "title": req.title, "obsidian": obsidian_result, "writeback": writeback}


@router.post("/api/improvement-log/delete")
def delete_improvement(req: DismissImprovementRequest):
    """改善案を一覧から削除する。監査できるよう ledger/GCS には deleted として残す。"""
    import json as _json
    from datetime import datetime as _dt
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    entry = {
        "key": req.key,
        "canonical_key": req.key,
        "status": "deleted",
        "title": req.title,
        "pr_url": "",
        "reason": "UI経由で改善ログから削除",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    writeback = record_cloudrun_input_event(
        event_type="improvement_delete",
        surface="improvement_log",
        payload=entry,
    )
    return {"ok": True, "key": req.key, "title": req.title, "writeback": writeback}


@router.post("/api/improvement-log/review")
def review_improvement(req: ReviewImprovementRequest):
    """改善案の承認・却下・deferredをledgerとObsidianに書き込む（REV-039）。"""
    import json as _json
    from datetime import datetime as _dt
    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    entry = {
        "key": req.key,
        "canonical_key": req.key,
        "status": req.action,
        "title": req.title,
        "pr_url": "",
        "reason": req.reason or f"UI経由で{req.action}",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_improvement_log_to_obsidian(entry)
    writeback = record_cloudrun_input_event(
        event_type="improvement_review",
        surface="improvement_log",
        payload={**entry, "obsidian": obsidian_result},
    )
    return {"ok": True, "key": req.key, "action": req.action, "obsidian": obsidian_result, "writeback": writeback}


# ── Phase 1: トリアージの構造化記録（planning/shion_improvement_loop_plan.md）──
# 判断は data/shion_improvement_triage.jsonl へ追記形式で記録し、最後のエントリが有効。
# 同定は canonical_key 単独に頼らず source_event_id とタイトルスナップショットも冗長に持つ。

_TRIAGE_DECISIONS = {"today", "later", "discard"}
_TRIAGE_CLASSIFIERS = {"rule", "llm", "user"}
_TRIAGE_DECISION_LABELS = {"today": "修正", "later": "保留", "discard": "却下"}


class ImprovementTriageRequest(BaseModel):
    canonical_key: str
    decision: str  # today / later / discard
    title: str = ""
    item_id: str = ""  # 改善レポート上のID（REV-XXX等）。Codexキューとの突き合わせ用
    source_event_id: str = ""
    reason: str = ""
    rule_decision: str = ""  # ルール分類（classifyPmImprovementItems）の初期値
    classified_by: str = "user"  # rule / llm / user


def _improvement_triage_path() -> Path:
    return Path(_REPO_ROOT) / "data" / "shion_improvement_triage.jsonl"


def _load_improvement_triage_latest() -> dict[str, dict]:
    """トリアージ記録を canonical_key ごとに解決する（最後のエントリが有効）。"""
    path = _improvement_triage_path()
    if not path.exists():
        return {}
    latest: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("canonical_key") or "")
            if key:
                latest[key] = row
    except OSError:
        return {}
    return latest


@router.post("/api/improvement/triage")
def record_improvement_triage(req: ImprovementTriageRequest):
    """改善候補のトリアージ判断（今日やる/後回し/捨てる）を構造化記録する（P1-1）。

    記録のみで、パイプラインの実行順序には影響しない（接続は Phase 2・シャドーモード経由）。
    """
    from datetime import datetime as _dt

    decision = str(req.decision or "").strip().lower()
    if decision not in _TRIAGE_DECISIONS:
        raise HTTPException(status_code=422, detail=f"decision は {sorted(_TRIAGE_DECISIONS)} のいずれかにしてください")
    classified_by = str(req.classified_by or "user").strip().lower()
    if classified_by not in _TRIAGE_CLASSIFIERS:
        raise HTTPException(status_code=422, detail=f"classified_by は {sorted(_TRIAGE_CLASSIFIERS)} のいずれかにしてください")
    canonical_key = str(req.canonical_key or "").strip()
    if not canonical_key:
        raise HTTPException(status_code=422, detail="canonical_key は空にできません")
    rule_decision = str(req.rule_decision or "").strip().lower()
    if rule_decision and rule_decision not in _TRIAGE_DECISIONS:
        rule_decision = ""
    record = {
        "canonical_key": canonical_key,
        "item_id": str(req.item_id or "").strip(),
        "source_event_id": str(req.source_event_id or "").strip(),
        "title": str(req.title or "").strip()[:120],
        "decision": decision,
        "rule_decision": rule_decision,
        "classified_by": classified_by,
        "reason": str(req.reason or "").strip()[:200],
        "decided_at": _dt.now().isoformat(timespec="seconds"),
    }
    path = _improvement_triage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _log_triage_action(record)
    return {"ok": True, "record": record}


def _log_triage_action(record: dict) -> None:
    """Agent Action Ledger（backlog §9.2）への記録。失敗しても本処理は止めない。"""
    try:
        from api.shion_action_ledger import log_action

        decision = str(record.get("decision") or "")
        title = str(record.get("title") or record.get("canonical_key") or "")
        log_action(
            "improvement_classified",
            summary=f"改善候補「{title}」を「{decision}」に分類",
            observed_sources=["data/shion_improvement_triage.jsonl"],
            risk_level="low",
            requires_user_approval=False,
            target=str(record.get("canonical_key") or ""),
            result=decision,
        )
    except Exception:
        pass


@router.get("/api/improvement/triage")
def get_improvement_triage():
    """トリアージの現在状態（キーごとに最後の判断）を返す。"""
    latest = _load_improvement_triage_latest()
    records = sorted(
        latest.values(),
        key=lambda row: str(row.get("decided_at") or ""),
        reverse=True,
    )
    counts: dict[str, int] = {}
    for row in records:
        decision = str(row.get("decision") or "")
        counts[decision] = counts.get(decision, 0) + 1
    return {"records": records[:100], "counts": counts}


def _build_codex_request_draft(record: dict) -> str:
    """実装承認済み候補の紫苑依頼文テンプレを生成する（下書きのみ・実行しない）。"""
    item_id = str(record.get("item_id") or "").strip()
    title = str(record.get("title") or record.get("canonical_key") or "").strip()
    reason = str(record.get("reason") or "").strip()
    head = f"{item_id} {title}".strip()
    lines = [
        f"紫苑依頼文: {head} を小さく実装してください。",
        f"- 目的: {title}" + (f"（{reason}）" if reason else ""),
        "- 変更範囲: 対象ファイルを最小限に。DB/API分岐/スコアリング/認証/デプロイ設定には触らない",
        "- 検証: python -m py_compile と対象テスト。フロント変更時は cd frontend && npx tsc --noEmit",
        "- data/・models/・.claude/state・.streamlit/secrets.toml は変更禁止",
        "- gitship/deploy の要否は User が判断する（自動実行しない）",
    ]
    return "\n".join(lines)


class ImprovementTriageApproveRequest(BaseModel):
    canonical_key: str


@router.post("/api/improvement/triage/approve")
def approve_improvement_triage(req: ImprovementTriageApproveRequest):
    """「今日やる」確定済みの候補に実装承認を記録する（P2-3）。

    承認レコード（approved_at）がある候補だけが紫苑依頼文生成・
    ライブモードのキュー最優先の対象になる。記録は同じ triage jsonl への
    追記で行い、最後のエントリが有効。
    """
    from datetime import datetime as _dt

    canonical_key = str(req.canonical_key or "").strip()
    if not canonical_key:
        raise HTTPException(status_code=422, detail="canonical_key は空にできません")
    latest = _load_improvement_triage_latest()
    current = latest.get(canonical_key)
    if not current:
        raise HTTPException(status_code=404, detail=f"トリアージ記録が見つかりません: {canonical_key}")
    if str(current.get("decision") or "") != "today":
        raise HTTPException(
            status_code=422,
            detail="実装承認は「今日やる」確定済みの候補にのみ記録できます",
        )
    record = dict(current)
    record["approved_at"] = _dt.now().isoformat(timespec="seconds")
    record["codex_request_draft"] = _build_codex_request_draft(record)
    path = _improvement_triage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        from api.shion_action_ledger import log_action

        log_action(
            "user_decision_recorded",
            summary=f"「{record.get('title') or record.get('canonical_key')}」の実装承認をUserが記録",
            observed_sources=["data/shion_improvement_triage.jsonl"],
            risk_level="low",
            requires_user_approval=True,
            user_approved=True,
            target=str(record.get("canonical_key") or ""),
            result="approved",
        )
    except Exception:
        pass
    return {"ok": True, "record": record}


def _action_ledger_report_path() -> Path | None:
    for reports_dir in _candidate_report_dirs():
        path = reports_dir / "agent_action_ledger_latest.json"
        if path.exists():
            return path
    return None


@router.get("/api/shion/action-ledger/summary")
def get_shion_action_ledger_summary(days: int = 7):
    """紫苑 Agent Action Ledger（backlog §9.2）の要約を返す（対話室UI・監査用、read-only）。

    直近レポート（run_daily_improvement_post.sh 生成）があればそれを使い、
    無ければ生ログから即時集計する。実行権限や承認とは無関係の可視化のみ。
    """
    from api.shion_action_ledger import read_actions, summarize

    report_path = _action_ledger_report_path()
    if report_path and report_path.exists():
        try:
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and int(cached.get("days") or 0) == days:
                return cached
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return summarize(read_actions(), days)


class MonitorReportRequest(BaseModel):
    failed_count: int = 0
    source: str = "dialogue_daily_report"


@router.post("/api/improvement/monitor-report")
def record_monitor_report(req: MonitorReportRequest):
    """紫苑のシステム監視報告がUserに表示された時刻を記録する（監視先行率KPI用）。

    notify_pipeline_alerts.py の Slack 通知時刻（data/pipeline_alert_notify_log.jsonl）
    と突き合わせて analyze_shion_pm_quality.py が監視先行率を算出する。
    """
    from datetime import datetime as _dt

    record = {
        "ts": _dt.now().isoformat(timespec="seconds"),
        "failed_count": max(0, int(req.failed_count or 0)),
        "source": str(req.source or "dialogue_daily_report")[:40],
    }
    path = Path(_REPO_ROOT) / "data" / "shion_monitor_report_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "record": record}


@router.post("/api/prompt-feedback/rules/register")
def register_prompt_rule(req: PromptRuleRegisterRequest):
    """UIから1クリックで修正ルールを登録する。

    強いPDCAルールに該当しないものは live prompt へ入れず、改善ログへ隔離する。
    """
    import json as _json
    from datetime import datetime as _dt
    from prompt_feedback import append_pdca_rule
    from memory_promotion_policy import classify_memory_destination, is_pdca_rule_candidate

    ledger_path = os.path.expanduser("~/Library/Logs/tunelease/ledger.jsonl")
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    rule_text = str(req.rule or "").strip()
    title = str(req.title or "").strip()
    reason = str(req.reason or "").strip()
    normalized_rule = rule_text or title
    destination = classify_memory_destination(normalized_rule)
    ledger_key = str(req.canonical_key or req.key or req.surface or title).strip()

    if not is_pdca_rule_candidate(normalized_rule):
        entry = {
            "key": ledger_key,
            "canonical_key": ledger_key,
            "status": "rule_review",
            "title": title or normalized_rule[:60],
            "rule": normalized_rule,
            "reason": reason or "PDCAルール条件外のため改善ログへ隔離",
            "source": req.source or "manual",
            "surface": req.surface or "",
            "destination": destination,
            "recorded_at": _dt.now().isoformat(),
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        obsidian_result = _save_improvement_log_to_obsidian(entry)
        return {
            "ok": True,
            "appended": False,
            "routed_to": "improvement_log",
            "reason": "pdca_rule_conditions_not_met",
            "rule": normalized_rule,
            "obsidian": obsidian_result,
        }

    append_result = append_pdca_rule(
        normalized_rule,
        source=req.source or "manual",
        reflection_summary=req.summary or None,
        ttl_days=90,
    )
    entry = {
        "key": ledger_key,
        "canonical_key": ledger_key,
        "status": "rule_registered",
        "title": title,
        "rule": normalized_rule,
        "reason": reason or normalized_rule,
        "source": req.source or "manual",
        "surface": req.surface or "",
        "destination": "pdca_rule",
        "recorded_at": _dt.now().isoformat(),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    obsidian_result = _save_prompt_rule_to_obsidian(entry)
    return {
        "ok": True,
        "appended": append_result.get("appended", False),
        "count": append_result.get("count", 0),
        "rule": append_result.get("rule", normalized_rule),
        "path": append_result.get("path"),
        "obsidian": obsidian_result,
    }


@router.get("/api/improvement-pipeline/summary")
def get_improvement_pipeline_summary():
    """最新パイプライン実行のサマリーを返す（REV-039）。latest.json直読み。"""
    import json as _json, re as _re
    path = _latest_improvement_report_path()
    if path is None:
        return {"run_date": None, "applied_count": 0, "needs_review_count": 0, "failed_count": 0, "commit_result": None}
    try:
        with open(path, encoding="utf-8") as f:
            report = _json.load(f)
    except Exception:
        return {"run_date": None, "applied_count": 0, "needs_review_count": 0, "failed_count": 0, "commit_result": None}
    run_date: str | None = None
    m = _re.search(r"(\d{8})", path.name)
    if m:
        d = m.group(1)
        run_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return {
        "run_date": run_date,
        "applied_count": report.get("applied_count", 0),
        "needs_review_count": report.get("needs_review_count", len(report.get("needs_review", []))),
        "failed_count": report.get("failed_count", 0),
        "commit_result": report.get("commit_result"),
    }
