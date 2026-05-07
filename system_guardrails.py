"""
システム監査・監査ログ・運用ルール・自動通知の共通基盤。

目的:
- データ品質の自動監査
- セグメント運用ルールの管理
- 監査ログの一元化
- Slack / 通知キューへの自動通知
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "data"
_RULES_FILE = _DATA_DIR / "system_guardrails_rules.json"
_STATE_FILE = _DATA_DIR / "system_guardrails_state.json"
_RUNS_FILE = _DATA_DIR / "system_guardrails_audit.jsonl"

_DEFAULT_RULES: dict[str, Any] = {
    "min_cases_total": 50,
    "min_cases_per_dept": 8,
    "min_cases_per_industry": 40,
    "min_cases_new_customer": 40,
    "min_cases_existing_customer": 40,
    "notify_cooldown_minutes": 240,
    "notify_enabled": True,
}

_VALID_STATUSES = {"成約", "失注", "検収", "検収完了", "保留", "未登録"}
_VALID_CUSTOMER_TYPES = {"既存先", "新規先"}
_VALID_DEPTS = {"宇都宮営業部", "小山営業部", "足利営業部", "埼玉営業部"}
_INDUSTRY_BASES = ["全体", "医療", "運送業", "サービス業", "製造業"]


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_guardrail_rules() -> dict[str, Any]:
    rules = dict(_DEFAULT_RULES)
    stored = _load_json(_RULES_FILE, {})
    if isinstance(stored, dict):
        rules.update(stored)
    return rules


def save_guardrail_rules(rules: dict[str, Any]) -> bool:
    try:
        merged = dict(_DEFAULT_RULES)
        merged.update(rules or {})
        _save_json(_RULES_FILE, merged)
        return True
    except Exception:
        return False


def _append_run_log(record: dict) -> None:
    _RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_dept(value: object) -> str:
    dept = str(value or "").strip()
    return dept if dept in _VALID_DEPTS else ""


def _normalize_customer_type(value: object) -> str:
    customer_type = str(value or "").strip()
    return customer_type if customer_type in _VALID_CUSTOMER_TYPES else ""


def _industry_base_from_case(case: dict) -> str:
    result = case.get("result") or {}
    major = result.get("industry_major") or case.get("industry_major") or (case.get("inputs") or {}).get("industry_major") or ""
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "")
    if major_code == "H":
        return "運送業"
    if major_code == "P":
        return "医療"
    if major_code in ["I", "K", "M", "R"]:
        return "サービス業"
    if major_code == "E":
        return "製造業"
    return "全体"


def _safe_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        v = float(value)
        return v
    except (TypeError, ValueError):
        return None


def _load_closed_cases() -> list[dict]:
    from data_cases import load_all_cases

    cases = [c for c in load_all_cases() if c.get("final_status") in ("成約", "失注", "検収", "検収完了")]
    cases.sort(key=lambda c: str(c.get("timestamp") or c.get("final_result_date") or ""))
    return cases


def _audit_data_quality(cases: list[dict]) -> list[dict]:
    issues: list[dict] = []
    total = len(cases)
    if total == 0:
        return [{"severity": "error", "kind": "no_data", "message": "監査対象の成約/失注データがありません。"}]

    required_fields = {
        "final_status": lambda c: c.get("final_status"),
        "sales_dept": lambda c: c.get("sales_dept") or (c.get("inputs") or {}).get("sales_dept"),
        "industry_major": lambda c: c.get("industry_major") or (c.get("result") or {}).get("industry_major") or (c.get("inputs") or {}).get("industry_major"),
        "customer_type": lambda c: c.get("customer_type") or (c.get("inputs") or {}).get("customer_type"),
    }
    for field, getter in required_fields.items():
        missing = sum(1 for c in cases if not str(getter(c) or "").strip())
        if missing:
            issues.append({
                "severity": "warn" if missing < total * 0.2 else "error",
                "kind": "missing_field",
                "field": field,
                "count": missing,
                "message": f"{field} の欠損が {missing} 件あります。",
            })

    invalid_status = sum(1 for c in cases if c.get("final_status") not in _VALID_STATUSES)
    if invalid_status:
        issues.append({
            "severity": "error",
            "kind": "invalid_status",
            "count": invalid_status,
            "message": f"final_status の想定外値が {invalid_status} 件あります。",
        })

    missing_dept = 0
    invalid_dept = 0
    for c in cases:
        raw_dept = str(c.get("sales_dept") or (c.get("inputs") or {}).get("sales_dept") or "").strip()
        norm_dept = _normalize_dept(raw_dept)
        if norm_dept:
            continue
        if raw_dept in ("", "0", "未設定", "未読取"):
            missing_dept += 1
        else:
            invalid_dept += 1
    if missing_dept:
        issues.append({
            "severity": "info",
            "kind": "missing_dept",
            "count": missing_dept,
            "message": f"営業部未入力の案件が {missing_dept} 件あります。",
        })
    if invalid_dept:
        issues.append({
            "severity": "warn" if invalid_dept < total * 0.2 else "error",
            "kind": "invalid_dept",
            "count": invalid_dept,
            "message": f"営業部名の表記が想定外の案件が {invalid_dept} 件あります。",
        })

    invalid_customer = sum(1 for c in cases if _normalize_customer_type(c.get("customer_type") or (c.get("inputs") or {}).get("customer_type")) == "")
    if invalid_customer:
        issues.append({
            "severity": "warn",
            "kind": "invalid_customer_type",
            "count": invalid_customer,
            "message": f"customer_type が想定外の案件が {invalid_customer} 件あります。",
        })

    score_values = []
    borrower_values = []
    bench_values = []
    ind_values = []
    final_rates = []
    for c in cases:
        res = c.get("result") or {}
        for key in ("score", "score_borrower", "bench_score", "ind_score"):
            val = res.get(key)
            if isinstance(val, (int, float)):
                if key == "score":
                    score_values.append(float(val))
                elif key == "score_borrower":
                    borrower_values.append(float(val))
                elif key == "bench_score":
                    bench_values.append(float(val))
                elif key == "ind_score":
                    ind_values.append(float(val))
        rate = _safe_float(c.get("final_rate") or res.get("final_rate") or (c.get("inputs") or {}).get("final_rate"))
        if rate is not None:
            final_rates.append(rate)

    def _range_issue(values: list[float], name: str, lo: float, hi: float) -> dict | None:
        invalid = sum(1 for v in values if not (lo <= v <= hi))
        if invalid:
            return {
                "severity": "warn" if invalid < max(5, total * 0.05) else "error",
                "kind": "range_issue",
                "field": name,
                "count": invalid,
                "message": f"{name} が範囲外の値を持つ案件が {invalid} 件あります。",
            }
        return None

    for values, name in [
        (score_values, "score"),
        (borrower_values, "score_borrower"),
        (bench_values, "bench_score"),
        (ind_values, "ind_score"),
    ]:
        issue = _range_issue(values, name, 0.0, 100.0)
        if issue:
            issues.append(issue)

    if final_rates:
        issue = _range_issue(final_rates, "final_rate", 0.0, 100.0)
        if issue:
            issues.append(issue)

    if not borrower_values:
        issues.append({
            "severity": "error",
            "kind": "missing_score",
            "field": "score_borrower",
            "count": total,
            "message": "score_borrower が全件未入力です。",
        })

    return issues


def _build_segment_policy(cases: list[dict], rules: dict[str, Any]) -> dict[str, Any]:
    from collections import Counter

    dept_counter: Counter[str] = Counter()
    industry_counter: Counter[str] = Counter()
    customer_counter: Counter[str] = Counter()

    for c in cases:
        dept = _normalize_dept(c.get("sales_dept") or (c.get("inputs") or {}).get("sales_dept"))
        customer_type = _normalize_customer_type(c.get("customer_type") or (c.get("inputs") or {}).get("customer_type"))
        major = _industry_base_from_case(c)
        if dept:
            dept_counter[dept] += 1
        if customer_type:
            customer_counter[customer_type] += 1
        industry_counter[major] += 1

    dept_rows = []
    for dept in _VALID_DEPTS:
        n = dept_counter.get(dept, 0)
        dept_rows.append({
            "営業部": dept,
            "件数": n,
            "運用": "共通フォールバック" if n < int(rules.get("min_cases_per_dept", 8)) else "個別運用",
        })

    industry_rows = []
    by_industry: dict[str, int] = {}
    for key, cnt in industry_counter.items():
        by_industry[key] = cnt
    for name, n in sorted(by_industry.items(), key=lambda x: (-x[1], x[0])):
        industry_rows.append({
            "業種大分類": name,
            "件数": n,
            "運用": "共通フォールバック" if n < int(rules.get("min_cases_per_industry", 40)) else "個別運用",
        })

    customer_rows = []
    for customer_type in _VALID_CUSTOMER_TYPES:
        n = customer_counter.get(customer_type, 0)
        threshold = int(rules.get("min_cases_new_customer" if customer_type == "新規先" else "min_cases_existing_customer", 40))
        customer_rows.append({
            "取引区分": customer_type,
            "件数": n,
            "運用": "共通フォールバック" if n < threshold else "個別運用",
            "閾値": threshold,
        })

    triggers = [r for r in dept_rows + industry_rows + customer_rows if r.get("運用") == "共通フォールバック"]
    return {
        "dept_rows": dept_rows,
        "industry_rows": industry_rows,
        "customer_rows": customer_rows,
        "trigger_count": len(triggers),
        "triggers": triggers,
    }


def _build_summary(cases: list[dict], issues: list[dict], policy: dict[str, Any], rules: dict[str, Any], force: bool) -> dict[str, Any]:
    severe = sum(1 for i in issues if i.get("severity") == "error")
    warned = sum(1 for i in issues if i.get("severity") == "warn")
    total = len(cases)
    segments_triggered = policy.get("trigger_count", 0)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "force": bool(force),
        "n_cases": total,
        "issue_count": len(issues),
        "severe_count": severe,
        "warn_count": warned,
        "segment_trigger_count": segments_triggered,
        "rules": rules,
        "issues": issues,
        "segment_policy": policy,
    }


def _summary_hash(summary: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "issue_count": summary.get("issue_count", 0),
            "severe_count": summary.get("severe_count", 0),
            "segment_trigger_count": summary.get("segment_trigger_count", 0),
            "issues": summary.get("issues", []),
            "segment_policy": summary.get("segment_policy", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _notify_slack(summary: dict[str, Any]) -> bool:
    text_lines = [
        "【リースシステム監査】",
        f"案件数: {summary.get('n_cases', 0)}",
        f"重大: {summary.get('severe_count', 0)} / 警告: {summary.get('warn_count', 0)} / セグメントフォールバック: {summary.get('segment_trigger_count', 0)}",
    ]
    issues = summary.get("issues") or []
    for issue in issues[:5]:
        text_lines.append(f"- {issue.get('message')}")
    if summary.get("segment_policy", {}).get("triggers"):
        text_lines.append(f"運用フォールバック: {len(summary['segment_policy']['triggers'])}件")

    text = "\n".join(text_lines)

    try:
        from secret_manager import get_slack_webhook_url
        import requests

        webhook_url = get_slack_webhook_url()
        if webhook_url:
            resp = requests.post(webhook_url, json={"text": text}, timeout=10)
            if resp.status_code == 200 and resp.text == "ok":
                return True
    except Exception:
        pass

    try:
        from slack_notify import push_notification
        push_notification(text)
        return True
    except Exception:
        return False


def _maybe_notify(summary: dict[str, Any]) -> bool:
    state = _load_json(_STATE_FILE, {})
    now = datetime.now()
    last_hash = state.get("last_notify_hash")
    last_ts_raw = state.get("last_notify_ts")
    cooldown_minutes = int(summary.get("rules", {}).get("notify_cooldown_minutes", 240))

    should_notify = bool(summary.get("severe_count", 0) > 0 or summary.get("segment_trigger_count", 0) > 0)
    if not should_notify or not summary.get("rules", {}).get("notify_enabled", True):
        return False

    cur_hash = _summary_hash(summary)
    if last_hash == cur_hash and last_ts_raw:
        try:
            last_ts = datetime.fromisoformat(last_ts_raw)
            if now - last_ts < timedelta(minutes=cooldown_minutes):
                return False
        except Exception:
            pass

    sent = _notify_slack(summary)
    if sent:
        state["last_notify_hash"] = cur_hash
        state["last_notify_ts"] = now.isoformat(timespec="seconds")
        _save_json(_STATE_FILE, state)
    return sent


def run_system_guardrails(force: bool = False) -> dict[str, Any]:
    """データ監査・運用ルール・通知をまとめて実行する。"""
    cases = _load_closed_cases()
    rules = load_guardrail_rules()
    issues = _audit_data_quality(cases)
    policy = _build_segment_policy(cases, rules)
    summary = _build_summary(cases, issues, policy, rules, force)
    _append_run_log(summary)
    _save_json(_STATE_FILE, summary)
    summary["notified"] = _maybe_notify(summary)
    return summary


def get_system_guardrails_status() -> dict[str, Any]:
    return {
        "rules": load_guardrail_rules(),
        "state": _load_json(_STATE_FILE, {}),
    }


def render_system_guardrails_panel() -> None:
    """Streamlit用の監査パネル。"""
    try:
        import streamlit as st
        import pandas as pd
    except Exception:
        return

    status = get_system_guardrails_status()
    state = status.get("state") or {}
    rules = status.get("rules") or {}

    with st.container(border=True):
        st.markdown("#### 🛡️ システム監査・運用ルール")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("案件数", f"{state.get('n_cases', 0)}")
        c2.metric("重大", f"{state.get('severe_count', 0)}")
        c3.metric("警告", f"{state.get('warn_count', 0)}")
        c4.metric("フォールバック", f"{state.get('segment_trigger_count', 0)}")

        st.caption(f"最終実行: {state.get('ts', '未実行')}")

        with st.expander("運用ルール", expanded=False):
            cols = st.columns(2)
            new_rules = dict(rules)
            new_rules["min_cases_total"] = cols[0].number_input("最少案件数", min_value=0, value=int(rules.get("min_cases_total", 50)), step=1)
            new_rules["min_cases_per_dept"] = cols[1].number_input("営業部の最少件数", min_value=0, value=int(rules.get("min_cases_per_dept", 8)), step=1)
            cols2 = st.columns(2)
            new_rules["min_cases_per_industry"] = cols2[0].number_input("業種の最少件数", min_value=0, value=int(rules.get("min_cases_per_industry", 40)), step=1)
            new_rules["min_cases_new_customer"] = cols2[1].number_input("新規先の最少件数", min_value=0, value=int(rules.get("min_cases_new_customer", 40)), step=1)
            cols3 = st.columns(2)
            new_rules["min_cases_existing_customer"] = cols3[0].number_input("既存先の最少件数", min_value=0, value=int(rules.get("min_cases_existing_customer", 40)), step=1)
            new_rules["notify_cooldown_minutes"] = cols3[1].number_input("通知クールダウン(分)", min_value=0, value=int(rules.get("notify_cooldown_minutes", 240)), step=5)
            new_rules["notify_enabled"] = st.toggle("Slack/通知を有効化", value=bool(rules.get("notify_enabled", True)))
            if st.button("💾 ルールを保存", key="btn_save_guardrail_rules"):
                if save_guardrail_rules(new_rules):
                    st.success("運用ルールを保存しました。")
                    st.rerun()
                else:
                    st.error("保存に失敗しました。")

        c_run, c_refresh = st.columns(2)
        if c_run.button("▶ 監査を実行", key="btn_run_guardrails", type="primary"):
            with st.spinner("データ監査・運用ルール判定・通知を実行中..."):
                res = run_system_guardrails(force=True)
            st.success(
                f"実行完了: 重大 {res['severe_count']} 件 / 警告 {res['warn_count']} 件 / "
                f"フォールバック {res['segment_trigger_count']} 件"
            )
            st.rerun()
        if c_refresh.button("↻ 再読込", key="btn_refresh_guardrails"):
            st.rerun()

        if state.get("issues"):
            st.markdown("##### データ監査")
            issues_df = pd.DataFrame(state["issues"])
            if not issues_df.empty:
                st.dataframe(issues_df, width="stretch", hide_index=True)

        policy = state.get("segment_policy") or {}
        if policy.get("dept_rows") or policy.get("industry_rows") or policy.get("customer_rows"):
            st.markdown("##### セグメント運用ルール")
            dept_df = pd.DataFrame(policy.get("dept_rows") or [])
            ind_df = pd.DataFrame(policy.get("industry_rows") or [])
            cust_df = pd.DataFrame(policy.get("customer_rows") or [])
            if not dept_df.empty:
                st.write("営業部")
                st.dataframe(dept_df, width="stretch", hide_index=True)
            if not ind_df.empty:
                st.write("業種")
                st.dataframe(ind_df, width="stretch", hide_index=True)
            if not cust_df.empty:
                st.write("取引区分")
                st.dataframe(cust_df, width="stretch", hide_index=True)
