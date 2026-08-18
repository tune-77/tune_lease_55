# -*- coding: utf-8 -*-
"""
tests/test_system_guardrails.py
================================
system_guardrails.py の統計キャッシュ不整合検知（_audit_cache_consistency）のテスト。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import system_guardrails
import data_cases


def _cases(n: int) -> list[dict]:
    return [{"id": f"c{i}", "final_status": "成約"} for i in range(n)]


def test_audit_cache_consistency_matching_counts_no_issue(monkeypatch):
    all_cases = _cases(3)
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: {"source_case_count": 3})
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: {"source_case_count": 3})

    issues = system_guardrails._audit_cache_consistency(all_cases)
    assert issues == []


def test_audit_cache_consistency_detects_drift(monkeypatch):
    """既知バグ回帰: save_all_cases() が refresh_stats_caches() を呼ばないと
    キャッシュの件数と SQLite の実件数が食い違う。この乖離を error として検知すること。
    """
    all_cases = _cases(5)
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: {"source_case_count": 3})
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: {"source_case_count": 3})

    issues = system_guardrails._audit_cache_consistency(all_cases)
    drift_issues = [i for i in issues if i["kind"] == "cache_drift"]
    assert len(drift_issues) == 2
    assert all(i["severity"] == "error" for i in drift_issues)
    assert {i["cache"] for i in drift_issues} == {"dashboard_stats_cache", "department_stats_cache"}
    assert drift_issues[0]["cached_count"] == 3
    assert drift_issues[0]["live_count"] == 5


def test_audit_cache_consistency_missing_cache(monkeypatch):
    all_cases = _cases(1)
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: None)
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: {"source_case_count": 1})

    issues = system_guardrails._audit_cache_consistency(all_cases)
    missing = [i for i in issues if i["kind"] == "cache_missing"]
    assert len(missing) == 1
    assert missing[0]["cache"] == "dashboard_stats_cache"
    assert missing[0]["severity"] == "warn"


def test_audit_cache_consistency_outdated_format(monkeypatch):
    all_cases = _cases(1)
    monkeypatch.setattr(data_cases, "load_dashboard_stats_cache", lambda: {"generated_at": "2026-01-01"})
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: {"source_case_count": 1})

    issues = system_guardrails._audit_cache_consistency(all_cases)
    outdated = [i for i in issues if i["kind"] == "cache_outdated_format"]
    assert len(outdated) == 1
    assert outdated[0]["cache"] == "dashboard_stats_cache"
    assert outdated[0]["severity"] == "info"
