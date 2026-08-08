"""Tests for scripts/audit_ledger_applied_claims.py (根本対応③のバックフィル監査)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_ledger_applied_claims.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("audit_ledger_applied_claims", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ledger(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dry_run_does_not_modify_ledger(tmp_path, monkeypatch):
    mod = load_module()
    ledger_path = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger_path,
        [
            {"key": "k1", "status": "applied", "title": "PR無しの怪しい適用",
             "pr_url": "", "recorded_at": "2026-05-30T00:00:00"},
            {"key": "k2", "status": "applied", "title": "PRありの本物の適用",
             "pr_url": "https://example.com/pr/9", "recorded_at": "2026-05-30T00:00:00"},
        ],
    )
    monkeypatch.setattr(mod, "has_matching_commit", lambda title: False)

    result = mod.audit(ledger_path, apply=False)

    assert result["applied_keys"] == 2
    assert result["applied_without_pr_url"] == 1
    assert len(result["unverified"]) == 1
    assert result["unverified"][0]["key"] == "k1"
    # dry-run: ファイルは変更されない
    assert len(_read_lines(ledger_path)) == 2


def test_apply_writes_correction_for_unverified_only(tmp_path, monkeypatch):
    mod = load_module()
    ledger_path = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger_path,
        [
            {"key": "k1", "status": "applied", "title": "PR無しの怪しい適用",
             "pr_url": "", "recorded_at": "2026-05-30T00:00:00"},
            {"key": "k2", "status": "applied", "title": "コミットは見つかるがPR記録漏れ",
             "pr_url": "", "recorded_at": "2026-05-30T00:00:00"},
        ],
    )
    # k1: コミットなし(未検証扱い) / k2: コミットあり(対象外)
    monkeypatch.setattr(mod, "has_matching_commit", lambda title: "見つかる" in title)

    result = mod.audit(ledger_path, apply=True)

    assert [item["key"] for item in result["unverified"]] == ["k1"]
    assert result["corroborated_by_git_log"] == 1

    lines = _read_lines(ledger_path)
    assert len(lines) == 3  # 元の2件 + 訂正1件
    correction = lines[-1]
    assert correction["key"] == "k1"
    assert correction["status"] == "apply_failed"

    import sys
    sys.path.insert(0, str(SCRIPT_PATH.resolve().parents[1] / ".agents" / "skills" / "auto-improvement-pipeline"))
    import pipeline_ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "LEDGER_PATH", ledger_path)
    processed, _status = ledger_mod.is_processed("k1")
    assert processed is False  # 再評価可能に戻っている

    # k2 (corroborated) は訂正されず applied のまま
    processed2, status2 = ledger_mod.is_processed("k2")
    assert processed2 is True
    assert status2 == "applied"


def test_only_latest_status_per_key_is_considered(tmp_path, monkeypatch):
    mod = load_module()
    ledger_path = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger_path,
        [
            {"key": "k1", "status": "applied", "title": "後でneeds_reviewに変わった案件",
             "pr_url": "", "recorded_at": "2026-05-30T00:00:00"},
            {"key": "k1", "status": "needs_review", "title": "後でneeds_reviewに変わった案件",
             "pr_url": "", "recorded_at": "2026-06-01T00:00:00"},
        ],
    )
    monkeypatch.setattr(mod, "has_matching_commit", lambda title: False)

    result = mod.audit(ledger_path, apply=False)

    # 最新ステータスが needs_review なので applied 集計には含まれない
    assert result["applied_keys"] == 0
    assert result["unverified"] == []
