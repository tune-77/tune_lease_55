"""Phase 3（紫苑中心・改善ループ移行計画）: 事後検証ループ。

- P3-1: 台帳で解決した候補の outcome をトリアージ記録へ書き戻す（冪等・冗長同定）
- P3-3: 的中率（classified_by 別）・Overrule率・リードタイムの集計
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import analyze_shion_pm_quality as pm
from scripts.shion_triage import load_triage_latest


@pytest.fixture()
def root(tmp_path, monkeypatch):
    # 実行環境の実台帳（~/Library/...）に依存しないよう home を隔離
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "data").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _seed(root: Path, triage_rows: list[dict], ledger_rows: list[dict]) -> None:
    _write_jsonl(root / "data" / "shion_improvement_triage.jsonl", triage_rows)
    _write_jsonl(root / "scripts" / "improvement_ledger.jsonl", ledger_rows)


def test_sync_outcomes_by_key_and_item_id_idempotent(root):
    _seed(
        root,
        triage_rows=[
            {"canonical_key": "misc_a", "decision": "today", "classified_by": "user",
             "decided_at": "2026-07-15T10:00:00"},
            {"canonical_key": "misc_drift", "item_id": "REV-302", "decision": "today",
             "classified_by": "user", "decided_at": "2026-07-15T11:00:00"},
            {"canonical_key": "misc_open", "decision": "later", "classified_by": "user"},
        ],
        ledger_rows=[
            {"canonical_key": "misc_a", "rev_id": "REV-301", "status": "applied",
             "recorded_at": "2026-07-17T01:00:00"},
            # misc_drift はキーが台帳と食い違うが rev_id で解決できる（冗長同定）
            {"canonical_key": "misc_other", "rev_id": "REV-302", "status": "rejected",
             "recorded_at": "2026-07-17T01:00:00"},
            {"canonical_key": "misc_open", "rev_id": "REV-303", "status": "needs_review",
             "recorded_at": "2026-07-17T01:00:00"},
        ],
    )
    triage = load_triage_latest(root)
    by_key, by_rev = pm.load_ledger_statuses(root)

    updates = pm.sync_outcomes(root, triage, by_key, by_rev, apply=True)

    outcomes = {u["canonical_key"]: u["outcome"] for u in updates}
    assert outcomes == {"misc_a": "applied", "misc_drift": "rejected"}
    # needs_review は解決扱いにしない
    after = load_triage_latest(root)
    assert "outcome" not in after["misc_open"]

    # 冪等: 2回目は追記なし
    second = pm.sync_outcomes(root, after, by_key, by_rev, apply=True)
    assert second == []


def test_compute_kpis_hit_overrule_leadtime(root):
    triage = {
        "k1": {"canonical_key": "k1", "decision": "today", "classified_by": "user",
               "rule_decision": "today", "outcome": "applied",
               "decided_at": "2026-07-15T10:00:00", "outcome_recorded_at": "2026-07-17T10:00:00"},
        "k2": {"canonical_key": "k2", "decision": "today", "classified_by": "user",
               "rule_decision": "later", "outcome": "rejected"},  # overrule かつ外した
        "k3": {"canonical_key": "k3", "decision": "today", "classified_by": "rule",
               "rule_decision": "today", "outcome": "applied",
               "decided_at": "2026-07-14T10:00:00", "outcome_recorded_at": "2026-07-18T10:00:00"},
        "k4": {"canonical_key": "k4", "decision": "discard", "classified_by": "user",
               "rule_decision": "today"},  # overrule・未解決
    }

    kpis = pm.compute_kpis(triage)

    assert kpis["triage_total"] == 4
    assert kpis["hit_rates_by_classifier"]["user"] == {"resolved": 2, "applied": 1, "hit_rate": 0.5}
    assert kpis["hit_rates_by_classifier"]["rule"] == {"resolved": 1, "applied": 1, "hit_rate": 1.0}
    assert kpis["overrule"] == {"with_rule_decision": 4, "overruled": 2, "rate": 0.5}
    # (2日 + 4日) / 2 = 3.0
    assert kpis["lead_time_days_avg"] == 3.0


def test_main_writes_reports(root, monkeypatch, capsys):
    _seed(
        root,
        triage_rows=[
            {"canonical_key": "misc_a", "decision": "today", "classified_by": "user",
             "decided_at": "2026-07-15T10:00:00"},
        ],
        ledger_rows=[
            {"canonical_key": "misc_a", "rev_id": "REV-301", "status": "applied",
             "recorded_at": "2026-07-17T01:00:00"},
        ],
    )
    monkeypatch.setattr(pm, "repo_root", lambda: root)
    monkeypatch.setattr("sys.argv", ["analyze_shion_pm_quality.py", "--date", "2026-07-18"])

    assert pm.main() == 0

    latest = json.loads((root / "reports" / "shion_pm_quality_latest.json").read_text(encoding="utf-8"))
    assert latest["outcomes_synced"] == 1
    assert latest["kpis"]["hit_rates_by_classifier"]["user"]["hit_rate"] == 1.0
    assert (root / "reports" / "shion_pm_quality_20260718.json").exists()
    md = (root / "reports" / "shion_pm_quality_latest.md").read_text(encoding="utf-8")
    assert "的中率" in md


def test_main_skips_without_triage(root, monkeypatch, capsys):
    monkeypatch.setattr(pm, "repo_root", lambda: root)
    monkeypatch.setattr("sys.argv", ["analyze_shion_pm_quality.py"])

    assert pm.main() == 0
    assert not (root / "reports" / "shion_pm_quality_latest.json").exists()
