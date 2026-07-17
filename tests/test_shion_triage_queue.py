"""Phase 2（紫苑中心・改善ループ移行計画）: トリアージ→パイプライン接続。

- P2-0: シャドーモードでは実キューは従来のまま、比較結果だけ出力する
- P2-1: ライブモードで「捨てる」除外・「今日やる（承認済み）」優先
- P2-2: User確定の「捨てる」は自動承認しない
- P2-4: off モードでトリアージの影響を即時に切り戻せる
"""

from __future__ import annotations

import json

import pytest

from scripts import build_codex_auto_queue as queue_builder
from scripts.shion_triage import (
    is_approved_today,
    is_user_discarded,
    load_triage_latest,
    resolve_triage_mode,
    triage_record_for_item,
)


@pytest.fixture(autouse=True)
def _no_policy_refresh(monkeypatch):
    # レポート内の auto_fix_policy をそのまま使う（再判定で書き換えない）
    monkeypatch.setattr(queue_builder, "evaluate_auto_fix_policy", None)


def _safe_item(rev_id: str, title: str, order: int, canonical_key: str = "") -> dict:
    return {
        "id": rev_id,
        "title": title,
        "canonical_key": canonical_key or f"key_{rev_id.lower()}",
        "recommended_order": order,
        "auto_fix_policy": {"auto_fix_allowed": True, "risk": "low", "max_files": 1},
    }


def _report() -> dict:
    return {
        "date": "2026-07-17",
        "needs_review": [
            _safe_item("REV-301", "表示ラベルの整理", 1),
            _safe_item("REV-302", "入力ヒントの文言修正", 2),
            _safe_item("REV-303", "ホーム画面の参照導線", 3),
        ],
    }


def _triage(records: list[dict]) -> dict[str, dict]:
    return {str(r["canonical_key"]): r for r in records}


def test_resolve_triage_mode_priority(monkeypatch):
    monkeypatch.delenv("SHION_TRIAGE_QUEUE_MODE", raising=False)
    assert resolve_triage_mode() == "shadow"
    monkeypatch.setenv("SHION_TRIAGE_QUEUE_MODE", "live")
    assert resolve_triage_mode() == "live"
    assert resolve_triage_mode("off") == "off"  # CLI が環境変数より優先
    monkeypatch.setenv("SHION_TRIAGE_QUEUE_MODE", "invalid")
    assert resolve_triage_mode() == "shadow"


def test_shadow_mode_keeps_baseline_queue_but_reports_divergence():
    triage = _triage([
        {"canonical_key": "key_rev-301", "decision": "discard", "classified_by": "user"},
        {"canonical_key": "key_rev-303", "decision": "today", "classified_by": "user",
         "approved_at": "2026-07-17T10:00:00"},
    ])

    queue = queue_builder.build_queue(_report(), limit=2, triage=triage, triage_mode="shadow")

    # 実キューは従来のまま（P2-0）
    assert [item["id"] for item in queue["items"]] == ["REV-301", "REV-302"]
    info = queue["triage"]
    assert info["mode"] == "shadow"
    assert info["applied"] is False
    assert info["baseline_ids"] == ["REV-301", "REV-302"]
    assert info["with_triage_ids"] == ["REV-303", "REV-302"]
    assert info["diverges"] is True
    assert [x["id"] for x in info["excluded_by_discard"]] == ["REV-301"]
    assert info["promoted_today_ids"] == ["REV-303"]


def test_live_mode_applies_triage_ordering_and_exclusion():
    triage = _triage([
        {"canonical_key": "key_rev-301", "decision": "discard", "classified_by": "user"},
        {"canonical_key": "key_rev-303", "decision": "today", "classified_by": "user",
         "approved_at": "2026-07-17T10:00:00"},
    ])

    queue = queue_builder.build_queue(_report(), limit=2, triage=triage, triage_mode="live")

    assert [item["id"] for item in queue["items"]] == ["REV-303", "REV-302"]
    assert queue["triage"]["applied"] is True
    first = queue["items"][0]
    assert first["triage_decision"] == "today"
    assert first["user_approved"] is True


def test_off_mode_ignores_triage_entirely():
    triage = _triage([
        {"canonical_key": "key_rev-301", "decision": "discard", "classified_by": "user"},
    ])

    queue = queue_builder.build_queue(_report(), limit=2, triage=triage, triage_mode="off")

    assert [item["id"] for item in queue["items"]] == ["REV-301", "REV-302"]
    assert queue["triage"]["mode"] == "off"
    assert "baseline_ids" not in queue["triage"]


def test_unapproved_today_ranks_below_approved_but_above_others():
    triage = _triage([
        {"canonical_key": "key_rev-303", "decision": "today", "classified_by": "user"},  # 未承認
        {"canonical_key": "key_rev-302", "decision": "today", "classified_by": "user",
         "approved_at": "2026-07-17T09:00:00"},
    ])

    queue = queue_builder.build_queue(_report(), limit=3, triage=triage, triage_mode="live")

    assert [item["id"] for item in queue["items"]] == ["REV-302", "REV-303", "REV-301"]


def test_triage_record_matches_by_item_id_fallback():
    latest = {
        "misc_drifted_key": {"canonical_key": "misc_drifted_key", "item_id": "REV-301",
                             "decision": "discard", "classified_by": "user"},
    }
    # canonical_key がドリフトして一致しなくても item_id で照合できる（冗長同定）
    item = {"id": "REV-301", "canonical_key": "misc_new_key"}
    record = triage_record_for_item(latest, item)
    assert record is not None
    assert is_user_discarded(record)


def test_rule_classified_discard_does_not_suppress():
    # 自動承認の抑制は User 確定の「捨てる」のみ（ルール分類の初期値では抑制しない）
    assert is_user_discarded({"decision": "discard", "classified_by": "rule"}) is False
    assert is_approved_today({"decision": "today"}) is False
    assert is_approved_today({"decision": "today", "approved_at": "2026-07-17T10:00:00"}) is True


def test_auto_approve_suppressed_by_user_discard():
    from scripts.auto_approve_safe_recipes import is_auto_approvable

    recipe = {
        "rev": "REV-301",
        "shion_recommendation": "auto",
        "risk_level": "low",
        "files": [{"path": "frontend/src/app/home/page.tsx"}],
    }
    triage = {
        "misc_x": {"canonical_key": "misc_x", "item_id": "REV-301",
                   "decision": "discard", "classified_by": "user"},
    }

    ok, reason = is_auto_approvable(recipe, triage)
    assert ok is False
    assert "捨てる" in reason

    # トリアージなしなら従来どおり承認可能
    ok, _ = is_auto_approvable(recipe, {})
    assert ok is True


def test_load_triage_latest_last_entry_wins(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"canonical_key": "misc_a", "decision": "later"},
        {"canonical_key": "misc_a", "decision": "today"},
        "壊れた行",
    ]
    (data_dir / "shion_improvement_triage.jsonl").write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    latest = load_triage_latest(tmp_path)

    assert latest["misc_a"]["decision"] == "today"
    assert load_triage_latest(tmp_path / "nonexistent") == {}
