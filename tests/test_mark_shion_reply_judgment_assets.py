"""判断資産に混入した紫苑の返答に出所を付けて除外する（REV-498）。削除はしない。"""
import json

from scripts import mark_shion_reply_judgment_assets as mark


def _rule(rule_id, statement, *, evidence="manual://screening/chat:lease-intelligence-dialogue", status="active"):
    return {"id": rule_id, "status": status, "canonical_statement": statement, "evidence_paths": [evidence]}


def test_shion_reply_markers_are_detected_only_for_dialogue_rules():
    assert mark.classify(_rule("a", "稟議テンプレ 承知いたしました。 ### 稟議コメントテンプレート")) == "shion"
    assert mark.classify(_rule("b", "① 推論：リース料回収のシミュレーション 投資額は高額ですね。")) == "shion"
    assert mark.classify(_rule("c", "顧客対話戦略 承知しました。 ```markdown ---")) == "shion"
    # ユーザーの口語の教示はユーザー由来のまま
    assert mark.classify(_rule("d", "銀行と取引のない企業とは付き合わない 延滞がちな企業にもリース増額しない")) == "user"
    # 調査ノート由来（manual:// 以外）は対象外
    assert mark.classify(_rule("e", "承知しました。### 見出し", evidence="Projects/tune_lease_55/Research/x.md")) == "user"


def test_polite_or_soft_only_is_left_for_user_confirmation():
    assert mark.classify(_rule("f", "残価が持つ多面性を示唆 その通りだ 覚えといて")) == "uncertain"


def test_apply_marks_keeps_text_and_records_previous_status():
    rules = [_rule("a", "承知いたしました。### 判断資産：型"), _rule("d", "延滞がちな企業にはリース増額しない")]

    assert mark.apply_marks(rules, now="2026-10-08T12:00:00") == 1

    shion, user = rules
    assert shion["content_source"] == "shion"
    assert shion["status"] == mark.EXCLUDED_STATUS
    assert shion["status_before_exclusion"] == "active"
    assert shion["canonical_statement"] == "承知いたしました。### 判断資産：型"  # 本文は消さない
    assert user["status"] == "active" and "content_source" not in user
    assert mark.apply_marks(rules, now="2026-10-08T13:00:00") == 0  # 2回目は何もしない


def test_main_dry_run_does_not_write_and_apply_backs_up(tmp_path, monkeypatch, capsys):
    canonical = tmp_path / "canonical.json"
    store = {"summary": {"active_rules": 1}, "rules": [_rule("a", "承知いたしました。### 型")]}
    canonical.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mark, "LOCK_PATH", tmp_path / ".lock")
    backups = tmp_path / "backups"

    monkeypatch.setattr("sys.argv", ["x", "--canonical", str(canonical), "--backup-dir", str(backups)])
    assert mark.main() == 0
    assert json.loads(canonical.read_text(encoding="utf-8")) == store

    monkeypatch.setattr("sys.argv", ["x", "--canonical", str(canonical), "--backup-dir", str(backups), "--apply"])
    assert mark.main() == 0
    written = json.loads(canonical.read_text(encoding="utf-8"))
    assert written["rules"][0]["content_source"] == "shion"
    assert written["summary"]["active_rules"] == 0
    assert json.loads(next(backups.iterdir()).read_text(encoding="utf-8")) == store
