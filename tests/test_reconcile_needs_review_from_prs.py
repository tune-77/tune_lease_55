"""reconcile_needs_review_from_prs のテスト。

経路1（REV番号リンク）は確実に applied 化し、経路2（タイトル類似）は提案のみで
自動適用しないこと、真の未着手（出荷痕跡なし）を誤って閉じないことを保証する。
"""

from __future__ import annotations

import importlib
import json


def _load(monkeypatch, ledger_path):
    monkeypatch.setenv("LEDGER_PATH", str(ledger_path))
    import scripts.reconcile_needs_review_from_prs as mod

    importlib.reload(mod)
    return mod


def _write_ledger(path, entries):
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


# ── 類似度 ────────────────────────────────────────────────────────────────

def test_bigram_similarity_bounds(tmp_path, monkeypatch):
    mod = _load(monkeypatch, tmp_path / "l.jsonl")
    assert mod.bigram_similarity("紫苑の記憶", "紫苑の記憶") == 1.0
    assert mod.bigram_similarity("abcdef", "zyxwvu") == 0.0
    assert mod.bigram_similarity("", "なにか") == 0.0


# ── コア reconcile ────────────────────────────────────────────────────────

def test_rev_linked_is_applied_with_real_key(tmp_path, monkeypatch):
    mod = _load(monkeypatch, tmp_path / "l.jsonl")
    open_entries = [
        {"key": "misc_phase_b_rev162", "rev_id": "REV-162",
         "title": "GCS → ローカル同期スクリプト実装", "status": "needs_review"},
    ]
    res = mod.reconcile(open_entries, {"REV-162"}, ["feat: REV-162 GCS→ローカル同期スクリプト実装"], "now")
    assert len(res["rev_linked"]) == 1
    u = res["rev_linked"][0]
    assert u["key"] == "misc_phase_b_rev162"  # 実キーを再利用
    assert u["status"] == "applied"
    assert res["suggested"] == [] and res["no_trace"] == []


def test_rev_without_merged_pr_is_not_linked(tmp_path, monkeypatch):
    mod = _load(monkeypatch, tmp_path / "l.jsonl")
    open_entries = [
        {"key": "k1", "rev_id": "REV-999", "title": "全く別の未出荷案件XYZ", "status": "needs_review"},
    ]
    # REV-999 は merged 集合に無い、かつ類似 PR も無い → no_trace（誤 applied しない）
    res = mod.reconcile(open_entries, {"REV-162"}, ["feat: REV-162 別物"], "now")
    assert res["rev_linked"] == []
    assert len(res["no_trace"]) == 1


def test_title_match_is_suggestion_only(tmp_path, monkeypatch):
    mod = _load(monkeypatch, tmp_path / "l.jsonl")
    open_entries = [
        {"key": "misc_abc", "rev_id": "", "title": "知識宇宙マップの視覚化機能強化", "status": "needs_review"},
    ]
    res = mod.reconcile(
        open_entries, set(),
        ["feat: REV-022 知識宇宙マップの視覚化機能強化"], "now", sim_threshold=0.4,
    )
    # rev_id 無し → 経路1 対象外。高類似なので提案に入るが applied 化はしない。
    assert res["rev_linked"] == []
    assert len(res["suggested"]) == 1
    assert res["suggested"][0]["key"] == "misc_abc"


def test_genuine_backlog_is_left_untouched(tmp_path, monkeypatch):
    mod = _load(monkeypatch, tmp_path / "l.jsonl")
    open_entries = [
        {"key": "misc_persona", "rev_id": "",
         "title": "紫苑の使い方が画一的で、ユーザーが個性を出しにくい。", "status": "needs_review"},
    ]
    # 出荷 PR が無い（無関係タイトルのみ）→ rev_linked にも suggested にも入らない
    res = mod.reconcile(
        open_entries, {"REV-162"},
        ["feat: REV-162 GCS同期スクリプト実装", "fix: REV-201 個人記憶キャプチャのバグ修正"],
        "now", sim_threshold=0.45,
    )
    assert res["rev_linked"] == []
    assert res["suggested"] == []
    assert len(res["no_trace"]) == 1


# ── I/O ───────────────────────────────────────────────────────────────────

def test_load_open_entries_filters_resolved(tmp_path, monkeypatch):
    ledger = tmp_path / "l.jsonl"
    _write_ledger(ledger, [
        {"key": "kA", "rev_id": "REV-162", "title": "A", "status": "needs_review", "recorded_at": "2026-07-01T00:00:00"},
        {"key": "kB", "rev_id": "REV-001", "title": "B", "status": "applied", "recorded_at": "2026-07-01T00:00:00"},
        {"key": "kA", "rev_id": "REV-162", "title": "A", "status": "applied", "recorded_at": "2026-07-10T00:00:00"},
    ])
    mod = _load(monkeypatch, ledger)
    entries = mod.load_open_entries()
    keys = {e["key"] for e in entries}
    # kA は最新 applied、kB も applied → どちらも open ではない
    assert keys == set()


def test_apply_path_reuses_real_key_end_to_end(tmp_path, monkeypatch):
    ledger = tmp_path / "l.jsonl"
    _write_ledger(ledger, [
        {"key": "misc_phase_b_rev164", "rev_id": "REV-164",
         "title": "GCS書き込みロック機構実装", "status": "needs_review", "recorded_at": "2026-06-26T00:00:00"},
    ])
    mod = _load(monkeypatch, ledger)
    open_entries = mod.load_open_entries()
    res = mod.reconcile(open_entries, {"REV-164"}, ["feat: REV-164 GCS書き込みロック機構実装"], "2026-07-24T00:00:00")
    assert len(res["rev_linked"]) == 1
    for u in res["rev_linked"]:
        mod._append_ledger(u)
    # 追記後、同キーの最新は applied → open から外れる
    assert mod.load_open_entries() == []
