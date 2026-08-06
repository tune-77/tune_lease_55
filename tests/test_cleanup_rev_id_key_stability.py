"""cleanup_improvement_reviews の REV番号→canonical_key 安定化のテスト。

REV_TITLES に載っていない REV 番号は、reconciliation時に rev_id 文字列自体を
タイトル代わりに canonical_key を計算していたため、起票時の実キーと食い違う
孤立キーへ "applied" が書き込まれ、元のエントリが needs_review のまま
取り残される事故が起きていた（実例: REV-230, REV-237）。
_get_rev_id_first_seen_key() が台帳上の「その REV 番号が最初に記録された
ときの key」を返し、reconciliation がそれを再利用することで解決する。
"""

from __future__ import annotations

import importlib
import json


def _load_module(monkeypatch, ledger_path):
    monkeypatch.setenv("LEDGER_PATH", str(ledger_path))
    import scripts.cleanup_improvement_reviews as mod

    importlib.reload(mod)
    return mod


def _write_ledger(path, entries):
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


def test_first_seen_key_is_the_original_entry_not_the_orphan(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    # REV-237 の実例を再現: 起票時は実タイトルで misc_64ce70442596、
    # 後の reconciliation が rev_id 文字列だけから孤立キー misc_665e5e075fed を生成。
    _write_ledger(ledger, [
        {
            "key": "misc_64ce70442596",
            "rev_id": "REV-237",
            "status": "needs_review",
            "title": "ユーザー関心に基づく能動的改善提案",
            "recorded_at": "2026-08-01T01:31:11",
        },
        {
            "key": "misc_665e5e075fed",
            "rev_id": "REV-237",
            "status": "applied",
            "title": "REV-237",
            "recorded_at": "2026-08-01T10:49:58",
        },
    ])
    mod = _load_module(monkeypatch, ledger)

    assert mod._get_rev_id_first_seen_key()["REV-237"] == "misc_64ce70442596"


def test_reconciliation_reuses_first_seen_key_instead_of_orphaning(tmp_path, monkeypatch):
    """REV_TITLES に無い REV 番号でも、実キーへ "applied" が書かれる（孤立キーを作らない）。"""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        {
            "key": "misc_6be43f8b3dce",
            "rev_id": "REV-230",
            "status": "needs_review",
            "title": "REV-230",
            "recorded_at": "2026-07-25T21:56:41",
        },
    ])
    mod = _load_module(monkeypatch, ledger)
    assert "REV-230" not in mod.REV_TITLES  # 前提: 未登録REVであること

    rev_id_first_key = mod._get_rev_id_first_seen_key()
    title = mod.REV_TITLES.get("REV-230", "REV-230")
    ck = rev_id_first_key.get("REV-230") or mod._canonical_key(title)

    assert ck == "misc_6be43f8b3dce"


def test_first_seen_key_missing_when_rev_id_unseen(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        {"key": "misc_aaa", "rev_id": "REV-100", "status": "applied", "title": "無関係"},
    ])
    mod = _load_module(monkeypatch, ledger)
    assert "REV-999" not in mod._get_rev_id_first_seen_key()
