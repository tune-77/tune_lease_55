"""cleanup_improvement_reviews の B群タイトル一致クローズのテスト。

出荷済みだが canonical_key（title+description 依存）が出荷REVと繋がらず
needs_review で churn し続ける生メモを、runtime 台帳のタイトル一致で
「実キーのまま」applied 化できることを保証する。description 差異でキーが
ブレても閉じられる（＝環境外の description を知らなくても効く）ことが要点。
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


def test_title_match_resolves_real_key_regardless_of_description(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    # runtime 台帳: description 由来の任意キーで needs_review になっている B群項目。
    _write_ledger(ledger, [
        {
            "key": "misc_deadbeef1234",  # description 込みで生成された未知キー
            "status": "needs_review",
            "title": "AIアシスタント「八奈見さん」の名称に関する課題",
            "recorded_at": "2026-07-01T00:00:00",
        },
        {
            "key": "misc_unrelated0001",
            "status": "needs_review",
            "title": "全く無関係な別の改善メモ",
            "recorded_at": "2026-07-01T00:00:00",
        },
    ])
    mod = _load_module(monkeypatch, ledger)

    updates = mod._apply_title_matched_closures("2026-07-24T00:00:00")

    # 八奈見の項目だけが applied 化され、実キー（misc_deadbeef1234）を再利用する。
    keys = {u["key"]: u for u in updates}
    assert "misc_deadbeef1234" in keys
    assert keys["misc_deadbeef1234"]["status"] == "applied"
    assert "misc_unrelated0001" not in keys  # マッチしないものは触らない


def test_no_op_when_title_absent(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        {"key": "k1", "status": "needs_review", "title": "無関係", "recorded_at": "2026-07-01T00:00:00"},
    ])
    mod = _load_module(monkeypatch, ledger)
    assert mod._apply_title_matched_closures("2026-07-24T00:00:00") == []


def test_already_applied_is_not_reclosed(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    # 同一キーが needs_review → applied と推移済みなら、最新=applied で対象外。
    _write_ledger(ledger, [
        {"key": "k9", "status": "needs_review", "title": "複数のAIモデル（紫苑）を同時に動かし、互いに意見を出し合わせる機能がない。", "recorded_at": "2026-07-01T00:00:00"},
        {"key": "k9", "status": "applied", "title": "複数のAIモデル（紫苑）を同時に動かし、互いに意見を出し合わせる機能がない。", "recorded_at": "2026-07-10T00:00:00"},
    ])
    mod = _load_module(monkeypatch, ledger)
    assert mod._apply_title_matched_closures("2026-07-24T00:00:00") == []


def test_norm_title_absorbs_whitespace(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [
        {"key": "kx", "status": "needs_review", "title": "  エージェントチームとの議論停滞  ", "recorded_at": "2026-07-01T00:00:00"},
    ])
    mod = _load_module(monkeypatch, ledger)
    # KNOWN_TITLE_APPLIED には議論停滞は含めていない（A群=cleanupがREV番号で拾う）ため空。
    # ただし _norm_title が前後空白を吸収することは単体で確認する。
    assert mod._norm_title("  エージェントチームとの議論停滞  ") == mod._norm_title("エージェントチームとの議論停滞")
