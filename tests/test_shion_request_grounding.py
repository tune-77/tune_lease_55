"""REV-482: 紫苑の依頼文の参照・約束・本番DB書き込み案の照合。"""

from __future__ import annotations

import json

from api import shion_request_grounding as grounding

# 2026-10-06 に対話室で実際に出た依頼文（社名なし）
REAL_REQUEST = """承知いたしました。過去の否決案件から、財務的に優良と判定される先を抽出するための検証用依頼文を作成します。

---
**紫苑依頼分:**
- **目的:** 財務健全性フラグを満たす否決案件の抽出と、スコア低迷原因の特定。
- **対象ファイル:** `scoring_core.py` (ロジック確認用), `screening_history.db` (案件データ)
- **変更範囲:**
    1. 既存の審査履歴DBに対し、財務健全性フラグを付与するクエリの実行。
    2. 該当案件のスコア要因（`get_score_detail`）と、否決時の審査コメントの突合。
- **検証コマンド:** `python3 -m scripts.analyze_financial_health --mode=audit_denied`
- **gitship/deploy:** 不要（今回は検証・分析のみ）
---

この依頼文で、まずは手元の環境にて検証を開始します。結果が出次第、報告いたします。"""


def test_request_turn_detection():
    assert grounding.is_request_turn("紫苑依頼分にして")
    assert grounding.is_request_turn("これを紫苑依頼文にまとめて")
    assert grounding.is_request_turn("Claude Code向けに書いて")
    assert not grounding.is_request_turn("今日の調子はどう？")


def test_reference_existence():
    assert grounding.check_reference("scoring_core.py").exists
    assert grounding.check_reference("get_score_detail").exists
    assert grounding.check_reference("lease_intelligence_tools.py").exists
    assert not grounding.check_reference("screening_history.db").exists
    missing_module = grounding.check_reference("python3 -m scripts.analyze_financial_health --mode=audit_denied")
    assert missing_module.kind == "command" and not missing_module.exists
    assert "analyze_financial_health" in missing_module.detail
    # 実在するスクリプトに無い引数
    bad_flag = grounding.check_reference("python scripts/analyze_scoring_drift.py --mode=audit_denied")
    assert not bad_flag.exists and "--mode" in bad_flag.detail
    assert grounding.check_reference("python -m pytest tests/test_scoring_core.py -q").exists
    # 普通の語・日本語は照合しない
    assert grounding.check_reference("needs") is None
    assert grounding.check_reference("審査履歴") is None


def test_real_request_is_grounded():
    result = grounding.ground_reply("紫苑依頼分にして", REAL_REQUEST, enabled=True)
    missing = {ref.ref for ref in result.missing_refs}
    assert "screening_history.db" in missing
    assert any("analyze_financial_health" in ref for ref in missing)
    assert "scoring_core.py" not in missing and "get_score_detail" not in missing
    assert "`screening_history.db`（要確認" in result.reply
    assert "analyze_financial_health --mode=audit_denied`（要確認: scripts/analyze_financial_health.py が無い）" in result.reply
    assert result.db_write_lines == 1
    assert grounding.DB_WRITE_NOTE in result.reply
    # 紫苑自身の実行・後での報告の約束は消え、実行者の注記が付く
    assert "手元の環境" not in result.reply
    assert "結果が出次第" not in result.reply
    assert result.reply.rstrip().endswith(grounding.EXECUTOR_NOTE)
    assert len(result.removed_promises) == 2
    # 依頼文そのものは残る
    assert "**紫苑依頼分:**" in result.reply and "`scoring_core.py`" in result.reply


def test_promises_outside_request():
    reply = (
        "承知いたしました。検証を進めます。\n\n過去の否決案件の傾向を記録から見ました。\n\n"
        "この検証結果は、次回対話時に傾向として報告しますね。"
    )
    result = grounding.ground_reply("いいよ", reply, enabled=True)
    assert "検証を進めます" not in result.reply
    assert "次回対話時" not in result.reply
    assert "記録から見ました" in result.reply
    assert result.reply.endswith(grounding.EXECUTOR_NOTE_PLAIN)
    assert not result.request_detected


def test_delegated_and_lookup_sentences_are_kept():
    reply = (
        "記録を調べます。\n紫苑依頼文:\n- 実行者: Claude Code\n"
        "- Claude Code が `python -m pytest tests/test_scoring_core.py` を実行します。\n"
        "- 審査履歴DBは読み取り専用で開き、フラグはコピー上で付与する。"
    )
    result = grounding.ground_reply("紫苑依頼文にして", reply, enabled=True)
    assert not result.removed_promises
    assert result.db_write_lines == 0
    assert not result.missing_refs
    assert result.reply == reply
    assert not result.changed


def test_new_file_is_not_flagged():
    reply = "紫苑依頼文:\n- 新規作成: `scripts/analyze_denied_financials.py`（読み取り専用で集計する）\n- 検証: `python -m py_compile scripts/analyze_denied_financials.py`"
    result = grounding.ground_reply("依頼文にして", reply, enabled=True)
    assert not [ref for ref in result.missing_refs if ref.kind == "path"]


def test_disabled_does_not_rewrite(monkeypatch):
    monkeypatch.setenv("SHION_REQUEST_GROUNDING", "off")
    result = grounding.ground_reply("紫苑依頼分にして", REAL_REQUEST)
    assert result.reply == REAL_REQUEST
    assert result.kept_promises and result.missing_refs


def test_request_block_lists_only_existing_refs():
    block = grounding.build_request_block("紫苑依頼分にして。否決案件のスコアを見たい", "`screening_history.db` を使う")
    assert "実行者: Claude Code" in block
    assert "scoring_core.py" in block and "scripts/analyze_scoring_drift.py" in block
    assert "screening_history.db" not in block.split("実在が確かめられた参照:")[1]
    assert "読み取り専用" in block


def test_verify_and_log_with_fake_jev(tmp_path, monkeypatch):
    monkeypatch.setattr(grounding, "verify_enabled", lambda: True)
    sent = {}

    def fake_request(payload):
        sent.update(payload)
        n = len(payload["state"]["sentences"])
        answers = {f"c{i}_verdict": {"choice": "other", "confidence": 0.9} for i in range(n)}
        answers["c0_verdict"] = {"choice": "unexecutable_self_action", "confidence": 0.9}
        return {"answers": answers, "model": "jev-test"}

    original = grounding.verify_reply_sentences
    monkeypatch.setattr(grounding, "verify_reply_sentences", lambda reply: original(reply, request_fn=fake_request))
    reply = "明日までにスクリプトを走らせておきます。よろしくお願いします。"
    result = grounding.ground_reply("いいよ", reply, enabled=True)
    log = tmp_path / "log.jsonl"
    record = grounding.verify_and_log("いいよ", reply, result.summary(), surface="test", path=log)
    saved = json.loads(log.read_text(encoding="utf-8").strip())
    assert saved["jev"]["status"] == "applied"
    assert saved["jev"]["unexecutable_promises"][0]["claim"].startswith("明日までに")
    # 決定的なパターンが拾えなかった約束は missed_by_rules に残る（パターン追加の候補）
    assert saved["jev"]["missed_by_rules"]
    assert record["surface"] == "test"
    assert "sentences" in sent["state"]


def test_sql_schema_check(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "lease_data.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE screening_records (id INTEGER, total_score REAL, outcome TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(grounding, "get_data_path", lambda *parts: str(tmp_path.joinpath(*parts)))
    grounding._db_schema.cache_clear()
    ok = grounding.check_reference("SELECT id, total_score FROM screening_records sr WHERE sr.outcome = 'x'")
    assert ok.exists
    bad_col = grounding.check_reference(
        "sqlite3 -readonly data/lease_data.db \"SELECT count(*) FROM screening_records WHERE status='rejected' AND equity_ratio > 30\""
    )
    assert not bad_col.exists and "status" in bad_col.detail and "equity_ratio" in bad_col.detail
    embedded = grounding.check_reference(
        "python3 -c \"import sqlite3; c = sqlite3.connect('data/lease_data.db'); c.execute('SELECT count(*) FROM screening_records WHERE outcome=1').fetchone()\""
    )
    assert embedded.exists
    bad_table = grounding.check_reference("SELECT * FROM screening_history")
    assert not bad_table.exists and "screening_history" in bad_table.detail
    reply = "紫苑依頼文:\n- 検証: `SELECT count(*) FROM screening_records WHERE equity_ratio > 30`"
    result = grounding.ground_reply("依頼文にして", reply, enabled=True)
    assert "equity_ratio > 30`（要確認: 列 equity_ratio" in result.reply
    grounding._db_schema.cache_clear()


def test_next_time_promise_is_removed():
    reply = (
        "紫苑依頼文:\n- 実行者: Claude Code\n- 目的: 見逃し候補の抽出\n\n"
        "この依頼文の内容で、実行の準備は整いました。次回の対話までに、新ロジック案を提示します。"
    )
    result = grounding.ground_reply("依頼文にして", reply, enabled=True)
    assert "準備は整いました" not in result.reply and "次回の対話までに" not in result.reply
    assert len(result.removed_promises) == 2


def test_in_reply_analysis_is_not_a_promise():
    reply = "以下の観点で分析を行います。記録上、同業種の否決は3件でした。こちらで分析した結果、価格負けが多いです。"
    result = grounding.ground_reply("否決の傾向は？", reply, enabled=True)
    assert not result.removed_promises
    assert result.reply == reply


def test_output_file_and_after_run_promise():
    reply = (
        "紫苑依頼文:\n- 実行者: Claude Code\n- 結果を `reports/financial_health_gap.md` に出力する。\n\n"
        "この依頼内容で進めてよろしいでしょうか。実行後は、抽出された先の傾向を報告します。"
    )
    result = grounding.ground_reply("依頼文にして", reply, enabled=True)
    assert not result.missing_refs
    assert "実行後は" not in result.reply and "進めてよろしいでしょうか" in result.reply
