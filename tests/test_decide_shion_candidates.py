"""SHION提案の一覧表示・台帳記録CLI（scripts/decide_shion_candidates.py）のテスト。"""

import json
import sys

import scripts.decide_shion_candidates as cli


def _write_queue(path, entries):
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_load_shion_candidates_filters_source_and_dedupes(tmp_path):
    path = tmp_path / "dispatch_queue.jsonl"
    _write_queue(path, [
        {"type": "improvement_candidates", "date": "2026-08-21", "source": "shion",
         "candidates": [{"id": "SHION-AAA", "title": "資金繰りヒアリングを具体化する"}]},
        {"type": "improvement_candidates", "date": "2026-08-22", "source": "shion",
         "candidates": [
             {"id": "SHION-BBB", "title": "資金繰りヒアリングを具体化する"},  # 重複タイトル
             {"id": "SHION-CCC", "title": "災害リスク地域の保全条件を強化する"},
         ]},
        {"type": "improvement_candidates", "date": "2026-08-22",
         "candidates": [{"id": "REV-999", "title": "REVパイプライン発の候補"}]},  # source=shion以外
    ])

    candidates = cli.load_shion_candidates(path)

    titles = [c["title"] for c in candidates]
    assert titles == ["資金繰りヒアリングを具体化する", "災害リスク地域の保全条件を強化する"]


def test_annotate_status_reflects_ledger(tmp_path, monkeypatch):
    queue_path = tmp_path / "dispatch_queue.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(cli.ledger, "LEDGER_PATH", ledger_path)
    _write_queue(queue_path, [
        {"type": "improvement_candidates", "date": "2026-08-22", "source": "shion",
         "candidates": [
             {"id": "SHION-AAA", "title": "却下済みの提案"},
             {"id": "SHION-BBB", "title": "未決着の提案"},
         ]},
    ])
    key = cli.ledger.compute_key("却下済みの提案", "")
    cli.ledger.record(key, "rejected", "却下済みの提案")

    candidates = cli.annotate_status(cli.load_shion_candidates(queue_path))
    by_title = {c["title"]: c["ledger_status"] for c in candidates}

    assert by_title["却下済みの提案"].startswith("rejected")
    assert by_title["未決着の提案"] == ""


def test_main_records_decision_by_index(tmp_path, monkeypatch, capsys):
    queue_path = tmp_path / "dispatch_queue.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(cli, "DISPATCH_QUEUE_PATH", queue_path)
    monkeypatch.setattr(cli.ledger, "LEDGER_PATH", ledger_path)
    _write_queue(queue_path, [
        {"type": "improvement_candidates", "date": "2026-08-22", "source": "shion",
         "candidates": [
             {"id": "SHION-AAA", "title": "候補1"},
             {"id": "SHION-BBB", "title": "候補2"},
         ]},
    ])

    monkeypatch.setattr(sys, "argv", ["decide_shion_candidates.py", "--reject", "1"])
    cli.main()

    key1 = cli.ledger.compute_key("候補1", "")
    processed, status = cli.ledger.is_processed(key1)
    assert processed and status.startswith("rejected")

    key2 = cli.ledger.compute_key("候補2", "")
    processed2, _ = cli.ledger.is_processed(key2)
    assert processed2 is False  # 未指定の候補は変更されない


def test_main_lists_only_undecided_by_default(tmp_path, monkeypatch, capsys):
    queue_path = tmp_path / "dispatch_queue.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(cli, "DISPATCH_QUEUE_PATH", queue_path)
    monkeypatch.setattr(cli.ledger, "LEDGER_PATH", ledger_path)
    _write_queue(queue_path, [
        {"type": "improvement_candidates", "date": "2026-08-22", "source": "shion",
         "candidates": [
             {"id": "SHION-AAA", "title": "決着済み"},
             {"id": "SHION-BBB", "title": "未決着"},
         ]},
    ])
    cli.ledger.record(cli.ledger.compute_key("決着済み", ""), "applied", "決着済み")

    monkeypatch.setattr(sys, "argv", ["decide_shion_candidates.py"])
    cli.main()

    out = capsys.readouterr().out
    assert "未決着" in out
    assert "決着済み" not in out
