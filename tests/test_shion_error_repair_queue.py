from __future__ import annotations

import json
from pathlib import Path

from scripts import build_shion_error_repair_queue as repair_queue


def test_classify_error_entry_accepts_single_file_name_error(tmp_path: Path) -> None:
    target = tmp_path / "api" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")

    entry = {
        "rev_id": "REV-900e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "description": "[エラーログ自動検出] NameError in api/example.py",
        "error_pattern": "NameError: name 'foo' is not defined at api/example.py",
        "affected_files": ["api/example.py"],
    }

    verdict = repair_queue.classify_error_entry(entry, tmp_path)

    assert verdict["is_safe"] is True
    assert verdict["item"]["target_module"] == "api/example.py"
    assert verdict["item"]["auto_fix_policy"]["auto_fix_allowed"] is True


def test_classify_error_entry_rejects_ambiguous_or_risky_entries(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "api" / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "model.py").write_text("", encoding="utf-8")

    ambiguous = {
        "rev_id": "REV-901e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "error_pattern": "ConnectionError: upstream failed",
        "affected_files": ["api/a.py"],
    }
    multi_file = {
        "rev_id": "REV-902e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "error_pattern": "NameError: missing symbol",
        "affected_files": ["api/a.py", "api/b.py"],
    }
    dangerous = {
        "rev_id": "REV-903e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "error_pattern": "NameError: missing symbol",
        "affected_files": ["data/model.py"],
    }

    assert repair_queue.classify_error_entry(ambiguous, tmp_path)["is_safe"] is False
    assert repair_queue.classify_error_entry(multi_file, tmp_path)["is_safe"] is False
    assert repair_queue.classify_error_entry(dangerous, tmp_path)["is_safe"] is False


def test_classify_error_entry_rejects_traversal_and_non_code_targets(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "example.py").write_text("", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "latest.json").write_text("{}", encoding="utf-8")

    traversal = {
        "rev_id": "REV-906e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "error_pattern": "NameError: missing symbol",
        "affected_files": ["../api/example.py"],
    }
    non_code = {
        "rev_id": "REV-907e",
        "category": "error_log_fix",
        "source": "analyze_error_logs",
        "error_pattern": "SyntaxError: malformed JSON at reports/latest.json",
        "affected_files": ["reports/latest.json"],
    }

    assert repair_queue.classify_error_entry(traversal, tmp_path)["is_safe"] is False
    assert repair_queue.classify_error_entry(non_code, tmp_path)["is_safe"] is False


def test_build_queue_skips_already_queued_ids(tmp_path: Path) -> None:
    target = tmp_path / "api" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    ledger = [
        {
            "rev_id": "REV-904e",
            "category": "error_log_fix",
            "source": "analyze_error_logs",
            "error_pattern": "ImportError: cannot import name x from api/example.py",
            "affected_files": ["api/example.py"],
        }
    ]

    queue = repair_queue.build_queue(ledger, already_queued_ids={"REV-904e"}, limit=1, root=tmp_path)

    assert queue["queued_count"] == 0
    assert queue["error_repair_safe_count"] == 0


def test_build_queue_outputs_ready_queue(tmp_path: Path) -> None:
    target = tmp_path / "api" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            [
                {
                    "rev_id": "REV-905e",
                    "category": "error_log_fix",
                    "source": "analyze_error_logs",
                    "error_pattern": "AttributeError: object has no attribute x at api/example.py",
                    "affected_files": ["api/example.py"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    queue = repair_queue.build_queue(json.loads(ledger_path.read_text(encoding="utf-8")), set(), 1, tmp_path)

    assert queue["status"] == "READY"
    assert queue["items"][0]["id"] == "REV-905e"
