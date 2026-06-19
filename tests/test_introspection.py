import datetime as dt
import json
from pathlib import Path

from scripts.introspection import build_introspection_report, render_markdown


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_introspection_detects_boredom_and_missing_action(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr("scripts.introspection.MEMORY_DIR", memory_dir)

    today = dt.date(2026, 6, 19)
    _write(
        memory_dir / "2026-06-19.md",
        "# 2026-06-19\n\n## Snapshot\n\n- 内省が全くされていない。つまらない。\n",
    )
    latest = tmp_path / "reports" / "latest.json"
    _write(latest, json.dumps({"needs_review_count": 3, "applied_count": 0}, ensure_ascii=False))
    prompt_log = tmp_path / "data" / "prompt_feedback_log.jsonl"
    _write(prompt_log, '{"pdca_applied": true, "previous_response_changed": false}\n')
    memory = tmp_path / "MEMORY.md"
    _write(memory, "# Memory\n\n内省と次の行動を残す。\n")

    report = build_introspection_report(
        today=today,
        days=1,
        latest_report_path=latest,
        recursive_report_path=tmp_path / "missing_recursive.json",
        loop_report_path=tmp_path / "missing_loop.json",
        prompt_log_path=prompt_log,
        memory_path=memory,
    )

    titles = {finding["title"] for finding in report["findings"]}
    assert report["status"] == "attention"
    assert "退屈・停滞シグナルが出ている" in titles
    assert "改善候補が観察だけで止まっている" in titles
    assert "再帰的自己改善レポートが欠けている" in titles
    assert report["next_actions"]


def test_introspection_markdown_contains_findings(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr("scripts.introspection.MEMORY_DIR", memory_dir)
    today = dt.date(2026, 6, 19)
    _write(
        memory_dir / "2026-06-19.md",
        (
            "# 2026-06-19\n\n"
            "## Promotable Items\n\n"
            "- 違和感を次の行動に変えた。\n"
        ),
    )
    _write(tmp_path / "MEMORY.md", "# Memory\n\n判断を変えた理由を記録する。\n")

    report = build_introspection_report(
        today=today,
        days=1,
        latest_report_path=tmp_path / "missing_latest.json",
        recursive_report_path=tmp_path / "recursive.json",
        loop_report_path=tmp_path / "loop.json",
        prompt_log_path=tmp_path / "prompt.jsonl",
        memory_path=tmp_path / "MEMORY.md",
    )
    markdown = render_markdown(report)

    assert "# Introspection Report" in markdown
    assert "## Findings" in markdown
    assert "Promotable items: 1" in markdown
