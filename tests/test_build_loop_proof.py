"""Tests for scripts/build_loop_proof.py — the judge-facing loop dashboard builder.

Fixtures are inline so the test does not depend on live report contents.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_loop_proof", ROOT / "scripts" / "build_loop_proof.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


GROWTH_FIXTURE = """# Judgment Asset Growth Score

## Current

- Date: 2026-07-20
- Score: 62.6

## Components

- Coverage: `███████████████████░` 94.0
- Reuse proxy: `████████████░░░░░░░░` 60.0
- Judgment change proxy: `███████████████░░░░░` 75.0
- Human alignment proxy: `██████████████░░░░░░` 70.0
- Field validation: `░░░░░░░░░░░░░░░░░░░░` 0.0
- Negative signal: `██████████░░░░░░░░░░` 49.0

## Counts

- Materials: 21
- Inbox candidates: 20
- Active rules: 10
- Risk axes: 5
- Concepts: 10
- User evidence: 18
"""

LOOP_FIXTURE = """# Loop Engineering Health

## Improvement Loop
- Applied: 0
- Needs review: 34

## Prompt Feedback Loop
- Total: 317
- PDCA applied: 317 (100.0%)
- Previous response diffs: 117 (36.9%)

## Scoring Coefficients
- Status: `attention`
"""

LEDGER_FIXTURE = "\n".join(json.dumps(r) for r in [
    {"status": "applied", "rev_id": "REV-001", "pr_url": "http://x/1",
     "recorded_at": "2026-06-14T08:00:00"},
    {"status": "applied", "rev_id": "REV-002", "pr_url": "http://x/2",
     "recorded_at": "2026-06-20T08:00:00"},
    {"status": "applied", "rev_id": "REV-003", "recorded_at": "2026-07-05T08:00:00"},
    {"status": "pending", "rev_id": "REV-004", "recorded_at": "2026-07-19T08:00:00"},
])


def test_parse_growth_components():
    g = _via_tmp(mod.parse_growth, GROWTH_FIXTURE)
    assert g["growth_score"] == 62.6
    assert g["coverage"] == 94.0
    assert g["reuse"] == 60.0
    assert g["field"] == 0.0
    assert g["materials"] == 21
    assert g["active_rules"] == 10


def test_parse_loop_feedback():
    lp = _via_tmp(mod.parse_loop, LOOP_FIXTURE)
    assert lp["feedback_total"] == 317
    assert lp["feedback_pct"] == 100.0
    assert lp["fb_diffs"] == 117
    assert lp["fb_diff_pct"] == 36.9
    assert lp["needs_review"] == 34
    assert lp["scoring_status"] == "attention"


def test_parse_ledger_counts_and_period():
    lg = _via_tmp(mod.parse_ledger, LEDGER_FIXTURE)
    assert lg["proposals"] == 4
    assert lg["applied"] == 3
    assert lg["pr_traced"] == 2
    assert lg["distinct_rev"] == 4
    assert lg["period_start"] == "2026-06-14"
    assert lg["period_end"] == "2026-07-19"
    assert lg["per_month"] == {"2026-06": 2, "2026-07": 1}


def test_render_contains_live_numbers(tmp_path, monkeypatch):
    m = {
        **_via_tmp(mod.parse_ledger, LEDGER_FIXTURE),
        **_via_tmp(mod.parse_growth, GROWTH_FIXTURE),
        **_via_tmp(mod.parse_loop, LOOP_FIXTURE),
    }
    proposals = m["proposals"] or 1
    m["applied_pct"] = round(m["applied"] / proposals * 100)
    m["fb_other"] = m["feedback_total"] - m["fb_diffs"]
    m["fb_other_pct"] = round(100 - m["fb_diff_pct"], 1)
    html = mod.render(m)
    assert "<style" in html and "class=\"page\"" in html
    # derived + parsed values reach the page
    assert "適用率 75%" in html          # 3/4 applied
    assert ">317<" in html               # feedback total
    assert "36.9%" in html               # response-diff rate
    assert "needs-review" in html
    # honesty: field-validation zero is shown, not hidden
    assert "Field validation はこれから" in html


def test_load_payload_snapshot_fallback(tmp_path, monkeypatch):
    """Cloud Run で reports/ が欠落しても、スナップショットで数値が埋まる。"""
    # ledger だけ本物、reports/ は存在しないパスに向ける（= Cloud Run の状態）
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(LEDGER_FIXTURE, encoding="utf-8")
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps({
        "proposals": 1, "applied": 1, "coverage": 94.0,
        "feedback_total": 317, "fb_diff_pct": 36.9, "needs_review": 34,
        "active_rules": 10, "per_month": {"2026-06": 2},
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "LEDGER", ledger)
    monkeypatch.setattr(mod, "GROWTH", tmp_path / "missing_growth.md")
    monkeypatch.setattr(mod, "LOOP", tmp_path / "missing_loop.md")
    monkeypatch.setattr(mod, "SNAPSHOT", snap)

    pl = mod.load_payload()
    # ledger 由来はライブ（本物）で上書き
    assert pl["proposals"] == 4
    assert pl["applied"] == 3
    # reports 由来はスナップショットで補完
    assert pl["coverage"] == 94.0
    assert pl["feedback_total"] == 317
    assert pl["needs_review"] == 34
    assert pl["source"] == "live+snapshot"


def test_write_snapshot_roundtrip(tmp_path, monkeypatch):
    snap = tmp_path / "snap.json"
    monkeypatch.setattr(mod, "SNAPSHOT", snap)
    mod.write_snapshot({"proposals": 265, "coverage": 94.0})
    loaded = mod.load_snapshot()
    assert loaded["proposals"] == 265
    assert "generated_at" in loaded


def _via_tmp(fn, text):
    """Call a path-based parser with fixture text through a temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(text)
        p = Path(fh.name)
    try:
        return fn(p)
    finally:
        p.unlink(missing_ok=True)
