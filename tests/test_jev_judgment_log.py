"""REV-424: Jev共通判定ログと、調査検証の呼び出し側配線。"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import jev_judgment_log as jlog
import typesafe_research_verify_guard as verify_guard

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "auto_research_lease_judgment", _ROOT / "scripts" / "auto_research_lease_judgment.py"
)
assert _SPEC and _SPEC.loader
research = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = research
_SPEC.loader.exec_module(research)

_NOTE = """## 判断に使える確認済み事実
- 2026年度のものづくり補助金は交付決定前の発注を補助対象外とする。
- 対象設備の法定耐用年数は7年である。
"""


def _item(subject: str, question: str, *, auto_passed: bool = True) -> dict:
    return {
        "subject": subject,
        "question": question,
        "probability": 0.4,
        "route": "send",
        "auto_passed": auto_passed,
        "thresholds": {"relevant_min": 0.35},
    }


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_keep_hash_not_text_and_carry_full_schema():
    records = jlog.build_records(
        guard="news",
        run_id="run1",
        mode="shadow",
        model="jev-test",
        items=[_item("社外秘の見出し", "repayment")],
        sample_rate=0.0,
    )

    assert "社外秘の見出し" not in json.dumps(records, ensure_ascii=False)
    record = records[0]
    assert set(record) == {
        "schema_version", "record_type", "ts", "judgment_id", "run_id", "guard", "mode",
        "model", "question", "subject_hash", "probability", "choice", "route", "thresholds",
        "auto_passed", "sampled_for_review", "label", "label_source", "labeled_at",
    }
    assert record["subject_hash"] == jlog.subject_hash("  社外秘の見出し ")
    assert record["label"] is None and record["record_type"] == "judgment"


def test_sampling_is_per_subject_and_only_for_auto_passed():
    items = [
        _item("記事A", "repayment"),
        _item("記事A", "injection"),
        _item("記事B", "repayment", auto_passed=False),
    ]
    records = jlog.build_records(
        guard="news", run_id="r", mode="shadow", model="m", items=items, sample_rate=1.0,
    )
    assert [r["sampled_for_review"] for r in records] == [True, True, False]

    mixed = jlog.build_records(
        guard="news", run_id="r", mode="shadow", model="m",
        items=[_item(f"記事{i}", q) for i in range(40) for q in ("repayment", "injection")],
        sample_rate=0.5, rng=random.Random(7),
    )
    pairs = [(a["sampled_for_review"], b["sampled_for_review"]) for a, b in zip(mixed[::2], mixed[1::2])]
    assert all(a == b for a, b in pairs), "同じ記事の質問は同じ抽出結果にする"
    assert 0 < sum(a for a, _ in pairs) < 40

    none = jlog.build_records(
        guard="news", run_id="r", mode="shadow", model="m", items=items, sample_rate=0.0,
    )
    assert not any(r["sampled_for_review"] for r in none)


def test_sample_rate_env_falls_back_on_invalid_values():
    assert jlog.configured_sample_rate({}) == jlog.DEFAULT_SAMPLE_RATE
    assert jlog.configured_sample_rate({"JEV_JUDGMENT_SAMPLE_RATE": "0.25"}) == 0.25
    assert jlog.configured_sample_rate({"JEV_JUDGMENT_SAMPLE_RATE": "1.5"}) == jlog.DEFAULT_SAMPLE_RATE
    assert jlog.configured_sample_rate({"JEV_JUDGMENT_SAMPLE_RATE": "abc"}) == jlog.DEFAULT_SAMPLE_RATE


def test_append_and_label_are_append_only(tmp_path):
    path = tmp_path / "log.jsonl"
    records = jlog.build_records(
        guard="news", run_id="r", mode="shadow", model="m", items=[_item("x", "q")], sample_rate=0.0,
    )
    assert jlog.append_records(records, path) == 1
    assert jlog.append_label(records[0]["judgment_id"], "relevant", label_source="human", path=path) == 1

    rows = _read(path)
    assert rows[0] == records[0], "既存行は書き換えない"
    assert rows[1]["record_type"] == "label" and rows[1]["label"] == "relevant"


def test_off_disables_and_write_failure_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", "off")
    assert jlog.log_path() is None
    assert jlog.append_records([{"a": 1}]) == 0

    blocker = tmp_path / "file"
    blocker.write_text("not a dir", encoding="utf-8")
    assert jlog.append_records([{"a": 1}], blocker / "log.jsonl") == 0


def _enable_verify(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("TYPESAFE_RESEARCH_VERIFY_MODE", mode)
    monkeypatch.setattr(verify_guard, "typesafe_available", lambda *a, **k: True)


def test_enforce_marks_note_unverified_when_guard_fails(monkeypatch):
    _enable_verify(monkeypatch, "enforce")

    def boom(*_a, **_k):
        raise TimeoutError("secret detail")

    monkeypatch.setattr(verify_guard, "verify_note_claims", boom)
    body, meta = research._verify_note_claims(_NOTE, "raw")

    assert body.startswith(_NOTE)
    assert "未検証" in body and "TimeoutError" in body
    assert "secret detail" not in body, "例外メッセージ本文はノートへ書かない"
    assert meta == {"status": "skipped", "reason": "TimeoutError"}


def test_enforce_marks_note_unverified_without_credential(monkeypatch):
    monkeypatch.setenv("TYPESAFE_RESEARCH_VERIFY_MODE", "enforce")
    monkeypatch.setattr(verify_guard, "typesafe_available", lambda *a, **k: False)
    body, _ = research._verify_note_claims(_NOTE, "raw")
    assert "credential_missing" in body and "未検証" in body


def test_shadow_keeps_note_unchanged_when_guard_fails(monkeypatch):
    _enable_verify(monkeypatch, "shadow")
    monkeypatch.setattr(
        verify_guard, "verify_note_claims", lambda *a, **k: (_ for _ in ()).throw(TimeoutError())
    )
    body, _ = research._verify_note_claims(_NOTE, "raw")
    assert body == _NOTE


def test_applied_verification_is_logged_per_claim(monkeypatch, tmp_path):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(log))
    _enable_verify(monkeypatch, "shadow")
    real = verify_guard.verify_note_claims

    def fake(body, evidence):
        return real(
            body,
            evidence,
            request_fn=lambda _p: {
                "answers": {
                    "c0_verdict": {"type": "choice", "choice": "verified", "confidence": 0.9},
                },
                "model": "jev-test",
            },
        )

    monkeypatch.setattr(verify_guard, "verify_note_claims", fake)
    body, meta = research._verify_note_claims(_NOTE, "raw")

    assert body == _NOTE
    assert meta["needs_review_count"] == 1
    text = log.read_text(encoding="utf-8")
    assert "ものづくり補助金" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert [(r["route"], r["auto_passed"], r["choice"]) for r in rows] == [
        ("verified", True, "verified"),
        ("needs_review", False, None),
    ]
    assert {r["guard"] for r in rows} == {"research_verify"}
