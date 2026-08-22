"""Regression tests for the applied/pending_applied confirmation gate in
step3_auto_apply.py.

Before this fix, `_applied` (and the ledger's "applied" record) were written
as soon as local tests passed, before git commit/push/PR creation were
confirmed. If the commit step then failed (e.g. `_pending_patches` ended up
empty, or push/PR creation failed), the item stayed marked "applied" forever
with no rollback: the ledger permanently skips "applied" items on future
runs, so the underlying bug resurfaced as a "new" user complaint in later
reports while the pipeline refused to ever retry a real fix for it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "auto-improvement-pipeline"
    / "scripts"
    / "step3_auto_apply.py"
)


def load_step3_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("step3_auto_apply_test_target", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_promote_pending_to_applied_moves_entries_and_records_ledger(tmp_path, monkeypatch):
    mod = load_step3_module()
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(mod._ledger, "LEDGER_PATH", ledger_path)

    applier = mod.Step3AutoApplier(workspace_root=tmp_path)
    applier._pending_applied = [
        {
            "id": "REV-001",
            "file": "a.py",
            "title": "テスト改善",
            "canonical_key": "ck1",
            "_ledger_key": "lk1",
            "pr_url": None,
        },
    ]

    applier._promote_pending_to_applied("https://example.com/pr/1")

    assert applier._pending_applied == []
    assert len(applier._applied) == 1
    assert applier._applied[0]["pr_url"] == "https://example.com/pr/1"
    assert "_ledger_key" not in applier._applied[0]

    entries = _read_ledger(ledger_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "applied"
    assert entries[0]["pr_url"] == "https://example.com/pr/1"


def test_demote_pending_to_needs_review_moves_entries_and_records_ledger(tmp_path, monkeypatch):
    mod = load_step3_module()
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(mod._ledger, "LEDGER_PATH", ledger_path)

    applier = mod.Step3AutoApplier(workspace_root=tmp_path)
    applier._pending_applied = [
        {
            "id": "REV-002",
            "file": "b.py",
            "title": "コミット失敗する改善",
            "canonical_key": "ck2",
            "_ledger_key": "lk2",
            "pr_url": None,
        },
    ]

    applier._demote_pending_to_needs_review("push失敗: テスト", ledger_status="push_pending")

    assert applier._pending_applied == []
    assert applier._applied == []
    assert len(applier._needs_review) == 1
    assert applier._needs_review[0]["id"] == "REV-002"

    entries = _read_ledger(ledger_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "push_pending"

    # push_pending must not permanently suppress the item like "applied" does.
    processed, _status = mod._ledger.is_processed("lk2")
    assert processed is False


def test_git_commit_and_push_without_pending_patches_demotes_instead_of_marking_applied(
    tmp_path, monkeypatch
):
    """Reproduces the exact anomaly observed in production (REV-019,
    2026-08-08 report): applied_count=1 while commit_result.success=False
    because pending_patches was empty. After the fix, such an item must
    never reach _applied nor be recorded as "applied" in the ledger.
    """
    mod = load_step3_module()
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(mod._ledger, "LEDGER_PATH", ledger_path)

    applier = mod.Step3AutoApplier(workspace_root=tmp_path)
    applier._pending_applied = [
        {
            "id": "REV-019",
            "file": "scoring_core.py",
            "title": "スコア80-100帯の成約率逆転",
            "canonical_key": "ck19",
            "_ledger_key": "lk19",
            "pr_url": None,
        },
    ]
    # _pending_patches intentionally left empty, matching the production bug.

    result = applier.git_commit_and_push()

    assert result["success"] is False
    assert applier._applied == []  # must NOT be marked applied
    assert applier._pending_applied == []
    assert len(applier._needs_review) == 1

    entries = _read_ledger(ledger_path)
    assert entries, "commit失敗時も台帳訂正が記録されるべき"
    assert entries[-1]["status"] != "applied"

    processed, _status = mod._ledger.is_processed("lk19")
    assert processed is False


def test_pending_ledger_key_fallback_uses_description_not_file(tmp_path, monkeypatch):
    """`_pending_ledger_key` の再計算フォールバックは canonical_key の実際の
    入力である description を使うべきで、無関係な file パスを渡すと
    apply_improvement() 側で最初に計算した _ledger_key とズレて同じ改善案が
    別キーとして台帳に記録されてしまう（needs_review 残留の一因）。
    """
    mod = load_step3_module()
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(mod._ledger, "LEDGER_PATH", ledger_path)

    applier = mod.Step3AutoApplier(workspace_root=tmp_path)
    title = "テスト改善タイトル"
    description = "説明文テキスト"
    entry = {
        "id": "REV-100",
        "file": "completely/unrelated/path.py",
        "title": title,
        "description": description,
        "canonical_key": "",
        "_ledger_key": "",
        "pr_url": None,
    }

    key = applier._pending_ledger_key(entry)

    assert key == mod._canonical_key(title, description)
    assert key != mod._canonical_key(title, entry["file"])
