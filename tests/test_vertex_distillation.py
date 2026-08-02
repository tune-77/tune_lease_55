"""Vertex Answer API 要約を Obsidian に永続化するモジュールのテスト。

Vertex AI Search は Obsidian の中身をコピーしただけの索引で、新しい知識を
生まない。唯一新しいのは Answer API の合成要約であり、これがローカルに
残らなければ Vertex 連携をオフにした瞬間に価値がゼロになる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.vertex_distillation import (
    OUTPUT_DIR_REL,
    build_note,
    capture_vertex_distillation,
    should_capture,
)

OK_ANSWER = {
    "used": True,
    "status": "ok",
    "answer_text": "再リース期間は法定耐用年数の60%を目安に確認する。",
    "refs": ["Projects/tune_lease_55/Research/foo.md"],
    "grounding_score": 0.83,
    "grounding_score_source": "grounding_api",
}


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


# ---------- should_capture ----------

def test_should_capture_true_for_ok_answer_with_text():
    assert should_capture(OK_ANSWER) is True


def test_should_capture_false_when_not_used():
    assert should_capture({**OK_ANSWER, "used": False}) is False


def test_should_capture_false_when_status_not_ok():
    assert should_capture({**OK_ANSWER, "status": "disabled"}) is False


def test_should_capture_false_when_answer_text_empty():
    assert should_capture({**OK_ANSWER, "answer_text": "  "}) is False


def test_should_capture_false_for_none():
    assert should_capture(None) is False


# ---------- build_note ----------

def test_build_note_contains_question_and_answer_and_refs():
    note = build_note("再リースの残価は？", OK_ANSWER)
    assert "再リースの残価は？" in note
    assert "再リース期間は法定耐用年数の60%を目安に確認する。" in note
    assert "Projects/tune_lease_55/Research/foo.md" in note
    assert "review_status: needs_human_review" in note
    assert "source: vertex_answer_api" in note


def test_build_note_marks_missing_refs():
    note = build_note("質問", {**OK_ANSWER, "refs": []})
    assert "(参照なし)" in note


# ---------- capture_vertex_distillation ----------

def test_capture_writes_note_and_returns_path(tmp_path):
    vault = _vault(tmp_path)
    state_path = tmp_path / "state.json"

    result = capture_vertex_distillation(
        "再リースの残価は？", OK_ANSWER, vault_path=vault, state_path=state_path
    )

    assert result["captured"] is True
    assert result["duplicate"] is False
    note_path = vault / result["note_path"]
    assert note_path.exists()
    assert note_path.parent == vault / OUTPUT_DIR_REL


def test_capture_deduplicates_same_query(tmp_path):
    vault = _vault(tmp_path)
    state_path = tmp_path / "state.json"

    first = capture_vertex_distillation(
        "同じ質問", OK_ANSWER, vault_path=vault, state_path=state_path
    )
    written_files_after_first = list((vault / OUTPUT_DIR_REL).glob("*.md"))

    second = capture_vertex_distillation(
        "同じ質問", OK_ANSWER, vault_path=vault, state_path=state_path
    )
    written_files_after_second = list((vault / OUTPUT_DIR_REL).glob("*.md"))

    assert first["captured"] is True
    assert first["duplicate"] is False
    assert second["captured"] is True
    assert second["duplicate"] is True
    assert len(written_files_after_first) == 1
    assert len(written_files_after_second) == 1  # 2回目で新規ファイルが増えない


def test_capture_skips_when_not_capturable(tmp_path):
    vault = _vault(tmp_path)
    state_path = tmp_path / "state.json"

    result = capture_vertex_distillation(
        "質問", {"used": False, "status": "disabled"}, vault_path=vault, state_path=state_path
    )

    assert result == {"captured": False, "reason": "not_capturable"}
    assert not (vault / OUTPUT_DIR_REL).exists()


def test_capture_skips_gracefully_when_vault_missing(tmp_path):
    missing_vault = tmp_path / "does-not-exist"
    state_path = tmp_path / "state.json"

    result = capture_vertex_distillation(
        "質問", OK_ANSWER, vault_path=missing_vault, state_path=state_path
    )

    assert result == {"captured": False, "reason": "vault_not_found"}


def test_capture_skips_gracefully_when_vault_path_empty(tmp_path):
    result = capture_vertex_distillation(
        "質問", OK_ANSWER, vault_path="", state_path=tmp_path / "state.json"
    )
    assert result == {"captured": False, "reason": "vault_not_found"}


def test_capture_persists_state_across_calls(tmp_path):
    vault = _vault(tmp_path)
    state_path = tmp_path / "state.json"

    capture_vertex_distillation("質問A", OK_ANSWER, vault_path=vault, state_path=state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert len(state) == 1
    entry = next(iter(state.values()))
    assert "note_path" in entry
    assert "captured_at" in entry
