import subprocess
from unittest.mock import Mock

from lease_intelligence_consultation import (
    consult_senior_reasoner,
    finalize_consultation_learning,
    sanitize_consultation_text,
)


def test_sanitize_consultation_text_redacts_sensitive_values():
    text = sanitize_consultation_text(
        "株式会社テスト test@example.com 03-1234-5678 "
        "法人番号1234567890123 売上12345678"
    )

    assert "株式会社テスト" not in text
    assert "test@example.com" not in text
    assert "03-1234-5678" not in text
    assert "1234567890123" not in text
    assert "12345678" not in text
    assert "[COMPANY]" in text


def test_consultation_requires_shions_hypothesis(tmp_path):
    result = consult_senior_reasoner(
        question="物件評価の扱い",
        shion_hypothesis="",
        confidence=0.4,
        evidence_summary="",
        vault=tmp_path,
    )

    assert result["consulted"] is False
    assert "初期仮説" in result["error"]


def test_consultation_uses_codex_and_records_learning(tmp_path, monkeypatch):
    completed = subprocess.CompletedProcess(
        args=["codex"],
        returncode=0,
        stdout="1. 妥当な点\n仮説の一部は妥当。\n2. 見落とし\n実装確認が必要。",
        stderr="",
    )
    run = Mock(return_value=completed)
    monkeypatch.setattr("lease_intelligence_consultation.subprocess.run", run)
    monkeypatch.setattr(
        "lease_intelligence_consultation.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "lease_intelligence_consultation.CONSULTATION_LOG_PATH",
        tmp_path / "consultations.jsonl",
    )

    result = consult_senior_reasoner(
        question="物件スコアは承認理由か",
        shion_hypothesis="警告には使うが最終点へは加算しないと考える。",
        confidence=0.6,
        evidence_summary="scoring_coreの定数を確認した。",
        vault=tmp_path,
    )

    assert result["consulted"] is True
    assert result["provider"] == "codex"
    assert "実装確認" in result["senior_advice"]
    assert "Learning" in result["learning_note"]
    command = run.call_args.args[0]
    assert "--sandbox" in command
    assert "read-only" in command
    assert "--ephemeral" in command


def test_consultation_falls_back_to_claude(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        if command[0].endswith("codex"):
            return subprocess.CompletedProcess(command, 1, "", "codex unavailable")
        return subprocess.CompletedProcess(command, 0, "Claudeによる助言", "")

    monkeypatch.setattr("lease_intelligence_consultation.subprocess.run", fake_run)
    monkeypatch.setattr(
        "lease_intelligence_consultation.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "lease_intelligence_consultation.CONSULTATION_LOG_PATH",
        tmp_path / "consultations.jsonl",
    )

    result = consult_senior_reasoner(
        question="難問",
        shion_hypothesis="仮説",
        confidence=0.3,
        evidence_summary="根拠",
        vault=tmp_path,
    )

    assert result["provider"] == "claude"
    assert result["senior_advice"] == "Claudeによる助言"


def test_finalize_records_shions_own_synthesis(tmp_path, monkeypatch):
    log_path = tmp_path / "consultations.jsonl"
    monkeypatch.setattr(
        "lease_intelligence_consultation.CONSULTATION_LOG_PATH",
        log_path,
    )
    finalize_consultation_learning(
        tmp_path,
        ["SHION-LEARN-TEST"],
        "初期仮説を修正し、実装確認を優先する。",
    )

    note = next((tmp_path / "Projects/tune_lease_55/Lease Intelligence/Learning").glob("*.md"))
    assert "紫苑による統合" in note.read_text(encoding="utf-8")

    from lease_intelligence_mind import build_mind_context

    context = build_mind_context(tmp_path)
    assert "上位検討から自分の判断へ統合した最近の学び" in context
    assert "実装確認を優先" in context
    log = log_path.read_text(encoding="utf-8")
    assert '"status": "integrated_by_shion"' in log
    assert "SHION-LEARN-TEST" in log
