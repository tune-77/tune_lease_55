from pathlib import Path

from api import user_personal_memory as upm


def test_capture_updates_dog_name(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("覚えて。犬の名前はポチです", source="test")

    assert result["captured"] is True
    assert result["dog_name"] == "ポチ"
    assert result["confidence"] == "confirmed"
    text = path.read_text(encoding="utf-8")
    assert "- [confirmed] Dog name: ポチ" in text
    assert "[confirmed/family_pet]" in text


def test_non_personal_message_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("リース審査のDSCRを説明して", source="test")

    assert result["captured"] is False
    assert not path.exists()


def test_preference_without_explicit_remember_is_candidate(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("僕は長い説明が苦手", source="test")

    assert result["captured"] is True
    assert result["confidence"] == "candidate"
    assert "[candidate/preference]" in path.read_text(encoding="utf-8")


def test_hurt_memory_is_sensitive(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("犬の名前を忘れられたのはショック", source="test")

    assert result["captured"] is True
    assert result["confidence"] == "sensitive"
    assert "[sensitive/family_pet]" in path.read_text(encoding="utf-8")
