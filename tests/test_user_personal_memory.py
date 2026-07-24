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


def test_capture_updates_preferred_name(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("私のことは太郎と呼んで", source="test")

    assert result["captured"] is True
    assert result["preferred_name"] == "太郎"
    assert result["confidence"] == "confirmed"
    assert "- [confirmed] Preferred name: 太郎" in path.read_text(encoding="utf-8")


def test_preferred_name_correction_replaces_not_appends(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    upm.capture_user_personal_memory("私のことは太郎と呼んで", source="test")
    result = upm.capture_user_personal_memory("やっぱり私のことは次郎と呼んで", source="test")

    assert result["preferred_name"] == "次郎"
    text = path.read_text(encoding="utf-8")
    assert text.count("Preferred name:") == 1
    assert "- [confirmed] Preferred name: 次郎" in text
    assert "太郎" not in [
        line.split(":", 1)[-1].strip() for line in text.splitlines() if line.startswith("- [confirmed] Preferred name")
    ]


def test_dog_name_correction_still_replaces_not_appends(tmp_path, monkeypatch):
    """_replace_or_append_fact への一般化後も、犬の名前の既存挙動が変わらないことを確認する。"""
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    upm.capture_user_personal_memory("覚えて。犬の名前はポチです", source="test")
    upm.capture_user_personal_memory("覚えて。犬の名前はタロウです", source="test")

    text = path.read_text(encoding="utf-8")
    assert text.count("Dog name:") == 1
    assert "- [confirmed] Dog name: タロウ" in text
