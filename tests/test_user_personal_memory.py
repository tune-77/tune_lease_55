from pathlib import Path

from api import user_personal_memory as upm


def test_capture_updates_dog_name(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("覚えて。犬の名前はポチです", source="test")

    assert result["captured"] is True
    assert result["dog_name"] == "ポチ"
    text = path.read_text(encoding="utf-8")
    assert "- Dog name: ポチ" in text
    assert "[family_pet]" in text


def test_non_personal_message_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "user_personal_memory.md"
    monkeypatch.setattr(upm, "LOCAL_MEMORY_PATH", path)

    result = upm.capture_user_personal_memory("リース審査のDSCRを説明して", source="test")

    assert result["captured"] is False
    assert not path.exists()

