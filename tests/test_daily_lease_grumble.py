from novelist_agent import (
    archive_old_grumble_illustrations,
    _daily_grumble_illustration_prompt,
    _save_gemini_image,
    generate_daily_grumble_illustration,
    generate_daily_lease_grumble,
)


def test_daily_grumble_fallback_is_three_or_four_lines(monkeypatch):
    monkeypatch.setattr("ai_chat._get_gemini_key_from_secrets", lambda: "")
    monkeypatch.setattr("ai_chat.GEMINI_API_KEY_ENV", "")

    lines = generate_daily_lease_grumble(
        "2026-06-12",
        ["設備更新案件では再リース余地を確認する。"],
        theme="設備投資",
    )

    assert 3 <= len(lines) <= 4
    assert any(word in " ".join(lines) for word in ("プリン", "昼食", "コーヒー", "八奈見"))


def test_daily_grumble_illustration_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr("novelist_agent._get_daily_gemini_api_key", lambda: "")
    url = generate_daily_grumble_illustration(
        "2026-06-12",
        ["今日はプリンより追加資料が多かった。"],
        output_dir=tmp_path,
    )
    assert url == "/lease-grumble/2026-06-12.webp"
    assert (tmp_path / "2026-06-12.webp").stat().st_size > 1000


def test_daily_grumble_prompt_keeps_the_fixed_heroine():
    prompt = _daily_grumble_illustration_prompt(
        "2026-06-14",
        ["プリンより先に追加資料が届いた。"],
    )

    assert "silver-white hair" in prompt
    assert "large purple eyes" in prompt
    assert "No monitor-headed robot" in prompt


def test_save_gemini_image_converts_bytes_to_webp(tmp_path):
    import io
    from types import SimpleNamespace

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 18), "pink").save(buffer, "JPEG")
    target = tmp_path / "gemini.webp"

    _save_gemini_image(SimpleNamespace(image_bytes=buffer.getvalue()), target)

    with Image.open(target) as saved:
        assert saved.format == "WEBP"
        assert saved.size == (32, 18)


def test_archive_old_grumble_illustrations_moves_only_expired_images(tmp_path):
    public_dir = tmp_path / "public"
    vault = tmp_path / "vault"
    public_dir.mkdir()
    vault.mkdir()
    old_image = public_dir / "2026-05-01.webp"
    recent_image = public_dir / "2026-06-01.webp"
    ignored_image = public_dir / "cover.webp"
    old_image.write_bytes(b"old-image")
    recent_image.write_bytes(b"recent-image")
    ignored_image.write_bytes(b"cover")

    archived = archive_old_grumble_illustrations(
        vault=vault,
        keep_days=30,
        today="2026-06-13",
        public_dir=public_dir,
    )

    destination = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Archive"
        / "Lease Grumble"
        / "Images"
        / "2026"
        / "05"
        / old_image.name
    )
    assert archived == [str(destination)]
    assert destination.read_bytes() == b"old-image"
    assert not old_image.exists()
    assert recent_image.exists()
    assert ignored_image.exists()
    index = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Archive"
        / "Lease Grumble"
        / "README.md"
    ).read_text(encoding="utf-8")
    assert "2026-05-01" in index
    assert "![[Images/2026/05/2026-05-01.webp]]" in index


def test_archive_old_grumble_illustrations_keeps_conflicting_archive(tmp_path):
    public_dir = tmp_path / "public"
    vault = tmp_path / "vault"
    public_dir.mkdir()
    source = public_dir / "2026-05-01.webp"
    source.write_bytes(b"new-content")
    destination = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Archive"
        / "Lease Grumble"
        / "Images"
        / "2026"
        / "05"
        / source.name
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing-content")

    archived = archive_old_grumble_illustrations(
        vault=vault,
        keep_days=30,
        today="2026-06-13",
        public_dir=public_dir,
    )

    assert archived == []
    assert source.exists()
    assert destination.read_bytes() == b"existing-content"
