from __future__ import annotations

import datetime as dt

from api import shion_illustration_gallery as gallery


def _touch(path, data=b"img"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_sync_collects_public_and_archive_skipping_non_dated(tmp_path):
    public = tmp_path / "public"
    archive = tmp_path / "archive"
    _touch(public / "2026-10-06.webp")
    _touch(public / "characters" / "shion.webp")
    _touch(archive / "2026" / "07" / "2026-07-01.webp")
    _touch(archive / "2026" / "07" / ".2026-07-02.webp.icloud")  # iCloud の未ダウンロード
    out = tmp_path / "gallery"

    assert gallery.sync_gallery(out, [public, archive]) == 2
    assert sorted(p.name for p in out.iterdir()) == ["2026-07-01.webp", "2026-10-06.webp"]
    assert gallery.sync_gallery(out, [public, archive]) == 0  # 2回目は写さない


def test_pick_daily_is_stable_and_random_varies():
    names = [f"2026-07-{d:02d}.webp" for d in range(1, 31)]
    day = dt.date(2026, 10, 7)
    assert gallery.pick("daily", names, day) == gallery.pick("daily", names, day)
    assert gallery.pick("daily", [], day) is None
    assert len({gallery.pick("random", names) for _ in range(40)}) > 1


def test_resolve_file_rejects_paths_and_falls_back_to_public(tmp_path):
    g, p = tmp_path / "g", tmp_path / "p"
    _touch(p / "2026-10-06.webp")
    assert gallery.resolve_file("2026-10-06.webp", g, p) == p / "2026-10-06.webp"
    assert gallery.resolve_file("../secrets.toml", g, p) is None
    assert gallery.resolve_file("2026-10-06.webp/../../x", g, p) is None
    assert gallery.resolve_file("2026-10-07.webp", g, p) is None


def test_payload_and_public_url_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "GALLERY_DIR", tmp_path / "g")
    monkeypatch.setattr(gallery, "PUBLIC_DIR", tmp_path / "p")
    _touch(tmp_path / "g" / "2026-08-01.webp")
    assert gallery.payload("2026-08-01.webp") == {
        "available": True,
        "date": "2026-08-01",
        "url": "/api/shion/illustrations/file/2026-08-01.webp",
    }
    assert gallery.payload(None) == {"available": False}
    assert gallery.public_url_to_api("/lease-grumble/2026-08-01.webp") == "/api/shion/illustrations/file/2026-08-01.webp"
    # 生成を止めた日の観察ログは画像が無いので空（壊れた画像を出さない）
    assert gallery.public_url_to_api("/lease-grumble/2026-10-08.webp") == ""
    assert gallery.public_url_to_api("") == ""


def test_router_serves_only_gallery_files(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers.shion_illustrations import router

    monkeypatch.setattr(gallery, "GALLERY_DIR", tmp_path / "g")
    monkeypatch.setattr(gallery, "PUBLIC_DIR", tmp_path / "p")
    monkeypatch.setattr(gallery, "ensure_synced_in_background", lambda: None)
    _touch(tmp_path / "g" / "2026-08-01.webp", b"RIFFxxxxWEBP")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    body = client.get("/api/shion/illustrations/random?mode=daily").json()
    assert body["available"] and body["date"] == "2026-08-01"
    res = client.get(body["url"])
    assert res.status_code == 200 and res.headers["content-type"] == "image/webp"
    assert client.get("/api/shion/illustrations/file/secrets.toml").status_code == 404
    assert client.get("/api/shion/illustrations/random?mode=bogus").status_code == 422
