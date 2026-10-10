from __future__ import annotations

import datetime as dt

import pytest

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
    monkeypatch.setattr(gallery, "EXTRA_DIR", tmp_path / "x")
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
    monkeypatch.setattr(gallery, "EXTRA_DIR", tmp_path / "x")
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


def test_extra_gallery_is_listed_served_and_labelled(tmp_path, monkeypatch):
    """REV-600: 既存の絵（gallery-*.webp）も候補に入り、日付の代わりに「紫苑ギャラリー」と出る。"""
    g, p, x = tmp_path / "g", tmp_path / "p", tmp_path / "x"
    _touch(g / "2026-08-01.webp")
    _touch(x / "gallery-0123456789.webp")
    _touch(x / "notes.txt")
    _touch(g / "gallery-abcdefabcd.webp")  # 日付の置き場に紛れたギャラリー名は配信しない

    assert gallery.list_names(g, p, x) == ["2026-08-01.webp", "gallery-0123456789.webp"]
    assert gallery.resolve_file("gallery-0123456789.webp", g, p, x) == x / "gallery-0123456789.webp"
    assert gallery.resolve_file("gallery-abcdefabcd.webp", g, p, x) is None
    assert gallery.resolve_file("gallery-../../secret", g, p, x) is None
    assert gallery.payload("gallery-0123456789.webp", g) == {
        "available": True,
        "label": "紫苑ギャラリー",
        "url": "/api/shion/illustrations/file/gallery-0123456789.webp",
    }


def test_build_extra_names_are_stable_and_servable():
    pytest.importorskip("PIL")
    from scripts import build_shion_gallery_extra as build

    names = [build.output_name(src, at) for src, at in build.SOURCES]
    assert len(set(names)) == len(names) == len(build.SOURCES)
    assert all(gallery.EXTRA_RE.match(n) for n in names)
    # 軍師の衣装・素材シート・紫苑が写っていない絵は入れない
    excluded = {"IMG_1913.PNG", "IMG_1914.PNG", "IMG_1915.PNG", "IMG_1916.PNG", "IMG_1917.jpeg",
                "IMG_1740.PNG", "IMG_1740 (1).PNG", "IMG_1741.PNG", "IMG_1754.JPG", "IMG_1760.JPG", "IMG_1780.PNG"}
    assert not excluded & {src.rsplit("/", 1)[-1] for src, _ in build.SOURCES}
    assert all((build.REPO_ROOT / src).is_file() for src, _ in build.SOURCES)


def test_fit_16x9_keeps_whole_picture_without_upscaling():
    Image = pytest.importorskip("PIL.Image")

    from scripts import build_shion_gallery_extra as build

    out = build.fit_16x9(Image.new("RGB", (1024, 1024), "red"))
    assert out.size == (1280, 720)
    small = build.fit_16x9(Image.new("RGB", (252, 353), "red"))
    assert small.height == 353 and abs(small.width / small.height - 16 / 9) < 0.01
