"""紫苑の過去イラストの集約と配信（REV-488, Gemini 呼び出しなし）。

毎朝のイラスト生成は止めたまま（SHION_DAILY_ILLUSTRATION_ENABLED 既定オフ）、これまでに作った
イラストからランダムに1枚を出す。

- 出どころ: frontend/public/lease-grumble/（直近30日）と、Obsidian の
  Projects/tune_lease_55/Archive/Lease Grumble/Images/YYYY/MM/（30日で移したもの）
- iCloud 上のアーカイブを要求ごとに読むと同期中に止まり得るので、裏のスレッドで
  data/lease_grumble_gallery/ へ写してから、そこ（ローカル）だけを配信する
- next start はビルド後に public へ足したファイルを配信しない（404）ため、画像は
  public ではなく API（/api/shion/illustrations/file/...）で出す
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import shutil
import threading
import time
from pathlib import Path

from ai_runtime_client import _MAIN_ROOT

NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(webp|png|jpg)$")
GALLERY_DIR = _MAIN_ROOT / "data" / "lease_grumble_gallery"
PUBLIC_DIR = _MAIN_ROOT / "frontend" / "public" / "lease-grumble"
ARCHIVE_PARTS = ("Projects", "tune_lease_55", "Archive", "Lease Grumble", "Images")
RESYNC_AFTER_S = 6 * 3600
FILE_URL_PREFIX = "/api/shion/illustrations/file/"

_sync_lock = threading.Lock()
_last_sync = {"at": 0.0}


def _archive_dir() -> Path | None:
    try:
        from lease_news_digest import find_vault

        vault = find_vault()
    except Exception:
        return None
    return Path(vault).joinpath(*ARCHIVE_PARTS) if vault else None


def sync_gallery(gallery_dir: Path | None = None, sources: list[Path] | None = None) -> int:
    """public とアーカイブの日付付き画像をギャラリーへ写す（既にある同サイズは飛ばす）。写した枚数を返す。"""
    gallery = gallery_dir or GALLERY_DIR
    if sources is None:
        sources = [PUBLIC_DIR]
        archive = _archive_dir()
        if archive is not None:
            sources.append(archive)
    copied = 0
    gallery.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not NAME_RE.match(path.name):  # iCloud の未ダウンロード（*.icloud）も除外される
                continue
            target = gallery / path.name
            try:
                if target.exists() and target.stat().st_size == path.stat().st_size:
                    continue
                shutil.copy2(path, target)
                copied += 1
            except OSError:
                continue
    return copied


def ensure_synced_in_background() -> None:
    """最後の同期から6時間以上たっていれば裏で同期する（要求は待たせない）。"""
    if time.monotonic() - _last_sync["at"] < RESYNC_AFTER_S and _last_sync["at"]:
        return
    if not _sync_lock.acquire(blocking=False):
        return
    _last_sync["at"] = time.monotonic()

    def run() -> None:
        try:
            sync_gallery()
        finally:
            _sync_lock.release()

    threading.Thread(target=run, name="shion-illustration-sync", daemon=True).start()


def list_names(gallery_dir: Path | None = None, public_dir: Path | None = None) -> list[str]:
    names = set()
    for directory in (gallery_dir or GALLERY_DIR, public_dir or PUBLIC_DIR):
        if directory.is_dir():
            names.update(p.name for p in directory.iterdir() if NAME_RE.match(p.name))
    return sorted(names)


def resolve_file(name: str, gallery_dir: Path | None = None, public_dir: Path | None = None) -> Path | None:
    """配信してよいローカルのファイル。名前は日付形式だけを受け付ける（パス指定は不可）。"""
    if not NAME_RE.match(name or ""):
        return None
    for directory in (gallery_dir or GALLERY_DIR, public_dir or PUBLIC_DIR):
        path = directory / name
        if path.is_file():
            return path
    return None


def pick(mode: str, names: list[str], today: dt.date | None = None) -> str | None:
    """daily: 日付から決まる1日1枚（同じ日は同じ絵）。random: 毎回ランダム。"""
    if not names:
        return None
    if mode == "daily":
        day = (today or dt.date.today()).isoformat()
        index = int(hashlib.sha256(f"shion-daily-{day}".encode()).hexdigest(), 16) % len(names)
        return names[index]
    return random.choice(names)


CAPTIONS_FILE = "captions.json"


def _captions(gallery_dir: Path | None = None) -> dict[str, str]:
    try:
        data = json.loads(((gallery_dir or GALLERY_DIR) / CAPTIONS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_caption(name: str, caption: str, gallery_dir: Path | None = None) -> None:
    """週1回の生成（REV-490）で、題材の一言を画像と一緒に残す。"""
    directory = gallery_dir or GALLERY_DIR
    captions = _captions(directory)
    captions[name] = str(caption or "")[:40]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CAPTIONS_FILE).write_text(json.dumps(captions, ensure_ascii=False, indent=1), encoding="utf-8")


def payload(name: str | None, gallery_dir: Path | None = None) -> dict:
    if not name:
        return {"available": False}
    result = {"available": True, "date": NAME_RE.match(name).group(1), "url": FILE_URL_PREFIX + name}
    caption = _captions(gallery_dir).get(name)
    if caption:
        result["caption"] = caption
    return result


def public_url_to_api(url: str) -> str:
    """観察ログの /lease-grumble/<日付>.webp を、実在すれば配信 API の URL に置き換える（なければ空）。"""
    name = str(url or "").rsplit("/", 1)[-1]
    if not url or not str(url).startswith("/lease-grumble/"):
        return url or ""
    return FILE_URL_PREFIX + name if resolve_file(name) else ""
