"""紫苑の過去イラストをランダムに出す API（REV-488, Gemini 呼び出しなし）。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api import shion_illustration_gallery as gallery

router = APIRouter(prefix="/api/shion/illustrations", tags=["shion"])

_MEDIA_TYPES = {"webp": "image/webp", "png": "image/png", "jpg": "image/jpeg"}


@router.get("/random")
def random_illustration(mode: Literal["daily", "random"] = "daily") -> dict:
    """daily=1日1枚固定（/chat の「今日の紫苑」）、random=毎回（歌の待ち時間）。"""
    gallery.ensure_synced_in_background()
    return gallery.payload(gallery.pick(mode, gallery.list_names()))


@router.get("/file/{name}")
def illustration_file(name: str) -> FileResponse:
    path = gallery.resolve_file(name)
    if path is None:
        raise HTTPException(status_code=404, detail="illustration not found")
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES[path.suffix.lstrip(".")],
        headers={"Cache-Control": "public, max-age=86400"},
    )
