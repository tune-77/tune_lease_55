"""REV-422: 気象庁の天気予報JSON（無料・APIキー不要）から当日の天気を取得し、紫苑のチャット文脈へ1行で渡す。

出典: 気象庁ホームページ（政府標準利用規約準拠）。公式APIではないため、形式変更・取得失敗時は空文字を返し
チャットは従来通り動く。取得は1日1回（失敗時は1時間）のモジュールキャッシュで抑える。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
import urllib.request
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")
_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{area}.json"
_FAILURE_TTL_SEC = 3600
_cache: dict[str, tuple[float, str]] = {}


def _area_code() -> str:
    code = os.environ.get("SHION_WEATHER_AREA_CODE", "130000").strip()
    return code if code.isdigit() else "130000"


def _parse(data: list, today: str) -> str:
    forecast = data[0]
    series = forecast["timeSeries"]
    area = series[0]["areas"][0]
    weather = re.sub(r"\s+", " ", str(area["weathers"][0])).strip()
    if not weather:
        return ""
    line = f"{forecast.get('publishingOffice', '気象庁')}発表 {area['area']['name']}: {weather}"
    if len(series) > 2:
        temps_series = series[2]
        temps = [
            int(v)
            for t, v in zip(temps_series["timeDefines"], temps_series["areas"][0]["temps"])
            if t.startswith(today) and str(v).lstrip("-").isdigit()
        ]
        if len(set(temps)) >= 2:
            line += f"（気温 {min(temps)}〜{max(temps)}℃）"
    return line


def fetch_today_weather_line(timeout: float = 3.0) -> str:
    area = _area_code()
    today = _dt.datetime.now(_JST).strftime("%Y-%m-%d")
    key = f"{area}:{today}"
    cached = _cache.get(key)
    if cached and (cached[1] or time.time() - cached[0] < _FAILURE_TTL_SEC):
        return cached[1]
    try:
        with urllib.request.urlopen(_URL.format(area=area), timeout=timeout) as resp:
            line = _parse(json.loads(resp.read().decode("utf-8")), today)
    except Exception:
        line = ""
    _cache.clear()
    _cache[key] = (time.time(), line)
    return line


def weather_context_block() -> str:
    line = fetch_today_weather_line()
    if not line:
        return ""
    return (
        "\n\n【今日の天気（気象庁）】\n"
        f"{line}\n"
        "天気の話題は会話に自然に合う時だけ軽く触れる。ここに無い気温・体感は断定しない。"
    )
