"""REV-422: 気象庁の天気予報JSON（無料・APIキー不要）から当日の天気を取得し、紫苑のチャット文脈へ1行で渡す。

出典: 気象庁ホームページ（政府標準利用規約準拠）。公式APIではないため、形式変更・取得失敗時は空文字を返し
チャットは従来通り動く。地域はフロントが渡すその日最初の現在地（約10km単位）から最寄りの予報区を選ぶ。取得は予報区ごと1日1回（失敗時は1時間）のモジュールキャッシュで抑える。
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import time
import urllib.request
from zoneinfo import ZoneInfo

_JST = ZoneInfo("Asia/Tokyo")
_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{area}.json"
_FAILURE_TTL_SEC = 3600
_cache: dict[str, tuple[float, str]] = {}

# 気象庁 府県予報区コード → 代表地点（県庁所在地・地方気象台）の緯度経度。北海道・沖縄は予報区ごと。
_AREA_POINTS: dict[str, tuple[float, float]] = {
    "011000": (45.42, 141.67), "012000": (43.77, 142.37), "013000": (44.02, 144.27),
    "014030": (42.92, 143.20), "014100": (42.98, 144.38), "015000": (42.32, 140.97),
    "016000": (43.06, 141.35), "017000": (41.77, 140.73),
    "020000": (40.82, 140.74), "030000": (39.70, 141.15), "040000": (38.27, 140.87),
    "050000": (39.72, 140.10), "060000": (38.24, 140.36), "070000": (37.75, 140.47),
    "080000": (36.34, 140.45), "090000": (36.57, 139.88), "100000": (36.39, 139.06),
    "110000": (35.86, 139.65), "120000": (35.61, 140.12), "130000": (35.69, 139.69),
    "140000": (35.45, 139.64), "150000": (37.90, 139.02), "160000": (36.70, 137.21),
    "170000": (36.59, 136.63), "180000": (36.07, 136.22), "190000": (35.66, 138.57),
    "200000": (36.65, 138.18), "210000": (35.39, 136.72), "220000": (34.98, 138.38),
    "230000": (35.18, 136.91), "240000": (34.73, 136.51), "250000": (35.00, 135.87),
    "260000": (35.02, 135.76), "270000": (34.69, 135.52), "280000": (34.69, 135.18),
    "290000": (34.69, 135.83), "300000": (34.23, 135.17), "310000": (35.50, 134.24),
    "320000": (35.47, 133.05), "330000": (34.66, 133.93), "340000": (34.40, 132.46),
    "350000": (34.19, 131.47), "360000": (34.07, 134.56), "370000": (34.34, 134.04),
    "380000": (33.84, 132.77), "390000": (33.56, 133.53), "400000": (33.61, 130.42),
    "410000": (33.25, 130.30), "420000": (32.74, 129.87), "430000": (32.79, 130.74),
    "440000": (33.24, 131.61), "450000": (31.91, 131.42), "460100": (31.56, 130.56),
    "471000": (26.21, 127.68), "473000": (24.80, 125.28), "474000": (24.34, 124.16),
}
_MAX_AREA_DISTANCE_DEG = 3.0  # これより遠い（国外など）座標は既定地域へ
# 予報区コード先頭2桁 = 都道府県JISコード
_PREFECTURES = (
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県",
    "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県",
    "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
)


def _nearest_area(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    scale = math.cos(math.radians(lat))
    code, dist = min(
        ((c, math.hypot(lat - p[0], (lon - p[1]) * scale)) for c, p in _AREA_POINTS.items()),
        key=lambda item: item[1],
    )
    return code if dist <= _MAX_AREA_DISTANCE_DEG else None


def prefecture_from_location(lat: float | None, lon: float | None) -> str:
    """その日最初の現在地から都道府県名を返す（地域ニュース・地域経済文脈用）。不明なら空文字。"""
    code = _nearest_area(lat, lon)
    return _PREFECTURES[int(code[:2]) - 1] if code else ""


def _area_code(lat: float | None = None, lon: float | None = None) -> str:
    nearest = _nearest_area(lat, lon)
    if nearest:
        return nearest
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


def fetch_today_weather_line(lat: float | None = None, lon: float | None = None, timeout: float = 3.0) -> str:
    area = _area_code(lat, lon)
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
    for old_key in [k for k in _cache if not k.endswith(today)]:
        del _cache[old_key]
    _cache[key] = (time.time(), line)
    return line


def weather_context_block(lat: float | None = None, lon: float | None = None) -> str:
    line = fetch_today_weather_line(lat, lon)
    if not line:
        return ""
    return (
        "\n\n【今日の天気（気象庁）】\n"
        f"{line}\n"
        "天気の話題は会話に自然に合う時だけ軽く触れる。ここに無い気温・体感は断定しない。"
    )
