import datetime as dt
import io
import json
from zoneinfo import ZoneInfo

import pytest

from api import shion_weather


def _payload(today: str) -> list:
    return [{
        "publishingOffice": "気象庁",
        "timeSeries": [
            {"areas": [{"area": {"name": "東京地方"}, "weathers": ["くもり　時々　晴れ"]}]},
            {"areas": []},
            {
                "timeDefines": [f"{today}T09:00:00+09:00", f"{today}T00:00:00+09:00", "2099-01-01T00:00:00+09:00"],
                "areas": [{"temps": ["24", "18", "30"]}],
            },
        ],
    }]


@pytest.fixture(autouse=True)
def _clear_cache():
    shion_weather._cache.clear()
    yield
    shion_weather._cache.clear()


def test_weather_line_includes_today_temps_and_caches(monkeypatch):
    today = dt.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps(_payload(today)).encode("utf-8"))

    monkeypatch.setattr(shion_weather.urllib.request, "urlopen", fake_urlopen)

    line = shion_weather.fetch_today_weather_line()
    assert line == "気象庁発表 東京地方: くもり 時々 晴れ（気温 18〜24℃）"
    shion_weather.fetch_today_weather_line()
    assert len(calls) == 1
    assert calls[0].endswith("/130000.json")
    assert "【今日の天気（気象庁）】" in shion_weather.weather_context_block()


def test_fetch_failure_returns_empty_block(monkeypatch):
    def boom(url, timeout):
        raise OSError("network down")

    monkeypatch.setattr(shion_weather.urllib.request, "urlopen", boom)
    assert shion_weather.weather_context_block() == ""


def test_invalid_area_code_falls_back_to_tokyo(monkeypatch):
    monkeypatch.setenv("SHION_WEATHER_AREA_CODE", "../x")
    assert shion_weather._area_code() == "130000"


@pytest.mark.parametrize(
    ("lat", "lon", "code"),
    [
        (35.4, 139.6, "140000"),  # 横浜 → 神奈川
        (34.7, 135.5, "270000"),  # 大阪
        (41.8, 140.7, "017000"),  # 函館 → 渡島・檜山
        (26.2, 127.7, "471000"),  # 那覇
    ],
)
def test_area_code_picks_nearest_forecast_area(lat, lon, code):
    assert shion_weather._area_code(lat, lon) == code


def test_area_code_outside_japan_falls_back(monkeypatch):
    monkeypatch.delenv("SHION_WEATHER_AREA_CODE", raising=False)
    assert shion_weather._area_code(48.9, 2.3) == "130000"  # パリ
    assert shion_weather._area_code(float("nan"), 139.0) == "130000"


def test_location_is_used_for_fetch_url(monkeypatch):
    today = dt.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps(_payload(today)).encode("utf-8"))

    monkeypatch.setattr(shion_weather.urllib.request, "urlopen", fake_urlopen)
    shion_weather.weather_context_block(34.7, 135.5)
    assert calls[0].endswith("/270000.json")


def test_prefecture_from_location():
    assert shion_weather.prefecture_from_location(35.4, 139.6) == "神奈川県"
    assert shion_weather.prefecture_from_location(41.8, 140.7) == "北海道"
    assert shion_weather.prefecture_from_location(31.6, 130.6) == "鹿児島県"
    assert shion_weather.prefecture_from_location(26.2, 127.7) == "沖縄県"
    assert shion_weather.prefecture_from_location(48.9, 2.3) == ""
    assert shion_weather.prefecture_from_location(None, None) == ""


def test_every_area_code_maps_to_matching_prefecture():
    for code, (lat, lon) in shion_weather._AREA_POINTS.items():
        pref = shion_weather.prefecture_from_location(lat, lon)
        assert pref == shion_weather._PREFECTURES[int(code[:2]) - 1]
