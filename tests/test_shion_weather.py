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
