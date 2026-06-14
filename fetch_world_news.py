"""外部情報（日銀・リース業界・経済指標）を取得して data/world_feed.jsonl に蓄積する。

取得できないソースは graceful degradation（エラーにしない）。
Usage: python fetch_world_news.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

_FEED_PATH = Path(__file__).parent / "data" / "world_feed.jsonl"
_TIMEOUT = 15
_MAX_FEED_LINES = 500  # ファイルが膨らみすぎないよう上限を設ける


def _write_entries(entries: list[dict]) -> int:
    if not entries:
        return 0
    _FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FEED_PATH.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def _trim_feed() -> None:
    """ファイルが _MAX_FEED_LINES を超えたら古い行を削除する。"""
    if not _FEED_PATH.exists():
        return
    lines = _FEED_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) > _MAX_FEED_LINES:
        _FEED_PATH.write_text("\n".join(lines[-_MAX_FEED_LINES:]) + "\n", encoding="utf-8")


def fetch_boj_rss() -> list[dict]:
    """日銀プレスリリースRSSから最新ニュースを取得する。"""
    url = "https://www.boj.or.jp/rss/index.rss"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tune_lease_55/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if not title:
                continue
            items.append({
                "source": "boj_rss",
                "title": title,
                "description": desc[:300],
                "url": link,
                "published": pub,
            })
        return items[:5]
    except Exception as exc:
        print(f"[fetch_world_news] BOJ RSS 取得失敗（スキップ）: {exc}", file=sys.stderr)
        return []


def fetch_leasing_assoc() -> list[dict]:
    """リース事業協会トップページからニュースらしきテキストを抽出する。"""
    url = "https://www.leasing.or.jp/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tune_lease_55/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        import re
        # <a>タグ内のテキストでニュースっぽい文字列を抽出
        candidates = re.findall(r'<a[^>]*>([^<]{12,100})</a>', html)
        items = []
        seen: set[str] = set()
        for text in candidates:
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 12 or text in seen:
                continue
            # ナビゲーションっぽい短い語は除外
            if re.match(r'^(トップ|ホーム|メニュー|検索|お問い合わせ|サイトマップ)$', text):
                continue
            seen.add(text)
            items.append({
                "source": "leasing_assoc",
                "title": text[:120],
                "description": "",
                "url": url,
                "published": "",
            })
        return items[:5]
    except Exception as exc:
        print(f"[fetch_world_news] リース協会 取得失敗（スキップ）: {exc}", file=sys.stderr)
        return []


def fetch_estat_gdp() -> list[dict]:
    """e-Stat APIからGDP統計データを取得する（ESTAT_API_KEY が必要）。"""
    api_key = os.environ.get("ESTAT_API_KEY", "").strip()
    if not api_key:
        return []
    # 内閣府「国民経済計算（GDP統計）」四半期速報 statsDataId 例
    stats_data_id = os.environ.get("ESTAT_GDP_STATS_ID", "0003109842")
    url = (
        "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
        f"?appId={api_key}&statsDataId={stats_data_id}&metaGetFlg=N&cntGetFlg=N&limit=3"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tune_lease_55/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        values = (
            data.get("GET_STATS_DATA", {})
                .get("STATISTICAL_DATA", {})
                .get("DATA_INF", {})
                .get("VALUE", [])
        )
        items = []
        for v in (values or [])[:3]:
            time_key = v.get("@time", "")
            val = v.get("$", "")
            items.append({
                "source": "estat_gdp",
                "title": f"GDP速報 時点:{time_key} 値:{val}",
                "description": json.dumps(v, ensure_ascii=False)[:200],
                "url": "https://www.e-stat.go.jp/",
                "published": "",
            })
        return items
    except Exception as exc:
        print(f"[fetch_world_news] e-Stat GDP 取得失敗（スキップ）: {exc}", file=sys.stderr)
        return []


def fetch_mof_bond() -> list[dict]:
    """財務省・国債金利情報ページから金利関連テキストを取得する。"""
    url = "https://www.mof.go.jp/jgbs/reference/interest_rate/index.htm"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tune_lease_55/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        import re
        # テーブルの数値行を粗く抽出
        rows = re.findall(r'<td[^>]*>([0-9.\-]{1,12})</td>', html)
        if rows:
            snippet = "長期金利参考値: " + " / ".join(rows[:6])
            return [{
                "source": "mof_bond",
                "title": snippet[:120],
                "description": "財務省国債金利情報（粗抽出）",
                "url": url,
                "published": "",
            }]
    except Exception as exc:
        print(f"[fetch_world_news] 財務省金利 取得失敗（スキップ）: {exc}", file=sys.stderr)
    return []


def main() -> None:
    fetched_at = dt.datetime.now().isoformat(timespec="seconds")
    results: list[tuple[str, list[dict]]] = [
        ("BOJ RSS", fetch_boj_rss()),
        ("リース事業協会", fetch_leasing_assoc()),
        ("e-Stat GDP", fetch_estat_gdp()),
        ("財務省金利", fetch_mof_bond()),
    ]
    all_entries = []
    for label, entries in results:
        for e in entries:
            e["fetched_at"] = fetched_at
        all_entries.extend(entries)
        print(f"  {label}: {len(entries)}件")

    count = _write_entries(all_entries)
    _trim_feed()
    print(f"[fetch_world_news] 計{count}件を {_FEED_PATH} に追記しました。")


if __name__ == "__main__":
    main()
