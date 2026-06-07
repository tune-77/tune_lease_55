#!/usr/bin/env python3
"""
e-Stat API から業種別営業利益率を取得し industry_benchmarks.json を動的更新する。

環境変数 ESTAT_APP_ID が未設定の場合はスキップ（graceful skip）。
取得したデータは static_data/industry_estat_cache.json にキャッシュ保存。

e-Stat API 公式: https://www.e-stat.go.jp/api/
使用統計: 法人企業統計調査（財務省）statsDataId = 0003084108
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import urlencode

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static_data"
BENCHMARKS_PATH = STATIC_DIR / "industry_benchmarks.json"
CACHE_PATH = STATIC_DIR / "industry_estat_cache.json"

# e-Stat API エンドポイント
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

# 法人企業統計（財務省）- 業種別売上高・営業利益
# statsDataId は「法人企業統計調査 > 年報 > 産業別・規模別集計」
STATS_DATA_IDS = {
    "法人企業統計_年報": "0003084108",
}

# e-Stat 業種コード → industry_benchmarks.json キーのマッピング
# e-Stat の産業分類コード（日本標準産業分類）→ benchmark キー
INDUSTRY_CODE_MAP: dict[str, str] = {
    "D": "09 食料品製造業",        # 製造業（食料品）
    "E06": "21 金属製品製造業",     # 金属製品製造業
    "E07": "24 生産用機械器具製造業",
    "E08": "26 情報通信機械器具製造業",
    "F": "06 総合工事業",           # 建設業（大分類で近似）
    "G": "50-55 各種卸売業",        # 卸売業
    "H": "56-61 各種小売業",        # 小売業
    "I": "44 道路貨物運送業",       # 運輸業（近似）
    "J": "76 飲食店",               # 宿泊業・飲食サービス業
    "K": "68 不動産代理・仲介",     # 不動産業
    "L": "70 物品賃貸業(リース・レンタル)",  # 物品賃貸業
    "N": "91 職業紹介・労働者派遣業",       # サービス業（生活関連）
    "P": "83 医療業(病院・診療所)",         # 医療・福祉
    "Q": "85 社会保険・社会福祉・介護事業",
}


def _get_app_id() -> Optional[str]:
    return os.environ.get("ESTAT_APP_ID")


def _fetch_json(url: str, params: dict, retries: int = 3) -> Optional[dict]:
    query = urlencode(params)
    full_url = f"{url}?{query}"
    for attempt in range(retries):
        try:
            with urlopen(full_url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            print(f"  警告: API呼び出し失敗 (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_corporate_enterprise_stats(app_id: str) -> Optional[dict]:
    """法人企業統計から業種別営業利益率を取得する。"""
    params = {
        "appId": app_id,
        "statsDataId": STATS_DATA_IDS["法人企業統計_年報"],
        "metaGetFlg": "Y",
        "cntGetFlg": "N",
        "sectionHeaderFlg": "1",
    }
    print(f"  e-Stat API: 法人企業統計取得中 (statsDataId={params['statsDataId']})...")
    data = _fetch_json(ESTAT_BASE_URL, params)
    if data is None:
        return None

    status = data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("STATUS", -1)
    if status != 0:
        msg = data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("ERROR_MSG", "不明なエラー")
        print(f"  警告: e-Stat API エラー (status={status}): {msg}")
        return None

    return data


def _parse_op_margins(raw: dict) -> dict[str, float]:
    """APIレスポンスから業種コード→営業利益率(%)を抽出する。

    法人企業統計の VALUE 配列:
      @cat01 = 業種コード, @cat02 = 指標コード, @time = 年次, $ = 値
    指標コード 010 = 売上高, 020 = 営業利益（概ね）
    """
    margins: dict[str, float] = {}
    try:
        values = (
            raw["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
        )
    except KeyError:
        return margins

    # 最新年次の売上高と営業利益をまとめる
    sales: dict[str, dict[str, float]] = {}    # {industry_code: {year: value}}
    op_profit: dict[str, dict[str, float]] = {}

    for v in values:
        cat01 = v.get("@cat01", "")   # 業種
        cat02 = v.get("@cat02", "")   # 指標
        year = v.get("@time", "")
        raw_val = v.get("$", "")
        if not raw_val or raw_val in ("-", "…", "x", "X"):
            continue
        try:
            val = float(str(raw_val).replace(",", ""))
        except ValueError:
            continue

        if cat02 in ("010", "010_1", "0100"):  # 売上高
            sales.setdefault(cat01, {})[year] = val
        elif cat02 in ("020", "020_1", "0200"):  # 営業利益
            op_profit.setdefault(cat01, {})[year] = val

    for code in set(sales) & set(op_profit):
        latest_year = max(set(sales[code]) & set(op_profit[code]), default=None)
        if latest_year is None:
            continue
        s = sales[code][latest_year]
        p = op_profit[code][latest_year]
        if s > 0:
            margins[code] = round(p / s * 100, 2)

    return margins


def update_benchmarks(margins: dict[str, float]) -> dict[str, str]:
    """industry_benchmarks.json の op_margin を更新し、変更サマリを返す。"""
    with open(BENCHMARKS_PATH, encoding="utf-8") as f:
        benchmarks: dict = json.load(f)

    changes: dict[str, str] = {}
    for estat_code, bench_key in INDUSTRY_CODE_MAP.items():
        if estat_code not in margins:
            continue
        new_margin = margins[estat_code]
        if bench_key not in benchmarks:
            continue
        old_margin = benchmarks[bench_key].get("op_margin")
        if old_margin != new_margin:
            changes[bench_key] = f"{old_margin} → {new_margin}"
            benchmarks[bench_key]["op_margin"] = new_margin
            benchmarks[bench_key]["op_margin_source"] = "e-Stat法人企業統計"
            benchmarks[bench_key]["op_margin_updated"] = datetime.now().strftime("%Y-%m-%d")

    with open(BENCHMARKS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, ensure_ascii=False, indent=4)

    return changes


def save_cache(raw: dict, margins: dict[str, float]) -> None:
    cache = {
        "fetched_at": datetime.now().isoformat(),
        "source": "e-Stat法人企業統計",
        "stats_data_id": STATS_DATA_IDS["法人企業統計_年報"],
        "op_margins_by_estat_code": margins,
        "raw_summary": {
            "total_values": len(
                raw.get("GET_STATS_DATA", {})
                .get("STATISTICAL_DATA", {})
                .get("DATA_INF", {})
                .get("VALUE", [])
            )
        },
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  キャッシュ保存: {CACHE_PATH}")


def main() -> int:
    app_id = _get_app_id()
    if not app_id:
        print("[fetch_estat_industry] ESTAT_APP_ID 未設定 → スキップ")
        return 0

    print("[fetch_estat_industry] e-Stat API 業種別統計更新開始...")

    raw = fetch_corporate_enterprise_stats(app_id)
    if raw is None:
        print("[fetch_estat_industry] データ取得失敗 → スキップ（既存データを維持）")
        return 0

    margins = _parse_op_margins(raw)
    print(f"  取得した業種別マージン数: {len(margins)}")

    save_cache(raw, margins)

    if not margins:
        print("[fetch_estat_industry] 解析可能なマージンデータなし → benchmarks 更新スキップ")
        return 0

    changes = update_benchmarks(margins)
    if changes:
        print(f"  industry_benchmarks.json 更新: {len(changes)} 業種")
        for k, v in changes.items():
            print(f"    {k}: op_margin {v}")
    else:
        print("  変更なし（既存値と同一）")

    print("[fetch_estat_industry] 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
